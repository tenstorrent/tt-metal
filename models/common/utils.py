# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor

# Backward compatibility: filter_none moved to models.common.sampling._utils
from models.common.sampling._utils import filter_none  # noqa: F401

# Backward compatibility: LogProbsCalculator moved to models.common.sampling.tt_log_probs
from models.common.sampling.tt_log_probs import LogProbsCalculator  # noqa: F401


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def blockcyclic_positions(sp: int, chunk_size_global: int, seq_len_cache: int) -> torch.Tensor:
    """Global natural position held by each block-cyclic shard row (device-major: an SP-contiguous
    split of the cache's seq dim yields each chip's rows).

    Shard row r -> chip c = r // seq_len_local, local row lr = r % seq_len_local, and that row holds
    global position (lr // chunk_local) * chunk_size_global + c * chunk_local + (lr % chunk_local) --
    the inverse of the update_padded_kv_cache writer. Returns a [seq_len_cache] index tensor.
    """
    seq_len_local = seq_len_cache // sp
    chunk_local = chunk_size_global // sp
    c = torch.arange(sp).repeat_interleave(seq_len_local)
    lr = torch.arange(seq_len_local).repeat(sp)
    slab, off = lr // chunk_local, lr % chunk_local
    return slab * chunk_size_global + c * chunk_local + off


def rotated_chunk_positions(chunk_start: int, sp: int, chunk_local: int) -> list[list[int]]:
    """Global position of each SP chip's local row for the chunk [chunk_start, chunk_start + sp*chunk_local).

    positions[c][r] is what update_padded_kv_cache writes chip c's r-th input row to: chip c owns the
    chunk_local-token blocks c, c+sp, ... of the global sequence, so a chunk that starts mid-slab rotates
    which chip holds its first block, and the boundary chip's rows run from the tail of one of its blocks
    into the head of its next one. Positions increase with r on every chip. Chunk-aligned starts reduce to
    chunk_start + c*chunk_local + r.
    """
    chunk_global = sp * chunk_local
    slab, boundary_chip, offset = (
        chunk_start // chunk_global,
        (chunk_start // chunk_local) % sp,
        chunk_start % chunk_local,
    )
    positions = []
    for c in range(sp):
        first_row = (slab + (c < boundary_chip)) * chunk_local + (offset if c == boundary_chip else 0)
        rows = range(first_row, first_row + chunk_local)
        positions.append([(lr // chunk_local) * chunk_global + c * chunk_local + lr % chunk_local for lr in rows])
    return positions


def rotate_chunk_tokens(chunk_token_ids: list, chunk_start: int, sp: int) -> list:
    """Reorder one chunk's natural-order tokens into the chip-major order a plain SP shard hands each chip.

    Chip c's shard receives the tokens at rotated_chunk_positions(chunk_start)[c], so the KV writer and the
    indexed RoPE (both keyed off chunk_start) see each token at its true position. This is the host-side
    reshuffle the engine's H2D prefill connector applies (tt-llm-engine ring_sdpa_reshuffle); identity for a
    chunk-aligned chunk_start.
    """
    assert len(chunk_token_ids) % sp == 0, f"chunk of {len(chunk_token_ids)} tokens does not split over sp={sp}"
    chunk_local = len(chunk_token_ids) // sp
    return [
        chunk_token_ids[p - chunk_start] for row in rotated_chunk_positions(chunk_start, sp, chunk_local) for p in row
    ]


def rotated_chunk_real_counts(chunk_start: int, actual_isl: int, sp: int, chunk_local: int) -> list[int]:
    """Per-chip count of real (non-pad) rows of a rotated chunk holding `actual_isl` real tokens.

    Positions increase with the local row, so each chip's real rows are a prefix of its rows (right padding).
    Reduces to min(chunk_local, actual_isl - c*chunk_local) for a chunk-aligned chunk_start.
    """
    end = chunk_start + actual_isl
    return [sum(p < end for p in row) for row in rotated_chunk_positions(chunk_start, sp, chunk_local)]


def block_cyclic_reorder(matrix: torch.Tensor, chunk_local: int, sp_factor: int, seq_dim: int = 2) -> torch.Tensor:
    """Reorder a [.., seq, ..] matrix into block-cyclic order keyed by `chunk_local`.

    Splits the sequence into blocks of `chunk_local` rows and concatenates them so that device c's
    contiguous shard (after a plain SP shard over `seq_dim`) holds blocks c, c+sp, c+2sp, ... — the
    same block-cyclic layout the per-chip KV cache writes into. This makes the indexed-RoPE op's
    contiguous, `update_idxt`-offset read of each device's cos/sin shard land on the right global
    positions, including the boundary chip's older-then-wrap rows.
    """
    seq_len = matrix.shape[seq_dim]
    assert seq_len % chunk_local == 0, f"seq_len {seq_len} must be a multiple of chunk_local {chunk_local}"
    num_blocks = seq_len // chunk_local
    assert num_blocks % sp_factor == 0, f"num_blocks {num_blocks} must be a multiple of sp_factor {sp_factor}"
    blocks = list(torch.split(matrix, chunk_local, dim=seq_dim))
    order = [b for c in range(sp_factor) for b in range(c, num_blocks, sp_factor)]
    return torch.cat([blocks[b] for b in order], dim=seq_dim)
