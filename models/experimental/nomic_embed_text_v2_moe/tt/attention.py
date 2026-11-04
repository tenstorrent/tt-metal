# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional MHA, the TTNN form of reference.NomicBertAttention.

    x (B, 1, S, H)
      Wqkv                    -> (B, 1, S, 3H)   three-major: [q | k | v], q and k head dims pair-permuted
      nlp_create_qkv_heads    -> three (B, A, S, D)
      rotary_embedding_llama  -> q and k rotated, v untouched
      SDPA, is_causal=False   -> (B, 1, S, H)    heads concatenated: output_concat_heads=True
      out_proj                -> (B, 1, S, H)

The rotary tables and the additive mask come from the caller, not from here: both depend only on
S, and building them per block would repeat the same host work 12 times.
"""

from __future__ import annotations

from functools import cache
from typing import Optional

import torch

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    FUSED_ATTENTION_MAX_TILES,
    FUSED_ATTENTION_MIN_TILES,
    m_tiles,
    rotary_pair_permutation,
    rotary_rotation_matrix,
    sharded_norm_layout,
    table_tensors,
    to_device,
    transpose_linear_weight,
)
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import (
    dense_linear,
    l1_bank_bytes,
    l1_floor,
    prepare_grouped_projection,
)
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup

# The largest fp32 score block, in tiles, that fit L1 beside the rest of SDPA's buffers.
_FP32_SCORE_TILES = 144

# rotary_embedding_llama splits prefill work over batch and sequence tiles, not heads, so (B, A, S, D)
# keeps at most B * ceil(S / 32) cores busy. Below this many, rotary folds the heads into the batch axis.
_ROTARY_FOLD_TILES = 96

# A core whose rows (its sequence tiles x its heads) number at most _ROTARY_FAST_ROWS runs rotary's
# faster path: it reads its cos/sin rows once for all of its heads, where above that it reads them
# again for every row. Up to _ROTARY_VIEW_SEQ_TILES sequence tiles, from _ROTARY_VIEW_MIN_TILES tile
# rows of M (1024 tokens) up, heads folded into the batch so a core keeps a few of them take that path
# on more cores (rotary_heads_per_view).
_ROTARY_FAST_ROWS = 8
_ROTARY_VIEW_SEQ_TILES = 4
_ROTARY_VIEW_MIN_TILES = 32
_ROTARY_HEAD_SPLIT_IDLE_RATIO = 4


@cache
def rotary_heads_per_view(batch: int, seq_tiles: int, heads: int, cores: int) -> int:
    """How many heads each batch entry of rotary's (B * A / k, k, S, D) view keeps, k a divisor of A.

    rotary_embedding_llama gives each core a share of the view's batch entries and sequence tiles, and
    of its k heads only when that leaves at least _ROTARY_HEAD_SPLIT_IDLE_RATIO times as many cores idle
    (its multi-core program factory). Up to _ROTARY_VIEW_SEQ_TILES sequence tiles the largest k whose
    cores keep at most _ROTARY_FAST_ROWS rows is taken: 3.9x faster than the native view at 256x32 (169
    -> 43 us a call), 2.3x at 128x64, 1.6x at 64x128, 1.5x at 85x96. Longer sequences keep the native
    view, A heads an entry, and inputs under 1024 tokens the small-input rule, one head an entry below
    _ROTARY_FOLD_TILES tile rows. Measured over every view at 25 shapes from 64x32 to 16x512, the pick
    is within 6% of the fastest at each. Every row is rotated alike in any view, so the output is the
    same bit for bit.
    """
    if seq_tiles > _ROTARY_VIEW_SEQ_TILES or batch * seq_tiles < _ROTARY_VIEW_MIN_TILES:
        return 1 if batch * seq_tiles < _ROTARY_FOLD_TILES else heads
    for kept in sorted((k for k in range(1, heads + 1) if heads % k == 0), reverse=True):
        entries = batch * heads // kept
        batch_cores = min(entries, cores)
        seq_cores = min(cores // batch_cores, seq_tiles)
        idle_ratio = cores // (batch_cores * seq_cores)
        head_cores = min(kept, idle_ratio) if idle_ratio >= _ROTARY_HEAD_SPLIT_IDLE_RATIO else 1
        rows = ttnn.core.divup(seq_tiles, seq_cores) * ttnn.core.divup(kept, head_cores)
        if rows <= _ROTARY_FAST_ROWS:
            return kept
    return 1


def _round_up_to_tile(n: int) -> int:
    return ttnn.core.divup(n, ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _balanced_chunk(length: int, limit: int) -> int:
    """The tile-multiple chunk covering `length` in the fewest chunks of at most `limit`, split evenly."""
    return _round_up_to_tile(ttnn.core.divup(length, ttnn.core.divup(length, limit)))


# SDPA's time per round of query chunks, in key-chunk-equivalents: a chunk of q queries against a
# whole sequence of S keys in chunks of k costs about q * (1 + _KEY_STEP_COST * S / k) + _CHUNK_COST.
# Fitted to the fp32 kernel with q, k, v in L1 at 15 shapes from 4x512 to 64x128 (2026-10-27): with the
# query chunk a whole sequence or half of one at 10 or more sequences, the cheapest pair by this estimate
# ran within 3% of the fastest pair measured at every shape.
_KEY_STEP_COST = 0.5
_CHUNK_COST = 128

# Below this many heads a call has more cores than (sequence, head) pairs, and smaller query chunks, up
# to a sequence in eight, spread it; at or above it a head runs as one query chunk or two.
_MANY_HEADS = 110


def _query_chunks_ok(batch: int, chunks: int) -> bool:
    """Three query chunks a head return wrong rows at 10 or more sequences (and two or four do not), so
    an odd count above one is left out there."""
    return chunks == 1 or chunks % 2 == 0 or batch < 10


@cache
def sdpa_chunk_candidates(
    batch: int, seqlen: int, heads: int, cores: int, fp32_dest_acc: bool, max_query_chunks: int = 0
) -> tuple[tuple[int, int], ...]:
    """(q_chunk, k_chunk) pairs for one (B, A, S, D) call, cheapest estimated time first.

    SDPA hands each core a contiguous run of the B * A * ceil(S / q_chunk) query chunks, so the time
    is the rounds of query chunks a core runs times the cost of one (_KEY_STEP_COST, _CHUNK_COST). A
    query chunk is a sequence, half of one or, when there are fewer heads than _MANY_HEADS, down to an
    eighth (max_query_chunks overrides the limit: the placement's L1 fallback splits into four). A key
    chunk is the sequence or an even split of at most 384, 256, 192 or 128 tokens; with fp32
    accumulation the kernel computes the padded keys of its last chunk, so they split evenly there,
    and its fp32 score block, q x k, fits L1 only up to 144 tiles (384 x 384).
    Ties go to the larger key chunk, then the larger query chunk.
    """
    whole = _round_up_to_tile(seqlen)
    tiles = whole // ttnn.TILE_SIZE
    most = max_query_chunks or (2 if batch * heads >= _MANY_HEADS else 8)
    queries = set()
    for split in (1, 2, 4, 8):
        if split <= min(most, tiles):
            q_chunk = _balanced_chunk(whole, ttnn.core.divup(whole, split))
            if _query_chunks_ok(batch, ttnn.core.divup(whole, q_chunk)):
                queries.add(q_chunk)
    keys = {
        _balanced_chunk(whole, limit) if fp32_dest_acc else min(whole, limit) for limit in (whole, 384, 256, 192, 128)
    }
    candidates = []
    for q_chunk in queries:
        rounds = ttnn.core.divup(batch * heads * ttnn.core.divup(whole, q_chunk), cores)
        for k_chunk in keys:
            if fp32_dest_acc and (q_chunk // ttnn.TILE_SIZE) * (k_chunk // ttnn.TILE_SIZE) > _FP32_SCORE_TILES:
                continue
            cost = rounds * (q_chunk * (1 + _KEY_STEP_COST * whole / k_chunk) + _CHUNK_COST)
            candidates.append((cost, -k_chunk, -q_chunk))
    return tuple((-q, -k) for _, k, q in sorted(candidates))


@cache
def sdpa_program_config(
    batch: int, seqlen: int, heads: int, grid: ttnn.CoreCoord, fp32_dest_acc: bool, masked: bool
) -> ttnn.SDPAProgramConfig:
    """The cheapest chunk pair of sdpa_chunk_candidates, for q, k and v in DRAM.

    On the bfloat16 kernel a mask adds a double-buffered block of the score's size, which leaves
    512 x 512 unable to allocate, and with a mask every shape measured ran fastest, or within 7%, at a
    key chunk of at most 256; so a masked call there takes the cheapest pair within that.
    """
    for q_chunk, k_chunk in sdpa_chunk_candidates(batch, seqlen, heads, grid.x * grid.y, fp32_dest_acc):
        if fp32_dest_acc or not masked or k_chunk <= 256:
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
            )
    raise ValueError(f"no SDPA chunk pair for {batch}x{seqlen}")


def sdpa_circular_buffer_bytes(
    seqlen: int,
    q_chunk: int,
    k_chunk: int,
    head_dim: int,
    fp32_dest_acc: bool,
    mask_dtype: ttnn.DataType | None,
    double_buffered_q: bool,
) -> int:
    """SDPA's circular-buffer bytes per core, as its program factory allocates them for bf16 q, k, v.

    A padded sequence without a mask gets a generated one: a single tile on the bfloat16 kernel,
    a full double-buffered block of bfloat4_b on the fp32 one. Against the two regions the
    allocator reported, 8x512 at 512 x 512 on the bfloat16 kernel and 8x450 at 256 x 480 on the
    fp32 one, this puts both regions' base at 110 KB to within 2 KB.
    """
    tile = ttnn.tile_size(ttnn.bfloat16)
    wide = ttnn.tile_size(ttnn.float32) if fp32_dest_acc else tile
    q_tiles, k_tiles, d_tiles = q_chunk // ttnn.TILE_SIZE, k_chunk // ttnn.TILE_SIZE, head_dim // ttnn.TILE_SIZE
    total = q_tiles * d_tiles * tile * (2 if double_buffered_q else 1)
    total += 2 * 2 * k_tiles * d_tiles * tile  # k and v, double-buffered
    total += q_tiles * k_tiles * wide  # the score block
    total += 2 * q_tiles * d_tiles * tile  # the two running outputs
    total += 3 * q_tiles * tile + 2 * q_tiles * wide  # the running maxima and their difference, the sums
    total += (q_tiles * d_tiles if fp32_dest_acc else 8) * tile  # the output block, a ping-pong when streaming
    total += 3 * tile  # the scalars
    padded = seqlen % q_chunk or seqlen % k_chunk
    if mask_dtype is not None:
        total += 2 * q_tiles * k_tiles * ttnn.tile_size(mask_dtype)
    elif padded:
        total += 2 * q_tiles * k_tiles * ttnn.tile_size(ttnn.bfloat4_b) if fp32_dest_acc else 2 * tile
    return total


# How far below the deepest tensor live before the head split (the residual, QKV's output) the
# deepest head tensor reaches at SDPA, in head tensors (one of q, k or v). L1 tensors are allocated
# from the top down, each in the highest gap it fits. The split writes into outputs allocated v
# first, then k, then q (TtNomicBertAttention.forward), so with QKV's output in L1 v sits right below
# it, the rotated q and k take the space QKV's output frees, and SDPA's output what is left of that:
# 1 head tensor. QKV's output takes ceil(3h / banks) tiles a bank against a head tensor's
# ceil(h / banks), so the rest can be short of a head tensor, and SDPA's output then goes below v: 2.
# With QKV's output in DRAM, v, k and q go below the residual, the rotated q and k below them, and
# SDPA's output where k was: 5.
_HEAD_TENSOR_SPAN_QKV_IN_DRAM = 5


def head_tensor_span(qkv_bank_bytes: int, head_bank: int) -> int:
    """Head tensors the heads reach below the deepest tensor live before the split; qkv_bank_bytes is
    QKV's output's share of a bank, 0 when it is in DRAM."""
    if not qkv_bank_bytes:
        return _HEAD_TENSOR_SPAN_QKV_IN_DRAM
    return 1 if qkv_bank_bytes - 2 * head_bank >= head_bank else 2


# Head room below the circular-buffer limit: the region started 6 KiB above the reported limit.
_L1_MARGIN_BYTES = 16 * 1024


def _first_fitting_pair(
    batch: int,
    seqlen: int,
    heads: int,
    head_dim: int,
    cores: int,
    fp32_dest_acc: bool,
    mask_dtype: ttnn.DataType | None,
    budget: int,
) -> tuple[int, int] | None:
    """The cheapest pair of sdpa_chunk_candidates, then of the four-way query splits, whose buffers take
    at most budget bytes a core."""
    candidates = sdpa_chunk_candidates(batch, seqlen, heads, cores, fp32_dest_acc)
    split = sdpa_chunk_candidates(batch, seqlen, heads, cores, fp32_dest_acc, max_query_chunks=4)
    for q_chunk, k_chunk in (*candidates, *(pair for pair in split if pair not in candidates)):
        double_buffered_q = batch * heads * ttnn.core.divup(seqlen, q_chunk) > cores
        footprint = sdpa_circular_buffer_bytes(
            seqlen, q_chunk, k_chunk, head_dim, fp32_dest_acc, mask_dtype, double_buffered_q
        )
        if footprint <= budget:
            return q_chunk, k_chunk
    return None


@cache
def sdpa_chunks_in_l1(
    batch: int,
    seqlen: int,
    heads: int,
    head_dim: int,
    grid: ttnn.CoreCoord,
    fp32_dest_acc: bool,
    mask_dtype: ttnn.DataType | None,
    cb_bytes: int,
) -> ttnn.SDPAProgramConfig | None:
    """SDPA's program config with its operands already in L1: the cheapest pair whose buffers fit cb_bytes,
    the L1 a bank below where SDPA's output will sit, with _L1_MARGIN_BYTES to spare; None if none does."""
    pair = _first_fitting_pair(
        batch, seqlen, heads, head_dim, grid.x * grid.y, fp32_dest_acc, mask_dtype, cb_bytes - _L1_MARGIN_BYTES
    )
    if pair is None:
        return None
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=pair[0], k_chunk_size=pair[1])


@cache
def attention_placement(
    batch: int,
    seqlen: int,
    heads: int,
    head_dim: int,
    grid: ttnn.CoreCoord,
    fp32_dest_acc: bool,
    mask_dtype: ttnn.DataType | None,
    cb_bytes: int,
    banks: int,
    qkv_bank_bytes: int = 0,
) -> tuple[ttnn.SDPAProgramConfig, ttnn.MemoryConfig]:
    """SDPA's program config and where the head tensors and SDPA's output live, for one call.

    In L1 the head split, both rotary calls and SDPA all read and write L1: at 8x384 that took them
    from 3.46 to 2.30 ms a forward. They need room beside SDPA's buffers, which at 8x512 take 1.07 MB
    of a core on the fp32 kernel at q256/k512. So the pairs of sdpa_chunk_candidates are tried
    cheapest first, and the first whose buffers fit beside the heads' span is taken; if none does, a
    head splits into four query chunks; if that does not fit either, the head tensors go to DRAM with
    the cheapest pair whose buffers fit alone. Each query row runs the same arithmetic whatever the
    query chunk and the placement, so for a given key chunk the output is the same bit for bit.

    cb_bytes is the L1 a bank below the deepest tensor live before the split; qkv_bank_bytes is QKV's
    output's share of a bank, 0 when it is in DRAM. The span is _HEAD_TENSOR_SPAN_QKV_IN_DRAM head
    tensors with QKV's output in DRAM, otherwise v and, when the space QKV's output frees cannot take
    SDPA's output beside the rotated q and k, SDPA's output too. That is the most the heads can reach;
    once they are placed, the forward re-picks the chunks against where they are (sdpa_chunks_in_l1).

    The query chunk is never a third of a head at 10 or more sequences: there SDPA returns wrong rows
    (10x384, 12x384, 16x384, 12x352), in DRAM as in L1. See test_sdpa_where_three_query_chunks_a_head_are_wrong.
    """
    cores = grid.x * grid.y
    head_bank = l1_head_bank_bytes(batch, seqlen, heads, head_dim, banks)

    def config(q_chunk: int, k_chunk: int) -> ttnn.SDPAProgramConfig:
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk)

    span = head_tensor_span(qkv_bank_bytes, head_bank)
    budget = cb_bytes - _L1_MARGIN_BYTES
    pair = _first_fitting_pair(
        batch, seqlen, heads, head_dim, cores, fp32_dest_acc, mask_dtype, budget - span * head_bank
    )
    if pair is not None:
        return config(*pair), ttnn.L1_MEMORY_CONFIG
    for q_chunk, k_chunk in sdpa_chunk_candidates(batch, seqlen, heads, cores, fp32_dest_acc):
        double_buffered_q = batch * heads * ttnn.core.divup(seqlen, q_chunk) > cores
        footprint = sdpa_circular_buffer_bytes(
            seqlen, q_chunk, k_chunk, head_dim, fp32_dest_acc, mask_dtype, double_buffered_q
        )
        if footprint <= budget:
            return config(q_chunk, k_chunk), ttnn.DRAM_MEMORY_CONFIG
    raise ValueError(f"no SDPA chunk pair fits L1 for {batch}x{seqlen}")


def l1_head_bank_bytes(batch: int, seqlen: int, heads: int, head_dim: int, banks: int) -> int:
    """One of q, k or v's share of an L1 bank, bfloat16 tiles; SDPA's output, the heads concatenated,
    takes the same."""
    head_tiles = batch * ttnn.core.divup(seqlen, ttnn.TILE_SIZE) * heads * head_dim // ttnn.TILE_SIZE
    return ttnn.core.divup(head_tiles, banks) * ttnn.tile_size(ttnn.bfloat16)


class TtNomicBertAttention(LightweightModule):
    """Fused three-major QKV projection, full-head rotary, then bidirectional SDPA.

    Two defaults differ from torch and both fail silently:

      - ttnn's SDPA defaults is_causal to True where torch defaults it to False. Left alone it
        applies a decoder mask to an encoder and still returns finite output, PCC 0.44.
      - the mask must materialise the query axis as (B, 1, S, S). Torch broadcasts (B, 1, 1, S)
        over queries; ttnn rejects that shape outright, which at least is loud.

    SDPA's 1/sqrt(head_dim) scale is folded into the q third of Wqkv instead, and the call takes
    scale=1.0. Given a mask and any other scale, SDPA multiplies the mask by 1/scale on every call,
    24 us a layer at 8x512. 1/sqrt(64) is a power of two, so q comes out scaled to the bit.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix, rotation_matrix=None):
        super().__init__()
        self.tt_config = tt_config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        def weight(name, group, tensor=None):
            return to_device(
                transpose_linear_weight(state_dict[f"{state_dict_prefix}{name}.weight"] if tensor is None else tensor),
                device,
                dtype=tt_config.matmul_weight_dtype(group),
            )

        def bias(name, tensor=None):
            return to_device(
                state_dict[f"{state_dict_prefix}{name}.bias"] if tensor is None else tensor,
                device,
                dtype=tt_config.weight_dtype,
            )

        # Wqkv is three-major, so the q projection is its first hidden_size output rows.
        query_rows = slice(0, config.hidden_size)
        qkv_weight = state_dict[f"{state_dict_prefix}Wqkv.weight"].clone()
        qkv_bias = state_dict[f"{state_dict_prefix}Wqkv.bias"].clone()
        qkv_weight[query_rows] *= self.head_dim**-0.5
        qkv_bias[query_rows] *= self.head_dim**-0.5
        # The q and k heads take rotary_pair_permutation's order, which rotary_embedding_llama rotates
        # in; v keeps its own, so SDPA's output and out_proj see the reference's head dims.
        query_and_key_heads = torch.arange(2 * self.num_heads)[:, None] * self.head_dim
        rows = torch.cat(
            (
                (query_and_key_heads + rotary_pair_permutation(self.head_dim)).flatten(),
                torch.arange(2 * config.hidden_size, 3 * config.hidden_size),
            )
        )
        qkv_weight, qkv_bias = qkv_weight[rows], qkv_bias[rows]
        self.qkv_weight, self.qkv_bias = weight("Wqkv", OpGroup.QKV, qkv_weight), bias("Wqkv", qkv_bias)
        # The generated QKV + head split + rotary op takes the checkpoint's Wqkv as it is, unscaled and unpermuted,
        # in bfloat16, and lays it out its own way (prepare_qkv_weights folds in the q scale).
        self.qkv_fused = None
        if tt_config.fused_attention_front:
            from ttnn.operations.nomic_qkv_rotary_heads import prepare_qkv_weights

            self.qkv_fused = prepare_qkv_weights(
                state_dict[f"{state_dict_prefix}Wqkv.weight"].to(torch.bfloat16),
                state_dict[f"{state_dict_prefix}Wqkv.bias"].to(torch.bfloat16),
                device,
            )
        # QKV with M <= N runs on grouped_matmul (dense_linear).
        self.qkv_grouped = prepare_grouped_projection(self.qkv_weight, self.qkv_bias, device)
        self.out_weight, self.out_bias = weight("out_proj", OpGroup.ATTN_OUT), bias("out_proj")
        # 2 KB of every core's L1, so the encoder builds one and hands it to all its layers.
        self.rotation_matrix = (
            rotary_rotation_matrix(device, tt_config.core_grid) if rotation_matrix is None else rotation_matrix
        )

    def _rotate(
        self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        """Apply rotary position embedding to one of q or k, whose head dims are pair-permuted.

        The op runs on the (B * A / k, k, S, D) view rotary_heads_per_view picks; cos/sin broadcast over
        both leading axes.

        Args:
            x: (B, A, S, D) queries or keys.
            cos: (1, 1, S, D) cosine table from tt.common.RotaryTables, or longer when S is on the
                tile grid.
            sin: the sine table, the same shape.

        Returns:
            ttnn.Tensor: (B, A, S, D), rotated.
        """
        shape = x.shape
        batch, heads, seqlen, head_dim = shape[0], shape[1], shape[2], shape[3]
        grid = self.tt_config.core_grid
        kept = rotary_heads_per_view(batch, ttnn.core.divup(seqlen, ttnn.TILE_SIZE), heads, grid.x * grid.y)
        folded = kept != heads
        source = ttnn.reshape(x, (batch * heads // kept, kept, seqlen, head_dim)) if folded else x
        rotated = ttnn.experimental.rotary_embedding_llama(
            source, cos, sin, self.rotation_matrix, is_decode_mode=False, memory_config=memory_config
        )
        return ttnn.reshape(rotated, (batch, heads, seqlen, head_dim)) if folded else rotated

    def _takes_fused(self, x: ttnn.Tensor) -> bool:
        """Whether the generated front reads x as it comes: 32 to 128 tile rows of M in the split norms' layout, the
        block input's layout there (tt.common.layer_norm keeps it above SMALL_M_TILES)."""
        if self.qkv_fused is None:
            return False
        if not FUSED_ATTENTION_MIN_TILES <= m_tiles(x.shape) <= FUSED_ATTENTION_MAX_TILES:
            return False
        layout = sharded_norm_layout(x.shape, self.tt_config.core_grid)
        return layout is not None and x.memory_config() == layout[0]

    def _fused_heads(
        self,
        x: ttnn.Tensor,
        tables,
        attn_mask: Optional[ttnn.Tensor],
        live: tuple[ttnn.Tensor, ...],
        compute_kernel_config,
    ):
        """q, k and v by the generated nomic_qkv_rotary_heads, one dispatch: QKV, the head split, q * 1/8 and
        rotary, with x read as it comes. Placed by the model's L1 plan as the four-op chain's are, less the QKV
        output it never writes. None where its circular buffers do not fit below the tensors live beside it, and the
        chain runs instead.

        Returns (query, key, value, SDPA program config, the heads' memory config, l1 floor or None, 0).
        """
        from ttnn.operations.nomic_qkv_rotary_heads import circular_buffer_bytes, nomic_qkv_rotary_heads

        batch, seqlen = x.shape[0], x.shape[-2]
        cb_bytes = circular_buffer_bytes(tuple(x.shape))
        floor = l1_floor((x, *live), self.tt_config)
        head_bank = l1_head_bank_bytes(batch, seqlen, self.num_heads, self.head_dim, self.tt_config.l1_banks)
        program_config = sdpa_program_config(
            batch,
            seqlen,
            self.num_heads,
            self.tt_config.core_grid,
            compute_kernel_config.fp32_dest_acc_en,
            attn_mask is not None,
        )
        heads_memory = ttnn.DRAM_MEMORY_CONFIG
        dram_config = program_config
        if self.tt_config.attention_l1:
            program_config, heads_memory = attention_placement(
                batch,
                seqlen,
                self.num_heads,
                self.head_dim,
                self.tt_config.core_grid,
                compute_kernel_config.fp32_dest_acc_en,
                None if attn_mask is None else attn_mask.dtype,
                self.tt_config.l1_cb_bytes - floor,
                self.tt_config.l1_banks,
                qkv_bank_bytes=0,
            )
        # The op's buffers sit below the heads it writes when those are in L1; where they do not fit there, the
        # heads go to DRAM, with SDPA's default program.
        l1_heads = heads_memory.buffer_type == ttnn.BufferType.L1
        if l1_heads and floor + 3 * head_bank + cb_bytes > self.tt_config.l1_cb_bytes:
            heads_memory, program_config = ttnn.DRAM_MEMORY_CONFIG, dram_config
        if floor + cb_bytes > self.tt_config.l1_cb_bytes:
            return None
        outputs = None
        if heads_memory.buffer_type == ttnn.BufferType.L1:
            head_shape = ttnn.Shape((batch, self.num_heads, seqlen, self.head_dim))
            value_out, key_out, query_out = (
                ttnn.allocate_tensor_on_device(head_shape, x.dtype, ttnn.TILE_LAYOUT, x.device(), heads_memory)
                for _ in range(3)
            )
            outputs = [query_out, key_out, value_out]
        query, key, value = nomic_qkv_rotary_heads(
            x, self.qkv_fused, tables, memory_config=heads_memory, output_tensors=outputs
        )
        return query, key, value, program_config, heads_memory, floor if self.tt_config.attention_l1 else None, 0

    def _chain_heads(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor],
        live: tuple[ttnn.Tensor, ...],
        compute_kernel_config,
    ):
        """q, k and v by the four-op chain: QKV (dense_linear), nlp_create_qkv_heads, rotary on q and k.

        Returns (query, key, value, SDPA program config, the heads' memory config, l1 floor or None, QKV's bank).
        """
        qkv = dense_linear(
            x, self.qkv_weight, self.qkv_bias, OpGroup.QKV, self.tt_config, live=live, grouped=self.qkv_grouped
        )
        batch, seqlen = x.shape[0], x.shape[-2]
        program_config = sdpa_program_config(
            batch,
            seqlen,
            self.num_heads,
            self.tt_config.core_grid,
            compute_kernel_config.fp32_dest_acc_en,
            attn_mask is not None,
        )
        heads_memory = ttnn.DRAM_MEMORY_CONFIG
        floor, qkv_bank = None, 0
        if self.tt_config.attention_l1:
            floor = l1_floor((x, qkv, *live), self.tt_config)
            if qkv.memory_config().buffer_type == ttnn.BufferType.L1:
                qkv_bank = l1_bank_bytes(
                    m_tiles(qkv.shape) * qkv.shape[-1] // ttnn.TILE_SIZE, self.tt_config, qkv.dtype
                )
            program_config, heads_memory = attention_placement(
                batch,
                seqlen,
                self.num_heads,
                self.head_dim,
                self.tt_config.core_grid,
                compute_kernel_config.fp32_dest_acc_en,
                None if attn_mask is None else attn_mask.dtype,
                self.tt_config.l1_cb_bytes - floor,
                self.tt_config.l1_banks,
                qkv_bank_bytes=qkv_bank,
            )

        # Three-major, heads contiguous inside each of q, k and v. transpose_k_heads stays False
        # because SDPA wants K as (B, A, S, D), not pre-transposed. qkv may sit in L1
        # (dense_linear); the heads are placed explicitly rather than inherit that. In L1 the outputs
        # are allocated v, k, q, so v lands highest and SDPA's buffers get the span the placement
        # planned.
        outputs = None
        if heads_memory.buffer_type == ttnn.BufferType.L1:
            head_shape = ttnn.Shape((batch, self.num_heads, seqlen, self.head_dim))
            value_out, key_out, query_out = (
                ttnn.allocate_tensor_on_device(head_shape, qkv.dtype, ttnn.TILE_LAYOUT, x.device(), heads_memory)
                for _ in range(3)
            )
            outputs = [query_out, key_out, value_out]
        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=heads_memory,
            output_tensors=outputs,
        )
        ttnn.deallocate(qkv)

        cos, sin = rot_mats
        rotated_query = self._rotate(query, cos, sin, heads_memory)
        rotated_key = self._rotate(key, cos, sin, heads_memory)
        # Freed here rather than inside _rotate: the reshape there aliases this buffer, so the
        # pre-rotation tensor is the one thing that owns it.
        ttnn.deallocate(query)
        ttnn.deallocate(key)
        return rotated_query, rotated_key, value, program_config, heads_memory, floor, qkv_bank

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
        fused_tables=None,
    ) -> ttnn.Tensor:
        """Project, rotate, attend and project back.

        Every L1 plan here is made below the deepest tensor live in L1 (l1_floor): x, QKV's output,
        the rotary tables and the mask.

        With fused_tables (tt.common.FusedRotaryTables) and x in a layout the generated front reads (_takes_fused),
        q, k and v come from nomic_qkv_rotary_heads in one dispatch (_fused_heads); otherwise from the four-op chain.

        Args:
            x: (B, 1, S, H) block input.
            rot_mats: (cos, sin), each (1, 1, S, D) or longer, from tt.common.RotaryTables.
            attn_mask: (B, 1, S, S) additive mask from tt.common.additive_attention_mask, or
                None for no masking.
            fused_tables: nomic_qkv_rotary_heads' tables for at least S, or None for the chain.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        live = (*rot_mats,) if attn_mask is None else (*rot_mats, attn_mask)
        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.SDPA)
        batch, seqlen = x.shape[0], x.shape[-2]
        heads = None
        if fused_tables is not None and self._takes_fused(x):
            heads = self._fused_heads(
                x, fused_tables, attn_mask, (*live, *table_tensors(fused_tables)), compute_kernel_config
            )
        if heads is None:
            heads = self._chain_heads(x, rot_mats, attn_mask, live, compute_kernel_config)
        rotated_query, rotated_key, value, program_config, heads_memory, floor, qkv_bank = heads
        head_bank = l1_head_bank_bytes(batch, seqlen, self.num_heads, self.head_dim, self.tt_config.l1_banks)

        if heads_memory.buffer_type == ttnn.BufferType.L1:
            # SDPA's output goes in the highest gap it fits: at worst right below the deepest tensor now
            # live, and no lower than attention_placement's span. The first is the tighter when v took a
            # gap above the residual, as from the second block on.
            deepest = l1_floor((x, rotated_query, rotated_key, value, *live), self.tt_config)
            output_bottom = min(deepest + head_bank, floor + head_tensor_span(qkv_bank, head_bank) * head_bank)
            program_config = (
                sdpa_chunks_in_l1(
                    batch,
                    seqlen,
                    self.num_heads,
                    self.head_dim,
                    self.tt_config.core_grid,
                    compute_kernel_config.fp32_dest_acc_en,
                    None if attn_mask is None else attn_mask.dtype,
                    self.tt_config.l1_cb_bytes - output_bottom,
                )
                or program_config
            )

        context = ttnn.transformer.scaled_dot_product_attention(
            rotated_query,
            rotated_key,
            value,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=heads_memory,
            output_concat_heads=True,
        )
        for tensor in (rotated_query, rotated_key, value):
            ttnn.deallocate(tensor)

        # SDPA's output can sit below v (attention_placement); dense_linear plans below it wherever it
        # is (l1_floor). Up to the split limit the output is written in the layout of norm1, its only
        # reader (tt/block.py).
        out_shape = (context.shape[0], 1, context.shape[-2], self.out_weight.shape[-1])
        norm_layout = sharded_norm_layout(out_shape, self.tt_config.core_grid)
        out = dense_linear(
            context,
            self.out_weight,
            self.out_bias,
            OpGroup.ATTN_OUT,
            self.tt_config,
            live=(x, *live),
            norm_layout=norm_layout,
        )
        ttnn.deallocate(context)
        return out
