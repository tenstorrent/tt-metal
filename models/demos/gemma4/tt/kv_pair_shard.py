# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pair-sharded global KV (``GEMMA4_KV_PAIR_SHARD``): host planning for the 31B's global layers on a (1, 8) Galaxy
row, where each of the 4 global KV heads sits on the two chips of a pair (column c holds head c // 2) and was stored
in full on both.

The served layout is the HALF-BLOCK one (``half_block_*`` below): vLLM's hybrid KV manager shares one physical tensor
and one block pool between each global layer and its sliding layers, so the blocks themselves cannot be split
between the chips; their tokens are. The global layers report half their per-token KV bytes
(``half_block_head_size``), vLLM's page-size unification gives the global group blocks of ``2 * B`` tokens (B = the
device page, 64), and the device of pair half ``h`` keeps the ``B``-token half ``h`` of every logical block in the
slot the logical page table names: one page table for both chips, the global KV per chip halved.

* decode / verify attention: ``SdpaDecodeLse(half_block=True)`` over the logical table, each chip deriving its own
  position from the row's (``half_block_cur_pos`` is the host reference) and merging with its partner through the
  log-sum-exp (``tt/sdpa_pair_split.py``);
* writes (stock ``paged_fill_cache`` / ``paged_update_cache`` / ``paged_fused_update_cache``): a per-chip WRITE
  table at the device page's granularity, column ``2 i + h`` = logical block ``i``'s slot on its owner and the
  vLLM null block 0 elsewhere (``half_block_write_tables``), so a chip writes only its own half;
* prefill attention over cached keys: each chip attends its own half of every block (the logical table at the
  device page) and the pair merges the per-half results by log-sum-exp
  (``attention.prefill.pair_split_prefill_attention``).

The block-parity scheme below (ownership by block id, compacted read tables) is not served: it needs a vLLM
allocator change. Its planner backs the ``owned`` mode of ``custom_ops/sdpa_decode_lse``, tested on its own.
Block-parity planner -- each chip stores only the logical blocks it OWNS:

* ownership by block-id parity: logical block ``b`` lives on pair half ``h = b % 2`` (column parity) at physical
  slot ``b // 2``; every chip's global-layer cache holds ``ceil(num_blocks / 2) + 1`` slots, the last one a dummy
  that non-owner writes land in (``dummy_slot``), so the global KV bytes per chip halve for the same logical pool;
* writes (``paged_fill_cache`` / the decode cache update, stock ops) go through a per-half WRITE table: the block at
  position ``j`` maps to ``b // 2`` on its owner and to the dummy on the other half (``write_tables``);
* decode attention: attention over a row's cached keys is order-invariant (RoPE is applied when K is written), so
  each half attends ONLY its owned blocks through a COMPACTED read table (owned blocks in position order, padded
  with the dummy) with an effective ``cur_pos_h`` = owned full blocks x block + the tail of the row's last
  (partial) block when this half owns it, minus 1 (``owned_read_tables``); a half that owns none of a row's blocks
  gets ``-1`` and writes the empty-half sentinel, and the two halves merge through their log-sum-exp exactly as the
  contiguous pair split does (``tt/sdpa_pair_split.py``).

Pure torch, no device: shared by the model's input preparation and the host tests.
"""

import os

import torch

KV_PAIR_SHARD_ENV = "GEMMA4_KV_PAIR_SHARD"


def kv_pair_shard_enabled() -> bool:
    """``GEMMA4_KV_PAIR_SHARD=1`` (default off)."""
    return os.environ.get(KV_PAIR_SHARD_ENV, "0") == "1"


def physical_slots(num_logical_blocks: int) -> int:
    """Slots one chip's global-layer cache holds under the shard: its half of the logical blocks plus the dummy."""
    return (int(num_logical_blocks) + 1) // 2 + 1


def dummy_slot(num_logical_blocks: int) -> int:
    """The slot non-owner writes land in (the last one)."""
    return physical_slots(num_logical_blocks) - 1


def write_tables(page_table: torch.Tensor, num_logical_blocks: int) -> torch.Tensor:
    """``[2, B, W]`` int32 per-half write tables for a ``[B, W]`` logical page table: owner -> ``b // 2``, the other
    half -> the dummy slot. Negative (unused) entries map to the dummy on both halves."""
    pt = page_table.to(torch.int64)
    dummy = dummy_slot(num_logical_blocks)
    out = torch.full((2,) + tuple(pt.shape), dummy, dtype=torch.int64)
    valid = pt >= 0
    for h in (0, 1):
        own = valid & (pt % 2 == h)
        out[h][own] = pt[own] // 2
    return out.to(torch.int32)


def owned_read_tables(
    page_table: torch.Tensor, cur_pos: torch.Tensor, block_size: int, num_logical_blocks: int, *, logical=False
) -> tuple:
    """Per-half compacted read tables and effective positions for decode attention.

    ``page_table [B, W]`` logical ids in position order, ``cur_pos [B]`` (``-1`` = inactive row). Returns
    ``(tables [2, B, W] int32, cur_pos_h [2, B] int32)``: half ``h``'s table lists the physical slots of the owned
    blocks among positions ``0 .. cur_pos // block_size`` in position order (dummy-padded), and ``cur_pos_h[h, b]``
    is the last valid index of that compacted sequence (``-1`` when the half owns none of the row's blocks or the
    row is inactive). ``logical=True``: the tables hold the logical ids themselves (padding 0) -- a single-chip test
    of the attention split over one unsharded cache."""
    pt = page_table.to(torch.int64)
    B, W = pt.shape
    dummy = 0 if logical else dummy_slot(num_logical_blocks)
    bs = int(block_size)
    tables = torch.full((2, B, W), dummy, dtype=torch.int64)
    pos_h = torch.full((2, B), -1, dtype=torch.int64)
    for b in range(B):
        cp = int(cur_pos[b])
        if cp < 0:
            continue
        last = cp // bs
        tail = cp % bs + 1
        for h in (0, 1):
            owned = [j for j in range(last + 1) if int(pt[b, j]) % 2 == h]
            if not owned:
                continue
            tables[h, b, : len(owned)] = pt[b, owned] if logical else pt[b, owned] // 2
            full = len(owned) - (1 if owned[-1] == last else 0)
            pos_h[h, b] = full * bs + (tail if owned[-1] == last else 0) - 1
    return tables.to(torch.int32), pos_h.to(torch.int32)


def merge_halves(out0, lse0, out1, lse1):
    """Reference log-sum-exp merge of the two halves' normalised outputs (fp32): ``[..., D]`` outputs, ``[...]``
    LSEs; a half with ``lse = -inf`` (owns nothing) contributes nothing."""
    m = torch.maximum(lse0, lse1)
    w0 = torch.exp(lse0 - m)
    w1 = torch.exp(lse1 - m)
    return (w0.unsqueeze(-1) * out0 + w1.unsqueeze(-1) * out1) / (w0 + w1).unsqueeze(-1)


# ── the half-block layout (served) ─────────────────────────────────────────────────────────────────────────────────
PAIR = 2  # chips holding one global KV head under the shard
NULL_BLOCK = 0  # vLLM's null block: never handed to a request, the write tables' sink for the other half


def half_block_head_size(head_dim: int) -> int:
    """The global layers' KV-spec head size under the shard: half the per-token bytes, so vLLM's page-size
    unification doubles the group's block size (the page bytes, and so the shared pool, are unchanged)."""
    assert int(head_dim) % PAIR == 0, f"head_dim {head_dim} does not halve"
    return int(head_dim) // PAIR


def half_block_cur_pos(cur_pos: torch.Tensor, half: int, page: int) -> torch.Tensor:
    """Host reference of the kernel transform (``SDPA_LSE_HALF_BLOCK``): pair half ``half``'s last key index in its
    compacted sequence for a row at global position ``cur_pos`` (``-1`` = inactive row, or this half holds none of
    the row's keys). Chunks of ``page`` tokens: the half owns chunks ``c % 2 == half``."""
    cp = cur_pos.to(torch.int64)
    n = torch.div(cp, page, rounding_mode="floor")
    own_last = (n % PAIR) == half
    length = torch.div(n + 1 - half, PAIR, rounding_mode="floor").clamp(min=0) * page + torch.where(
        own_last, cp % page + 1, torch.zeros_like(cp)
    )
    return torch.where((cp >= 0) & (length > 0), length - 1, torch.full_like(cp, -1)).to(torch.int32)


def half_block_write_table(page_table: torch.Tensor, half: int) -> torch.Tensor:
    """Pair half ``half``'s write table for a ``[B, W]`` logical table (blocks of ``2 * page`` tokens): ``[B, 2 W]``
    int32 at the device page's granularity, column ``2 i + half`` = ``page_table[:, i]`` (this chip's half of
    logical block ``i``), column ``2 i + 1 - half`` = the null block (the partner's half, written into the sink)."""
    pt = page_table if page_table.dim() > 1 else page_table.unsqueeze(0)
    out = torch.full((int(pt.shape[0]), PAIR * int(pt.shape[1])), NULL_BLOCK, dtype=torch.int32)
    out[:, int(half) :: PAIR] = pt.to(torch.int32)
    return out


def half_block_write_tables(page_table: torch.Tensor, num_columns: int) -> torch.Tensor:
    """The write tables of a one-row mesh stacked for ``ShardTensorToMesh(dim=0)``: ``[num_columns * B, 2 W]``,
    rows ``[c * B, (c + 1) * B)`` = column ``c``'s (pair half ``c % 2``)."""
    assert int(num_columns) % PAIR == 0, f"{num_columns} columns do not form pairs"
    halves = [half_block_write_table(page_table, h) for h in range(PAIR)]
    return torch.cat([halves[c % PAIR] for c in range(int(num_columns))], dim=0)
