# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pair-sharded global KV (``GEMMA4_KV_PAIR_SHARD``): host planning for the 31B's global layers on a (1, 8) Galaxy
row, where each of the 4 global KV heads sits on the two chips of a pair (column c holds head c // 2), which the
unsharded layout stores in full on both.

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
* prefill attention over cached keys: on the layer's pair view (the q heads of the chip's whole KV group) each chip
  attends its own half of every block (the logical table at the device page) and the chunk's own keys, and the pair
  merges by log-sum-exp, exchanging one LSE per head and row (``attention.prefill.pair_split_prefill_attention``).

The block-parity layout (ownership by block id, compacted read tables) is not served: it needs a vLLM allocator
change. The ``owned`` mode of ``custom_ops/sdpa_decode_lse`` that would read it is tested with its own read-table
helper (tests/unit/custom_ops/test_sdpa_decode_lse.py).

Pure torch, no device: shared by the model's input preparation and the host tests.
"""

import os

import torch

KV_PAIR_SHARD_ENV = "GEMMA4_KV_PAIR_SHARD"


def kv_pair_shard_enabled() -> bool:
    """``GEMMA4_KV_PAIR_SHARD=1`` (default off)."""
    return os.environ.get(KV_PAIR_SHARD_ENV, "0") == "1"


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
