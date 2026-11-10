# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Prefix spread (MINIMAX_H3_SOL_PREFIX_SPREAD=1): exact prefix query rows at routed-prefix speed.

With Sol's `fused` prefix rule every 64-token query block holding a prefix (text / condition / audio) token keeps
every valid key block. The packed sequence puts the whole prefix at the start, so all of those dense query blocks
sit on the first SP chip, which then carries ~1.5x (t2v) to ~2x (ref2va) the attention work of every other chip.

This module moves whole 64-row blocks, so the dense query blocks spread over the SP chips:
- Prefix block j (j >= 1) goes to shard j mod S, slot 1 + j // S; the video block in that slot takes its place.
- Block 0 of every shard never moves. Sol forces block 0 of each shard exact, so those positions keep their meaning.
- The slots used are inside each shard's first 256-key piece, which the forced block 0 already makes every query
  stream, so the moved sink key blocks add no K pieces.
- Only full blocks inside the logical length move. Padding stays at the end, where `logical_n` excludes it.

Everything in the model is per row (RoPE, adaLN indices, the source-table gather) or permutation-equivariant
(attention over all keys with padding at the end), so permuting the host-built row tables moves the rows and
nothing else. Sol's positional rules use the block map instead (`sol_prepass.build_constants(block_map=...)`):
- band on logical block indices;
- the sink and dense-prefix rows on the blocks' logical identity;
- force / length / validity per position, unchanged because only equal, full blocks swap.
The kept set is then exactly the `fused` one, permuted, and the outputs equal `fused` up to reassociation.

Pure torch / Python: no ttnn here.
"""

from __future__ import annotations

BLOCK = 64


def spread_enabled() -> bool:
    import os

    return os.environ.get("MINIMAX_H3_SOL_PREFIX_SPREAD", "0") == "1"


def prefix_spread_block_map(
    *, rows: int, shards: int, logical_n: int, prefix_rows: int, block: int = BLOCK
) -> tuple[int, ...] | None:
    """Logical global block index of each packed block, global block g = shard * nb + b (nb = ceil(rows / block)).

    Returns None when nothing moves (identity): one shard, no prefix, or a prefix inside block 0.
    """
    if shards <= 1 or prefix_rows <= block or rows <= 0:
        return None
    nb = -(-rows // block)
    perm = list(range(shards * nb))

    def full(s: int, b: int) -> bool:
        # A whole block of `block` rows inside the shard and inside the logical length.
        lo = s * rows + b * block
        return b * block + block <= rows and lo + block <= logical_n

    def logical_start(g: int) -> int:
        s, b = divmod(g, nb)
        return s * rows + b * block

    n_prefix = -(-prefix_rows // block)  # blocks holding a prefix token, in packed (= logical) order
    moved = False
    for j in range(1, n_prefix):
        src_s, src_b = divmod(j, nb)  # the prefix block's logical position (shard 0 for every real geometry)
        dst_s = j % shards
        dst_b = 1 + j // shards
        if dst_s == src_s or dst_b >= nb or src_b == 0:
            continue  # stays on its shard (j = 8, 16, ... with 8 shards), or a shard's block 0 (forced exact: fixed)
        src, dst = src_s * nb + src_b, dst_s * nb + dst_b
        if not (full(src_s, src_b) and full(dst_s, dst_b)):
            continue
        if logical_start(perm[dst]) < prefix_rows:
            continue  # the slot holds a prefix block itself (a prefix longer than one shard): leave it
        perm[src], perm[dst] = perm[dst], perm[src]
        moved = True
    return tuple(perm) if moved else None


def row_permutation(block_map: tuple[int, ...] | None, *, rows: int, shards: int, padded_len: int, block: int = BLOCK):
    """Packed row p -> logical row, as a torch.long tensor of `padded_len` rows (None: identity)."""
    import torch

    if block_map is None:
        return None
    assert padded_len == rows * shards, (padded_len, rows, shards)
    nb = -(-rows // block)
    perm = torch.arange(padded_len, dtype=torch.long)
    for g, lg in enumerate(block_map):
        if g == lg:
            continue
        s, b = divmod(g, nb)
        ls, lb = divmod(lg, nb)
        n = min(block, rows - b * block)
        assert n == min(block, rows - lb * block) == block, "only full blocks move"
        perm[s * rows + b * block : s * rows + b * block + n] = torch.arange(ls * rows + lb * block, ls * rows + lb * block + n)
    return perm


def inverse(perm):
    """Logical row -> packed row."""
    import torch

    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv


def dense_local_blocks(block_map: tuple[int, ...] | None, *, rows: int, shards: int, prefix_rows: int, block: int = BLOCK):
    """Per shard, the packed local Q-block indices that hold a prefix token (the dense query blocks of `fused`)."""
    nb = -(-rows // block)
    out = []
    for s in range(shards):
        local = []
        for b in range(nb):
            g = s * nb + b
            lg = block_map[g] if block_map is not None else g
            ls, lb = divmod(lg, nb)
            if ls * rows + lb * block < prefix_rows:
                local.append(b)
        out.append(tuple(local))
    return tuple(out)


def rows_in_packed_order(values, row_perm, rung: int):
    """A per-row table (logical order, up to `rung` rows) in packed order: zero-padded to `rung`, then permuted.
    Without a permutation it is returned unchanged (the callers pad as before)."""
    if row_perm is None:
        return values
    import torch

    pad = rung - values.shape[0]
    if pad:
        values = torch.cat([values, torch.zeros(pad, *values.shape[1:], dtype=values.dtype)])
    return values[row_perm]
