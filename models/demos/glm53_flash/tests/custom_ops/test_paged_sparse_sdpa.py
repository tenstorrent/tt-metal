# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/paged_sparse_sdpa`` (the sparse DSA prefill attention over the paged latent cache, a model-local generic
op) vs ``ttnn.transformer.sparse_sdpa`` on the whole-prefix ROW_MAJOR copy (the stock contract): bit-identical output
at the DSA shapes (H = 32 heads, K_DIM = V_DIM = 512, 128-row chunks, 512 pools of 4 + a tail pool + sentinels), a
second user's slot (identity page table), a short index row (one active chunk), a second call with new indices
(program reuse), and the timing of both ops at 256 queries over a 1 M-position cache.
    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_paged_sparse_sdpa.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.paged_sparse_sdpa.op import SENTINEL, PagedSparseSdpa

L = 512  # the latent width (K_DIM = V_DIM)
H = 32  # heads after the stock op's padding to a tile row
P = 4  # tokens per pool
K_POOLS = 512  # ranked pools per query (index_select_pools)
TOPK = 2176  # 512 pools x 4 + up to 3 tail ids, padded to a multiple of 128 (the expansion's width)


def _index_rows(S: int, positions: list[int], bps: int, g: torch.Generator) -> torch.Tensor:
    """Index rows like the indexer's expansion: for query at position a, 512 distinct complete pools below a's pool
    (or every complete pool when fewer exist), their 4 token ids each, then the tail pool's ids below a, then
    sentinels."""
    rows = torch.full((S, TOPK), SENTINEL, dtype=torch.int64)
    for s, a in enumerate(positions):
        n_complete = (a + 1) // P  # complete pools strictly below the query's own tail pool
        tail_pool = (a + 1) // P
        tail_n = (a + 1) - tail_pool * P  # ids of the incomplete pool: 4*tail_pool .. a
        k = min(K_POOLS, n_complete)
        pools = torch.randperm(n_complete, generator=g)[:k]
        ids = (pools[:, None] * P + torch.arange(P)).reshape(-1)
        row = [ids]
        if tail_n:
            row.append(torch.arange(tail_pool * P, tail_pool * P + tail_n))
        row = torch.cat(row)
        rows[s, : row.numel()] = row
    assert int(rows[rows != SENTINEL].max()) < bps * 32  # every real id inside the user's pages
    return rows


def _run_pair(mesh_device, S: int, bps: int, slot: int, slots: int, positions: list[int], seed: int):
    g = torch.Generator().manual_seed(seed)
    cache = torch.randn(slots * bps, 1, 32, L, generator=g).bfloat16()
    q = torch.randn(1, H, S, L, generator=g).bfloat16()
    rows = _index_rows(S, positions, bps, g)
    PagedSparseSdpa.check_indices(rows)
    cache_t = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    q_t = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    idx_t = ttnn.from_torch(
        rows.reshape(1, 1, S, TOPK), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )
    # the stock contract: the user's rows as one ROW_MAJOR [1,1,T,L] tensor
    kv_rm = ttnn.from_torch(
        cache[slot * bps : (slot + 1) * bps].reshape(1, 1, bps * 32, L),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    scale = L**-0.5
    t0 = time.perf_counter()
    ref_t = ttnn.transformer.sparse_sdpa(
        q_t, kv_rm, idx_t, v_dim=L, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=scale, k_chunk_size=128
    )
    ttnn.synchronize_device(mesh_device)
    t_ref = time.perf_counter() - t0
    op = PagedSparseSdpa(mesh_device)
    t0 = time.perf_counter()
    out_t = op(q_t, cache_t, idx_t, slot_block0=slot * bps, blocks_per_seq=bps, v_dim=L, scale=scale)
    ttnn.synchronize_device(mesh_device)
    t_op = time.perf_counter() - t0
    ref = ttnn.to_torch(ref_t)
    out = ttnn.to_torch(out_t)
    return ref, out, t_ref, t_op, (op, q_t, cache_t, idx_t, kv_rm, scale)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "S,bps,slot,slots",
    [(128, 640, 0, 1), (128, 640, 2, 4), (32, 96, 1, 2)],  # 20 K positions; another user's slot; a short prefix
)
def test_paged_vs_stock(mesh_device, S, bps, slot, slots):
    T = bps * 32
    g = torch.Generator().manual_seed(3 + S)
    # queries at positions in the sparse region (>= 2048) up to the prefix end
    positions = (
        sorted(torch.randint(2048, T, (S,), generator=g).tolist())
        if T > 2048
        else sorted(torch.randint(8, T, (S,), generator=g).tolist())
    )
    ref, out, t_ref, t_op, env = _run_pair(mesh_device, S, bps, slot, slots, positions, seed=7 + slot)
    equal = torch.equal(ref, out)
    diff = (ref.float() - out.float()).abs().max().item()
    logger.info(
        f"S={S} bps={bps} slot={slot}: equal {equal}, max |diff| {diff:.3g}; stock {t_ref * 1e3:.1f} ms, paged {t_op * 1e3:.1f} ms"
    )
    assert equal, f"paged_sparse_sdpa differs from the stock op: max |diff| {diff}"
    # a second call with new indices reuses the program (same shapes) and still matches
    op, q_t, cache_t, idx_t, kv_rm, scale = env
    g2 = torch.Generator().manual_seed(99)
    rows2 = _index_rows(S, positions, bps, g2)
    idx2 = ttnn.from_torch(
        rows2.reshape(1, 1, S, TOPK), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )
    ref2 = ttnn.to_torch(
        ttnn.transformer.sparse_sdpa(
            q_t, kv_rm, idx2, v_dim=L, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=scale, k_chunk_size=128
        )
    )
    out2 = ttnn.to_torch(op(q_t, cache_t, idx2, slot_block0=slot * bps, blocks_per_seq=bps, v_dim=L, scale=scale))
    assert torch.equal(ref2, out2), "the second call (new indices) differs from the stock op"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_paged_timing_1m(mesh_device):
    """256 queries at ~1 M positions over a 1 M-position cache: the stock op on the ROW_MAJOR copy (whose copy, the
    untilize of 1 GiB, is the cost this op removes) vs the paged op; bit-identical and the two times logged."""
    bps = 32768  # 1 048 576 positions
    g = torch.Generator().manual_seed(5)
    positions = sorted(torch.randint(bps * 32 - 4096, bps * 32, (256,), generator=g).tolist())
    cache = torch.randn(bps, 1, 32, L, generator=g).bfloat16()
    q = torch.randn(1, H, 256, L, generator=g).bfloat16()
    rows = _index_rows(256, positions, bps, g)
    cache_t = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    q_t = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    idx_t = ttnn.from_torch(
        rows.reshape(1, 1, 256, TOPK), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )
    scale = L**-0.5
    op = PagedSparseSdpa(mesh_device)
    for _ in range(2):  # the first call compiles
        out_t = op(q_t, cache_t, idx_t, slot_block0=0, blocks_per_seq=bps, v_dim=L, scale=scale)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    out_t = op(q_t, cache_t, idx_t, slot_block0=0, blocks_per_seq=bps, v_dim=L, scale=scale)
    ttnn.synchronize_device(mesh_device)
    t_paged = time.perf_counter() - t0
    t0 = time.perf_counter()
    kv_rm = ttnn.untilize_with_unpadding(cache_t, [bps - 1, 0, 31, L - 1])  # the stock path's whole-prefix copy
    kv_rm = ttnn.reshape(kv_rm, [1, 1, bps * 32, L])
    ttnn.synchronize_device(mesh_device)
    t_copy = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref_t = ttnn.transformer.sparse_sdpa(
        q_t, kv_rm, idx_t, v_dim=L, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=scale, k_chunk_size=128
    )
    ttnn.synchronize_device(mesh_device)
    t_stock = time.perf_counter() - t0
    equal = torch.equal(ttnn.to_torch(ref_t), ttnn.to_torch(out_t))
    logger.info(
        f"1 M cache, 256 queries: paged {t_paged * 1e3:.1f} ms; stock copy {t_copy * 1e3:.1f} ms + attention "
        f"{t_stock * 1e3:.1f} ms; equal {equal}"
    )
    assert equal
