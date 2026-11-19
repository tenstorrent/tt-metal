# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/paged_pool_gather`` (the sparse-decode latent gather, a model-local generic op): the 512 ranked pool
ids + the tail pool of every user (dense users: the pools in order) gathered out of a paged bf16 TILE cache
(permuted page table) into a compact paged cache vs the torch golden, with sentinel and out-of-range pool ids
(zero rows), B = 1 / 4 / 32 users, a short ranked list, the padding rows, a second call on new ids (program
reuse), trace capture + replay with refreshed ids, and the timing at the decode geometry (1 / 4 / 8 users, cache
sized for 1 M positions).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_paged_pool_gather.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.paged_pool_gather.op import SENTINEL, PagedPoolGather

L = 512


def _ids_tensor(ids: torch.Tensor, device):
    return ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _tail_tensor(tail: torch.Tensor, device):
    return ttnn.from_torch(tail.reshape(-1, 1, 1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B,K,bps", [(1, 512, 80), (4, 512, 80), (32, 512, 16), (2, 16, 8), (4, 16, 4)])
def test_pool_gather_vs_golden(mesh_device, B, K, bps):
    P = 4
    op = PagedPoolGather(mesh_device, pool_size=P)
    g = torch.Generator().manual_seed(11)
    cache = torch.randn(B * bps + 3, 1, 32, L, generator=g).bfloat16()  # 3 spare blocks nobody owns
    pt = torch.randperm(B * bps, generator=g).to(torch.int32).reshape(B, bps)
    max_pool = bps * 32 // P
    ids = torch.randint(0, max_pool, (B, 1, 1, K), generator=g)
    # tail pools: user 0 sparse (n >= K: ranked ids + tail), user 1 (if any) dense (n < K: pools in order), others mixed
    tail = torch.randint(0, max_pool, (B,), generator=g)
    tail[0] = max(K, min(max_pool - 1, K + 3))
    if B > 1:
        tail[1] = min(K - 1, max_pool - 1, 5)
    if K >= 16:
        ids[:, 0, 0, 5] = SENTINEL  # a sentinel mid-row
        ids[:, 0, 0, 7] = max_pool  # out of the user's range
        ids[0, 0, 0, -1] = SENTINEL
    ref = op.golden(cache, pt, ids, tail, P)
    cache_t = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    pt_t = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    ids_t = _ids_tensor(ids, mesh_device)
    tail_t = _tail_tensor(tail, mesh_device)
    out_t = ttnn.zeros(op.output_shape(ids_t, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    op(cache_t, pt_t, ids_t, tail_t, out_t)
    got = ttnn.to_torch(out_t)
    assert tuple(got.shape) == tuple(ref.shape), (got.shape, ref.shape)
    n_bad = int((got != ref).sum())
    logger.info(f"paged_pool_gather B={B} K={K} bps={bps}: output {tuple(got.shape)}, {n_bad} mismatches vs golden")
    assert n_bad == 0
    # program reuse on new ids / tails
    ids2 = torch.randint(0, max_pool, (B, 1, 1, K), generator=g)
    tail2 = torch.randint(0, max_pool, (B,), generator=g)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(ids2, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), ids_t)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(tail2.reshape(-1, 1, 1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), tail_t
    )
    op(cache_t, pt_t, ids_t, tail_t, out_t)
    assert torch.equal(ttnn.to_torch(out_t), op.golden(cache, pt, ids2, tail2, P))
    for t in (cache_t, pt_t, ids_t, tail_t, out_t):
        ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 16 * 1024 * 1024, "l1_small_size": 32768}], indirect=True
)
def test_pool_gather_trace_replay(mesh_device):
    """The op captured in a trace and replayed with refreshed ids / tails (persistent io tensors): every replay == golden."""
    P, B, K, bps = 4, 4, 512, 80
    op = PagedPoolGather(mesh_device, pool_size=P)
    g = torch.Generator().manual_seed(5)
    cache = torch.randn(B * bps, 1, 32, L, generator=g).bfloat16()
    pt = torch.randperm(B * bps, generator=g).to(torch.int32).reshape(B, bps)
    cache_t = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    pt_t = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    max_pool = bps * 32 // P
    ids = torch.randint(0, max_pool, (B, 1, 1, K), generator=g)
    tail = torch.randint(0, max_pool, (B,), generator=g)
    ids_t = _ids_tensor(ids, mesh_device)
    tail_t = _tail_tensor(tail, mesh_device)
    out_t = ttnn.zeros(op.output_shape(ids_t, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    op(cache_t, pt_t, ids_t, tail_t, out_t)  # compile
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(cache_t, pt_t, ids_t, tail_t, out_t)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for step in range(3):
        ids = torch.randint(0, max_pool, (B, 1, 1, K), generator=g)
        tail = torch.randint(0, max_pool, (B,), generator=g)
        tail[step % B] = step  # a dense user
        ids[1, 0, 0, step] = SENTINEL
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), ids_t)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(tail.reshape(-1, 1, 1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), tail_t
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        eq = torch.equal(ttnn.to_torch(out_t), op.golden(cache, pt, ids, tail, P))
        logger.info(f"trace replay {step}: bit-equal to golden {eq}")
        assert eq
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_pool_gather_timing_1m(mesh_device):
    """Cache allocated for 8 users x 1 048 640 positions (8 GiB of pages), 512 ranked pools + tail per user."""
    P, K, bps, B_max = 4, 512, 32770, 8
    op = PagedPoolGather(mesh_device, pool_size=P)
    cache_t = ttnn.zeros((B_max * bps, 1, 32, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    pt_t = ttnn.from_torch(
        torch.arange(B_max * bps, dtype=torch.int32).reshape(B_max, bps),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    g = torch.Generator().manual_seed(3)
    for B in (1, 4, 8):
        ids_t = _ids_tensor(torch.randint(0, bps * 32 // P, (B, 1, 1, K), generator=g), mesh_device)
        tail_t = _tail_tensor(torch.full((B,), 200000, dtype=torch.int64), mesh_device)
        out_t = ttnn.zeros(op.output_shape(ids_t, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        op(cache_t, pt_t, ids_t, tail_t, out_t)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            op(cache_t, pt_t, ids_t, tail_t, out_t)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / n
        logger.info(
            f"paged_pool_gather B={B} x {K + 1} pools (1 M-position cache): {dt * 1e3:.3f} ms per call ({B * (K + 1) * P} KB gathered)"
        )
        ttnn.deallocate(out_t)
        ttnn.deallocate(ids_t)
        ttnn.deallocate(tail_t)
