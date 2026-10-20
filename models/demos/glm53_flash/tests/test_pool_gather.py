# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.paged_pool_gather`` (the sparse-decode latent gather): pools of 4 token rows gathered out
of a paged bf16 TILE cache (permuted page table) into a compact paged cache, vs a torch gather, including
sentinel and out-of-range pool ids (zero rows), several users, a short pool list and the padding rows; then
the timing at the decode geometry (513 pools per user, 1 / 4 / 8 users, cache sized for 1 M positions).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_pool_gather.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn

SENT = 0xFFFFFFFF
L = 512


def torch_pool_gather(cache: torch.Tensor, pt: torch.Tensor, ids: torch.Tensor, P: int) -> torch.Tensor:
    """cache [blocks,1,32,L], pt [B,bps] int, ids [B,N] int64 -> [B*OB,1,32,L] (zeros for invalid pools)."""
    B, N = ids.shape
    bps = pt.shape[1]
    OB = -(-(N * P) // 32)
    out = torch.zeros(B * OB, 1, 32, cache.shape[-1], dtype=cache.dtype)
    rows = out.reshape(B, OB * 32, -1)
    for b in range(B):
        for i in range(N):
            p = int(ids[b, i])
            if p == SENT or p * P >= bps * 32:
                continue
            r0 = p * P
            blk = int(pt[b, r0 // 32])
            rows[b, i * P : (i + 1) * P] = cache[blk, 0, r0 % 32 : r0 % 32 + P]
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B,N,bps", [(1, 513, 80), (3, 513, 80), (2, 20, 8), (4, 1, 4)])
def test_pool_gather_vs_torch(mesh_device, B, N, bps):
    P = 4
    g = torch.Generator().manual_seed(11)
    cache = torch.randn(B * bps + 3, 1, 32, L, generator=g).bfloat16()  # 3 spare blocks nobody owns
    pt = torch.randperm(B * bps, generator=g).to(torch.int32).reshape(B, bps)
    max_pool = bps * 32 // P
    ids = torch.randint(0, max_pool, (B, N), generator=g)
    if N >= 20:
        ids[:, 5] = SENT  # a sentinel mid-row
        ids[:, 7] = max_pool  # out of the user's range
        ids[0, -1] = SENT
    ref = torch_pool_gather(cache, pt, ids, P)
    cache_t = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    pt_t = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    ids_t = ttnn.from_torch(
        ids.reshape(B, 1, 1, N), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )
    out = ttnn.experimental.paged_pool_gather(cache_t, pt_t, ids_t, pool_size=P)
    got = ttnn.to_torch(out)
    assert tuple(got.shape) == tuple(ref.shape), (got.shape, ref.shape)
    eq = torch.equal(got, ref)
    n_bad = int((got != ref).sum())
    logger.info(
        f"paged_pool_gather B={B} N={N} bps={bps}: output {tuple(got.shape)}, bit-equal {eq} ({n_bad} mismatches)"
    )
    assert eq
    # a second call with different ids reuses the program (runtime args patched)
    ids2 = torch.randint(0, max_pool, (B, N), generator=g)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(ids2.reshape(B, 1, 1, N), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), ids_t
    )
    out2 = ttnn.experimental.paged_pool_gather(cache_t, pt_t, ids_t, pool_size=P)
    assert torch.equal(ttnn.to_torch(out2), torch_pool_gather(cache, pt, ids2, P))
    for t in (cache_t, pt_t, ids_t, out, out2):
        ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_pool_gather_timing_1m(mesh_device):
    """Cache allocated for 8 users x 1 048 640 positions (8 GiB of pages), 513 pools per user."""
    P, N = 4, 513
    bps = 32770
    B_max = 8
    cache_t = ttnn.zeros((B_max * bps, 1, 32, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    pt_t = ttnn.from_torch(
        torch.arange(B_max * bps, dtype=torch.int32).reshape(B_max, bps),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    g = torch.Generator().manual_seed(3)
    for B in (1, 4, 8):
        ids = torch.randint(0, bps * 32 // P, (B, 1, 1, N), generator=g)
        ids_t = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        out = ttnn.experimental.paged_pool_gather(cache_t, pt_t, ids_t, pool_size=P)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            o = ttnn.experimental.paged_pool_gather(cache_t, pt_t, ids_t, pool_size=P)
            ttnn.deallocate(o)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / n
        logger.info(
            f"paged_pool_gather B={B} x {N} pools (1 M-position cache): {dt * 1e3:.3f} ms per call ({B * N * P} KB gathered)"
        )
        ttnn.deallocate(out)
        ttnn.deallocate(ids_t)
