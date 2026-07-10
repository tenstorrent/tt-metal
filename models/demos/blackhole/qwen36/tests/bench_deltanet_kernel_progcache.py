# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-cache probe for the fused deltanet_decode_full op.

Calls the op N times with IDENTICAL shapes and times each. If the program cache works,
call 1 pays the JIT compile and calls 2..N are fast (cache hit). If every call is slow,
the op is recompiling per step — the root cause of the fused-decode-path slowdown.
"""
import os
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp


@parametrize_mesh_tp()
def test_progcache_deltanet_decode_full(mesh_device, ensure_gc):
    from loguru import logger

    mesh_device.enable_program_cache()

    Nk, Nv, Dk, Dv, K = 4, 12, 128, 128, 4
    rf = Nv // Nk
    key_dim, val_dim = Nk * Dk, Nv * Dv
    conv_dim = 2 * key_dim + val_dim
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    torch.manual_seed(0)
    qkv = dev(torch.randn(1, 1, 1, conv_dim) * 0.2)
    z = dev(torch.randn(1, 1, 1, val_dim) * 0.2)
    b = dev(torch.rand(1, 1, 1, Nv))
    a = dev(torch.rand(1, 1, 1, Nv))
    dummy = dev(torch.zeros(1, 1, conv_dim, 32))
    rec = dev(torch.zeros(1, Nv, Dk, Dv))
    zh = dev(torch.zeros(1, 1, 1, Nv))
    nw = dev(torch.ones(1, 1, 1, Dv))

    def call():
        out = ttnn.experimental.deltanet_decode_full(
            qkv, z, b, a, dummy, rec, dummy, zh, zh, nw,
            num_heads=Nv, num_k_heads=Nk, k_head_dim=Dk, v_head_dim=Dv,
            conv_dim=conv_dim, conv_kernel_size=K, head_expand_ratio=rf,
        )
        ttnn.synchronize_device(mesh_device)
        for t in out:
            ttnn.deallocate(t)

    times = []
    for i in range(12):
        t0 = time.time()
        call()
        dt = (time.time() - t0) * 1e3
        times.append(dt)
        logger.info(f"call {i:2d}: {dt:8.1f} ms   (cache entries: {mesh_device.num_program_cache_entries()})")

    warm = times[2:]
    logger.info(f"PROGCACHE: call0={times[0]:.0f}ms  warm_median={sorted(warm)[len(warm)//2]:.1f}ms  "
                f"cache_entries={mesh_device.num_program_cache_entries()}")
    # If warm calls are ~compile-time (100s of ms) and cache_entries keeps growing → no caching.
    assert sorted(warm)[len(warm) // 2] < 50.0, "op appears to recompile per call (no program-cache hit)"
