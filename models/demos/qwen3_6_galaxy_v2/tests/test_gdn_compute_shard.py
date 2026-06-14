# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify the GDN-compute sharding lever (device-kernel) on BH GLX 8x4.

GDN layer is ~everything DRAM. Per the rms_norm result, COMPUTE ops shard ~2.3x. But the two
GDN compute buckets differ:
  - elementwise/norm (recurrent BinaryNg ~34us + LayerNorm ~12us): activation-bound -> should
    shard ~2x (clean, model-preserving lever).
  - matmul (~142us, 9/layer): a M=32 decode matmul is WEIGHT-READ bound (DRAM weight) -> sharding
    the ACTIVATION should give little (that's the prefetcher's domain). Test L1-interleaved vs DRAM
    activation: if ~1.1x, weight-bound (confirms matmul lever != placement).

Run (device-kernel), one mode per invocation:
  for MEM in dram l1shard; do
    MESH_DEVICE=BH_GLX QWEN36_GDN_MEM=$MEM python -m tracy -p -v -r --op-support-count 20000 \
      -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_gdn_compute_shard.py -s
  done
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ImportError:
    signpost = lambda *_a, **_k: None  # noqa: E731

_MESH = (8, 4)
_M, _K, _N = 32, 1280, 1024  # representative GDN projection (K=dim_per_tp, N~per-chip heads)
_ITERS = int(os.environ.get("QWEN36_GDN_ITERS", "30"))
_MEM = os.environ.get("QWEN36_GDN_MEM", "dram").lower()


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH), trace_region_size=20_000_000)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.hardware
def test_gdn_compute_shard(bh_glx_mesh):
    mesh = bh_glx_mesh
    print(f"\n[gdn] MEM={_MEM} matmul[{_M},{_K}]@[{_K},{_N}] + mul[{_M},{_N}] iters={_ITERS}", flush=True)
    sharded = _MEM == "l1shard"

    def shard(width, ncores):
        return ttnn.create_sharded_memory_config(
            shape=(_M, width // ncores),
            core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(ncores, 0))]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    torch.manual_seed(0)
    act_mc = shard(_N, 8) if sharded else ttnn.DRAM_MEMORY_CONFIG  # for the mul (elementwise)
    # elementwise mul (recurrent BinaryNg proxy) — activation-bound, should shard
    a = ttnn.from_torch(
        torch.randn(1, 1, _M, _N) * 0.1,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=act_mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    b = ttnn.from_torch(
        torch.randn(1, 1, _M, _N) * 0.1,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=act_mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    # matmul: weight DRAM (always) — test whether activation placement matters (weight-bound?)
    mm_act_mc = ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG  # L1-interleaved vs DRAM activation
    x = ttnn.from_torch(
        torch.randn(1, 1, _M, _K) * 0.1,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mm_act_mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, _K, _N) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)

    def mul_op():
        return ttnn.mul(a, b, memory_config=act_mc, dtype=ttnn.bfloat16)

    def mm_op():
        return ttnn.linear(x, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=mm_act_mc)

    for fn in (mul_op, mm_op):  # warmup
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    signpost("start")
    for _ in range(_ITERS):
        o1 = mul_op()
        o2 = mm_op()
        ttnn.deallocate(o1)
        ttnn.deallocate(o2)
    ttnn.synchronize_device(mesh)
    signpost("stop")
    print(f"[gdn] done MEM={_MEM}: aggregate Binary(mul) + Matmul device-kernel /{_ITERS}", flush=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
