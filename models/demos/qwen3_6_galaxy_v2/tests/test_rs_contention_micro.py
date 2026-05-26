# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RS inter-op contention micro-benchmark.

Question being tested: in the qwen3.6 FA decode path, the cluster_axis=0 ring=8
RS (the post-WO row-axis reduce, 1280×BF16) runs at 666 µs/call mean vs DN's
identical-shape RS at 64 µs/call. The isolated chain test
(test_wo_chain_micro.py) showed that the same chain in isolation runs at
~47 µs. So the slowdown comes from the FULL-MODEL CONTEXT, not the chain.

This driver tests whether the slowdown is caused by INTER-CCL contention:

  Variant A: 4 warm iters of (matmul → cluster_axis=0 all_reduce)              — clean axis-0 baseline
  Variant B: 4 warm iters of (matmul → cluster_axis=1 all_reduce)              — clean axis-1 baseline
  Variant C: 4 warm iters of (matmul → axis=1 RS → matmul → axis=0 RS)          — axis=1 then axis=0 (the FA pattern)
  Variant D: 4 warm iters of (matmul → axis=0 RS → matmul → axis=0 RS)          — two consecutive axis=0
  Variant E: 4 warm iters of (matmul → axis=0 RS) but with 1 dummy matmul       — adds work between calls

Each variant: 1 compile pass (NOT signposted) + 4 warm iters bracketed by
``<var>_warm_start`` ... ``<var>_warm_done`` signposts.

Reads back DEVICE KERNEL DURATION + OP_TO_OP_LATENCY of the final cluster_axis=0
RS across variants. If any variant's axis-0 RS DK matches the 666 µs/call model
number, that variant's preceding pattern explains the slowdown.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_rs_contention_micro.py -v -s
"""
from __future__ import annotations

import pytest
import torch

import ttnn

_MESH = (8, 4)
_M = 32  # tile-padded T
_PER_CHIP_W_0 = 1280  # for axis=0 RS (the slow one in FA): H / 4 cols = 1280
_PER_CHIP_W_1 = 2048  # for axis=1 RS (FA WQKVG output): 16384 / 8 rows = 2048
_K = 768  # per-chip K dim for the matmul producing the axis=0 RS input
_K1 = 1280  # per-chip K dim for the matmul producing the axis=1 RS input

_N_WARM = 4


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*_MESH),
        trace_region_size=184915840,
        worker_l1_size=1345000,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _make_weight(mesh, k, n, seed=7):
    torch.manual_seed(seed)
    w = torch.randn(_MESH[0], _MESH[1], k, n, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        w,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
    )


def _make_act(mesh, m, k, seed=99):
    torch.manual_seed(seed)
    a = torch.randn(_MESH[0], _MESH[1], m, k, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        a,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
    )


@pytest.mark.hardware
def test_rs_contention_variants(bh_glx_mesh):
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *a, **k: None  # noqa: E731

    mesh = bh_glx_mesh

    # Inputs / weights for axis-0 RS path (matches FA WO shape)
    a0 = _make_act(mesh, _M, _K, seed=11)
    w0 = _make_weight(mesh, _K, _PER_CHIP_W_0, seed=21)
    # Inputs / weights for axis-1 RS path (matches FA WQKVG shape)
    a1 = _make_act(mesh, _M, _K1, seed=12)
    w1 = _make_weight(mesh, _K1, _PER_CHIP_W_1, seed=22)

    def axis0_chain():
        out = ttnn.linear(a0, w0, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r = ttnn.all_reduce(out, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(r)

    def axis1_chain():
        out = ttnn.linear(a1, w1, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(r)

    def axis1_then_axis0():
        axis1_chain()
        axis0_chain()

    def axis0_then_axis0():
        axis0_chain()
        axis0_chain()

    variants = [
        ("A_axis0_alone", lambda: axis0_chain()),
        ("B_axis1_alone", lambda: axis1_chain()),
        ("C_axis1_then_axis0", lambda: axis1_then_axis0()),
        ("D_axis0_then_axis0", lambda: axis0_then_axis0()),
    ]

    for name, fn in variants:
        print(f"\n=== Variant {name} ===")
        # Compile pass (NOT signposted)
        fn()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost(f"{name}_warm_start")
        for _ in range(_N_WARM):
            fn()
        ttnn.synchronize_device(mesh)
        signpost(f"{name}_warm_done")
        print(f"  {_N_WARM} warm iterations completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
