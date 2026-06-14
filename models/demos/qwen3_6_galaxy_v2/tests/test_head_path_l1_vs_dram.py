# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prototype: FA head-path ops on DRAM (current) vs L1-width-sharded — device-kernel.

The FA decode head dance (qk_norm rms_norm + slice + concat + elementwise) currently runs
DRAM-interleaved in+out (confirmed in the raw Tracy report) — for tiny [32, n_heads*hd] decode
tensors that round-trips DRAM every op. Hypothesis: keeping them L1-resident cuts the per-op
device-kernel (DRAM access latency removed). This isolates the DRAM-vs-L1 delta on the
representative per-chip head shape, WITHOUT a model build.

Per-chip FA Q at decode: n_q=24 / 8 rows = 3 q-heads/chip x head_dim=256 = 768 wide, M=32.

Run under Tracy (device-kernel), one mem mode per invocation:
  for MEM in dram l1; do
    MESH_DEVICE=BH_GLX QWEN36_HEAD_MEM=$MEM python -m tracy -p -v -r --op-support-count 20000 \
      -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_head_path_l1_vs_dram.py -s
  done
then aggregate RMSNorm/Slice/Concat device-kernel over the signpost window for each.
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
_M, _NH, _HD = 32, 3, 256  # per-chip: 3 q-heads x head_dim 256 = 768
_W = _NH * _HD
_ITERS = int(os.environ.get("QWEN36_HEAD_ITERS", "30"))
_MEM = os.environ.get("QWEN36_HEAD_MEM", "dram").lower()


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
def test_head_path_mem(bh_glx_mesh):
    mesh = bh_glx_mesh
    print(f"\n[head-mem] MEM={_MEM} shape=[{_M},{_W}] (n_heads={_NH} hd={_HD}) iters={_ITERS}", flush=True)

    # Three modes:
    #   dram     = DRAM-interleaved (current)
    #   l1       = L1-interleaved (same layout, L1 tier)
    #   l1shard  = L1 WIDTH-SHARDED by head (1 head/core) — the COMPUTE-op layout (rms_norm reduces
    #              over the full head_dim within a core). Manipulation (concat) needs interleaved, so
    #              in l1shard mode we time the qk_norm rms_norm ONLY (the compute op).
    if _MEM == "l1shard":
        mc = ttnn.create_sharded_memory_config(
            shape=(_M, _HD),
            core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0))]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif _MEM == "l1":
        mc = ttnn.L1_MEMORY_CONFIG
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG

    torch.manual_seed(0)
    x = torch.randn(1, 1, _M, _W) * 0.1
    g = torch.ones(1, 1, 1, _HD)
    x_t = ttnn.from_torch(
        x,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    # per-head view [1,1,M*NH? ] - keep it simple: norm over the last hd within a [M, NH, hd] view
    g_t = ttnn.from_torch(
        g,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

    def dance():
        # qk_norm rms_norm (the COMPUTE op) — always measured.
        n = ttnn.rms_norm(x_t, weight=None, epsilon=1e-6, memory_config=mc, compute_kernel_config=ck)
        if _MEM == "l1shard":
            return n  # compute-op-only comparison (concat needs interleaved; not in this mode)
        # full manipulation dance (interleaved-friendly): slice (rope rotate_half) + concat
        a = ttnn.slice(n, [0, 0, 0, 0], [1, 1, _M, _W // 2], memory_config=mc)
        b = ttnn.slice(n, [0, 0, 0, _W // 2], [1, 1, _M, _W], memory_config=mc)
        c = ttnn.concat([b, a], dim=3, memory_config=mc)
        n.deallocate(True)
        a.deallocate(True)
        b.deallocate(True)
        return c

    o = dance()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    signpost("start")
    for _ in range(_ITERS):
        o = dance()
        o.deallocate(True)
    ttnn.synchronize_device(mesh)
    signpost("stop")
    print(f"[head-mem] done MEM={_MEM}: aggregate RMSNorm/Slice/Concat device-kernel /{_ITERS}", flush=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
