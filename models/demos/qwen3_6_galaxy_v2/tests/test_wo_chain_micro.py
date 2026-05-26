# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolate the FA-style pre-RS op chain vs the DN-style pre-RS op chain.

Empirical observation from the warm-only 1L tracy (2026_05_20_07_15_53):
- FA WO 1280×BF16 RS device kernel duration: mean 666 µs, max 2786 µs (bimodal)
- DN out_proj 1280×BF16 RS device kernel duration: mean 64 µs, max 110 µs
- Same shape, same dtype, same cluster_axis, same ring size, same hardware
- Op preceding FA RS: ShardedToInterleavedDeviceOperation (L1→DRAM)
- Op preceding DN RS: MatmulDeviceOperation (writes directly to DRAM)

Hypothesis: FA's V2-DRAM-P1 fast path inserts an extra L1→DRAM conversion
between the WO matmul and the all_reduce. This conversion has variable kernel
duration across chips, causing some chips to enter the RS ring late.
The other chips' RS kernels wait, inflating their `DEVICE KERNEL DURATION`.

This driver replicates both chains in isolation with tracy signposts around
the warm calls only, so the per-op DEVICE KERNEL DURATION and OP_TO_OP_LATENCY
can be compared cleanly.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_wo_chain_micro.py -v -s
"""
from __future__ import annotations

import pytest
import torch

import ttnn

_MESH = (8, 4)
_H = 5120  # full residual dim
_PER_CHIP_W = 1280  # H / 4 cols
_M = 32  # tile-padded T
_K = 768  # WO input dim per chip (n_q_per_chip × hd = 3 × 256 in qwen3.6 FA, 3 × 256 in DN out_proj per-row)


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


def _make_weight(mesh, k, n):
    """Per-chip weight [K, N] in DRAM-interleaved tile layout, BFLOAT16."""
    torch.manual_seed(7)
    w_full = torch.randn(_MESH[0] * _MESH[1], k, n, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        w_full.reshape(_MESH[0], _MESH[1], k, n),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
    )


def _make_act(mesh, m, k):
    """Per-chip activation [M=32, K=768] in DRAM-interleaved tile layout, BFLOAT16."""
    torch.manual_seed(99)
    a_full = torch.randn(_MESH[0] * _MESH[1], m, k, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        a_full.reshape(_MESH[0], _MESH[1], m, k),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
    )


def _l1_width_sharded_memcfg(width_tot, n_cores=10):
    """L1 width-sharded memcfg analogous to the V2-DRAM-P1 wo output."""
    per_core_w = width_tot // n_cores
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet([core_range]), [_M, per_core_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.hardware
def test_wo_chain_dn_vs_fa(bh_glx_mesh):
    """Compare DN-style chain (matmul → all_reduce) vs FA-style chain
    (matmul → L1→DRAM → all_reduce) for the same final RS shape/dtype.

    Each variant runs:
      1. Build a fresh activation + weight on the mesh
      2. Compile pass (1 iter, NOT signposted)
      3. signpost('warm_start')
      4. 4 warm iterations of the FULL chain
      5. signpost('stop')

    Tracy CSV per-op:
      - DEVICE KERNEL DURATION on the final RS for each warm iter
      - OP_TO_OP_LATENCY on the final RS for each warm iter
      - The op preceding the RS (Matmul for DN, ShardedToInterleaved for FA)

    Compare distributions across the two variants.
    """
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *a, **k: None  # noqa: E731

    print(f"\n=== DN-style chain (matmul → DRAM → all_reduce) ===")
    a = _make_act(bh_glx_mesh, _M, _K)
    w = _make_weight(bh_glx_mesh, _K, _PER_CHIP_W)

    def dn_chain():
        out = ttnn.linear(a, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        red = ttnn.all_reduce(out, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(red)

    # Compile pass (NOT signposted)
    dn_chain()
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"  compile pass done")

    ttnn.ReadDeviceProfiler(bh_glx_mesh)
    signpost("dn_warm_start")
    for _ in range(4):
        dn_chain()
    ttnn.synchronize_device(bh_glx_mesh)
    signpost("dn_warm_done")
    print(f"  4 warm DN chain iters done")

    print(f"\n=== FA-style chain (matmul → L1 → DRAM → all_reduce) ===")
    wo_l1_memcfg = _l1_width_sharded_memcfg(_PER_CHIP_W, n_cores=10)

    def fa_chain():
        out_l1 = ttnn.linear(a, w, dtype=ttnn.bfloat16, memory_config=wo_l1_memcfg)
        out_dram = ttnn.to_memory_config(out_l1, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_l1)
        red = ttnn.all_reduce(out_dram, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_dram)
        ttnn.deallocate(red)

    # Compile pass
    fa_chain()
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"  compile pass done")

    signpost("fa_warm_start")
    for _ in range(4):
        fa_chain()
    ttnn.synchronize_device(bh_glx_mesh)
    signpost("fa_warm_done")
    print(f"  4 warm FA chain iters done")

    print(f"\n=== DONE — see tracy CSV for per-op kernel durations ===")
    print(f"Signposts: dn_warm_start..dn_warm_done (DN chain), fa_warm_start..fa_warm_done (FA chain)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
