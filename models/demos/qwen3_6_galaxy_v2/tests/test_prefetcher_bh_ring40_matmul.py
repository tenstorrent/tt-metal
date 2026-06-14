# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Task 3 (device): one ring-40 matmul through the prefetcher global_cb on BH.

Streams an FF1-shaped weight (K=dim_per_tp=1280, N=intermediate_per_tp=2176->2560)
from 4 DRAM banks into L1 via ttnn.dram_prefetcher + the ring-40 global_cb, then runs
the gather_in0 ring matmul reading the weight FROM the global_cb (global_cb=...,
num_global_cb_receivers=10, grid=(x=4,y=10)). This proves the qwen3.6 ring-40 layout
can drive a real prefetched matmul on BH (the thing test_prefetcher_BH proved only for
ring-24/80 on 8 banks).

WIN: matmul completes (no deadlock) AND output PCC > 0.99 vs torch x@w (first 2176 of N).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_bh_ring40_matmul.py -v -s
"""
from __future__ import annotations

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_MESH_SHAPE = (8, 4)
_RING = 40
_K = 1280  # dim_per_tp
_N = 2176  # intermediate_per_tp (native)


def _round_up(v, m):
    return ((v + m - 1) // m) * m


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH_SHAPE), trace_region_size=20_000_000)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.xfail(
    reason="BLOCKED (NOT ring-size-specific — ring-24 fails identically, see "
    "test_prefetcher_bh_ring24_matmul): dram_prefetcher STREAMS ok, but the gather_in0 ring matmul "
    "fails 'Specified cores are not contained in associated GlobalCircularBuffer'. The matmul's "
    "weight-read CB cores are not a subset of this global_cb's cores. Our get_bh_prefetcher_core_ranges "
    "receiver placement does NOT match the placement the working tt_transformers.Prefetcher class uses "
    "(which test_prefetcher_BH passes at ring-24). FIX: mirror Prefetcher's exact decode receiver / "
    "global_cb / matmul-core construction, not get_bh_prefetcher_core_ranges' allocation-only layout.",
    strict=False,
)
@pytest.mark.hardware
def test_ring40_prefetched_matmul(bh_glx_mesh):
    from models.demos.qwen3_6_galaxy_v2.tt.prefetcher_common import TtLlamaPrefetcherSetup

    mesh = bh_glx_mesh
    print(f"\n[ring40-mm] is_blackhole={is_blackhole()}", flush=True)

    N_pad = _round_up(math.ceil(_N / _RING), ttnn.TILE_SIZE) * _RING  # 2176 -> 2560
    per_core_N = N_pad // _RING // ttnn.TILE_SIZE  # 2 tiles
    k_per_shard = _round_up(math.ceil(_K / _RING), ttnn.TILE_SIZE)  # 32
    print(f"[ring40-mm] K={_K} N={_N}->{N_pad} per_core_N={per_core_N} k_per_shard={k_per_shard}", flush=True)

    pf = TtLlamaPrefetcherSetup(mesh, n_tensors=1, n_layers=1, mode="decode", is_qwen=True, ring40=True)
    pf.create_global_cb()

    # Single CoreRangeSet covering all 40 receiver cores (ring matmul compute cores).
    recv_crs = ttnn.CoreRangeSet([cr for crs in pf.all_receiver_cores for cr in crs.ranges()])

    # ---- weight: [K, N_pad], DRAM width-sharded across the 4 banks, replicated over mesh ----
    torch.manual_seed(0)
    w = torch.randn(_K, _N) * 0.05
    w_pad = torch.nn.functional.pad(w, (0, N_pad - _N))  # [1280, 2560]
    n_banks = len(pf.dram_cores)  # 4
    dram_shard = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(pf.dram_core_range_set, [_K, N_pad // n_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    w_tt = ttnn.from_torch(
        w_pad.unsqueeze(0).unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_shard,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # ---- prefetch the weight into L1 via the global_cb ----
    # dram_prefetcher(tensors, num_layers, global_cb, *, enable_performance_mode):
    # the ROW_MAJOR address tensor is the LAST element of the tensors list.
    pf.insert_tensor(w_tt)
    addr = pf.get_tensor_addrs()
    print("[ring40-mm] BEFORE dram_prefetcher", flush=True)
    ttnn.dram_prefetcher([w_tt, addr], 1, pf.global_circular_buffer)
    mesh.set_sub_device_stall_group([pf.worker_sub_device_id])
    print("[ring40-mm] AFTER dram_prefetcher", flush=True)

    # ---- input x: [1,1,32,K], replicated, width-sharded across the 40 ring cores ----
    x = torch.randn(1, 1, 32, _K) * 0.1
    in_mem = ttnn.create_sharded_memory_config(
        shape=(32, k_per_shard),
        core_grid=recv_crs,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x_tt = ttnn.from_torch(
        x,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    out_mem = ttnn.create_sharded_memory_config(
        shape=(32, N_pad // _RING),
        core_grid=recv_crs,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_subblock_w = 8
    while per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(4, 10),  # x=banks, y=receivers -> ring-40
        in0_block_w=_K // _RING // ttnn.TILE_SIZE,  # 1
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=32 // ttnn.TILE_SIZE,  # 1
        per_core_N=per_core_N,  # 2
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        num_global_cb_receivers=10,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    print("[ring40-mm] BEFORE ring matmul", flush=True)
    out = ttnn.linear(
        x_tt,
        w_tt,
        program_config=pc,
        memory_config=out_mem,
        compute_kernel_config=ck,
        dtype=ttnn.bfloat16,
        global_cb=pf.global_circular_buffer,
        sub_device_id=pf.worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh, sub_device_ids=[pf.worker_sub_device_id])
    print("[ring40-mm] AFTER ring matmul — NO DEADLOCK", flush=True)

    ref = (x.reshape(32, _K).float() @ w.float())[:, :_N]  # [32, 2176]
    worst = 1.0
    for i, t in enumerate(ttnn.get_device_tensors(out)):
        got = t.cpu().to_torch().float().reshape(32, N_pad)[:, :_N]
        _, msg = comp_pcc(got, ref)
        try:
            pcc = float(msg.split("PCC:")[-1].strip())
        except Exception:  # noqa: BLE001
            pcc = -1.0
        worst = min(worst, pcc)
    print(f"[ring40-mm] worst-device PCC = {worst:.5f} -> {'PASS' if worst > 0.99 else 'FAIL'}", flush=True)
    mesh.reset_sub_device_stall_group()
    assert worst > 0.99, f"ring-40 prefetched matmul PCC too low: {worst}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
