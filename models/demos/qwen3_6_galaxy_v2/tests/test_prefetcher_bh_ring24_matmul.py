# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decisive experiment: ring-24 (8 banks x 3 receivers) prefetched matmul on BH.

This is the test_prefetcher_BH PASSING class (num_receiver_cores <= 3, default
placement). If this PCC-passes in the qwen3.6 prefetcher setup, the prefetcher
matmul path works here and the ring-40 blocker is specifically the >3-receiver
override placement (so ring-40 is not prefetcher-viable; prefetcher => ring-24).
If this fails the same 'cores not contained in global_cb' way, the matmul wiring
has a deeper issue independent of receiver count.

FF1 shape: K=dim_per_tp=1280, N=intermediate_per_tp=2176, ring-24.

Run:
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_bh_ring24_matmul.py -v -s
"""
from __future__ import annotations

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_MESH_SHAPE = (8, 4)
_RING = 24
_BANKS = 8
_K = 1280
_N = 2176


def _round_up(v, m):
    return ((v + m - 1) // m) * m


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH_SHAPE), trace_region_size=20_000_000)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.xfail(
    reason="DECISIVE RESULT: ring-24 (8 banks x 3, the test_prefetcher_BH PASSING class) fails the SAME "
    "'cores not contained in global_cb' as ring-40 -> the blocker is NOT ring size / receiver count, it is "
    "that get_bh_prefetcher_core_ranges' receiver placement (used to build our global_cb) does not match the "
    "matmul's weight-read CB cores. The working tt_transformers.Prefetcher uses a different decode receiver / "
    "global_cb construction. FIX = mirror that, not our allocation-only layout.",
    strict=False,
)
@pytest.mark.hardware
def test_ring24_prefetched_matmul(bh_glx_mesh):
    from models.demos.qwen3_6_galaxy_v2.tt.prefetcher_common import TtLlamaPrefetcherSetup

    mesh = bh_glx_mesh
    print(f"\n[ring24-mm] is_blackhole={is_blackhole()}", flush=True)

    N_pad = _round_up(math.ceil(_N / _RING), ttnn.TILE_SIZE) * _RING  # 2176 -> 2304
    K_pad = _round_up(math.ceil(_K / _RING), ttnn.TILE_SIZE) * _RING  # 1280 -> 1536
    per_core_N = N_pad // _RING // ttnn.TILE_SIZE
    k_per_shard = _round_up(math.ceil(_K / _RING), ttnn.TILE_SIZE)
    print(f"[ring24-mm] K={_K}->{K_pad} N={_N}->{N_pad} per_core_N={per_core_N} k_per_shard={k_per_shard}", flush=True)

    pf = TtLlamaPrefetcherSetup(mesh, n_tensors=1, n_layers=1, mode="decode", is_qwen=True)  # ring40=False -> 8x3
    pf.create_global_cb()
    recv_crs = ttnn.CoreRangeSet([cr for crs in pf.all_receiver_cores for cr in crs.ranges()])

    torch.manual_seed(0)
    w = torch.randn(_K, _N) * 0.05
    w_pad = torch.nn.functional.pad(w, (0, N_pad - _N), value=0.0)
    w_pad = torch.nn.functional.pad(w_pad, (0, 0, 0, K_pad - _K), value=0.0)  # [K_pad, N_pad]
    n_banks = len(pf.dram_cores)
    dram_shard = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(pf.dram_core_range_set, [K_pad, N_pad // n_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    w_tt = ttnn.from_torch(
        w_pad.unsqueeze(0).unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_shard,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    pf.insert_tensor(w_tt)
    addr = pf.get_tensor_addrs()
    print("[ring24-mm] BEFORE dram_prefetcher", flush=True)
    ttnn.dram_prefetcher([w_tt, addr], 1, pf.global_circular_buffer)
    mesh.set_sub_device_stall_group([pf.worker_sub_device_id])
    print("[ring24-mm] AFTER dram_prefetcher", flush=True)

    x = torch.randn(1, 1, 32, _K) * 0.1
    x_pad = torch.nn.functional.pad(x, (0, K_pad - _K), value=0.0)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(32, k_per_shard),
        core_grid=recv_crs,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x_tt = ttnn.from_torch(
        x_pad,
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
        compute_with_storage_grid_size=(_BANKS, _RING // _BANKS),  # (8, 3)
        in0_block_w=K_pad // _RING // ttnn.TILE_SIZE,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=32 // ttnn.TILE_SIZE,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        num_global_cb_receivers=3,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    print("[ring24-mm] BEFORE ring matmul", flush=True)
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
    print("[ring24-mm] AFTER ring matmul — NO DEADLOCK", flush=True)

    ref = (x.reshape(32, _K).float() @ w.float())[:, :_N]
    worst = 1.0
    for t in ttnn.get_device_tensors(out):
        got = t.cpu().to_torch().float().reshape(32, N_pad)[:, :_N]
        _, msg = comp_pcc(got, ref)
        try:
            pcc = float(msg.split("PCC:")[-1].strip())
        except Exception:  # noqa: BLE001
            pcc = -1.0
        worst = min(worst, pcc)
    print(f"[ring24-mm] worst-device PCC = {worst:.5f} -> {'PASS' if worst > 0.99 else 'FAIL'}", flush=True)
    mesh.reset_sub_device_stall_group()
    assert worst > 0.99, f"ring-24 prefetched matmul PCC too low: {worst}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
