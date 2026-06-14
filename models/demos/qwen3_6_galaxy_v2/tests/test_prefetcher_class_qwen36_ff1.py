# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate the REUSE-Prefetcher-class approach on a qwen3.6 FF1-shaped ring-24 matmul.

Context: qwen3.6's hand-rolled prefetcher_common fails the prefetched ring matmul
('cores not contained in global_cb') even at ring-24, while the OFFICIAL
test_prefetcher_BH passes ring-24 (num_receiver_cores=3) on this box. So the fix is
to REUSE tt_transformers.tt.prefetcher.Prefetcher's global_cb / receiver-core /
matmul-core construction. This unit test proves that path works for qwen3.6's FF1
per-device shape (K=dim_per_tp=1280, N=intermediate_per_tp=2176 -> 2304 @ ring-24)
BEFORE refactoring prefetcher_common.

Uses the Prefetcher class exactly like test_prefetcher_BH (prefetcher.global_cb,
prefetcher.receiver_cores, prefetcher.to_core_range_set, prefetcher.run()), on the
SUPPORTED 1x8 mesh (P150x8). HF_MODEL=Qwen3-32B (verified; dim 5120 matches qwen3.6)
only to satisfy the prefetcher support gate — the matmul shape is qwen3.6's FF1.

WIN: matmul completes (no deadlock) AND PCC > 0.99 vs torch x@w.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=P150x8 HF_MODEL=Qwen3-32B python -m pytest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_class_qwen36_ff1.py -v -s
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def round_up(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


# qwen3.6 FF1 per-device shape (2D galaxy TP): K=dim_per_tp, N=intermediate_per_tp.
_K = 1280
_N = 2176
_NUM_RECV = 3  # ring-24 = 8 banks * 3 (the test_prefetcher_BH PASSING class)


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.parametrize(
    "mesh_device",
    [{"P300": (1, 2), "P150x4": (1, 4), "P150x8": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 8))],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_prefetcher_class_qwen36_ff1(mesh_device, function_level_defaults):
    from models.tt_transformers.tt.common import Mode
    from models.tt_transformers.tt.prefetcher import Prefetcher

    os.environ.setdefault("HF_MODEL", "Qwen3-32B")
    print(f"\n[pf-class-ff1] is_blackhole={is_blackhole()} mesh={tuple(mesh_device.shape)}", flush=True)

    prefetcher = Prefetcher(mesh_device, num_tensors=1, num_layers=1, num_receiver_cores=_NUM_RECV)
    prefetcher.init(mode=Mode.DECODE)
    ring_size = prefetcher.ring_size  # 24
    dram_cores = len(prefetcher.dram_banks())
    print(
        f"[pf-class-ff1] ring_size={ring_size} dram_banks={dram_cores} num_recv={prefetcher.num_receiver_cores}",
        flush=True,
    )

    N_pad = round_up(math.ceil(_N / ring_size), ttnn.TILE_SIZE) * ring_size
    per_core_N = N_pad // ring_size // ttnn.TILE_SIZE
    k_per_shard = round_up(math.ceil(_K / ring_size), ttnn.TILE_SIZE)
    print(f"[pf-class-ff1] K={_K} N={_N}->{N_pad} per_core_N={per_core_N} k_per_shard={k_per_shard}", flush=True)

    # ---- weight [K, N_pad], DRAM width-sharded across the logical dram grid (cols 0..dram_cores-1, row 0) ----
    torch.manual_seed(0)
    w = torch.randn(_K, _N) * 0.05
    w_pad = torch.nn.functional.pad(w, (0, N_pad - _N), value=0.0)
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    dram_shard = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (_K, N_pad // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )
    w_tt = ttnn.as_tensor(
        w_pad.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_shard,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    prefetcher.insert_tensor(w_tt)
    print("[pf-class-ff1] BEFORE prefetcher.run()", flush=True)
    prefetcher.run()  # dram_prefetcher via the class
    print("[pf-class-ff1] AFTER prefetcher.run()", flush=True)

    # ---- receiver core range set (the matmul compute cores), via the class ----
    recv_crs = prefetcher.to_core_range_set(prefetcher.receiver_cores(sender_active=True, receiver_active=True))

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
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_mem = ttnn.create_sharded_memory_config(
        shape=(32, N_pad // ring_size),
        core_grid=recv_crs,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    out_subblock_w = 8
    while per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    grid_x = dram_cores
    grid_y = ring_size // dram_cores
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=_K // ring_size // ttnn.TILE_SIZE if (_K // ring_size // ttnn.TILE_SIZE) > 0 else 1,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        num_global_cb_receivers=prefetcher.num_receiver_cores,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    print("[pf-class-ff1] BEFORE ring matmul", flush=True)
    out = ttnn.linear(
        x_tt,
        w_tt,
        program_config=pc,
        memory_config=out_mem,
        compute_kernel_config=ck,
        dtype=ttnn.bfloat16,
        global_cb=prefetcher.global_cb,
        sub_device_id=prefetcher.worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[prefetcher.worker_sub_device_id])
    print("[pf-class-ff1] AFTER ring matmul — NO DEADLOCK", flush=True)

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
    print(f"[pf-class-ff1] worst-device PCC = {worst:.5f} -> {'PASS' if worst > 0.99 else 'FAIL'}", flush=True)
    mesh_device.reset_sub_device_stall_group()
    assert worst > 0.99, f"Prefetcher-class qwen3.6 FF1 ring-24 matmul PCC too low: {worst}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
