# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate the reuse-Prefetcher-class FF1 matmul on the 32-chip 8x4 BH_GLX galaxy.

test_prefetcher_class_qwen36_ff1 proved the Prefetcher class drives a qwen3.6 FF1
ring-24 matmul on the 1x8 mesh. But qwen3.6's real target is the 32-chip 8x4 galaxy,
and the prefetcher's support gate SKIPS num_devices not in {2,4,8} (and is_prefetcher_supported
fails for 32: n_kv 8%32 and a lower L1 budget). The prefetcher is per-chip, so it SHOULD
behave identically on 8x4 — this test confirms that on real hardware.

The support gate (KV-head divisibility + L1 fit) is irrelevant to a single replicated FF1
matmul, so we monkeypatch is_prefetcher_supported -> True to exercise the per-chip mechanism.

WIN: matmul completes (no deadlock) AND PCC > 0.99 across all 32 devices.

Run:
    MESH_DEVICE=BH_GLX HF_MODEL=Qwen3-32B python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_class_qwen36_ff1_glx.py -v -s
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_MESH_SHAPE = (8, 4)
_K = 1280
_N = 2176
_NUM_RECV = 3


def round_up(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH_SHAPE), trace_region_size=23887872)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.hardware
def test_prefetcher_class_qwen36_ff1_glx(bh_glx_mesh):
    import models.tt_transformers.tt.prefetcher as pf_mod
    from models.tt_transformers.tt.common import Mode
    from models.tt_transformers.tt.prefetcher import Prefetcher

    os.environ.setdefault("HF_MODEL", "Qwen3-32B")
    # The support gate concerns KV-head sharding + L1 fit for a full model; irrelevant to a
    # single replicated FF1 matmul. Patch it so we can exercise the per-chip mechanism on 8x4.
    pf_mod.is_prefetcher_supported = lambda *a, **k: True

    mesh = bh_glx_mesh
    print(
        f"\n[pf-glx-ff1] is_blackhole={is_blackhole()} mesh={tuple(mesh.shape)} ndev={mesh.get_num_devices()}",
        flush=True,
    )

    prefetcher = Prefetcher(mesh, num_tensors=1, num_layers=1, num_receiver_cores=_NUM_RECV)
    prefetcher.init(mode=Mode.DECODE)
    ring_size = prefetcher.ring_size
    dram_cores = len(prefetcher.dram_banks())
    print(f"[pf-glx-ff1] ring_size={ring_size} dram_banks={dram_cores}", flush=True)

    N_pad = round_up(math.ceil(_N / ring_size), ttnn.TILE_SIZE) * ring_size
    per_core_N = N_pad // ring_size // ttnn.TILE_SIZE
    k_per_shard = round_up(math.ceil(_K / ring_size), ttnn.TILE_SIZE)

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
        device=mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_shard,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    prefetcher.insert_tensor(w_tt)
    print("[pf-glx-ff1] BEFORE prefetcher.run()", flush=True)
    prefetcher.run()
    print("[pf-glx-ff1] AFTER prefetcher.run()", flush=True)

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
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
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
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(dram_cores, ring_size // dram_cores),
        in0_block_w=max(1, _K // ring_size // ttnn.TILE_SIZE),
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
    print("[pf-glx-ff1] BEFORE ring matmul", flush=True)
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
    ttnn.synchronize_device(mesh, sub_device_ids=[prefetcher.worker_sub_device_id])
    print("[pf-glx-ff1] AFTER ring matmul — NO DEADLOCK", flush=True)

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
    print(f"[pf-glx-ff1] worst-of-32-device PCC = {worst:.5f} -> {'PASS' if worst > 0.99 else 'FAIL'}", flush=True)
    mesh.reset_sub_device_stall_group()
    assert worst > 0.99, f"Prefetcher-class qwen3.6 FF1 ring-24 matmul on 8x4 PCC too low: {worst}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
