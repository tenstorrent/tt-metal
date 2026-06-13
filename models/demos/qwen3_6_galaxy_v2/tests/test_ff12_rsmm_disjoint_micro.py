# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Probe: does the FF12 rs-matmul fusion (matmul_line_reduce_scatter) work at
ring-40 when the matmul is placed DISJOINT from the RS packet workers?

matmul_line_reduce_scatter runs the w3-matmul AND the w1 reduce-scatter
concurrently. It DEADLOCKED at ring-40 because the matmul (RING40_MM_CRS, cols
1,2,4,5) overlapped PACKET_WORKER_CRS (cols 1-3, the RS workers) — confirmed in
test_ff12_matmul_rs_poolstate_micro (hangs iter 0 even in isolation). NOT a
pool-state issue, NOT a missing op: llama_reduce_scatter_matmul is ring-40
capable (ring_devices=4, input 2560 % 4 == 0).

FIX (same recipe as the FF2 all_gather_matmul disjoint-core fix): place the
rs-matmul's matmul on RING40_RSMM_MM_CRS = cols 4-11 x rows 0-4 (40 cores),
disjoint from PACKET_WORKER_CRS (cols 1-3) and dispatch (col 0). Then the
concurrent w3-matmul + w1-RS don't collide.

This drives ONE matmul_line_reduce_scatter at ring-40 with the disjoint
placement, decode-mode tt_ccl (so reduce_scatter_buffers exist), in ISOLATION.
WIN = (a) it completes (no deadlock) and (b) the w1 reduce-scatter output PCC > 0.99.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_ff12_rsmm_disjoint_micro.py -s -x
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    bh_glx_mesh,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_CLUSTER_SHAPE = (8, 4)


@pytest.mark.hardware
def test_ff12_rsmm_disjoint(bh_glx_mesh):
    mesh_device = bh_glx_mesh
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FUSE_RS_MATMUL": "1",
    }.items():
        os.environ.setdefault(_k, _v)

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    state_dict = _load_full_state_dict(_SNAPSHOT)
    pfx0 = "model.language_model.layers.0."
    all_pfx = "model.language_model.layers."
    state_dict = {k: v for k, v in state_dict.items() if (not k.startswith(all_pfx)) or k.startswith(pfx0)}
    paged = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh_device, state_dict, ["linear_attention"], 1, paged)

    # decode-mode tt_ccl so reduce_scatter_buffers / packet workers exist.
    model.switch_mode("decode")
    mlp = model.layers[0].feed_forward
    mc = model.model_config
    tt_ccl = mlp.tt_ccl
    ck = args.compute_kernel_config_hifi2
    wsd = tt_ccl.worker_sub_device_id
    M, dim_per_tp = 32, args.dim_per_tp  # 1280

    torch.manual_seed(0)
    in_full = torch.randn(*_CLUSTER_SHAPE, M, dim_per_tp)  # [8,4,32,1280]
    ff_in = ttnn.from_torch(
        in_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_CLUSTER_SHAPE),
    )
    # ring_in on the DISJOINT RSMM placement (cols 4-11).
    ring_in = ttnn.to_memory_config(ff_in, mc["SHARDED_FF12_RSMM_IN_MEMCFG"])

    # w1 plain matmul on RSMM cores (DRAM-interleaved weight fed directly).
    w1_raw = ttnn.linear(
        ring_in,
        mlp.w1_ring40,
        compute_kernel_config=ck,
        dtype=ttnn.bfloat8_b,
        program_config=mc["FF1_3_RING40_PROGCFG"],
        memory_config=mc["SHARDED_FF12_RSMM_OUT_MEMCFG"],
        sub_device_id=wsd,
    )
    print("[rsmm] BEFORE matmul_line_reduce_scatter (disjoint cols 4-11; packet workers cols 1-3)", flush=True)
    # FUSED: w3-matmul (on RSMM cols 4-11) + w1 reduce-scatter (packet workers cols 1-3).
    w1_red, w3_out = tt_ccl.matmul_line_reduce_scatter(
        ring_in,
        mlp.w3_ring40,
        w1_raw,
        cluster_axis=1,
        num_links=mc["GALAXY_NUM_LINKS"],
        RS_memory_config=mc["REDUCE_SCATTER_OUT_RING40_FF12_MEMCFG"],
        compute_kernel_config=ck,
        dtype=ttnn.bfloat8_b,
        program_config=mc["FF1_3_RING40_PROGCFG"],
        memory_config=mc["SHARDED_FF12_RSMM_OUT_MEMCFG"],
        global_cb=None,
        sub_device_id=wsd,
        use_noc1_only=False,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd])
    print("[rsmm] AFTER matmul_line_reduce_scatter — NO DEADLOCK", flush=True)

    # ---- Torch golden for w1_red (the reduce-scatter of w1_raw) ----
    # w1_raw[r,c] = in_full[r,c] @ w1_dev[r,c]  (2560 partial). RS sums over the 4
    # cols -> full 2560, then scatters this col's 640 slice.
    w1_dev = [ttnn.get_device_tensors(mlp.w1_ring40)[i].cpu().to_torch().float() for i in range(32)]
    w1_dev = [w.reshape(w.shape[-2], w.shape[-1]) for w in w1_dev]  # [1280,2560]
    worst = 1.0
    for i, t in enumerate(ttnn.get_device_tensors(w1_red)):
        r, c = i // _CLUSTER_SHAPE[1], i % _CLUSTER_SHAPE[1]
        full = sum(in_full[r, cc] @ w1_dev[r * _CLUSTER_SHAPE[1] + cc] for cc in range(_CLUSTER_SHAPE[1]))  # [32,2560]
        ref = full[:, c * 640 : (c + 1) * 640]  # this col's 640 slice
        got = t.cpu().to_torch().float().reshape(M, -1)[:, :640]
        eq, msg = comp_pcc(got, ref)
        try:
            pcc = float(msg.split("PCC:")[-1].strip())
        except Exception:  # noqa: BLE001
            pcc = -1.0
        worst = min(worst, pcc)
    print(
        f"[rsmm] w1 reduce-scatter worst-device PCC = {worst:.5f}  -> {'PASS' if worst > 0.99 else 'FAIL'}", flush=True
    )
    assert worst > 0.99, f"FF12 rs-matmul fusion (disjoint placement) PCC too low: {worst}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
