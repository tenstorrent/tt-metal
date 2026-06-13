# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Micro-test isolating the FF12 ring-40 stage of the fused decode MLP.

The fused ``_fuse_rs`` branch in ``tt/llama_decoder.py::_mlp_decode_qwen36``
(env ``QWEN36_FUSE_RS_MATMUL=1``, default OFF) crashes on the FIRST FF12 op:
``TT_FATAL: Dimension K must match in bmm_op``. Task 2 only validated the FF2
all_gather_matmul wrapper; the FF12 ring-40 stage
(``matmul_line_reduce_scatter`` + ``FF1_3_RING40_PROGCFG`` +
``SHARDED_FF12_RING40_MEMCFG``) was never unit-tested.

This drives ONLY the FF12 w1 ring-40 matmul on the REAL ``mlp.w1_ring40``
weight, with the EXACT decode input layout ``_mlp_decode_qwen36`` receives
(``ff_in_sharded`` = DRAM, col-sharded ``[1,1,32,1280]`` across cluster_axis=1),
resharded to ``SHARDED_FF12_RING40_MEMCFG`` -> ``ring_in``. It PRINTS the shapes
that pinpoint the K-mismatch and asserts FF12 output PCC > 0.99 vs torch.

Built on the same scaffold as the PASSING
``tests/test_all_gather_matmul_wrapper_pcc.py`` (real bh_glx mesh + 1-layer model
+ clean teardown).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_ff12_ring40_rs_micro.py -s -x
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


def _describe_memcfg(name, mc):
    try:
        ss = mc.shard_spec
        ncores = 0
        for r in ss.grid.ranges():
            s, e = r.start, r.end
            ncores += (e.x - s.x + 1) * (e.y - s.y + 1)
        print(f"[ff12_micro]   {name}: layout={mc.memory_layout} shard_shape={list(ss.shape)} num_cores={ncores}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ff12_micro]   {name}: (no shard_spec) {mc}  ({exc})")


def _describe_progcfg(name, pc):
    print(
        f"[ff12_micro]   {name}: grid={pc.compute_with_storage_grid_size} "
        f"in0_block_w={pc.in0_block_w} per_core_M={pc.per_core_M} per_core_N={pc.per_core_N} "
        f"out_subblock_h={pc.out_subblock_h} out_subblock_w={pc.out_subblock_w} gather_in0={pc.gather_in0}"
    )


@pytest.mark.hardware
def test_ff12_ring40_rs_micro(bh_glx_mesh):
    mesh_device = bh_glx_mesh

    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FUSE_RS_MATMUL": "1",  # ensures the decode-mode ring-40 weights are built
    }.items():
        os.environ.setdefault(_k, _v)

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    pattern = ["linear_attention"]
    state_dict = _load_full_state_dict(_SNAPSHOT)
    out = {}
    pfx0 = "model.language_model.layers.0."
    all_pfx = "model.language_model.layers."
    for k, v in state_dict.items():
        if k.startswith(all_pfx):
            if k.startswith(pfx0):
                out[k] = v
            else:
                continue
        else:
            out[k] = v
    state_dict = out

    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh_device, state_dict, pattern, 1, paged_attention_config)

    mlp = model.layers[0].feed_forward
    mc = model.model_config

    # ---- Decode input layout: DRAM, col-sharded [1,1,32,1280] across cluster_axis=1 ----
    # This is exactly what _mlp_decode_qwen36 receives as ``ff_in_sharded``.
    M, dim_per_tp = 32, args.dim_per_tp  # 1280
    torch.manual_seed(0)
    # Full hidden dim across the 4 cols = dim_per_tp * cluster_axis(4)... but the
    # decode input is ALREADY col-sharded: each col holds its own 1280 slice of
    # activations and matmuls its own K-shard of w1, producing a PARTIAL N-sum that
    # the reduce-scatter then sums across the 4 cols. So per (row,col) device the
    # input is an independent [32,1280].
    in_full = torch.randn(*_CLUSTER_SHAPE, M, dim_per_tp)  # [8,4,32,1280]
    ff_in_sharded = ttnn.from_torch(
        in_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_CLUSTER_SHAPE),
    )

    print("[ff12_micro] ===== INPUT / CONFIG SHAPES =====")
    print(
        f"[ff12_micro]   ff_in_sharded (DRAM col-sharded) shape = {list(ff_in_sharded.shape)}  dtype={ff_in_sharded.dtype}"
    )
    print(f"[ff12_micro]   mlp.w1_ring40 shape = {list(mlp.w1_ring40.shape)}  dtype={mlp.w1_ring40.dtype}")
    print(f"[ff12_micro]   mlp.w3_ring40 shape = {list(mlp.w3_ring40.shape)}  dtype={mlp.w3_ring40.dtype}")
    _describe_memcfg("SHARDED_FF12_RING40_MEMCFG", mc["SHARDED_FF12_RING40_MEMCFG"])
    _describe_memcfg("SHARDED_FF12_OUT_RING40_MEMCFG", mc["SHARDED_FF12_OUT_RING40_MEMCFG"])
    _describe_progcfg("FF1_3_RING40_PROGCFG", mc["FF1_3_RING40_PROGCFG"])

    # ---- Reshard to the ring-40 FF12 input layout (the line under test) ----
    ring_in = ttnn.to_memory_config(ff_in_sharded, mc["SHARDED_FF12_RING40_MEMCFG"])
    print(f"[ff12_micro]   ring_in shape = {list(ring_in.shape)}")
    _describe_memcfg("ring_in.memory_config()", ring_in.memory_config())

    fused_ck = args.compute_kernel_config_hifi2

    # HYPOTHESIS PROBE: the non-prefetcher gather_in0 ring matmul needs the weight
    # L1-WIDTH-SHARDED over the SAME 40 ring cores (exactly like the PASSING FF2
    # wrapper test's ``in1_sharded_mem_config``), not DRAM-sharded. Reshard the
    # DRAM ``w1_ring40`` to an L1 [K, N/40] layout on the 40 ring cores when
    # FF12_W1_L1=1 to confirm.
    # The fix: w1_ring40 is built DRAM-INTERLEAVED (column-order-preserving),
    # N zero-padded to 2560, then resharded JUST-IN-TIME to L1 WIDTH-SHARDED over
    # the 40 ring cores (W1W3_RING40_L1_MEMCFG) — the layout the gather_in0 ring
    # matmul requires. (DRAM-BANK-sharded weights scramble columns for this kernel.)
    _describe_memcfg("mlp.w1_ring40.memory_config() (DRAM interleaved)", mlp.w1_ring40.memory_config())
    w1_ring40_l1 = ttnn.to_memory_config(mlp.w1_ring40, mc["W1W3_RING40_L1_MEMCFG"])
    _describe_memcfg("w1_ring40_l1.memory_config() (L1 width-sharded)", w1_ring40_l1.memory_config())

    # ---- FF12 w1 ring-40 matmul (the op the fused decode MLP crashes on) ----
    w1_raw = ttnn.linear(
        ring_in,
        w1_ring40_l1,
        compute_kernel_config=fused_ck,
        dtype=ttnn.bfloat8_b,
        program_config=mc["FF1_3_RING40_PROGCFG"],
        memory_config=mc["SHARDED_FF12_OUT_RING40_MEMCFG"],
        sub_device_id=mlp.tt_ccl.worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh_device)
    print(f"[ff12_micro]   w1_raw (matmul out) shape = {list(w1_raw.shape)}")
    _describe_memcfg("w1_raw.memory_config()", w1_raw.memory_config())

    # ---- Torch golden: each device computes its own [32,1280] @ [1280,2560] partial-N.
    # The matmul is a per-device gather_in0 ring matmul: in0 is width-sharded across the
    # 40 ring cores (each holds 1280/40=32 of K), gathered to full 1280, then @ w1 (K=1280,
    # N=2560 per device). So the per-device output is in_full[r,c] @ w1_weight[r,c]. We
    # recover the per-device w1 weight from the on-device tensor to avoid re-deriving the
    # ShardTensor2dMesh split. ----
    w1_dev0 = ttnn.get_device_tensors(mlp.w1_ring40)[0].cpu().to_torch().float()
    # weight per-device shape: [1,1,K,N] = [1,1,1280,2560]
    w1_w = w1_dev0.reshape(w1_dev0.shape[-2], w1_dev0.shape[-1])

    worst = 1.0
    last = None
    for i, t in enumerate(ttnn.get_device_tensors(w1_raw)):
        r = i // _CLUSTER_SHAPE[1]
        c = i % _CLUSTER_SHAPE[1]
        w1_wi = ttnn.get_device_tensors(mlp.w1_ring40)[i].cpu().to_torch().float()
        w1_wi = w1_wi.reshape(w1_wi.shape[-2], w1_wi.shape[-1])
        ref = in_full[r, c] @ w1_wi  # [32, 2560]
        got = t.cpu().to_torch().float().reshape(M, -1)[:, : ref.shape[-1]]
        eq, msg = comp_pcc(got, ref)
        last = msg
        if not eq:
            print(f"[ff12_micro]   device {i} PCC: {msg}")
    print(f"[ff12_micro] FF12 w1 ring-40 matmul PCC last msg: {last}")
    # Focused single-device diagnostic: dev0, first 64 cols (core 0's N-slice).
    _t0 = ttnn.get_device_tensors(w1_raw)[0].cpu().to_torch().float().reshape(M, -1)
    _w0 = ttnn.get_device_tensors(mlp.w1_ring40)[0].cpu().to_torch().float()
    _w0 = _w0.reshape(_w0.shape[-2], _w0.shape[-1])
    _ref0 = in_full[0, 0] @ _w0
    print(f"[ff12_micro]   dev0 got[:2,:4]={_t0[:2,:4].tolist()}")
    print(f"[ff12_micro]   dev0 ref[:2,:4]={_ref0[:2,:4].tolist()}")
    print(f"[ff12_micro]   dev0 got width={_t0.shape[-1]} ref width={_ref0.shape[-1]}")
    # PCC on just cols [0:64] (core 0) vs cols [64:128] (core 1) to localize.
    for lo in (0, 64, 2048, 2112):
        eqs, ms = comp_pcc(_t0[:, lo : lo + 64], _ref0[:, lo : lo + 64])
        print(f"[ff12_micro]   dev0 cols[{lo}:{lo+64}] PCC: {ms}")
    # Full valid region PCC (cols 0:2176).
    eqf, msf = comp_pcc(_t0[:, :2176], _ref0[:, :2176])
    print(f"[ff12_micro]   dev0 cols[0:2176] PCC: {msf}")
    # Trimmed to a 64-multiple BELOW the boundary (cols 0:2112 = 33 cores worth).
    eqt, mst = comp_pcc(_t0[:, :2112], _ref0[:, :2112])
    print(f"[ff12_micro]   dev0 cols[0:2112] PCC: {mst}")
    for i, t in enumerate(ttnn.get_device_tensors(w1_raw)):
        r = i // _CLUSTER_SHAPE[1]
        c = i % _CLUSTER_SHAPE[1]
        w1_wi = ttnn.get_device_tensors(mlp.w1_ring40)[i].cpu().to_torch().float()
        w1_wi = w1_wi.reshape(w1_wi.shape[-2], w1_wi.shape[-1])
        ref = in_full[r, c] @ w1_wi
        got = t.cpu().to_torch().float().reshape(M, -1)[:, : ref.shape[-1]]
        eq, msg = comp_pcc(got, ref)
        assert eq, f"device {i} FAILED FF12 w1 ring-40 PCC: {msg}"
    print("[ff12_micro] FF12 w1 ring-40 matmul PCC OK across all 32 devices")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
