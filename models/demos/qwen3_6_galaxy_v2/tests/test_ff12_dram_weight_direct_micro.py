# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Probe: can the ring-40 gather_in0 matmul read its weight DIRECTLY from DRAM
(no per-token L1 reshard)?

Motivation: the fused ring-40 decode MLP currently reshards w1/w3/w2 DRAM->L1
(InterleavedToSharded) EVERY decode token because the gather_in0 ring matmul was
assumed to need L1-width-sharded weights. That reshard costs ~73 us/step and
cancels the FF2 fusion's ~60 us/step device-kernel saving. qwen3-32b avoids any
reshard by streaming the DRAM weight via the PREFETCHER (global_cb) — which is a
NO-GO on BH. So the question: on BH (no prefetcher), can the ring matmul read the
weight straight from DRAM (column-order-preserving) with NO L1 copy?

Variants on the FF12 w1 ring-40 matmul (M=32, K=1280, N=2560, gather_in0):
  * BASELINE (known-good): weight resharded DRAM->L1 width-sharded (W1W3_RING40_L1_MEMCFG)
    then matmul — PCC 0.99997 (this is the current per-token-reshard path).
  * VARIANT A: weight fed straight from DRAM-INTERLEAVED (mlp.w1_ring40 as-built) — NO reshard.
  * VARIANT B: weight fed from the DRAM-bank-sharded W1W3_RING40_MEMCFG — NO L1 reshard
    (expected to FAIL PCC: bank-sharding scrambles columns for gather_in0 — confirms WHY
    we reshard).

If VARIANT A passes PCC > 0.99, the per-token L1 weight reshard can be dropped (feed the
DRAM-interleaved weight directly), turning the FF2 fusion from net-neutral into a real win.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_ff12_dram_weight_direct_micro.py -s -x
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


def _run_variant(name, mesh_device, ring_in, weight, progcfg, out_memcfg, ck, wsd, in_full, w1_ring40):
    """Run the FF12 w1 ring matmul with `weight` (whatever layout) and report per-device PCC."""
    try:
        w1_raw = ttnn.linear(
            ring_in,
            weight,
            compute_kernel_config=ck,
            dtype=ttnn.bfloat8_b,
            program_config=progcfg,
            memory_config=out_memcfg,
            sub_device_id=wsd,
        )
        ttnn.synchronize_device(mesh_device)
    except Exception as exc:  # noqa: BLE001
        print(f"[dram-weight] {name}: ERROR {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return False
    worst = 1.0
    for i, t in enumerate(ttnn.get_device_tensors(w1_raw)):
        r, c = i // _CLUSTER_SHAPE[1], i % _CLUSTER_SHAPE[1]
        w_i = ttnn.get_device_tensors(w1_ring40)[i].cpu().to_torch().float()
        w_i = w_i.reshape(w_i.shape[-2], w_i.shape[-1])
        ref = in_full[r, c] @ w_i
        got = t.cpu().to_torch().float().reshape(in_full.shape[-2], -1)[:, : ref.shape[-1]]
        eq, msg = comp_pcc(got, ref)
        try:
            pcc = float(msg.split("PCC:")[-1].strip())
        except Exception:  # noqa: BLE001
            pcc = -1.0
        worst = min(worst, pcc)
    w1_raw.deallocate(True)
    print(f"[dram-weight] {name}: worst-device PCC = {worst:.5f}  -> {'PASS' if worst > 0.99 else 'FAIL'}", flush=True)
    return worst > 0.99


@pytest.mark.hardware
def test_ff12_dram_weight_direct(bh_glx_mesh):
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

    mlp = model.layers[0].feed_forward
    mc = model.model_config
    ck = args.compute_kernel_config_hifi2
    wsd = mlp.tt_ccl.worker_sub_device_id
    M, dim_per_tp = 32, args.dim_per_tp

    torch.manual_seed(0)
    in_full = torch.randn(*_CLUSTER_SHAPE, M, dim_per_tp)
    ff_in = ttnn.from_torch(
        in_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_CLUSTER_SHAPE),
    )
    ring_in = ttnn.to_memory_config(ff_in, mc["SHARDED_FF12_RING40_MEMCFG"])
    progcfg = mc["FF1_3_RING40_PROGCFG"]
    out_memcfg = mc["SHARDED_FF12_OUT_RING40_MEMCFG"]

    print(
        f"[dram-weight] mlp.w1_ring40 layout = {mlp.w1_ring40.memory_config().memory_layout} "
        f"buffer_type={mlp.w1_ring40.memory_config().buffer_type}"
    )

    # BASELINE: DRAM-interleaved -> L1 width-sharded reshard (current per-token path)
    w1_l1 = ttnn.to_memory_config(mlp.w1_ring40, mc["W1W3_RING40_L1_MEMCFG"])
    base = _run_variant(
        "BASELINE_L1_reshard", mesh_device, ring_in, w1_l1, progcfg, out_memcfg, ck, wsd, in_full, mlp.w1_ring40
    )
    w1_l1.deallocate(True)

    # VARIANT A: feed DRAM-INTERLEAVED weight directly (NO reshard) — the candidate fix
    varA = _run_variant(
        "VARIANT_A_dram_interleaved_direct",
        mesh_device,
        ring_in,
        mlp.w1_ring40,
        progcfg,
        out_memcfg,
        ck,
        wsd,
        in_full,
        mlp.w1_ring40,
    )

    # VARIANT B: feed DRAM-bank-sharded weight directly (expected FAIL — confirms bank-shard scramble)
    try:
        w1_bank = ttnn.to_memory_config(mlp.w1_ring40, mc["W1W3_RING40_MEMCFG"])
        varB = _run_variant(
            "VARIANT_B_dram_bank_sharded_direct",
            mesh_device,
            ring_in,
            w1_bank,
            progcfg,
            out_memcfg,
            ck,
            wsd,
            in_full,
            mlp.w1_ring40,
        )
        w1_bank.deallocate(True)
    except Exception as exc:  # noqa: BLE001
        print(f"[dram-weight] VARIANT_B build ERROR: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        varB = False

    print(
        f"\n[dram-weight] SUMMARY: baseline(L1)={base}  A(DRAM-interleaved-direct)={varA}  B(DRAM-bank-direct)={varB}"
    )
    print("[dram-weight] => If A is True, the per-token L1 weight reshard can be DROPPED (feed DRAM weight directly).")
    assert base, "baseline L1-reshard path should pass (sanity)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
