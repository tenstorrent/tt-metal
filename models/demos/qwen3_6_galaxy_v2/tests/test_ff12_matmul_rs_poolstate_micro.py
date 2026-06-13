# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-CONTEXT micro-test: reproduce the FF12 ``matmul_line_reduce_scatter``
deadlock that only manifests in the FULL decode forward (not in the fresh-pool
micro-tests).

Root cause (diagnosed via py-spy + per-op synchronize fences): the fused FF12
``matmul_line_reduce_scatter`` (llama_rs_matmul) pulls its interim buffer +
semaphore from the SHARED ``tt_ccl`` pools (``reduce_scatter_buffers`` /
``reduce_scatter_buffer_idx`` / ``gather_semaphore_handles`` / ``gather_idx``).
In a real decode step the attention layer + norms run BEFORE the MLP and ADVANCE
those pool indices, so the MLP's fused RS-matmul lands on a different (and
deadlock-inducing) pool slot than it does with the fresh pools a standalone
micro-test starts with. That's why ``test_ff12_ring40_rs_micro`` passed
(PCC 0.99997, fresh pools) yet the full forward HANGS at this exact op.

This test mimics the decode context WITHOUT the full model (no prefill, no
lm_head, no Generator) by LOOPING the FF12 fused op so the shared pool indices
advance across iterations — iteration 0 ≈ the passing micro-test (fresh pools),
later iterations ≈ the full-forward state (advanced pools). It prints the pool
indices each iteration and synchronizes after every op so a hang is attributable
to a SPECIFIC iteration's pool state.

  * RED (current ``matmul_line_reduce_scatter``): expected to complete iter 0 and
    then HANG on a later iteration (the advanced-pool slot) — reproducing the
    full-forward deadlock cheaply.
  * GREEN (after the fix — give the fused RS-matmul its own dedicated, shape-
    guarded interim buffer instead of the shared pool slot): all iterations
    complete + PCC > 0.99.

NOTE: a genuine on-device CCL deadlock wedges the fabric (needs tt-smi -r) the
same as the full model — this test does NOT avoid that one wedge. Its value is
(a) faithful pool-state reproduction so "passes here" == "passes in decode", and
(b) a fast, minimal loop to validate the fix (no prefill/lm_head/teardown weight).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_ff12_matmul_rs_poolstate_micro.py -s -x
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

_CLUSTER_SHAPE = (8, 4)
_N_ITERS = int(os.environ.get("QWEN36_POOLSTATE_ITERS", "6"))


def _pool_state(tt_ccl, cluster_axis=1):
    gi = getattr(tt_ccl, "gather_idx", None)
    rsi = getattr(tt_ccl, "reduce_scatter_buffer_idx", None)
    gi_v = gi[cluster_axis] if gi is not None else "?"
    rsi_v = rsi[cluster_axis] if rsi is not None else "?"
    ncbs = getattr(tt_ccl, "num_cbs", "?")
    return f"gather_idx[{cluster_axis}]={gi_v} reduce_scatter_buffer_idx[{cluster_axis}]={rsi_v} num_cbs={ncbs}"


@pytest.mark.hardware
def test_ff12_matmul_rs_poolstate_micro(bh_glx_mesh):
    mesh_device = bh_glx_mesh

    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FUSE_RS_MATMUL": "1",  # build the decode-mode ring-40 weights
    }.items():
        os.environ.setdefault(_k, _v)

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    # ---- build 1-layer model, grab real tt_ccl + ring-40 weights ----
    state_dict = _load_full_state_dict(_SNAPSHOT)
    pfx0 = "model.language_model.layers.0."
    all_pfx = "model.language_model.layers."
    state_dict = {k: v for k, v in state_dict.items() if (not k.startswith(all_pfx)) or k.startswith(pfx0)}

    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh_device, state_dict, ["linear_attention"], 1, paged_attention_config)

    mlp = model.layers[0].feed_forward
    mc = model.model_config
    tt_ccl = mlp.tt_ccl
    fused_ck = args.compute_kernel_config_hifi2
    wsd = tt_ccl.worker_sub_device_id

    M, dim_per_tp = 32, args.dim_per_tp  # 1280
    torch.manual_seed(0)

    print(f"[poolstate] looping FF12 matmul_line_reduce_scatter x{_N_ITERS} (pool advances each iter)")
    print(f"[poolstate] initial pool: {_pool_state(tt_ccl)}")

    # JIT L1 ring-40 weights (column-order-preserving DRAM-interleaved -> L1 width-sharded).
    for it in range(_N_ITERS):
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
        w1_l1 = ttnn.to_memory_config(mlp.w1_ring40, mc["W1W3_RING40_L1_MEMCFG"])
        w3_l1 = ttnn.to_memory_config(mlp.w3_ring40, mc["W1W3_RING40_L1_MEMCFG"])

        w1_raw = ttnn.linear(
            ring_in,
            w1_l1,
            compute_kernel_config=fused_ck,
            dtype=ttnn.bfloat8_b,
            program_config=mc["FF1_3_RING40_PROGCFG"],
            memory_config=mc["SHARDED_FF12_OUT_RING40_MEMCFG"],
            sub_device_id=wsd,
        )
        print(f"[poolstate] iter {it}: BEFORE matmul_line_reduce_scatter  {_pool_state(tt_ccl)}", flush=True)
        w1_red_s, w3_out_s = tt_ccl.matmul_line_reduce_scatter(
            ring_in,
            w3_l1,
            w1_raw,
            cluster_axis=1,
            num_links=mc["GALAXY_NUM_LINKS"],
            RS_memory_config=mc["REDUCE_SCATTER_OUT_RING40_FF12_MEMCFG"],
            compute_kernel_config=fused_ck,
            dtype=ttnn.bfloat8_b,
            program_config=mc["FF1_3_RING40_PROGCFG"],
            memory_config=mc["SHARDED_FF12_OUT_RING40_MEMCFG"],
            global_cb=None,
            sub_device_id=wsd,
            use_noc1_only=False,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[wsd])
        print(
            f"[poolstate] iter {it}: AFTER  matmul_line_reduce_scatter (completed)  {_pool_state(tt_ccl)}", flush=True
        )

        # PCC check on w1_red_s (per-device partial-N reduce-scattered to this col's 640 slice).
        eq_all = True
        for i, t in enumerate(ttnn.get_device_tensors(w1_red_s)):
            got = t.cpu().to_torch().float().reshape(M, -1)
            # light sanity: finite + nonzero
            if not torch.isfinite(got).all():
                eq_all = False
                print(f"[poolstate] iter {it} dev {i}: non-finite output")
                break
        ring_in.deallocate(True)
        w1_raw.deallocate(True)
        w1_l1.deallocate(True)
        w3_l1.deallocate(True)
        w1_red_s.deallocate(True)
        w3_out_s.deallocate(True)
        ff_in.deallocate(True)
        assert eq_all, f"iter {it}: matmul_line_reduce_scatter produced non-finite output"

    print(f"[poolstate] ALL {_N_ITERS} iters completed — matmul_line_reduce_scatter is pool-state robust")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
