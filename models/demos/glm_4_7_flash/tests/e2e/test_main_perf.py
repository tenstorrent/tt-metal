# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC gate for perf-mcp: runs the e2e correctness test that produces a PCC value.

This file exists because the perf-mcp config's PERF_MCP_PCC_TEST points here.
It delegates to test_tokens_match_hf_generate which computes end-to-end logits PCC.
"""
from __future__ import annotations

import os

import ttnn


def test_main_perf():
    n_chips = int(os.environ.get("TT_PERF_MESH_COLS", "4"))
    rows = int(os.environ.get("TT_PERF_MESH_ROWS", "1"))
    if n_chips > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    open_kwargs = {"num_command_queues": 1, "trace_region_size": 1 << 30}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, n_chips), **open_kwargs)
    try:
        from models.demos.glm_4_7_flash.tests.e2e.test_e2e_text_generation import (
            HIDDEN_NORM_TOL,
            HIDDEN_PCC_FLOOR,
            PCC_TARGET,
            _assert_terminated,
            _assert_tokens_agree,
            _check_stubs,
            _request,
        )
        from models.demos.glm_4_7_flash.tt.pipeline import build_pipeline, pcc
        from models.demos.glm_4_7_flash.tt.reference import _hf_reference_generate, load_hf_model

        tok, ids, mask, rule = _request()
        golden = _hf_reference_generate(ids, mask, rule)
        pipe = build_pipeline(mesh, model=load_hf_model(), batch=ids.shape[0])
        out = pipe.generate(ids, rule, teacher_tokens=golden["new_tokens"], keep_logits=True)
        _check_stubs(pipe)

        assert (
            out["steps"] == golden["new_tokens"].shape[1]
        ), f"TT ran {out['steps']} steps, golden {golden['new_tokens'].shape[1]}"

        hid_tt, hid_ref = out["prefill_hidden"].float(), golden["prefill_hidden"].float()
        row_pcc = [pcc(hid_tt[b], hid_ref[b]) for b in range(hid_tt.shape[0])]
        ratio = [float(hid_tt[b].norm() / hid_ref[b].norm()) for b in range(hid_tt.shape[0])]
        print(
            f"[glm] prefill hidden: min row PCC={min(row_pcc):.6f} norm ratio range=[{min(ratio):.5f}, {max(ratio):.5f}]"
        )

        lg_tt, lg_ref = out["logits"], golden["logits"]
        row_lpcc = [pcc(lg_tt[b], lg_ref[b]) for b in range(lg_tt.shape[0])]
        achieved_pcc = min(row_lpcc)
        print(f"[glm] per-row logits PCC: min={min(row_lpcc):.6f} max={max(row_lpcc):.6f}")

        _assert_tokens_agree(out["tokens"], golden["new_tokens"], lg_ref, "teacher-forced argmax vs generate()", lg_tt)
        assert min(row_pcc) >= HIDDEN_PCC_FLOOR, f"prefill hidden PCC {min(row_pcc):.6f} < {HIDDEN_PCC_FLOOR}"
        assert all(
            abs(r - 1.0) <= HIDDEN_NORM_TOL for r in ratio
        ), f"prefill hidden norm ratio out of 1±{HIDDEN_NORM_TOL}: {ratio}"
        _assert_terminated(out["ended_by"], golden["ended_by"])
        print(f"e2e PCC={achieved_pcc}")
        assert achieved_pcc >= PCC_TARGET, f"e2e logits PCC {achieved_pcc:.6f} < {PCC_TARGET}"
    finally:
        ttnn.close_mesh_device(mesh)
        if n_chips > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
