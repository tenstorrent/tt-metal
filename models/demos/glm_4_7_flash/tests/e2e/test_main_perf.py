# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC gate for perf-mcp: self-contained e2e correctness test that produces a PCC value.

All constants and helpers are inlined so the worktree copy works without
test_e2e_text_generation.py present.
"""
from __future__ import annotations

import os

import torch

import ttnn

PCC_TARGET = 0.99
HIDDEN_PCC_FLOOR = 0.995
HIDDEN_NORM_TOL = 0.005


def _bf16_tie_tol(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.ldexp(torch.ones_like(x), torch.frexp(x.abs().clamp_min(1e-6))[1] - 8)


def _assert_tokens_agree(tt_tok, ref_tok, ref_logits, what, tt_logits=None):
    assert tt_tok.shape == ref_tok.shape, f"{what}: shape {tuple(tt_tok.shape)} vs {tuple(ref_tok.shape)}"
    diff = (tt_tok != ref_tok).nonzero().tolist()
    bad = []
    for b, s in diff:
        lg = ref_logits[b, s].float()
        top = lg[ref_tok[b, s]]
        gap = float(top - lg[tt_tok[b, s]])
        tol = float(_bf16_tie_tol(top))
        sig = float((tt_logits[b, s].float() - lg).pow(2).mean().sqrt()) if tt_logits is not None else float("nan")
        print(
            f"[glm] {what} row {b} step {s}: tt={int(tt_tok[b, s])} ref={int(ref_tok[b, s])} gap={gap:.4f} tie_tol={tol:.4f} sigma={sig:.4f} top={float(top):.3f}"
        )
        if gap > tol:
            bad.append((b, s, gap))
    n = tt_tok.numel()
    print(
        f"[glm] {what}: {n - len(diff)}/{n} tokens equal, {len(diff) - len(bad)} bf16 near-ties, {len(bad)} real mismatches"
    )
    assert not bad, f"{what}: {len(bad)} token(s) differ from the reference beyond a bf16 tie: {bad[:8]}"


def _assert_terminated(ended_by, golden_ended_by):
    if os.environ.get("TT_PERF_OSL_TOKENS"):
        print("TT_PERF_OSL_TOKENS caps the horizon for profiling by design; termination not asserted")
        return
    assert ended_by != "safety_cap", "run ended on the safety cap, not the model's stop rule"
    assert ended_by in ("eos", "max_new_tokens"), f"run ended by {ended_by}"
    assert ended_by == golden_ended_by, f"TT ended by {ended_by}, HF generate() by {golden_ended_by}"


def test_main_perf():
    n_chips = int(os.environ.get("TT_PERF_MESH_COLS", "4"))
    rows = int(os.environ.get("TT_PERF_MESH_ROWS", "1"))
    if n_chips > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    open_kwargs = {"num_command_queues": 1, "trace_region_size": 1 << 30}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, n_chips), **open_kwargs)
    try:
        from models.demos.glm_4_7_flash.tt import inputs as glm_inputs
        from models.demos.glm_4_7_flash.tt.pipeline import GRADUATED_STUBS, build_pipeline, pcc
        from models.demos.glm_4_7_flash.tt.reference import StopRule, _hf_reference_generate, load_hf_model

        B = glm_inputs.batch_size()
        tok = glm_inputs.load_tokenizer()
        ids, mask = glm_inputs.encode_example(tok, B)
        rule = StopRule.for_prompt(ids.shape[1])

        golden = _hf_reference_generate(ids, mask, rule)
        pipe = build_pipeline(mesh, model=load_hf_model(), batch=ids.shape[0])
        out = pipe.generate(ids, rule, teacher_tokens=golden["new_tokens"], keep_logits=True)

        missing = [n for n in GRADUATED_STUBS if pipe.counts.get(n, 0) == 0]
        print(f"[glm] stub invocations: {dict(sorted(pipe.counts.items()))}")
        assert not missing, f"graduated stubs never invoked: {missing}"

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
