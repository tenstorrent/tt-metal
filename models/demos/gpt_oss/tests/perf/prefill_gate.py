# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy gate for GPT-OSS batch-1 prefill on a 1x4 mesh. Exit code 0 = PASS, 1 = FAIL.

    python models/demos/gpt_oss/tests/perf/prefill_gate.py --lengths fast     # 128,1K,8K (every iteration)
    python models/demos/gpt_oss/tests/perf/prefill_gate.py --lengths all      # all 10 lengths (keeps)

Checks, all in one command:
  (a) goldens: for each length L, prefill the frozen real-text prompt (first L tokens of the repo's tech reports,
      GOLDENS_DIR/prompt_tokens.pt) through Generator.prefill_forward_text and compare the last-token logits with
      the clean-tree golden GOLDENS_DIR/tt_L{L}.pt (PCC, top-1, top-5 overlap).
  (b) reference: where GOLDENS_DIR/hf_ref_L{L}.pt exists (CPU HuggingFace logits of the same prompt, made by
      prefill_hf_ref.py; 128, 1K, 8K), compare the same logits with the HF model.
  (c) unit: test_modules.py::test_decoder at 1x4 for prefill_128 / prefill_1024 with the existing thresholds in
      models/demos/gpt_oss/unit_test_thresholds.json (skip with --no-unit).

Thresholds are the module constants below, chosen from the measured clean-tree run-to-run variation and the
clean tree's distance to the HF reference (see runs/gpt-oss-20b-prefill/baseline/README.md). Never loosen them to
make a change pass.

--save-goldens writes tt_L{L}.pt from the current tree (clean tree only) instead of comparing.

--mode v2 / v3 / v4: distribution gates for numerics-changing levers (see the V2_* / V3_* / V4_* constants). The keep
gate is v4: python models/demos/gpt_oss/tests/perf/prefill_gate.py --mode v4 --result <json> [--v4-reference <dense>]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---- thresholds (see module docstring) -----------------------------------------------------------------------
# vs clean-tree golden (same hardware, same prompt). Clean tree measured: every length >= 512 is bit-exact run to run
# (in-process and across processes); L=128 is NOT deterministic: call-to-call PCC 0.986-0.989, top-5 overlap 0.8-1.0,
# top-1 stable. So L=128 gets its own floor below its noise; the others keep 0.99 as the bar for a legitimate
# numerics change (accumulation order, program config).
GOLDEN_MIN_PCC = 0.99
GOLDEN_MIN_PCC_BY_LEN = {128: 0.97}
GOLDEN_MIN_TOP5_OVERLAP = 0.6  # >= 3 of 5
GOLDEN_TOP1_MARGIN = 0.25  # top-1 must match unless the golden's top-1/top-2 logit gap is below this
# vs CPU HuggingFace reference (bf16). Clean tree measured: PCC 0.944 / 0.970 / 0.967, top-5 overlap 0.8 / 0.8 / 0.6,
# top-1 equal at 128 / 1K / 8K (HF top-1 margins 7.1 / 1.4 / 2.8, so top-1 must match). L=128 is nondeterministic
# (HF PCC 0.941-0.953 over 5 clean calls), hence its lower floor. A planted 10% prefill SDPA scale error measured
# 0.947 (1K) / 0.944 (8K), below the 0.955 floor.
HF_MIN_PCC = 0.955
HF_MIN_PCC_BY_LEN = {128: 0.92}
HF_MIN_TOP5_OVERLAP = 0.4
HF_TOP1_MARGIN = 0.5
# Gate v2 (--mode v2, for levers that change numerics; spec set by the orchestrator 2026-10-14): the last-token PCC
# above swings 0.97 -> 0.91 at 1K when only the layer-0 MoE output of the clean tree is scaled by 0.97 or 1.03, so
# it cannot judge a numerics change. v2 scores the last 32 positions (the LM-head tile) at 128 / 1K / 8K against
# HF CPU logits at the same positions (GOLDENS_DIR/hf_last32, prefill_hf_ref.py --last32): top-1 agreement, mean
# top-5 overlap, mean full-vocab KL(HF || TT). A candidate passes against a reference run of the dense tree
# (--v2-reference <json>) iff top1 >= ref - V2_TOP1_DROP, KL <= V2_KL_RATIO x ref KL, and no NaN/Inf.
V2_LENGTHS = [128, 1024, 8192]
V2_TOP1_DROP = 0.03
V2_KL_RATIO = 1.15
# Gate v3 (--mode v3; acceptance fixed by the orchestrator 2026-10-14): v2's 96 last-tile positions passed an SDPA x1.1
# plant. v3 scores EVERY position of the frozen real-text prompt at 128 / 1K / 2K (3200 next-token predictions)
# against CPU HF bf16 logits at the same positions (GOLDENS_DIR/hf_full, prefill_hf_ref.py --full): top-1 agreement
# (TT argmax == HF argmax), top-5 hit (HF argmax in TT top-5), mean full-vocab KL(HF || TT). A candidate passes
# against a dense-tree run (--v3-reference <json>) iff pooled top1 >= ref - V3_TOP1_DROP, pooled KL <= V3_KL_RATIO x
# ref KL, and no NaN/Inf.
V3_LENGTHS = [128, 1024, 2048]
# Optional long window (--v3-window L:N, e.g. 8192:1024): also prefill the first L prompt tokens and score the LAST N
# positions against GOLDENS_DIR/hf_full/hf_full_L{L}_last{N}.pt (prefill_hf_ref.py --full --last N --lengths L).
# It is its own row "L_lastN" and joins the pooled numbers, so the reference run must use the same window (checked).
# Same acceptance rule. Without the flag v3 is unchanged.
V3_TOP1_DROP = 0.01
V3_KL_RATIO = 1.05
# Gate v4 (--mode v4; spec set by the orchestrator 2026-10-15): v3 scores one prompt (its 128/1K/2K are prefixes of
# one stream) and its KL moved .175 -> .265 at 1K under a rounding-only change (o_proj bias fusion). v4 scores ALL
# positions of V4_PROMPTS independent 1024-token real-text prompts (one prose Markdown doc each, no shared 8-gram;
# GOLDENS_DIR/hf_v4/prompts_v4.pt, made with their HF logits by prefill_hf_ref.py --v4) plus the v3 8K window (last
# 1024 positions of the 8192-token v3 prompt): 9 cases, 9216 positions. Same metrics as v3. A candidate passes against
# a dense-tree v4 run (--v4-reference <json>) iff pooled (all 9 cases) top1 >= ref - V4_TOP1_DROP, pooled KL <=
# V4_KL_RATIO x ref KL, every case's KL <= V4_CASE_KL_RATIO x the same case's ref KL, and no NaN/Inf.
V4_DIR = "hf_v4"
V4_PROMPTS = 8
V4_PROMPT_LEN = 1024
V4_WINDOW = (8192, 1024)
V4_TOP1_DROP = 0.01
V4_KL_RATIO = 1.05
V4_CASE_KL_RATIO = 1.25
# ---------------------------------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[5]


def _check(metrics, min_pcc, min_top5, top1_margin):
    fails = []
    if metrics["pcc"] < min_pcc:
        fails.append(f"pcc {metrics['pcc']:.5f} < {min_pcc}")
    if metrics["top5_overlap"] < min_top5:
        fails.append(f"top5_overlap {metrics['top5_overlap']:.2f} < {min_top5}")
    if not metrics["top1_match"]:
        if metrics["ref_top1_margin"] >= top1_margin:
            fails.append(
                f"top1 {metrics['cur_top1']} != ref {metrics['ref_top1']} (ref margin {metrics['ref_top1_margin']:.3f})"
            )
        elif not metrics["ref_top1_in_cur_top5"]:
            fails.append(f"near-tie top1 flipped and ref top1 {metrics['ref_top1']} not in current top-5")
    return fails


def _score_positions(tt, hf, chunk=256):
    """Per-position top-1 agreement, top-5 hit (HF argmax in TT top-5) and KL(HF || TT); float32, chunked.

    tt, hf: [positions, vocab] logits (vocab may differ by padding; the common prefix is scored).
    Returns (top1, top5, kl, has_nan) with one float per position."""
    import torch

    n = min(tt.shape[-1], hf.shape[-1])
    top1, top5, kl, nan = [], [], [], False
    for i in range(0, tt.shape[0], chunk):
        t, h = tt[i : i + chunk, :n].float(), hf[i : i + chunk, :n].float()
        nan = nan or not bool(torch.isfinite(t).all())
        h1 = h.argmax(-1)
        top1.append(t.argmax(-1) == h1)
        top5.append((torch.topk(t, 5, dim=-1).indices == h1[:, None]).any(-1))
        lp_h, lp_t = torch.log_softmax(h, -1), torch.log_softmax(t, -1)
        kl.append((lp_h.exp() * (lp_h - lp_t)).sum(-1))
    return torch.cat(top1).float(), torch.cat(top5).float(), torch.cat(kl), nan


# The CLI parent only spawns pytest; importing test_factory would query the devices (ttnn.get_num_devices) from the
# parent while the child process owns them, so the test is defined only when pytest imports this module.
if __name__ != "__main__":
    import pytest

    from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

    @pytest.mark.timeout(4 * 3600)
    @parametrize_mesh_with_fabric([(1, 4)])
    def test_prefill_gate(mesh_device, device_params):
        import torch
        from loguru import logger

        from models.demos.gpt_oss.tests.perf.prefill_common import (
            GOLDENS_DIR,
            PREFILL_LENGTHS,
            build_prefill_generator,
            compare_logits,
            env_snapshot,
            git_info,
            load_or_build_prompt_tokens,
            parse_lengths,
            run_prefill,
        )

        lengths = parse_lengths(os.getenv("PREFILL_GATE_LENGTHS", "fast"))
        save = os.getenv("PREFILL_GATE_SAVE") == "1"
        result_path = os.getenv("PREFILL_GATE_RESULT")

        t0 = time.perf_counter()
        ctx = build_prefill_generator(mesh_device, max(PREFILL_LENGTHS))
        build_s = time.perf_counter() - t0
        tokens = load_or_build_prompt_tokens(ctx["tokenizer"], create=save)

        report = {
            "git": git_info(),
            "env": env_snapshot(),
            "goldens_dir": str(GOLDENS_DIR),
            "model_build_s": build_s,
            "lengths": {},
        }
        all_fails = []
        repeats = int(os.getenv("PREFILL_GATE_REPEATS", "1"))  # diagnostic: extra in-process calls per length
        for L in lengths:
            t1 = time.perf_counter()
            dt, logits = run_prefill(ctx, tokens[:L].unsqueeze(0))
            entry = {"prefill_s": dt, "top1": int(logits.argmax())}
            for r in range(1, repeats):
                _, again = run_prefill(ctx, tokens[:L].unsqueeze(0))
                entry[f"repeat{r}_vs_first"] = compare_logits(again, logits)
                logger.info(f"[gate] L={L} in-process repeat {r}: {entry[f'repeat{r}_vs_first']}")
            if save:
                top = torch.topk(logits, 10)
                torch.save(
                    {
                        "length": L,
                        "logits": logits,
                        "top10_ids": top.indices,
                        "top10_vals": top.values,
                        "git": git_info(),
                    },
                    GOLDENS_DIR / f"tt_L{L}.pt",
                )
                entry["saved"] = str(GOLDENS_DIR / f"tt_L{L}.pt")
            else:
                gold = torch.load(GOLDENS_DIR / f"tt_L{L}.pt")
                m = compare_logits(logits, gold["logits"])
                f = _check(m, GOLDEN_MIN_PCC_BY_LEN.get(L, GOLDEN_MIN_PCC), GOLDEN_MIN_TOP5_OVERLAP, GOLDEN_TOP1_MARGIN)
                entry["golden"] = {**m, "fails": f}
                all_fails += [f"L={L} golden: {x}" for x in f]
            hf_path = GOLDENS_DIR / f"hf_ref_L{L}.pt"
            if hf_path.exists():
                ref = torch.load(hf_path)
                m = compare_logits(logits, ref["logits"])
                f = _check(m, HF_MIN_PCC_BY_LEN.get(L, HF_MIN_PCC), HF_MIN_TOP5_OVERLAP, HF_TOP1_MARGIN)
                entry["hf"] = {**m, "fails": f}
                all_fails += [f"L={L} hf: {x}" for x in f]
            entry["length_s"] = time.perf_counter() - t1
            report["lengths"][str(L)] = entry
            g = entry.get("golden", {})
            h = entry.get("hf", {})
            logger.info(
                f"[gate] L={L} {entry['length_s']:.1f}s top1={entry['top1']} "
                f"golden pcc={g.get('pcc', float('nan')):.5f} top5={g.get('top5_overlap', float('nan'))} "
                f"hf pcc={h.get('pcc', float('nan')):.5f} top5={h.get('top5_overlap', float('nan'))} "
                f"fails={g.get('fails', []) + h.get('fails', [])}"
            )
        report["fails"] = all_fails
        if result_path:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(json.dumps(report, indent=2))
        assert not all_fails, "prefill gate FAIL:\n" + "\n".join(all_fails)

    @pytest.mark.timeout(4 * 3600)
    @parametrize_mesh_with_fabric([(1, 4)])
    def test_prefill_gate_v2(mesh_device, device_params):
        """Gate v2 (numerics-changing levers): last-tile logits vs HF over V2_LENGTHS x 32 positions."""
        import torch
        from loguru import logger

        from models.demos.gpt_oss.tests.perf.prefill_common import (
            GOLDENS_DIR,
            PREFILL_LENGTHS,
            build_prefill_generator,
            env_snapshot,
            git_info,
            load_or_build_prompt_tokens,
            run_prefill,
        )

        result_path = os.getenv("PREFILL_GATE_RESULT")
        ref_path = os.getenv("PREFILL_GATE_V2_REF")
        ctx = build_prefill_generator(mesh_device, max(PREFILL_LENGTHS))
        tokens = load_or_build_prompt_tokens(ctx["tokenizer"])

        # The prefill LM head computes the whole 32-row tile holding the last token; keep all of it.
        tt_model = ctx["generator"].model[0]
        orig = tt_model.process_output_prefill
        captured = {}

        def capture(tt_out, last_token_idx):
            tile = orig(tt_out, slice(None))  # [..., 32, vocab]
            captured["tile"] = tile.reshape(-1, tile.shape[-1]).float().clone()
            return tile[..., last_token_idx, :]

        tt_model.process_output_prefill = capture
        report = {"mode": "v2", "git": git_info(), "env": env_snapshot(), "lengths": {}}
        top1s, top5s, kls, nan = [], [], [], False
        try:
            for L in V2_LENGTHS:
                assert L % 32 == 0, L
                captured.clear()
                run_prefill(ctx, tokens[:L].unsqueeze(0))
                tt = captured["tile"]
                assert tt.shape[0] == 32, f"expected the 32-row last tile, got {tuple(tt.shape)}"
                if os.getenv("PREFILL_GATE_V2_DUMP"):  # diagnostic: keep the TT tile
                    torch.save(tt, Path(os.environ["PREFILL_GATE_V2_DUMP"]) / f"tt_last32_L{L}.pt")
                ref = torch.load(GOLDENS_DIR / "hf_last32" / f"hf_last32_L{L}.pt")
                assert ref["positions"] == list(range(L - 32, L)), "HF reference positions do not match the tile"
                hf = ref["logits"].float()
                n = min(tt.shape[-1], hf.shape[-1])
                tt, hf = tt[:, :n].double(), hf[:, :n].double()
                nan = nan or bool(torch.isnan(tt).any() or torch.isinf(tt).any())
                top1 = (tt.argmax(-1) == hf.argmax(-1)).double()
                t5, h5 = torch.topk(tt, 5, dim=-1).indices, torch.topk(hf, 5, dim=-1).indices
                top5 = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in zip(t5, h5)]).double()
                lp_hf, lp_tt = torch.log_softmax(hf, -1), torch.log_softmax(tt, -1)
                kl = (lp_hf.exp() * (lp_hf - lp_tt)).sum(-1)
                top1s.append(top1)
                top5s.append(top5)
                kls.append(kl)
                entry = {
                    "top1": float(top1.mean()),
                    "top5_overlap": float(top5.mean()),
                    "kl_hf_tt": float(kl.mean()),
                    "kl_max": float(kl.max()),
                }
                report["lengths"][str(L)] = entry
                logger.info(f"[gate v2] L={L} {entry}")
        finally:
            tt_model.process_output_prefill = orig
        overall = {
            "positions": int(sum(t.numel() for t in top1s)),
            "top1": float(torch.cat(top1s).mean()),
            "top5_overlap": float(torch.cat(top5s).mean()),
            "kl_hf_tt": float(torch.cat(kls).mean()),
            "nan": nan,
        }
        report["overall"] = overall
        fails = ["NaN/Inf in TT logits"] if nan else []
        if ref_path:
            base = json.loads(Path(ref_path).read_text())["overall"]
            report["reference"] = {"path": ref_path, **base}
            if overall["top1"] < base["top1"] - V2_TOP1_DROP:
                fails.append(f"top1 {overall['top1']:.4f} < reference {base['top1']:.4f} - {V2_TOP1_DROP}")
            if overall["kl_hf_tt"] > V2_KL_RATIO * base["kl_hf_tt"]:
                fails.append(f"KL {overall['kl_hf_tt']:.5f} > {V2_KL_RATIO} x reference {base['kl_hf_tt']:.5f}")
        report["fails"] = fails
        logger.info(f"[gate v2] overall {overall} fails={fails}")
        if result_path:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(json.dumps(report, indent=2))
        assert not fails, "prefill gate v2 FAIL:\n" + "\n".join(fails)

    @pytest.mark.timeout(4 * 3600)
    @parametrize_mesh_with_fabric([(1, 4)])
    def test_prefill_gate_v3(mesh_device, device_params):
        """Gate v3 (numerics-changing levers): logits at ALL positions vs HF over V3_LENGTHS (3200 positions).

        All-position logits come from the gate only: for this test, ttnn_prefill_forward is called with
        get_last_token=-1 (no 32-row slice before the final norm + LM head) and process_output_prefill keeps every
        row. Default model behaviour is untouched; both wrappers are removed in the finally block.
        """
        import torch
        from loguru import logger

        from models.demos.gpt_oss.tests.perf.prefill_common import (
            GOLDENS_DIR,
            PREFILL_LENGTHS,
            build_prefill_generator,
            compare_logits,
            env_snapshot,
            git_info,
            load_or_build_prompt_tokens,
            run_prefill,
        )

        result_path = os.getenv("PREFILL_GATE_RESULT")
        ref_path = os.getenv("PREFILL_GATE_V3_REF")
        # cases: (row label, prompt length, first scored position); the base lengths score every position
        cases = [(str(L), L, 0) for L in V3_LENGTHS]
        window = os.getenv("PREFILL_GATE_V3_WINDOW", "")
        if window:
            win_len, win_rows = (int(v) for v in window.split(":"))
            assert 0 < win_rows <= win_len, window
            cases.append((f"{win_len}_last{win_rows}", win_len, win_len - win_rows))
        ctx = build_prefill_generator(mesh_device, max(PREFILL_LENGTHS))
        tokens = load_or_build_prompt_tokens(ctx["tokenizer"])

        tt_model = ctx["generator"].model[0]
        orig_fwd, orig_out = tt_model.ttnn_prefill_forward, tt_model.process_output_prefill
        captured = {}

        def fwd_all_positions(*a, **kw):
            assert kw.get("batch_size", 1) == 1 and kw.get("chunk_start_idx") is None, "v3 needs unchunked batch-1"
            kw["get_last_token"] = -1
            return orig_fwd(*a, **kw)

        def out_all_positions(tt_out, last_token_idx):
            full = orig_out(tt_out, slice(None))
            full = full.reshape(-1, full.shape[-1])
            captured["full"] = full
            return full[captured["L"] - 1]

        tt_model.ttnn_prefill_forward = fwd_all_positions
        tt_model.process_output_prefill = out_all_positions
        report = {
            "mode": "v3",
            "git": git_info(),
            "env": env_snapshot(),
            "cases": [label for label, _, _ in cases],
            "lengths": {},
        }
        top1s, top5s, kls, nan = [], [], [], False
        try:
            for label, L, first in cases:
                captured.clear()
                captured["L"] = L
                _, last = run_prefill(ctx, tokens[:L].unsqueeze(0))
                tt = captured.pop("full")
                assert tt.shape[0] == L, f"expected {L} rows, got {tuple(tt.shape)}"
                if first:
                    ref = torch.load(GOLDENS_DIR / "hf_full" / f"hf_full_L{L}_last{L - first}.pt")
                    assert ref["positions"] == [first, L], (ref["positions"], first, L)
                    tt = tt[first:]
                else:
                    ref = torch.load(GOLDENS_DIR / "hf_full" / f"hf_full_L{L}.pt")
                hf = ref["logits"]
                rows = L - first
                assert ref["length"] == L and hf.shape[0] == rows, (ref["length"], tuple(hf.shape), rows)
                n = min(tt.shape[-1], hf.shape[-1])
                top1, top5, kl = [], [], []
                for i in range(0, rows, 256):  # float32 math, chunked over positions
                    t, h = tt[i : i + 256, :n].float(), hf[i : i + 256, :n].float()
                    nan = nan or not bool(torch.isfinite(t).all())
                    h1 = h.argmax(-1)
                    top1.append(t.argmax(-1) == h1)
                    top5.append((torch.topk(t, 5, dim=-1).indices == h1[:, None]).any(-1))
                    lp_h, lp_t = torch.log_softmax(h, -1), torch.log_softmax(t, -1)
                    kl.append((lp_h.exp() * (lp_h - lp_t)).sum(-1))
                top1, top5, kl = torch.cat(top1).float(), torch.cat(top5).float(), torch.cat(kl)
                top1s.append(top1)
                top5s.append(top5)
                kls.append(kl)
                entry = {
                    "positions": rows,
                    "top1": float(top1.mean()),
                    "top5_hit": float(top5.mean()),
                    "kl_hf_tt": float(kl.mean()),
                    "kl_max": float(kl.max()),
                }
                gold = GOLDENS_DIR / f"tt_L{L}.pt"
                if gold.exists():  # sanity: the hooked last row vs the normal path's last-token golden (dense tree)
                    entry["last_row_vs_tt_golden_pcc"] = compare_logits(last.float(), torch.load(gold)["logits"])["pcc"]
                report["lengths"][label] = entry
                logger.info(f"[gate v3] L={label} {entry}")
                del tt, hf, ref
        finally:
            tt_model.ttnn_prefill_forward, tt_model.process_output_prefill = orig_fwd, orig_out

        def pooled(n_cases):
            return {
                "positions": int(sum(t.numel() for t in top1s[:n_cases])),
                "top1": float(torch.cat(top1s[:n_cases]).mean()),
                "top5_hit": float(torch.cat(top5s[:n_cases]).mean()),
                "kl_hf_tt": float(torch.cat(kls[:n_cases]).mean()),
            }

        overall = {**pooled(len(cases)), "nan": nan}
        report["overall"] = overall
        if window:  # the 128/1K/2K pool alone, comparable with runs without the window
            report["overall_base"] = pooled(len(V3_LENGTHS))
        fails = ["NaN/Inf in TT logits"] if nan else []
        if ref_path:
            ref_report = json.loads(Path(ref_path).read_text())
            ref_cases = ref_report.get("cases", [str(L) for L in V3_LENGTHS])
            assert ref_cases == report["cases"], (
                f"reference {ref_path} was run with cases {ref_cases}, this run with {report['cases']}: "
                "pooled numbers are not comparable (rerun the reference with the same --v3-window)"
            )
            base = ref_report["overall"]
            report["reference"] = {"path": ref_path, **base}
            if overall["top1"] < base["top1"] - V3_TOP1_DROP:
                fails.append(f"top1 {overall['top1']:.4f} < reference {base['top1']:.4f} - {V3_TOP1_DROP}")
            if overall["kl_hf_tt"] > V3_KL_RATIO * base["kl_hf_tt"]:
                fails.append(f"KL {overall['kl_hf_tt']:.5f} > {V3_KL_RATIO} x reference {base['kl_hf_tt']:.5f}")
        report["fails"] = fails
        logger.info(f"[gate v3] overall {overall} fails={fails}")
        if result_path:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(json.dumps(report, indent=2))
            # per-position vectors for paired (run vs run) analysis
            torch.save(
                {
                    "lengths": [L for _, L, _ in cases],
                    "cases": report["cases"],
                    "top1": top1s,
                    "top5": top5s,
                    "kl": kls,
                },
                Path(result_path).with_suffix(".positions.pt"),
            )
        assert not fails, "prefill gate v3 FAIL:\n" + "\n".join(fails)

    @pytest.mark.timeout(4 * 3600)
    @parametrize_mesh_with_fabric([(1, 4)])
    def test_prefill_gate_v4(mesh_device, device_params):
        """Gate v4 (numerics-changing levers): all positions of V4_PROMPTS independent 1K prompts + the 8K window.

        Same hooks as v3 (get_last_token=-1, every LM-head row kept); both are removed in the finally block.
        """
        import torch
        from loguru import logger

        from models.demos.gpt_oss.tests.perf.prefill_common import (
            GOLDENS_DIR,
            PREFILL_LENGTHS,
            build_prefill_generator,
            env_snapshot,
            git_info,
            load_or_build_prompt_tokens,
            run_prefill,
        )

        result_path = os.getenv("PREFILL_GATE_RESULT")
        ref_path = os.getenv("PREFILL_GATE_V4_REF")
        v4_dir = GOLDENS_DIR / V4_DIR
        prompts = torch.load(v4_dir / "prompts_v4.pt")
        assert tuple(prompts["tokens"].shape) == (V4_PROMPTS, V4_PROMPT_LEN), tuple(prompts["tokens"].shape)
        docs = [d["file"] for d in prompts["docs"]]
        win_len, win_rows = V4_WINDOW
        # cases: (label, prompt tokens or None for the v3 stream window, first scored position, HF reference file)
        cases = [(f"p{i}", prompts["tokens"][i], 0, v4_dir / f"hf_v4_p{i}.pt") for i in range(V4_PROMPTS)]
        cases.append(
            (
                f"{win_len}_last{win_rows}",
                None,
                win_len - win_rows,
                GOLDENS_DIR / "hf_full" / f"hf_full_L{win_len}_last{win_rows}.pt",
            )
        )
        ctx = build_prefill_generator(mesh_device, max(PREFILL_LENGTHS))
        stream = load_or_build_prompt_tokens(ctx["tokenizer"])

        tt_model = ctx["generator"].model[0]
        orig_fwd, orig_out = tt_model.ttnn_prefill_forward, tt_model.process_output_prefill
        captured = {}

        def fwd_all_positions(*a, **kw):
            assert kw.get("batch_size", 1) == 1 and kw.get("chunk_start_idx") is None, "v4 needs unchunked batch-1"
            kw["get_last_token"] = -1
            return orig_fwd(*a, **kw)

        def out_all_positions(tt_out, last_token_idx):
            full = orig_out(tt_out, slice(None))
            full = full.reshape(-1, full.shape[-1])
            captured["full"] = full
            return full[captured["L"] - 1]

        tt_model.ttnn_prefill_forward = fwd_all_positions
        tt_model.process_output_prefill = out_all_positions
        report = {
            "mode": "v4",
            "git": git_info(),
            "env": env_snapshot(),
            "cases": [c[0] for c in cases],
            "docs": docs,
            "lengths": {},
        }
        top1s, top5s, kls, nan = [], [], [], False
        try:
            for label, toks, first, ref_file in cases:
                if toks is None:
                    toks = stream[:win_len]
                L = toks.numel()
                captured.clear()
                captured["L"] = L
                run_prefill(ctx, toks.unsqueeze(0))
                tt = captured.pop("full")
                assert tt.shape[0] == L, f"expected {L} rows, got {tuple(tt.shape)}"
                ref = torch.load(ref_file)
                if first:
                    assert ref["length"] == L and ref["positions"] == [first, L], (ref["positions"], first, L)
                else:
                    assert torch.equal(ref["tokens"], toks), f"{ref_file} was made from other tokens"
                tt, hf = tt[first:], ref["logits"]
                assert hf.shape[0] == L - first, (tuple(hf.shape), L - first)
                top1, top5, kl, case_nan = _score_positions(tt, hf)
                nan = nan or case_nan
                top1s.append(top1)
                top5s.append(top5)
                kls.append(kl)
                entry = {
                    "positions": L - first,
                    "top1": float(top1.mean()),
                    "top5_hit": float(top5.mean()),
                    "kl_hf_tt": float(kl.mean()),
                    "kl_max": float(kl.max()),
                }
                if first == 0:
                    entry["doc"] = docs[int(label[1:])]
                report["lengths"][label] = entry
                logger.info(f"[gate v4] {label} {entry}")
                del tt, hf, ref
        finally:
            tt_model.ttnn_prefill_forward, tt_model.process_output_prefill = orig_fwd, orig_out

        def pooled(sel):
            return {
                "positions": int(sum(top1s[i].numel() for i in sel)),
                "top1": float(torch.cat([top1s[i] for i in sel]).mean()),
                "top5_hit": float(torch.cat([top5s[i] for i in sel]).mean()),
                "kl_hf_tt": float(torch.cat([kls[i] for i in sel]).mean()),
            }

        overall = {**pooled(range(len(cases))), "nan": nan}
        report["overall"] = overall
        report["overall_prompts"] = pooled(range(V4_PROMPTS))  # the independent 1K prompts alone
        case_kl = [float(k.mean()) for k in kls]
        report["prompt_kl_spread"] = {"min": min(case_kl[:V4_PROMPTS]), "max": max(case_kl[:V4_PROMPTS])}
        fails = ["NaN/Inf in TT logits"] if nan else []
        if ref_path:
            ref_report = json.loads(Path(ref_path).read_text())
            assert (
                ref_report.get("mode") == "v4" and ref_report["cases"] == report["cases"]
            ), f"reference {ref_path} is not a v4 run with cases {report['cases']}"
            assert ref_report["docs"] == docs, "reference was run on other v4 prompts"
            base = ref_report["overall"]
            report["reference"] = {"path": ref_path, **base}
            if overall["top1"] < base["top1"] - V4_TOP1_DROP:
                fails.append(f"top1 {overall['top1']:.4f} < reference {base['top1']:.4f} - {V4_TOP1_DROP}")
            if overall["kl_hf_tt"] > V4_KL_RATIO * base["kl_hf_tt"]:
                fails.append(f"KL {overall['kl_hf_tt']:.5f} > {V4_KL_RATIO} x reference {base['kl_hf_tt']:.5f}")
            report["case_kl_ratio"] = {}
            for label, kl_c in zip(report["cases"], case_kl):
                kl_ref = ref_report["lengths"][label]["kl_hf_tt"]
                report["case_kl_ratio"][label] = kl_c / kl_ref
                if kl_c > V4_CASE_KL_RATIO * kl_ref:
                    fails.append(f"{label} KL {kl_c:.5f} > {V4_CASE_KL_RATIO} x reference {kl_ref:.5f}")
            ref_pos = Path(ref_path).with_suffix(".positions.pt")
            if ref_pos.exists():  # informational: paired per-position KL difference vs the reference run
                d = torch.cat([a - b for a, b in zip(kls, torch.load(ref_pos)["kl"])]).double()
                se = float(d.std()) / d.numel() ** 0.5
                report["paired_dkl"] = {"mean": float(d.mean()), "z": float(d.mean()) / se if se > 0 else 0.0}
        report["fails"] = fails
        logger.info(f"[gate v4] overall {overall} prompts {report['overall_prompts']} fails={fails}")
        if result_path:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(json.dumps(report, indent=2))
            torch.save(
                {"cases": report["cases"], "top1": top1s, "top5": top5s, "kl": kls},
                Path(result_path).with_suffix(".positions.pt"),
            )
        assert not fails, "prefill gate v4 FAIL:\n" + "\n".join(fails)


def _run(cmd, env, log_name):
    t0 = time.time()
    print(f"[gate] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    dt = time.time() - t0
    print(f"[gate] {log_name}: rc={rc} in {dt:.0f}s", flush=True)
    return rc, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="fast", help="'fast' (128,1K,8K) | 'all' | comma list")
    ap.add_argument("--save-goldens", action="store_true")
    ap.add_argument("--no-unit", action="store_true", help="skip test_modules decoder PCC at 1x4")
    ap.add_argument("--result", default=None, help="JSON report path")
    ap.add_argument(
        "--mode",
        default="v1",
        choices=["v1", "v2", "v3", "v4"],
        help="v1: last-token gate (default); v2/v3/v4: see V2_*/V3_*/V4_*",
    )
    ap.add_argument("--v2-reference", default=None, help="v2: JSON report of the dense reference run to accept against")
    ap.add_argument("--v3-reference", default=None, help="v3: JSON report of the dense reference run to accept against")
    ap.add_argument("--v4-reference", default=None, help="v4: JSON report of the dense reference run to accept against")
    ap.add_argument(
        "--v3-window",
        default="",
        help="v3: add a long case L:N scoring the last N positions of an L-token prefill (e.g. 8192:1024); "
        "the reference must have been run with the same window",
    )
    args = ap.parse_args()

    env = dict(os.environ)
    env["PREFILL_GATE_LENGTHS"] = args.lengths
    if args.v3_window:
        env["PREFILL_GATE_V3_WINDOW"] = args.v3_window
    env["PREFILL_GATE_SAVE"] = "1" if args.save_goldens else "0"
    if args.result:
        env["PREFILL_GATE_RESULT"] = str(Path(args.result).resolve())
    py = sys.executable
    if args.mode in ("v2", "v3", "v4"):
        m = args.mode
        reference = {"v2": args.v2_reference, "v3": args.v3_reference, "v4": args.v4_reference}[m]
        if reference:
            env[f"PREFILL_GATE_{m.upper()}_REF"] = str(Path(reference).resolve())
        rc, dt = _run(
            [py, "-m", "pytest", "-x", "-q", str(Path(__file__).resolve()) + f"::test_prefill_gate_{m}", "-k", "1x4"],
            env,
            f"gate {m}",
        )
        verdict = "PASS" if rc == 0 else "FAIL"
        if not reference and rc == 0:
            verdict = "MEASURED (no reference given; NaN check only)"
        print(f"[gate] {m}: {verdict} ({dt:.0f}s)")
        print(f"[gate] RESULT: {'PASS' if rc == 0 else 'FAIL'}", flush=True)
        return 0 if rc == 0 else 1
    stages = []
    rc, dt = _run(
        [py, "-m", "pytest", "-x", "-q", str(Path(__file__).resolve()) + "::test_prefill_gate", "-k", "1x4"],
        env,
        "goldens+hf",
    )
    stages.append(("goldens+hf", rc, dt))
    if not args.no_unit and not args.save_goldens:
        rc, dt = _run(
            [
                py,
                "-m",
                "pytest",
                "-q",
                "models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder",
                "-k",
                "1x4 and (prefill_128 or prefill_1024) and layer_0",
            ],
            env,
            "unit decoder 1x4",
        )
        stages.append(("unit decoder 1x4", rc, dt))
    ok = all(rc == 0 for _, rc, _ in stages)
    print("[gate] " + " | ".join(f"{n}: {'PASS' if rc == 0 else 'FAIL'} ({dt:.0f}s)" for n, rc, dt in stages))
    print(f"[gate] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
