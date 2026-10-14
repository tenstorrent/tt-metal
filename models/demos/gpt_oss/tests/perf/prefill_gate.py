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
    args = ap.parse_args()

    env = dict(os.environ)
    env["PREFILL_GATE_LENGTHS"] = args.lengths
    env["PREFILL_GATE_SAVE"] = "1" if args.save_goldens else "0"
    if args.result:
        env["PREFILL_GATE_RESULT"] = str(Path(args.result).resolve())
    py = sys.executable
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
