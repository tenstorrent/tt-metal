# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WER and speaker similarity over noise draws: the mean and the range per utterance and for the corpus.

RUN IN THE REFERENCE VENV. Each run directory (one draw) is scored exactly as scripts/eval_wer_sim.py scores a run:
its own functions, unchanged (the same Whisper call and similarity model), and its scores.json is written next to
the results. Groups are compared side by side, e.g. TT against the reference, Stage 1 and streaming:

    $COSYVOICE2_REF_ENV/bin/python eval_draws.py --out <json> \\
        --group "TT Stage 1" <draw dir> ... --group "reference Stage 1" <draw dir> ...

scripts/noise_draws.py (TT) and run_reference.py / streaming_reference.py with --noise-seed (the reference) make the
draws.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_wer_sim as ews  # noqa: E402


def spread(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 2), "min": round(float(np.min(values)), 2),
            "max": round(float(np.max(values)), 2), "n": len(values)}  # fmt: skip


def fmt(s: dict) -> str:
    return f"{s['mean']:.2f} ({s['min']:.2f}-{s['max']:.2f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--reuse", action="store_true", help="load a draw's scores.json, when it has one, instead of scoring it"
    )
    args = ap.parse_args()
    groups = [(g[0], g[1:]) for g in args.group]
    assert all(dirs for _, dirs in groups), "every --group needs a name and at least one run directory"

    asr = sim = None
    report = {}
    for name, dirs in groups:
        per_case, corpus_wer, corpus_sim = {}, [], []
        for d in dirs:
            print(f"[{name}] {d}", flush=True)
            path = os.path.join(d, "scores.json")
            if args.reuse and os.path.exists(path):
                with open(path) as fh:
                    run = json.load(fh)
            else:
                if asr is None:
                    asr, sim = ews.ASR(), ews.SpeakerSim()
                run = ews.score_run(d, asr, sim, None)
                with open(path, "w") as fh:
                    json.dump(run, fh, indent=2, ensure_ascii=False)
            corpus_wer.append(run["aggregate"]["corpus_wer_percent"])
            corpus_sim.append(run["aggregate"]["sim_mean"])
            for r in run["scored"]:
                c = per_case.setdefault(r["case_id"], {"wer": [], "sim": [], "words": r["ref_units"], "hyp": []})
                c["wer"].append(r["error_rate_percent"])
                c["sim"].append(r.get("sim", float("nan")))
                c["hyp"].append(r["asr_hypothesis"])
        report[name] = {
            "draws": dirs,
            "corpus": {"wer_percent": spread(corpus_wer), "sim": spread(corpus_sim)},
            "cases": {k: {"words": v["words"], "wer_percent": spread(v["wer"]), "sim": spread(v["sim"]),
                          "hypotheses": v["hyp"]} for k, v in sorted(per_case.items())},  # fmt: skip
        }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    names = [n for n, _ in groups]
    print("\nWER % and SIM, mean (range) over the draws:\n")
    print(
        "| case | words | "
        + " | ".join(f"WER {n}" for n in names)
        + " | "
        + " | ".join(f"SIM {n}" for n in names)
        + " |"
    )
    print("|---|---|" + "---|" * (2 * len(names)))
    for case in sorted(set().union(*(report[n]["cases"] for n in names))):
        cs = [report[n]["cases"].get(case) for n in names]
        words = next(c["words"] for c in cs if c)
        print(f"| {case} | {words} | " + " | ".join(fmt(c["wer_percent"]) if c else "" for c in cs) + " | "
              + " | ".join(fmt(c["sim"]) if c else "" for c in cs) + " |")  # fmt: skip
    print("| **corpus** | | " + " | ".join(f"**{fmt(report[n]['corpus']['wer_percent'])}**" for n in names) + " | "
          + " | ".join(f"**{fmt(report[n]['corpus']['sim'])}**" for n in names) + " |")  # fmt: skip
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
