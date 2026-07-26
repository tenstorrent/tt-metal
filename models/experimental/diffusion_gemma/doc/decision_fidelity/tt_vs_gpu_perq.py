#!/usr/bin/env python3
"""Per-question TT-vs-GPU comparison. Aggregate rates hide the question that matters.

The rates alone cannot distinguish two very different worlds:

  * TT collapses on the SAME questions the model already struggles with -> the collapse is the
    diffusion-loop expression of an inherent difficulty, and the ceiling is the model's;
  * TT collapses on questions the GPU answers cleanly -> those are TT defects, and each one is a
    concrete bug with a reproducer.

So this cross-tabulates, question by question, what the CUDA reference did against what TT did.
The GPU answer set is the union of its 8192 run and the 16384 retry of the 45 it failed, which is
the fairest reading of "could the reference do this question at all".
"""
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = {int(k): v for k, v in json.load(open(f"{HERE}/gold.json")).items()}
ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])")


def load_gpu():
    best = {}
    for path, label in ((f"{HERE}/gpu_8192.jsonl", "8192"), (f"{HERE}/gpu_16k_retry.jsonl", "16k")):
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            index = row["index"]
            entry = {
                "answered": bool(row.get("parsed")),
                "correct": bool(row.get("correct")),
                "tokens": row.get("completion_tokens") or 0,
                "at": label,
            }
            # the 16k retry supersedes the 8192 attempt for the questions it covers
            if index not in best or label == "16k":
                best[index] = entry
    return best


def load_tt():
    summary = {}
    path = f"{HERE}/tt_logs/hifi2_summary.txt"
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 3 or not parts[0].startswith("q"):
            continue
        index = int(parts[0][1:])
        guard = int(parts[1].split("=")[1])
        blocks = int(parts[2].split("=")[1])
        summary[index] = {"guard": guard, "blocks": blocks}
    for index, entry in summary.items():
        metrics = f"{HERE}/tt_hifi2/m_q{index:03d}.json"
        entry["answered"] = False
        entry["correct"] = False
        entry["chars"] = 0
        if os.path.isfile(metrics):
            try:
                data = json.load(open(metrics))
            except Exception:
                continue
            text = data.get("text") or ""
            found = ANSWER.findall(text)
            entry["chars"] = len(text)
            if found:
                entry["answered"] = True
                entry["correct"] = found[-1] == GOLD.get(index)
    return summary


def main():
    gpu, tt = load_gpu(), load_tt()
    both = sorted(set(gpu) & set(tt))
    print(f"questions with both arms: {len(both)}\n")

    cross = Counter()
    tt_collapse_gpu_ok = []
    tt_collapse_gpu_also_fails = []
    for index in both:
        g, t = gpu[index], tt[index]
        collapsed = bool(t["guard"])
        cross[(g["answered"], collapsed)] += 1
        if collapsed:
            (tt_collapse_gpu_ok if g["answered"] else tt_collapse_gpu_also_fails).append(index)

    print("cross-tab, GPU answered x TT collapsed:")
    print(f"  GPU answered & TT clean       {cross[(True, False)]:>4}")
    print(f"  GPU answered & TT COLLAPSED   {cross[(True, True)]:>4}   <-- TT-specific failures")
    print(f"  GPU failed   & TT clean       {cross[(False, False)]:>4}")
    print(f"  GPU failed   & TT collapsed   {cross[(False, True)]:>4}   <-- hard for both")

    n = len(both)
    gpu_ans = sum(1 for i in both if gpu[i]["answered"])
    gpu_cor = sum(1 for i in both if gpu[i]["correct"])
    tt_ans = sum(1 for i in both if tt[i]["answered"])
    tt_cor = sum(1 for i in both if tt[i]["correct"])
    tt_col = sum(1 for i in both if tt[i]["guard"])
    print(f"\n{'':<12} {'answered':>10} {'correct':>9}")
    print(f"{'GPU':<12} {gpu_ans:>4}/{n} {gpu_ans / n:>4.0%} {gpu_cor:>4} {gpu_cor / n:>4.0%}")
    print(f"{'TT (HiFi2)':<12} {tt_ans:>4}/{n} {tt_ans / n:>4.0%} {tt_cor:>4} {tt_cor / n:>4.0%}")
    print(f"TT collapsed: {tt_col}/{n} ({tt_col / n:.0%})")

    print(f"\nTT collapses where the GPU ANSWERED FINE ({len(tt_collapse_gpu_ok)}):")
    print(f"  {tt_collapse_gpu_ok}")
    print(f"TT collapses where the GPU also failed ({len(tt_collapse_gpu_also_fails)}):")
    print(f"  {tt_collapse_gpu_also_fails}")

    if tt_collapse_gpu_ok:
        share = len(tt_collapse_gpu_ok) / max(1, tt_col)
        print(f"\n  -> {share:.0%} of TT's collapses are on questions the reference handles.")
        toks = [gpu[i]["tokens"] for i in tt_collapse_gpu_ok]
        toks.sort()
        print(f"     GPU token use on those: median {toks[len(toks) // 2]}, min {toks[0]}, max {toks[-1]}")


if __name__ == "__main__":
    main()
