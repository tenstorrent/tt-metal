#!/usr/bin/env python3
"""Regression check for DG_DENOISE_SLIDING_WINDOW=1 on the questions that were already CLEAN.

The collapsed-question arm can only look like an improvement, so on its own it cannot tell a fix
from a change that trades one failure mode for another. These 67 questions answered fine under the
baseline and cross 1023 too (they run 1-17 blocks), so their behaviour changes as well -- and a
change that repairs 56 collapses while corrupting clean answers is not a fix.

Three questions, in increasing strictness:

  1. does the guard now fire on a question that used to be clean?      (new collapses)
  2. does the answer change, and if so, in which direction?            (baseline -> window)
  3. does agreement with the CUDA reference go up or down?             (the fidelity claim)

Agreement with the reference is the one that matters most: the whole argument for this flag is that
it moves TT toward HF's geometry, so agreement should not fall.
"""
import glob
import json
import os
import re
import sys

BASE = os.environ.get("BASE_DIR", "/tmp/dg_gpqa198_stoparm")
ARM = os.environ.get("ARM_DIR", "/tmp/dg_gpqa_sw_clean")
GOLD = "/tmp/gpqa198/gold.json"
GPU = os.environ.get("GPU_JSONL", "/tmp/gpu_reference.jsonl")
ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\{?([A-D])")


def answer_of(metrics_path):
    try:
        text = json.load(open(metrics_path, encoding="utf-8")).get("text", "")
    except (OSError, ValueError):
        return None
    found = ANSWER.findall(text)
    return found[-1].upper() if found else None


def main() -> int:
    gold = json.load(open(GOLD, encoding="utf-8"))
    reference = {}
    if os.path.exists(GPU):
        for line in open(GPU, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                reference[row["index"]] = row.get("pred")

    rows = []
    for arm_metrics in sorted(glob.glob(f"{ARM}/m_q*.json")):
        index = int(os.path.basename(arm_metrics)[3:6])
        base_metrics = f"{BASE}/m_q{index:03d}.json"
        if not os.path.exists(base_metrics):
            continue
        arm_log = f"{ARM}/q{index:03d}.log"
        raw = open(arm_log, encoding="utf-8", errors="replace").read() if os.path.exists(arm_log) else ""
        rows.append(
            {
                "q": index,
                "guard": raw.count("ending request at block"),
                "window_active": "DG_DENOISE_SLIDING_WINDOW=1" in raw,
                "base": answer_of(base_metrics),
                "arm": answer_of(arm_metrics),
                "ref": reference.get(index),
                "gold": gold.get(str(index)),
            }
        )

    if not rows:
        print(f"no results yet in {ARM}")
        return 0

    new_collapse = [r for r in rows if r["guard"]]
    lost_answer = [r for r in rows if r["base"] and not r["arm"]]
    changed = [r for r in rows if r["base"] and r["arm"] and r["base"] != r["arm"]]
    inactive = [r for r in rows if not r["window_active"]]

    def correct(key):
        return sum(1 for r in rows if r[key] and r["gold"] and r[key] == r["gold"])

    print(f"clean-question regression arm: {len(rows)} of 67 done\n")
    print(f"  NEW collapses (guard fired) : {len(new_collapse):>3}  {[r['q'] for r in new_collapse] or ''}")
    print(f"  lost an extractable answer  : {len(lost_answer):>3}  {[r['q'] for r in lost_answer] or ''}")
    print(f"  answer changed              : {len(changed):>3}")
    if inactive:
        print(f"  !! window flag not active in {len(inactive)}: {[r['q'] for r in inactive]}")
    print(f"\n  correct, baseline : {correct('base'):>3}/{len(rows)}")
    print(f"  correct, window   : {correct('arm'):>3}/{len(rows)}")

    if reference:
        agree_base = sum(1 for r in rows if r["base"] and r["ref"] and r["base"] == r["ref"])
        agree_arm = sum(1 for r in rows if r["arm"] and r["ref"] and r["arm"] == r["ref"])
        print(f"\n  agrees with the reference, baseline : {agree_base:>3}/{len(rows)}")
        print(
            f"  agrees with the reference, window   : {agree_arm:>3}/{len(rows)}"
            f"   <- the flag's claim is that this does not fall"
        )
    else:
        print(f"\n  (no reference predictions at {GPU}; agreement not computed)")

    if changed:
        print(f"\n{'q':>5} {'base':>5} {'window':>7} {'ref':>4} {'gold':>5}  direction")
        for r in sorted(changed, key=lambda r: r["q"]):
            was, now = r["base"] == r["gold"], r["arm"] == r["gold"]
            direction = "FIXED" if now and not was else ("BROKE" if was and not now else "wrong->wrong")
            print(
                f"{r['q']:>5} {str(r['base']):>5} {str(r['arm']):>7} {str(r['ref']):>4} "
                f"{str(r['gold']):>5}  {direction}"
            )

    json.dump(rows, open("/tmp/sw_regression_scored.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
