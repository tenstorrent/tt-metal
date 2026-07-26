#!/usr/bin/env python3
"""Score the DG_DENOISE_SLIDING_WINDOW=1 arm against the shipped-config baseline.

The arm ran exactly the 64 questions that collapsed in the baseline, so the baseline value for
every question here is "collapsed". Three views, same discipline as the earlier runs:

  GUARD  -- did the degeneracy guard refuse a canvas and end the request?
  BLOCKS -- how many blocks were emitted (a clean finish means EOS, not a refusal)
  ANSWER -- did an `Answer: $LETTER` come out, and is it the GPQA gold letter?

A fix has to move all three: no guard fire, a natural finish, and an extractable correct answer.
Reporting only the guard would count "ran longer, still garbage" as a win.
"""
import glob
import json
import os
import re
import sys

ARM = os.environ.get("ARM_DIR", "/tmp/dg_gpqa_slidingwindow")
GOLD = "/tmp/gpqa198/gold.json"
ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\{?([A-D])")


def committed_text(metrics_path):
    """The generated text, from the metrics JSON's `text` field (not scraped from the log)."""
    try:
        return json.load(open(metrics_path, encoding="utf-8")).get("text", "")
    except (OSError, ValueError):
        return ""


def main() -> int:
    gold = json.load(open(GOLD, encoding="utf-8"))
    if isinstance(gold, dict):
        gold_of = lambda i: gold.get(str(i), gold.get(i))  # noqa: E731
    else:
        gold_of = lambda i: gold[i]  # noqa: E731

    rows = []
    for log in sorted(glob.glob(f"{ARM}/q*.log")):
        index = int(os.path.basename(log)[1:4])
        raw = open(log, encoding="utf-8", errors="replace").read()
        metrics = f"{ARM}/m_q{index:03d}.json"
        done = os.path.exists(metrics)
        guard = raw.count("ending request at block")
        blocks = raw.count("DG_DEGENERACY ")
        window = "DG_DENOISE_SLIDING_WINDOW=1" in raw
        text = committed_text(metrics)
        found = ANSWER.findall(text)
        letter = found[-1].upper() if found else None
        want = gold_of(index)
        want = want.get("answer") if isinstance(want, dict) else want
        rows.append(
            {
                "q": index,
                "done": done,
                "guard": guard,
                "blocks": blocks,
                "window_active": window,
                "answer": letter,
                "gold": want,
                "correct": bool(letter and want and letter == str(want).strip().upper()[:1]),
            }
        )

    done = [r for r in rows if r["done"]]
    print(f"arm dir: {ARM}")
    print(f"finished: {len(done)} / {len(rows)} started  (baseline: ALL of these collapsed)\n")
    if not done:
        return 0

    no_guard = [r for r in done if not r["guard"]]
    answered = [r for r in done if r["answer"]]
    correct = [r for r in done if r["correct"]]
    inactive = [r for r in done if not r["window_active"]]
    print(f"  guard did NOT fire : {len(no_guard):>3} / {len(done)}   <- collapse fixed")
    print(f"  answer extracted   : {len(answered):>3} / {len(done)}")
    print(f"  answer correct     : {len(correct):>3} / {len(done)}")
    if inactive:
        print(f"  !! window flag not active in {len(inactive)} logs: {[r['q'] for r in inactive]}")

    print(f"\n{'q':>5} {'guard':>6} {'blocks':>7} {'ans':>4} {'gold':>5} {'ok':>3}")
    for r in sorted(done, key=lambda r: r["q"]):
        print(
            f"{r['q']:>5} {r['guard']:>6} {r['blocks']:>7} {str(r['answer']):>4} "
            f"{str(r['gold']):>5} {'Y' if r['correct'] else '.':>3}"
        )

    block0 = {7, 64, 90, 95, 96, 106, 122}
    ctrl = [r for r in done if r["q"] in block0]
    if ctrl:
        still = sum(1 for r in ctrl if r["guard"])
        print(f"\nnegative control (block-0 collapses, prompt_len 167-481, window never binds):")
        print(f"  {still} / {len(ctrl)} still collapse -- these MUST NOT be fixed by this arm")

    json.dump(rows, open("/tmp/sw_arm_scored.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
