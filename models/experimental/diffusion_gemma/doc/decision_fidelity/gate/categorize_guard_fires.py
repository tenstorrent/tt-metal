#!/usr/bin/env python3
"""Categorise every guard fire in the sliding-window arm, so the final count is not one number.

"56 of 64 fixed" would be a misleading headline if the residue is a mix of three different things.
A question where the reference ALSO fails was never this fix's target; a question where the reference
rambles to a wrong answer is a shared weakness; only a question the reference answers CORRECTLY and
TT still collapses on is a residual TT defect worth localising next.

Also reports how far the collapse moved, since a fire at block 33 where the baseline fired at 13 is a
different outcome from one that did not move at all.
"""
import glob
import json
import os
import re
import sys

ARM = os.environ.get("ARM_DIR", "/tmp/dg_gpqa_slidingwindow")
BASE = os.environ.get("BASE_DIR", "/tmp/dg_gpqa198_stoparm")
GPU = os.environ.get("GPU_JSONL", "/tmp/gpu_reference.jsonl")
BLOCK = re.compile(
    r"ending request at block (\d+): degenerate committed canvas: (\d+)/(\d+) distinct ids, "
    r"top id (\d+) covers ([\d.]+)%, longest run (\d+)"
)


def collapse_of(log_path):
    if not os.path.exists(log_path):
        return None
    for line in reversed(open(log_path, encoding="utf-8", errors="replace").read().splitlines()):
        m = BLOCK.search(line)
        if m:
            return {
                "block": int(m.group(1)),
                "distinct": int(m.group(2)),
                "top_id": int(m.group(4)),
                "top_pct": float(m.group(5)),
                "run": int(m.group(6)),
            }
    return None


def main() -> int:
    reference = {}
    for line in open(GPU, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            reference[row["index"]] = row

    # The seven block-0 collapses are the retention arm's negative control -- their prefixes never
    # cross 1023, so the mask is bit-identical and they MUST still fire here. They are the target of
    # the OTHER fix (hiding the prefill pad keys), so counting them as residue of THIS one would mix
    # the two defects' books.
    BLOCK0 = {7, 64, 90, 95, 96, 106, 122}

    fires, clean = [], 0
    for metrics in sorted(glob.glob(f"{ARM}/m_q*.json")):
        index = int(os.path.basename(metrics)[3:6])
        arm = collapse_of(f"{ARM}/q{index:03d}.log")
        if arm is None:
            clean += 1
            continue
        fires.append((index, arm, collapse_of(f"{BASE}/q{index:03d}.log"), reference.get(index, {})))

    print(f"{clean + len(fires)} finished: {clean} clean, {len(fires)} guard fires\n")
    buckets = {
        "block-0 set (pad fix's target)": [],
        "residual after both fixes": [],
        "reference also fails": [],
        "reference rambles to a wrong answer": [],
    }
    print(
        f"{'q':>5} {'base blk':>9} {'arm blk':>8} {'moved':>7} {'distinct':>9} {'top%':>6} "
        f"{'ref':>4} {'ref ok':>7} {'ref tok':>8}  bucket"
    )
    for index, arm, base, ref in fires:
        moved = "" if base is None else f"+{arm['block'] - base['block']}"
        if index in BLOCK0:
            bucket = "block-0 set (pad fix's target)"
        elif not ref.get("pred"):
            bucket = "reference also fails"
        elif ref.get("correct"):
            bucket = "residual after both fixes"
        else:
            bucket = "reference rambles to a wrong answer"
        buckets[bucket].append(index)
        print(
            f"{index:>5} {str(base and base['block']):>9} {arm['block']:>8} {moved:>7} "
            f"{arm['distinct']:>9} {arm['top_pct']:>6.1f} {str(ref.get('pred')):>4} "
            f"{str(ref.get('correct')):>7} {str(ref.get('completion_tokens')):>8}  {bucket}"
        )

    print()
    for name, qs in buckets.items():
        print(f"  {name:>36}: {len(qs):>2}  {qs}")
    print(f"\n  {'cleanly fixed by the retention mask':>36}: {clean:>2}")
    moved0 = [i for i, arm, base, _ in fires if i in BLOCK0 and base and arm["block"] == base["block"]]
    if moved0:
        print(f"\n  negative control holding (bit-identical block, mask never binds below 1023): {moved0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
