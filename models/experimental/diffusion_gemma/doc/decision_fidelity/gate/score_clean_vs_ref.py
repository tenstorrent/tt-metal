#!/usr/bin/env python3
"""Score the retention arm's clean-question run against the CUDA reference.

The TT baseline metrics were lost when the box wiped /tmp, so the baseline-vs-arm answer diff is not
available. That costs less than it sounds: the flag's claim is that it moves TT toward HF's geometry,
so agreement with the REFERENCE is the quantity the claim actually rests on, and the reference
predictions survived off-box.

What is genuinely lost is the ability to say "this specific answer changed"; what remains is whether
the arm's accuracy and reference-agreement are where they should be.
"""
import glob, json, os, re, sys

ARM = os.environ.get("ARM_DIR", "/tmp/dg_gpqa_sw_clean")
GOLD = "/tmp/gpqa198/gold.json"
GPU = "/tmp/gpu_reference.jsonl"
ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\{?([A-D])")


def main() -> int:
    gold = json.load(open(GOLD, encoding="utf-8"))
    ref = {}
    for line in open(GPU, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            ref[row["index"]] = row

    rows = []
    for path in sorted(glob.glob(f"{ARM}/m_q*.json")):
        index = int(os.path.basename(path)[3:6])
        metrics = json.load(open(path, encoding="utf-8"))
        found = ANSWER.findall(metrics.get("text", ""))
        log = f"{ARM}/q{index:03d}.log"
        raw = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
        rows.append(
            {
                "q": index,
                "answer": found[-1].upper() if found else None,
                "gold": gold.get(str(index)),
                "ref": ref.get(index, {}).get("pred"),
                "ref_ok": ref.get(index, {}).get("correct"),
                "guard": raw.count("ending request at block"),
                "blocks": metrics.get("blocks_emitted"),
                "halted": metrics.get("halted_per_block") or [],
            }
        )

    answered = [r for r in rows if r["answer"]]
    tt_ok = sum(1 for r in rows if r["answer"] and r["answer"] == r["gold"])
    ref_ok = sum(1 for r in rows if r["ref_ok"])
    agree = sum(1 for r in rows if r["answer"] and r["ref"] and r["answer"] == r["ref"])
    halted = sum(sum(1 for x in r["halted"] if x) for r in rows)
    blocks = sum(len(r["halted"]) for r in rows)

    print(f"clean-question regression arm, retention ON: {len(rows)}/67 done\n")
    print(
        f"  NEW collapses (guard fired) : {sum(1 for r in rows if r['guard'])}/{len(rows)}"
        f"   {[r['q'] for r in rows if r['guard']]}"
    )
    print(f"  answered                    : {len(answered)}/{len(rows)}")
    print(f"  correct                     : {tt_ok}/{len(rows)}")
    print(f"  reference correct (same set): {ref_ok}/{len(rows)}")
    print(f"  agrees with the reference   : {agree}/{len(rows)}")
    print(f"  blocks that halt            : {halted}/{blocks} = {100*halted/blocks:.0f}%")
    missing = [r["q"] for r in rows if not r["answer"]]
    if missing:
        print(f"  no extractable answer       : {missing}")
    json.dump(rows, open("/tmp/clean_vs_ref.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
