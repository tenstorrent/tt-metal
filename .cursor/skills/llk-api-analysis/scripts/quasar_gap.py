#!/usr/bin/env python3
"""First-pass Quasar (QSR) gap detector for an LLK base-API report.

For each base LLK API in the by-base CSV, counts how many arch LLK source files
mention that API name, per architecture. An API present on a reference arch
(blackhole / wormhole_b0) but absent on the target arch (quasar) is flagged as a
candidate GAP.

Search scope per arch <A>:  tt_metal/hw/ckernels/<A>  +  tt_metal/tt-llk/tt_llk_<A>

This is a NAME-substring heuristic and a first pass only: confirm every flagged
GAP by reading the actual headers (an API may be implemented under a different
name, or routed through a datacopy/unpack-pack path). Rows found on no arch are
INCONCLUSIVE (likely a shared/common-layer or higher-level API, not arch-specific).

Usage:
  python3 quasar_gap.py <by_base_api.csv> <tt_metal_root> [out.csv] \
      [--target quasar] [--ref blackhole,wormhole_b0]
"""
import csv
import os
import sys

SRC_EXT = (".h", ".hpp", ".hh", ".inc", ".cpp", ".cc", ".c")


def parse_args(argv):
    pos, target, ref = [], "quasar", "blackhole,wormhole_b0"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--target":
            target = argv[i + 1]
            i += 2
        elif a == "--ref":
            ref = argv[i + 1]
            i += 2
        else:
            pos.append(a)
            i += 1
    csv_in = pos[0] if pos else "llk_report_by_base_api.csv"
    root = pos[1] if len(pos) > 1 else "."
    out = pos[2] if len(pos) > 2 else "llk_report_quasar_gap.csv"
    return csv_in, root, out, target, [r for r in ref.split(",") if r]


def arch_corpus(root, arch):
    """List of file-contents for one arch's LLK source trees."""
    dirs = [
        os.path.join(root, "tt_metal/hw/ckernels", arch),
        os.path.join(root, "tt_metal/tt-llk", f"tt_llk_{arch}"),
    ]
    texts = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for dp, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(SRC_EXT):
                    try:
                        with open(os.path.join(dp, fn), errors="ignore") as fh:
                            texts.append(fh.read())
                    except OSError:
                        pass
    return texts


def count_files(corpus, token):
    return sum(1 for t in corpus if token in t)


def main():
    csv_in, root, out, target, refs = parse_args(sys.argv[1:])
    archs = [target] + refs
    corpora = {a: arch_corpus(root, a) for a in archs}
    for a in archs:
        if not corpora[a]:
            print(f"warning: no LLK source found for arch '{a}' under {root}", file=sys.stderr)

    rows = list(csv.DictReader(open(csv_in, newline="")))
    out_rows, gaps = [], []
    for r in rows:
        name = r["LLK API"].strip().strip("`").strip().rstrip("_")
        counts = {a: count_files(corpora[a], name) for a in archs}
        tgt = counts[target]
        ref_hits = max((counts[a] for a in refs), default=0)
        if tgt > 0:
            status = "present"
        elif ref_hits > 0:
            status = "GAP"
            gaps.append(name)
        else:
            status = "inconclusive"
        row = {"LLK API": name, "status": status}
        row.update({a: counts[a] for a in archs})
        out_rows.append(row)

    cols = ["LLK API", "status"] + archs
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    n_gap = sum(1 for r in out_rows if r["status"] == "GAP")
    n_inc = sum(1 for r in out_rows if r["status"] == "inconclusive")
    print(f"base APIs           : {len(out_rows)}")
    print(f"present on {target:9}: {sum(1 for r in out_rows if r['status']=='present')}")
    print(f"candidate GAPs      : {n_gap}")
    print(f"inconclusive        : {n_inc}")
    print(f"wrote               : {out}")
    if gaps:
        print("\n=== candidate Quasar GAPs (confirm by reading headers) ===")
        for g in gaps:
            print(f"  {g}")


if __name__ == "__main__":
    main()
