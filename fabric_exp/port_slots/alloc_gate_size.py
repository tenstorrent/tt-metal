#!/usr/bin/env python3
"""alloc_gate_size.py ARM=CASES_DIR ...: router code size and stack per mock case and arm (the staged allocator's gate,
SLOT_ALLOCATOR.md). CASES_DIR holds one directory per case (cov.py's COV/cases, or rc_capture.sh's RC/port); every
fabric_erisc_router ELF in a case's kernel cache is measured as size_report.py does (loadable bytes; deepest static
stack from _start against the ERISC's entry sp: free stack). Per case and arm: the largest E0 plus the largest E1 (an upper bound
for one router's two images), the largest E0 and E1, the least free stack per ERISC; then the worst over every
case, against the 26,624 B kernel config buffer."""
import collections
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../consolidation"))
import size_report as sr  # noqa: E402


def measure(case_dir):
    per_hash = collections.defaultdict(dict)
    for proc in sr.ENTRY_SP:
        for e in glob.glob(os.path.join(case_dir, "**", "kernels", "fabric_erisc_router", "*", proc, f"{proc}.elf"),
                           recursive=True):
            r = sr.one(e, proc)
            per_hash[e.split("/")[-3]][sr.ERISC[proc]] = r
    if not per_hash:
        return None
    code = {x: max((v[x]["code"] for v in per_hash.values() if x in v), default=0) for x in ("E0", "E1")}
    # E0 and E1 of one router are separate builds (their args differ): the largest E0 plus the largest E1 bounds a
    # router's two images from above
    both = code["E0"] + code["E1"]
    free = {x: min((v[x]["free"] for v in per_hash.values() if x in v), default=0) for x in ("E0", "E1")}
    return {"builds": len(per_hash), "both": both, "code": code, "free": free}


def main():
    arms = [a.partition("=")[::2] for a in sys.argv[1:]]
    cases = sorted({os.path.basename(p) for _, d in arms for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)})
    print("| case | " + " | ".join(f"{n}: builds, E0+E1, E0, E1, free E0 / E1" for n, _ in arms) + " |")
    print("| --- | " + " | ".join("---" for _ in arms) + " |")
    worst = {n: {"both": 0, "E0": 0, "E1": 0, "freeE0": 1 << 30, "freeE1": 1 << 30, "case": {}} for n, _ in arms}
    for c in cases:
        cells = []
        for n, d in arms:
            m = measure(os.path.join(d, c))
            if not m:
                cells.append("-")
                continue
            w = worst[n]
            for k, v in (("both", m["both"]), ("E0", m["code"]["E0"]), ("E1", m["code"]["E1"])):
                if v > w[k]:
                    w[k] = v
                    w["case"][k] = c
            for x in ("E0", "E1"):
                # a 1-ERISC router builds no E0 image: no stack to count
                if m["code"][x] and m["free"][x] < w["free" + x]:
                    w["free" + x] = m["free"][x]
                    w["case"]["free" + x] = c
            cells.append(f"{m['builds']}, {m['both']}, {m['code']['E0']}, {m['code']['E1']}, "
                         f"{m['free']['E0']} / {m['free']['E1']}")
        print(f"| {c} | " + " | ".join(cells) + " |")
    print()
    for n, _ in arms:
        w = worst[n]
        print(f"{n}: largest E0+E1 {w['both']} B ({w['case'].get('both')}; buffer {sr.KERNEL_CONFIG} B, "
              f"{sr.KERNEL_CONFIG - w['both']} B left); largest E0 {w['E0']} ({w['case'].get('E0')}), E1 {w['E1']} "
              f"({w['case'].get('E1')}); least free stack E0 {w['freeE0']} ({w['case'].get('freeE0')}), "
              f"E1 {w['freeE1']} ({w['case'].get('freeE1')})")


if __name__ == "__main__":
    main()
