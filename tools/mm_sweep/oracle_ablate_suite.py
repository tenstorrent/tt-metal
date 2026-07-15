#!/usr/bin/env python3
# Part 5 ablation suite: run the controlled counterfactuals for representative Mt=8 winner geometries in
# the retained C++ oracle, hang-safely (reset device between failures). Emits ORACLE_ABLATION.md + JSON.
# Deltas are NOT additive (stages overlap) -> read as critical-path counterfactuals (see oracle_ablate.py).
import json, os, subprocess, sys

import regime_a_bench as b
import oracle_ablate as oa

HERE = os.path.dirname(__file__)

# (label, M,K,N, Ns,Pk,Sm,kb,nsb) — TTNN winners from the characterization.
CONFIGS = [
    ("256x2048x1024 (N=1024, 37%, KxM winner)", 256, 2048, 1024, 1, 4, 2, 2, 2),
    ("256x6144x768 (N=768, 47%, pure-K winner)", 256, 6144, 768, 1, 12, 1, 2, 1),
    ("256x6144x4608 (N=4608, 78%, wide-N control)", 256, 6144, 4608, 1, 12, 1, 2, 1),
]
ABLATIONS = [
    ("full", []),
    ("skipin1", ["--skipin1"]),
    ("skipin0", ["--skipin0"]),
    ("skipfwd", ["--skipfwd"]),
    ("noreduce", ["--noreduce"]),
    ("skipin0+in1", ["--skipin0", "--skipin1"]),
    ("skipin0+in1+noreduce", ["--skipin0", "--skipin1", "--noreduce"]),
    ("nsbcontig", ["--nsbcontig"]),
]


def safe_run(M, K, N, Ns, Pk, Sm, kb, nsb, flags, coremap=False):
    try:
        return oa.run(M, K, N, Ns, Pk, Sm, kb, nsb, flags, coremap=coremap)
    except subprocess.TimeoutExpired:
        subprocess.run(["tt-smi", "-r"], capture_output=True)
        return {
            "cfg": [Ns, Pk, Sm, kb, nsb],
            "flags": flags,
            "pass": False,
            "wall_us": None,
            "risc": {},
            "returncode": "timeout",
        }


def main():
    results = []
    for label, M, K, N, Ns, Pk, Sm, kb, nsb in CONFIGS:
        print(f"\n=== {label} ===", flush=True)
        runs = {}
        for i, (aname, flags) in enumerate(ABLATIONS):
            r = safe_run(M, K, N, Ns, Pk, Sm, kb, nsb, flags, coremap=(aname == "full"))
            runs[aname] = r
            w = r.get("wall_us")
            print(
                f"  {aname:22} wall={w if w is None else round(w,1)}us pass={r.get('pass')} "
                f"rc={r.get('returncode')}",
                flush=True,
            )
            if not r.get("wall_us") and r.get("returncode") in ("timeout",):
                subprocess.run(["tt-smi", "-r"], capture_output=True)
        results.append({"label": label, "M": M, "K": K, "N": N, "cfg": [Ns, Pk, Sm, kb, nsb], "runs": runs})
        json.dump(results, open(f"{HERE}/oracle_ablation.json", "w"), indent=2)
    write_md(results)
    print("WROTE oracle_ablation.json + ORACLE_ABLATION.md")


def write_md(results):
    L = [
        "# Part 5: oracle ablation counterfactuals (Mt=8 winner geometries)\n",
        "Retained C++ oracle (`test_regime_a_mm --unified`). Wall = median kernel us (max over cores, "
        "device profiler). Deltas are critical-path counterfactuals, NOT additive (stages overlap). "
        "`full full-path` should track the TTNN op; a large speedup under an ablation implicates that "
        "stage. skipin0+in1 = compute-only floor; compare to the ideal DRAM time.\n",
    ]
    for r in results:
        M, K, N = r["M"], r["K"], r["N"]
        ideal = b.logical_bytes(M, K, N) / b.PEAK512 * 1e6
        full = r["runs"].get("full", {}).get("wall_us")
        L.append(f"\n## {r['label']}  cfg={tuple(r['cfg'])}  ideal@512={ideal:.1f}us\n")
        L.append("| ablation | wall us | vs full | interpretation cue |")
        L.append("|---|---|---|---|")
        cue = {
            "full": "baseline (tracks TTNN op)",
            "skipin1": "in1 read removed -> if ~=full, in1 NOT limiting",
            "skipin0": "in0 read removed",
            "skipfwd": "in0 ring forward removed",
            "noreduce": "split-K reduction removed -> if faster, reduction limiting",
            "skipin0+in1": "compute-only floor (vs ideal DRAM)",
            "skipin0+in1+noreduce": "compute w/o reduction",
            "nsbcontig": "layout-optimal in1 read -> if faster, in1 layout limiting",
        }
        for aname, _ in ABLATIONS:
            w = r["runs"].get(aname, {}).get("wall_us")
            vs = f"{(w/full-1)*100:+.0f}%" if (w and full) else "-"
            L.append(f"| {aname} | {w if w is None else round(w,1)} | {vs} | {cue.get(aname,'')} |")
    open(f"{HERE}/ORACLE_ABLATION.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
