#!/usr/bin/env python3
"""Phase A / step 0: replay v3 reconcile.py over every v2 cell's own committed artefacts.

For each (cell, layer kind) the v2 corpus published a reconciliation. This script finds the inputs that
produced it, re-runs the v3 tool on them, and diffs the two.

The CSV is identified by MATCHING THE WINDOW v2 recorded, not by filename. Filenames are inconsistent
across cells (per-kind, per-scope, some under invalid_prior/), and pairing the wrong CSV would produce a
confident wrong comparison -- the exact failure mode the v2 analysis kept hitting. If no CSV reproduces
the window to 0.01 us, the pair is reported as NOT REPRODUCIBLE rather than guessed at.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

CORPUS = Path("/tmp/claude-1000/-home-mvasiljevic/60266cca-1b03-46c6-a07a-f79b5d4bd278/scratchpad/ar/"
              "shard-advisor-experiments/03-advisor-stage-v2/autoports")
RECONCILE = Path("/home/mvasiljevic/_v3-verify-ro/.agents/skills/advisor-challenger/scripts/reconcile.py")
OUT = Path("/tmp/claude-1000/-home-mvasiljevic/60266cca-1b03-46c6-a07a-f79b5d4bd278/scratchpad/step0")


def read_csv_total(path: Path) -> float | None:
    """Total device time in us, by the same column rules reconcile.py uses."""
    try:
        raw = gzip.open(path, "rt", errors="replace") if path.suffix == ".gz" else path.open(newline="")
        with raw as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return None
    if not rows:
        return None
    keys = {(k or "").strip().lower(): k for k in rows[0]}

    def find(*needles):
        for n in needles:
            if n in keys:
                return keys[n]
        for n in needles:
            for k, orig in keys.items():
                if n in k:
                    return orig
        return None

    dur = find("device time", "device kernel duration [ns]", "device kernel duration")
    code = find("op code", "op type", "name")
    if not dur or not code:
        return None
    scale = 1 / 1000.0 if ("ns" in dur.lower() or "cycle" in dur.lower()) else 1.0
    total = 0.0
    for r in rows:
        if not (r.get(code) or "").strip():
            continue
        try:
            total += float(str(r[dur]).replace(",", "")) * scale
        except (TypeError, ValueError):
            pass
    return total


def main() -> int:
    results = []
    for cell_dir in sorted(CORPUS.glob("*/*/")):
        model, arm = cell_dir.parts[-2], cell_dir.parts[-1]
        ac = cell_dir / "doc" / "advisor_challenger"
        if not ac.is_dir():
            continue
        # every CSV anywhere under doc/, excluding the stale copies one cell kept
        csvs = [p for p in (cell_dir / "doc").rglob("*") if p.suffix in (".csv", ".gz")
                and p.name.endswith((".csv", ".csv.gz")) and "invalid_prior" not in p.parts]
        totals = {p: read_csv_total(p) for p in csvs}
        for rec_path in sorted(ac.glob("reconciliation*.json")):
            v2 = json.loads(rec_path.read_text())
            kind = v2.get("layer_kind", "unknown")
            row = {"model": model, "arm": arm, "kind": kind, "v2_reconciliation": rec_path.name,
                   "v2_window_us": v2.get("measured_window_us")}
            sa = ac / "shard_advise" / kind
            report, ir = sa / "report.json", sa / "final_ir.mlir"
            if not report.is_file():                       # some cells name the dir differently
                cand = [d for d in (ac / "shard_advise").glob("*") if d.is_dir()]
                if len(cand) == 1:
                    report, ir = cand[0] / "report.json", cand[0] / "final_ir.mlir"
            if not report.is_file():
                row["status"] = "NO REPORT"
                results.append(row); continue
            row["report"] = str(report.relative_to(CORPUS))
            row["has_ir"] = ir.is_file()

            want = v2.get("measured_window_us")
            match = [p for p, t in totals.items()
                     if t is not None and want is not None and abs(t - want) < 0.011]
            if not match:
                row["status"] = "WINDOW NOT REPRODUCIBLE"
                near = [t for t in totals.values() if t is not None]
                row["csvs_found"] = len(near)
                row["closest_csv_us"] = (min((abs(t - want), t) for t in near)[1]
                                         if near and want else None)
                results.append(row); continue
            perf = sorted(match, key=lambda p: len(str(p)))[0]
            row["perf"] = str(perf.relative_to(CORPUS))

            # Pick the incumbent whose SCOPE matches this window. One cell's incumbent.json is a derived
            # full-model composite (27.6 ms against a 0.5 ms layer), and its per-kind files are named for
            # the kind by a different convention than the reconciliation is -- so match on the number,
            # not the filename, and require it to be within 5x of the window.
            cands = []
            for q in sorted(ac.glob("incumbent*.json")):
                try:
                    ims = json.loads(q.read_text()).get("incumbent_ms")
                except Exception:
                    continue
                if isinstance(ims, (int, float)) and ims > 0 and want:
                    ratio = want / (ims * 1000.0)
                    if 0.2 < ratio < 5:
                        cands.append((abs(1 - ratio), q))
            inc = min(cands)[1] if cands else None
            row["incumbent"] = inc.name if inc else None
            row["incumbent_scope_note"] = None if inc else (
                "no incumbent record has a scope compatible with this window")

            src = perf
            if perf.suffix == ".gz":                       # reconcile.py does not read gzip
                src = OUT / "tmp" / (perf.name[:-3])
                src.parent.mkdir(parents=True, exist_ok=True)
                src.write_bytes(gzip.decompress(perf.read_bytes()))

            outp = OUT / "v3" / f"{model}__{arm}__{kind}.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(RECONCILE), "--report", str(report), "--perf", str(src),
                   "--layer-kind", kind, "--out", str(outp),
                   "--layers-of-kind", str(v2.get("layers_of_kind", 1)),
                   "--total-layers", str(v2.get("total_layers", 1))]
            if ir.is_file():
                cmd += ["--ir", str(ir)]
            if inc:
                cmd += ["--incumbent", str(inc)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            row["rc"] = p.returncode
            if p.returncode != 0:
                row["status"] = "TOOL ABORTED"
                row["stderr"] = (p.stderr or "").strip().splitlines()[-1][:300] if p.stderr else ""
                results.append(row); continue

            v3 = json.loads(outp.read_text())
            row["status"] = "ok"
            row.update(compare(v2, v3))
            results.append(row)

    (OUT / "step0-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"{len(results)} (cell, kind) pairs -> {OUT/'step0-results.json'}")
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['status']:24s} {r['model']}/{r['arm']}/{r['kind']}"
                  + (f"  {r.get('stderr','')}" if r.get("stderr") else ""))
    return 0


def bucket(d):
    return {k: (round(v["us"], 3), v["ops"]) for k, v in (d.get("accounting") or {}).items()}


def compare(v2: dict, v3: dict) -> dict:
    a2, a3 = bucket(v2), bucket(v3)
    adv2 = {(r.get("device"), i): r for i, r in enumerate(v2.get("disagreements") or [])}
    adv3 = {(r.get("device"), i): r for i, r in enumerate(v3.get("disagreements") or [])}
    understated = sum(1 for k, r in adv3.items()
                      if r.get("advised_cores") is not None
                      and r.get("advised_cores_bbox") is not None
                      and r["advised_cores_bbox"] < r["advised_cores"])
    with_cores = sum(1 for r in adv3.values() if r.get("advised_cores") is not None)
    # rows v2 called `chain` that v3 does not
    moved = {}
    for k in set(adv2) & set(adv3):
        b2, b3 = adv2[k].get("bucket"), adv3[k].get("bucket")
        if b2 != b3:
            moved[f"{b2}->{b3}"] = moved.get(f"{b2}->{b3}", 0) + 1
    ch3 = v3.get("chains") or []
    ceil3 = (v3.get("feasibility") or {}).get("ceiling_us")
    return {
        "window_same": abs((v2.get("measured_window_us") or 0) - (v3.get("measured_window_us") or -1)) < 0.011,
        "v3_closes": v3.get("accounting_closes_100pct"),
        "v2_closes": v2.get("accounting_closes_100pct"),
        "v2_buckets": a2, "v3_buckets": a3, "bucket_moves": moved,
        "advised_ops_with_cores": with_cores, "advised_cores_understated": understated,
        "v2_verdict": (v2.get("feasibility") or {}).get("verdict"),
        "v3_verdict": (v3.get("feasibility") or {}).get("verdict"),
        "v2_ceiling_us": (v2.get("feasibility") or {}).get("ceiling_us"),
        "v3_ceiling_us": ceil3,
        "v3_regrid_pool_us": (v3.get("feasibility") or {}).get("regrid_pool_us"),
        "v2_chains": len(v2.get("chains") or []), "v3_chains": len(ch3),
        "chain_attrib_sum_us": round(sum(c.get("advisor_removes_us") or 0 for c in ch3), 3),
        "ceiling_reconciles": (abs(sum(c.get("advisor_removes_us") or 0 for c in ch3) - (ceil3 or 0)) < 0.02),
        "v2_chain_attrib_sum_us": round(sum(c.get("advisor_removes_us") or 0
                                            for c in (v2.get("chains") or [])), 3),
        "cliff_ops": len(v3.get("cliff_candidates") or []),
        "cliff": [{"op": c["op"], "cores": c["shipped_cores"], "advised": c["advised_cores"],
                   "share_pct": c["share_pct"], "per_model_us": c["per_model_us"],
                   "ladder": c.get("legal_ladder")} for c in (v3.get("cliff_candidates") or [])],
        "unfixable_declared": sorted((v3.get("advised_plan") or {}).get("unfixable_ops") or {}),
        "unfixable_bucket_us": a3.get("advisor_unfixable", (0, 0))[0],
        "positional_pct": (v3.get("confidence") or {}).get("pct_paired_by_position"),
    }


if __name__ == "__main__":
    sys.exit(main())
