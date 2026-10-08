#!/usr/bin/env python3
"""Write eval/score.json, summary.md, ops.csv and error.txt for one attempt, or build baseline.json.

Called by eval_attempt.sh; not meant to be run by workers directly.

    score.py --campaign C --node N --out-dir <node>/eval --status ok --log run.log --report-dir R --rc 0
    score.py --campaign C --node N --out-dir <node>/eval --status build_error --error-file build.log
    score.py --campaign C --measure-only --log run.log --report-dir R --rc 0 --out m.json
    score.py --campaign C --make-baseline m1.json m2.json m3.json --out baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream.campaign import load_campaign  # noqa: E402
from dream.scoring import (  # noqa: E402
    make_baseline,
    measure,
    score_vs_baseline,
    summary_md,
)

PRE_RUN = {"build_error", "forbidden_edit", "hang", "infra"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--node")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--status", default="ok", help="ok (test ran) or a pre-run fail_class")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--report-dir", type=Path)
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--error-file", type=Path)
    ap.add_argument("--error", default=None)
    ap.add_argument("--meta", default="{}", help="extra JSON merged into score.json")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--make-baseline", nargs="+", type=Path)
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    cfg = c.cfg

    if args.make_baseline:
        runs = [json.loads(p.read_text()) for p in args.make_baseline]
        bad = [p for p, r in zip(args.make_baseline, runs) if r["fail_class"] != "ok"]
        if bad:
            sys.exit(f"baseline runs failed: {bad}")
        base = make_baseline(runs, float(cfg.get("eval", {}).get("min_noise_pct", 1.0)))
        args.out.write_text(json.dumps(base, indent=2) + "\n")
        print(json.dumps({k: v["us_chip_mean"] for k, v in base["shapes"].items()}), f"noise_pct={base['noise_pct']}")
        return

    if args.measure_only:
        m = measure(cfg, args.log, args.report_dir, args.rc)
        m.pop("rows")
        args.out.write_text(json.dumps(m, indent=2) + "\n")
        print(m["fail_class"], {k: v.get("us_chip_mean") for k, v in m["shapes"].items()})
        return

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("score.json", "summary.md", "ops.csv", "error.txt"):
        (out / stale).unlink(missing_ok=True)
    base_path = c.ledger / "baseline.json"
    baseline = json.loads(base_path.read_text()) if base_path.exists() else None

    res = {"valid": False, "fail_class": args.status, "error": args.error, "score": 0.0}
    if args.status == "ok":
        m = measure(cfg, args.log, args.report_dir, args.rc)
        res.update(fail_class=m["fail_class"], error=m["error"], shapes=m["shapes"], ops_csv=m["ops_csv"])
        if m["rows"] is not None:
            m["rows"].to_csv(out / "ops.csv", index=False)
        if m["fail_class"] == "ok":
            if baseline is None:
                res.update(fail_class="infra", error=f"no baseline at {base_path}; run eval_attempt.sh --baseline")
            else:
                res.update(valid=True, score=score_vs_baseline(res["shapes"], baseline))
    elif args.status not in PRE_RUN:
        sys.exit(f"unknown status {args.status}")
    if args.error_file and args.error_file.exists():
        tail = args.error_file.read_text(errors="replace").splitlines()[-80:]
        res["error"] = res["error"] or "\n".join(tail[-15:])
        (out / "error.txt").write_text("\n".join(tail) + "\n")
    elif res["error"]:
        (out / "error.txt").write_text(str(res["error"]) + "\n")

    res["score_def"] = (
        "geomean over shapes of baseline_us / attempt_us; per-chip mean device kernel time, measured calls only"
    )
    res["noise_pct"] = baseline["noise_pct"] if baseline else None
    res.update(json.loads(args.meta))
    (out / "score.json").write_text(json.dumps(res, indent=2) + "\n")
    if "shapes" in res:
        (out / "summary.md").write_text(summary_md(args.node or "?", res))
    print(json.dumps({k: res[k] for k in ("valid", "fail_class", "score")}))


if __name__ == "__main__":
    main()
