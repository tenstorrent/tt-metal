# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 of the LLK perf gate: re-measure stage-1 regressions at other code positions.

A candidate is a confirmed regression only when the current side is slower at every code position and the code of
its measured TILE_LOOP window changed. Anything else is reported as unclear, with the reason.

    placement_check.py nodes  --candidates R.regressions.csv --map 'current/**/*.windowhash' --out nodes.txt
    placement_check.py decide --candidates R.regressions.csv --baseline DIR --current DIR --report stage2.md
"""

import argparse
import collections
import csv
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_compare import (  # noqa: E402
    _TOP_N,
    DEFAULT_MIN_CYCLES,
    DEFAULT_THRESHOLD,
    MODULE_COL,
    _medians,
    _module_thresholds,
    _params,
    _read_frames,
    _run_type_of,
)

POSITIONS = ("p0", "p4", "p20", "p36", "p52")
DEFAULT_CAP = 1000
_STATS = ("current", "baseline", "delta_pct", "delta_cycles", "marker", "run_type")


def _value(v):
    if isinstance(v, bool) or v in ("True", "False"):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else repr(f)


def _module_of_node(nodeid):
    return nodeid.split("::")[0].rsplit("/", 1)[-1].removesuffix(".py")


def _config_key(module, run_type, pairs):
    return module, run_type, frozenset((c, _value(v)) for c, v in pairs if pd.notna(v))


def read_window_map(paths):
    """{(module, run type, config): {"nodeid": ..., thread: hash}} from the stage runs' .windowhash files."""
    out = {}
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        if "nodeid" not in frame.columns:
            continue
        config = [
            c
            for c in frame.columns
            if not c.startswith("TEXT_SIZE(")
            and not c.startswith("window_")
            and c != "nodeid"
        ]
        for rec in frame.to_dict("records"):
            key = _config_key(
                _module_of_node(rec["nodeid"]),
                rec["window_run_type"],
                ((c, rec[c]) for c in config),
            )
            entry = out.setdefault(key, {"nodeid": rec["nodeid"]})
            entry[rec["window_thread"]] = rec["window_hash"]
    return out


def read_candidates(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        if row["marker"] != "TILE_LOOP":
            continue
        config = [
            (c, _value(v))
            for c, v in row.items()
            if c not in _STATS and c != MODULE_COL and v != ""
        ]
        out.append(
            {
                "module": row[MODULE_COL],
                "run_type": row["run_type"],
                "config": config,
                "delta_pct": float(row["delta_pct"]),
            }
        )
    return out


def nodes(candidates, window_map, cap):
    ranked = sorted(candidates, key=lambda c: -c["delta_pct"])
    selected, seen = [], set()
    for c in ranked:
        if len(selected) >= cap:
            break
        entry = window_map.get(_config_key(c["module"], c["run_type"], c["config"]))
        if entry and entry["nodeid"] not in seen:
            seen.add(entry["nodeid"])
            selected.append(entry["nodeid"])
    return selected


def _side(root):
    values, windows = [], {}
    for pos in POSITIONS:
        files = [
            p
            for p in glob.glob(f"{root}/{pos}/**/*.csv", recursive=True)
            if not p.endswith(".post.csv")
        ]
        medians = _medians(_read_frames(files)) if files else {}
        values.append(
            {
                (k[0][1], _run_type_of(k[1])): v
                for k, v in medians.items()
                if k[0][0] == "TILE_LOOP"
            }
        )
        if pos == "p0":
            windows = read_window_map(
                glob.glob(f"{root}/{pos}/**/*.windowhash", recursive=True)
            )
    return values, windows


def decide(
    candidates,
    baseline_root,
    current_root,
    threshold,
    min_cycles,
    module_thresholds,
):
    base, base_windows = _side(baseline_root)
    cur, cur_windows = _side(current_root)
    results = []
    index = {}
    for key in base[0]:
        cfg, run_type = key
        config = dict(cfg)
        module = config.pop(MODULE_COL, None)
        index[
            (module, run_type, frozenset((k, _value(v)) for k, v in config.items()))
        ] = key
    for c in candidates:
        limit = module_thresholds.get(c["module"], threshold)
        lookup = index.get(
            (
                c["module"],
                c["run_type"],
                frozenset((k, _value(v)) for k, v in c["config"]),
            )
        )
        row = dict(c, deltas=[], threads=[])
        if lookup is None or not all(lookup in side for side in base + cur):
            row.update(
                outcome="unclear", why="not measured at every position on both sides"
            )
            results.append(row)
            continue
        deltas, slower = [], []
        for b, h in zip(base, cur):
            bv, hv = b[lookup], h[lookup]
            deltas.append(round((hv - bv) / bv * 100, 2))
            slower.append((hv - bv) / bv > limit and hv - bv > min_cycles)
        wkey = _config_key(c["module"], c["run_type"], c["config"])
        bw, cw = base_windows.get(wkey), cur_windows.get(wkey)
        threads = []
        if bw and cw:
            threads = sorted(t for t in bw if t != "nodeid" and bw.get(t) != cw.get(t))
        missing = (
            not bw
            or not cw
            or any(
                str(v).startswith("no-") for v in list(bw.values()) + list(cw.values())
            )
        )
        if not all(slower):
            outcome, why = "unclear", "slower at some code positions only"
        elif missing:
            outcome, why = "unclear", "no window fingerprint"
        elif not threads:
            outcome, why = (
                "unclear",
                "slower everywhere, but the loop code did not change",
            )
        else:
            outcome, why = "regression", ""
        row.update(outcome=outcome, why=why, deltas=deltas, threads=threads)
        results.append(row)
    return results


def _row(r):
    positions = " / ".join(f"{d:+.1f}%" for d in r["deltas"]) or "—"
    return (
        f"| {r['module']} | {_params(r['config']) or '—'} | {r['run_type']} | {positions} | "
        f"{', '.join(r['threads']) or '—'} |"
    )


def render(results, arch):
    regressions = [r for r in results if r["outcome"] == "regression"]
    unclear = [r for r in results if r["outcome"] == "unclear"]
    name = f"{arch.capitalize()} stage 2" if arch else "Stage 2"
    if regressions:
        verdict = f"❌ {len(regressions)} confirmed regression(s)"
    elif unclear:
        verdict = "⚠️ no confirmed regression, some moves are unclear"
    else:
        verdict = "✅ no regressions"
    head = "| test | configuration | run type | change at positions +0 / +4 / +20 / +36 / +52 B | loop code changed in |"
    rule = "|---|---|---|---|---|"
    lines = [
        f"## {name}: {verdict}",
        "",
        "Stage 2 re-measured the stage-1 regressions with the code at 5 positions. A regression is confirmed "
        "only when it is slower at every position and the code of the measured loop changed.",
        "",
        f"- {len(results)} stage-1 regression(s): **{len(regressions)} confirmed**, {len(unclear)} unclear",
    ]
    reasons = collections.Counter(r["why"] for r in unclear)
    lines += [f"- unclear, {why}: {n}" for why, n in reasons.most_common()]
    for title, rows, folded in (
        ("Confirmed regressions", regressions, False),
        ("Unclear", unclear, True),
    ):
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: -min(r["deltas"] or [r["delta_pct"]]))
        caption = f"{title}: " + (
            f"top {_TOP_N} of {len(rows)}" if len(rows) > _TOP_N else f"all {len(rows)}"
        )
        table = [head, rule] + [_row(r) for r in rows[:_TOP_N]]
        if folded:
            lines += ["", "<details>", f"<summary>{caption}</summary>", ""] + table
            lines += ["", "</details>"]
        else:
            lines += ["", f"### {caption}", ""] + table
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nodes")
    n.add_argument("--candidates", required=True)
    n.add_argument("--map", required=True)
    n.add_argument("--cap", type=int, default=DEFAULT_CAP)
    n.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--candidates", required=True)
    d.add_argument("--baseline", required=True)
    d.add_argument("--current", required=True)
    d.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    d.add_argument("--min-cycles", type=float, default=DEFAULT_MIN_CYCLES)
    d.add_argument("--module-threshold", default="")
    d.add_argument("--arch", default="")
    d.add_argument("--report", required=True)
    a = ap.parse_args(argv)
    candidates = read_candidates(a.candidates)
    if a.cmd == "nodes":
        selected = nodes(
            candidates, read_window_map(glob.glob(a.map, recursive=True)), a.cap
        )
        Path(a.out).write_text("\n".join(selected) + ("\n" if selected else ""))
        print(
            f"{len(selected)} test(s) for stage 2 from {len(candidates)} candidate point(s)"
        )
        return 0
    results = decide(
        candidates,
        a.baseline,
        a.current,
        a.threshold,
        a.min_cycles,
        _module_thresholds(a.module_threshold),
    )
    Path(a.report).write_text(render(results, a.arch) + "\n")
    Path(a.report).with_suffix(".json").write_text(json.dumps(results, default=str))
    confirmed = sum(r["outcome"] == "regression" for r in results)
    print(f"stage2 confirmed={confirmed} unclear={len(results) - confirmed}")
    return 1 if confirmed else 0


if __name__ == "__main__":
    raise SystemExit(main())
