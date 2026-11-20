# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 of the LLK perf gate: re-measure stage-1 regressions at other code positions.

Each stage-1 regression, on any marker, gets one outcome; none is dropped from the report:

- regression: slower at every code position.
- review: a TILE_LOOP regression whose loop code did not change (a change outside the loop, e.g. in init).
- placement: slower at some code positions only, so the move follows where the code sits (a hardware effect).
- partial: a placement-like TILE_LOOP row whose loop code changed.
- not-reproduced: slower at no code position when measured again.
- unmeasured: over the test limit, or not measured at every position on both sides.

The positions decide; the loop fingerprint only names which thread's loop code changed. A blocking gate fails on
regression, review and unmeasured.

    placement_check.py nodes  --candidates R.regressions.csv --map 'current/**/*.windowhash' --out nodes.txt
    placement_check.py decide --candidates R.regressions.csv --baseline DIR --current DIR --report stage2.md
    placement_check.py render --results 'legs/**/stage2.json' --report stage2.md   (exit 1: it blocks)
    placement_check.py gate   --results 'legs/**/stage2.json' --legs 4 --stage1 R.regressions.csv --bases 'legs/**/stage2.base' --report stage2.md
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
    """The stage-1 regressions, every marker: stage 2 re-runs the whole test, so it measures them all."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        config = [
            (c, _value(v))
            for c, v in row.items()
            if c not in _STATS and c != MODULE_COL and v != ""
        ]
        out.append(
            {
                "module": row[MODULE_COL],
                "run_type": row["run_type"],
                "marker": row["marker"],
                "config": config,
                "delta_pct": float(row["delta_pct"]),
            }
        )
    return out


def nodes(candidates, window_map, cap):
    """(test node ids to re-measure, indices of the candidates left out by the cap).

    The cap is shared evenly across the test modules, largest slowdown first within each, so every module
    with a candidate is re-measured.
    """
    queues = collections.defaultdict(list)
    for i, c in sorted(enumerate(candidates), key=lambda ic: -ic[1]["delta_pct"]):
        entry = window_map.get(_config_key(c["module"], c["run_type"], c["config"]))
        if entry:
            queues[c["module"]].append((i, entry["nodeid"]))
    selected, chosen = [], {}
    while len(selected) < cap and any(queues.values()):
        for module in list(queues):
            if not queues[module] or len(selected) >= cap:
                continue
            i, nodeid = queues[module].pop(0)
            if nodeid not in chosen:
                chosen[nodeid] = True
                selected.append(nodeid)
    over = sorted(i for q in queues.values() for i, nodeid in q if nodeid not in chosen)
    return selected, over


def _side(root):
    values, windows = [], {}
    for pos in POSITIONS:
        # A position's reports, measured in one part (<pos>/) or in several (<pos>-<part>/).
        dirs = [f"{root}/{pos}"] + glob.glob(f"{root}/{pos}-*")
        files = [
            p
            for d in dirs
            for p in glob.glob(f"{d}/**/*.csv", recursive=True)
            if not p.endswith(".post.csv")
        ]
        medians = _medians(_read_frames(files)) if files else {}
        values.append(
            {(k[0][0], k[0][1], _run_type_of(k[1])): v for k, v in medians.items()}
        )
        if pos == "p0":
            windows = read_window_map(
                [
                    w
                    for d in dirs
                    for w in glob.glob(f"{d}/**/*.windowhash", recursive=True)
                ]
            )
    return values, windows


def decide(
    candidates,
    baseline_root,
    current_root,
    threshold,
    min_cycles,
    module_thresholds,
    over_cap=(),
):
    over_cap = set(over_cap)
    base, base_windows = _side(baseline_root)
    cur, cur_windows = _side(current_root)
    results = []
    index = {}
    for key in set(base[0]) | set(cur[0]):
        marker, cfg, run_type = key
        config = dict(cfg)
        module = config.pop(MODULE_COL, None)
        index[
            (
                marker,
                module,
                run_type,
                frozenset((k, _value(v)) for k, v in config.items()),
            )
        ] = key
    for i, c in enumerate(candidates):
        limit = module_thresholds.get(c["module"], threshold)
        lookup = index.get(
            (
                c["marker"],
                c["module"],
                c["run_type"],
                frozenset((k, _value(v)) for k, v in c["config"]),
            )
        )
        row = dict(c, deltas=[], threads=[], slower=0)
        if i in over_cap:
            row.update(outcome="unmeasured", why="over the test limit")
            results.append(row)
            continue
        on_base = lookup is not None and all(lookup in side for side in base)
        on_head = lookup is not None and all(lookup in side for side in cur)
        if not (on_base and on_head):
            why = "not measured at every position on both sides"
            if on_head:
                why = "not measured on the base (renamed or new test?)"
            row.update(outcome="unmeasured", why=why)
            results.append(row)
            continue
        deltas, slower = [], []
        for b, h in zip(base, cur):
            bv, hv = b[lookup], h[lookup]
            deltas.append(round((hv - bv) / bv * 100, 2))
            slower.append((hv - bv) / bv > limit and hv - bv > min_cycles)
        wkey = _config_key(c["module"], c["run_type"], c["config"])
        bw, cw = base_windows.get(wkey), cur_windows.get(wkey)
        # The loop fingerprint only labels TILE_LOOP rows; the positions decide. Other markers, and TILE_LOOP rows
        # with no fingerprint, get the outcomes that say nothing about the loop code: regression or placement.
        labelled = (
            c["marker"] == "TILE_LOOP"
            and bw
            and cw
            and not any(
                str(v).startswith("no-") for v in list(bw.values()) + list(cw.values())
            )
        )
        threads = []
        if labelled:
            threads = sorted(t for t in bw if t != "nodeid" and bw.get(t) != cw.get(t))
        why = ""
        if not any(slower):
            outcome = "not-reproduced"
        elif all(slower):
            outcome = "review" if labelled and not threads else "regression"
        else:
            outcome = "partial" if threads else "placement"
        row["slower"] = sum(slower)
        row.update(outcome=outcome, why=why, deltas=deltas, threads=threads)
        results.append(row)
    return results


def _fixed(results):
    """{module: parameters with one value in all of the module's results}, left out of the tables."""
    values = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in results:
        for k, v in r["config"]:
            values[r["module"]][k].add(v)
    return {
        m: {k for k, vs in params.items() if len(vs) == 1}
        for m, params in values.items()
    }


def _row(r, hidden=()):
    positions = " / ".join(f"{d:+.1f}%" for d in r["deltas"]) or "—"
    return (
        f"| {r['module']} | {_params(r['config'], hidden) or '—'} | {r['run_type']} | {positions} | "
        f"{', '.join(r['threads']) or '—'} |"
    )


_SECTIONS = (
    (
        "regression",
        "Confirmed regressions",
        "Slower at every code position. For a TILE_LOOP row, the last column names the threads whose loop code "
        "changed; INIT, UNINIT and KERNEL rows have no loop fingerprint, so it is empty for them.",
        False,
    ),
    (
        "review",
        "Slower at every position, loop code unchanged: needs a look",
        "Not caused by the loop's own instructions. Either a change outside the loop (for example a config "
        "value written in init) or a placement effect that every position shares.",
        False,
    ),
    (
        "partial",
        "Loop code changed, slower at some positions only: needs a look",
        "Probably real, but how much slower depends on where the code sits, and at least one position stays "
        "under the threshold. Not confirmed.",
        False,
    ),
    (
        "placement",
        "Code-placement effects (hardware)",
        "Slower at some code positions only: the move follows where the code sits, not what it does. For a "
        "TILE_LOOP row, the loop code also did not change. This is real hardware behaviour (branch predictor, "
        "instruction fetch or similar), reported so it is not hidden.",
        False,
    ),
    (
        "not-reproduced",
        "Not reproduced",
        "Slower at no code position when measured again.",
        True,
    ),
    ("unmeasured", "Not re-measured", "", True),
)


# A blocking gate fails on these: slower at every position, or not re-measured at all.
BLOCKING = ("regression", "review", "unmeasured")


def blocks(results):
    return any(r["outcome"] in BLOCKING for r in results)


def render(results, arch, limit_note=""):
    by = collections.defaultdict(list)
    for r in results:
        by[r["outcome"]].append(r)
    fixed = _fixed(results)
    name = f"{arch.capitalize()} stage 2" if arch else "Stage 2"
    if by["regression"] or by["review"]:
        verdict = f"❌ {len(by['regression']) + len(by['review'])} slower at every code position"
    elif by["unmeasured"]:
        verdict = f"❌ {len(by['unmeasured'])} not re-measured"
    elif by["partial"]:
        verdict = f"⚠️ no regression at every position, {len(by['partial'])} to review"
    elif by["placement"]:
        verdict = "ℹ️ no confirmed regression, code-placement effects only"
    else:
        verdict = "✅ no confirmed regression"
    head = "| test | configuration | run type | change at positions +0 / +4 / +20 / +36 / +52 B | loop code changed in |"
    rule = "|---|---|---|---|---|"
    lines = [
        f"## {name}: {verdict}",
        "",
        "Stage 2 measured every stage-1 regression again with the TRISC code at 5 positions, on the base "
        "and on the head.",
        "",
        f"- {len(results)} stage-1 regression(s)",
    ]
    note = "The configuration column leaves out the parameters that are the same in all of a test's rows; `stage2.json` has every parameter."
    lines += [f"- {title}: {len(by[key])}" for key, title, _, _ in _SECTIONS if by[key]]
    unmeasured = collections.Counter(r["why"] for r in by["unmeasured"])
    lines += [f"  - {why}: {n}" for why, n in unmeasured.most_common()]
    over = sum(r["why"] == "over the test limit" for r in by["unmeasured"])
    if over:
        lines += [
            "",
            f"> **The confirmed count is a lower bound.** {over} stage-1 regression(s) were over the test limit "
            f"and were not re-measured. {limit_note}".rstrip(),
        ]
    if results:
        lines += ["", note]
    for key, title, text, folded in _SECTIONS:
        rows = by[key]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: -min(r["deltas"] or [r["delta_pct"]]))
        caption = f"{title}: " + (
            f"top {_TOP_N} of {len(rows)}" if len(rows) > _TOP_N else f"all {len(rows)}"
        )
        table = (
            ([text, ""] if text else [])
            + [head, rule]
            + [_row(r, fixed.get(r["module"], ())) for r in rows[:_TOP_N]]
        )
        if folded:
            lines += ["", "<details>", f"<summary>{caption}</summary>", ""] + table
            lines += ["", "</details>"]
        else:
            lines += ["", f"### {caption}", ""] + table
    return "\n".join(lines)


def _row_key(row):
    return (
        row["marker"],
        row["module"],
        row["run_type"],
        frozenset((c, _value(v)) for c, v in row["config"]),
    )


def gate(results_by_leg, legs, stage1, gate_base, leg_bases):
    """(results, notes): the stage-2 rows of every leg, plus a not-re-measured row for each stage-1 regression
    no leg reported, and the notes that explain why stage 2 cannot clear the gate.

    The merge gate blocks when notes is not empty or blocks(results). It needs a result from every leg, every leg
    compared against the gate's own baseline, and every stage-1 regression judged by some leg.
    """
    results = [row for rows in results_by_leg for row in rows]
    notes = []
    if not legs or len(results_by_leg) < legs:
        notes.append(f"{len(results_by_leg)} of {legs} leg(s) left a stage-2 result.")
    if len(leg_bases) < len(results_by_leg):
        notes.append(
            f"{len(results_by_leg) - len(leg_bases)} leg(s) left no record of their baseline."
        )
    other = sorted({b for b in leg_bases if b != gate_base})
    if other:
        notes.append(
            f"A leg compared against {', '.join(other)}, not the gate's baseline {gate_base or '(none)'}."
        )
    judged = {_row_key(r) for r in results}
    missing = [c for c in stage1 if _row_key(c) not in judged]
    results += [
        dict(
            c,
            deltas=[],
            threads=[],
            slower=0,
            outcome="unmeasured",
            why="no stage-2 result",
        )
        for c in missing
    ]
    return results, notes


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
    d.add_argument(
        "--over-cap", default="", help="file of candidate indices left out by the cap"
    )
    d.add_argument(
        "--limit-note", default="", help="shown when candidates were over the cap"
    )
    d.add_argument("--report", required=True)
    r = sub.add_parser("render")
    r.add_argument("--results", required=True, help="glob of decide's .json files")
    r.add_argument("--arch", default="")
    r.add_argument("--report", required=True)
    g = sub.add_parser(
        "gate", help="the merge gate's stage-2 verdict (exit 1: it blocks)"
    )
    g.add_argument(
        "--results", required=True, help="glob of the legs' stage2.json files"
    )
    g.add_argument(
        "--legs", type=int, required=True, help="measuring legs of this arch"
    )
    g.add_argument("--stage1", required=True, help="the gate's stage-1 regressions CSV")
    g.add_argument("--gate-base", default="", help="commit of the gate's baseline")
    g.add_argument("--bases", required=True, help="glob of the legs' stage2.base files")
    g.add_argument("--arch", default="")
    g.add_argument("--report", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        paths = sorted(glob.glob(a.results, recursive=True))
        results, notes = gate(
            [json.loads(Path(path).read_text()) for path in paths],
            a.legs,
            read_candidates(a.stage1) if Path(a.stage1).exists() else [],
            a.gate_base,
            [
                Path(b).read_text().strip()
                for b in sorted(glob.glob(a.bases, recursive=True))
            ],
        )
        report = render(results, a.arch)
        if notes:
            report += "\n\n" + "\n".join(f"> ⚠️ {n}" for n in notes)
        Path(a.report).write_text(report + "\n")
        return 1 if notes or blocks(results) else 0
    if a.cmd == "render":
        results = [
            row
            for path in sorted(glob.glob(a.results, recursive=True))
            for row in json.loads(Path(path).read_text())
        ]
        Path(a.report).write_text(render(results, a.arch) + "\n")
        return 1 if blocks(results) else 0
    candidates = read_candidates(a.candidates)
    if a.cmd == "nodes":
        selected, over = nodes(
            candidates, read_window_map(glob.glob(a.map, recursive=True)), a.cap
        )
        Path(a.out).write_text("".join(f"{n}\n" for n in selected))
        Path(a.out).with_suffix(".over").write_text("".join(f"{i}\n" for i in over))
        print(
            f"{len(selected)} test(s) for stage 2 from {len(candidates)} candidate point(s), "
            f"{len(over)} over the cap of {a.cap}"
        )
        return 0
    over_cap = []
    if a.over_cap and Path(a.over_cap).exists():
        over_cap = [int(x) for x in Path(a.over_cap).read_text().split()]
    results = decide(
        candidates,
        a.baseline,
        a.current,
        a.threshold,
        a.min_cycles,
        _module_thresholds(a.module_threshold),
        over_cap,
    )
    Path(a.report).write_text(render(results, a.arch, a.limit_note) + "\n")
    Path(a.report).with_suffix(".json").write_text(json.dumps(results, default=str))
    counts = collections.Counter(r["outcome"] for r in results)
    print("stage2", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if blocks(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
