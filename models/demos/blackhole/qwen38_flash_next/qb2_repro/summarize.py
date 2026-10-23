#!/usr/bin/env python3
"""Summarize a run_repro.sh results dir: one markdown table per step, measured vs expected (medians).

  summarize.py RESULTS_DIR [--print-expected]

Reads RESULTS_DIR/<step>/<suite>/summary.json (bench_native.py output) and RESULTS_DIR/acceptance/acceptance.json.
Expected values (EXPECTED.md) are medians from the reference QB2 run (pr/58690 head 0fb7f9d95b3).
--print-expected prints the reference tables only (this is how EXPECTED.md was generated).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STEPS = ["P1_plain", "P2_slab2048", "P3_mtp4_slab2048", "P4a_lanes4_mtp4", "P4b_lanes8_mtp3"]
SUITES = ["decode", "sampled", "concurrency", "prefill"]
# keys: ttft = server TTFT s, dec = decode ms/token, tps = server tok/s, pre = prefill ms/token, agg = aggregate tok/s
EXPECTED = {
    "P1_plain": {
        "concurrency": {
            "c1": {"ttft": 0.3641, "dec": 25.901, "tps": 38.609, "pre": 2.644, "agg": 36.714},
            "c2": {"ttft": 0.3681, "dec": 25.909, "tps": 38.597, "pre": 2.675, "agg": 36.692},
            "c4": {"ttft": 0.3679, "dec": 25.907, "tps": 38.599, "pre": 2.671, "agg": 36.695},
        },
        "decode": {
            "decode": {"ttft": 0.3646, "dec": 25.914, "tps": 38.589, "pre": 2.648},
        },
        "prefill": {
            "isl1024": {"ttft": 1.464, "dec": 26.993, "tps": 37.046, "pre": 1.403},
            "isl4096": {"ttft": 5.293, "dec": 28.47, "tps": 35.127, "pre": 1.285},
            "isl8192": {"ttft": 10.463, "dec": 28.467, "tps": 35.128, "pre": 1.274},
            "isl16384": {"ttft": 20.952, "dec": 28.407, "tps": 35.206, "pre": 1.277},
            "isl31000": {"ttft": 39.764, "dec": 28.533, "tps": 35.047, "pre": 1.282},
        },
        "sampled": {
            "sampled": {"ttft": 0.3691, "dec": 25.998, "tps": 38.465, "pre": 2.644},
        },
    },
    "P2_slab2048": {
        "decode": {
            "decode": {"ttft": 0.3641, "dec": 25.904, "tps": 38.605, "pre": 2.643},
        },
        "prefill": {
            "isl1024": {"ttft": 1.459, "dec": 26.963, "tps": 37.087, "pre": 1.398},
            "isl4096": {"ttft": 3.492, "dec": 28.427, "tps": 35.175, "pre": 0.846},
            "isl8192": {"ttft": 4.81, "dec": 28.45, "tps": 35.147, "pre": 0.5835},
            "isl16384": {"ttft": 7.434, "dec": 28.437, "tps": 35.165, "pre": 0.452},
            "isl31000": {"ttft": 10.342, "dec": 28.413, "tps": 35.197, "pre": 0.3325},
        },
    },
    "P3_mtp4_slab2048": {
        "decode": {
            "decode": {"ttft": 0.3729, "dec": 20.4, "tps": 49.02, "pre": 2.706},
        },
        "prefill": {
            "isl4096": {"ttft": 3.684, "dec": 27.673, "tps": 36.623, "pre": 0.892},
            "isl16384": {"ttft": 8.204, "dec": 21.207, "tps": 47.16, "pre": 0.499},
            "isl31000": {"ttft": 11.796, "dec": 24.36, "tps": 44.182, "pre": 0.38},
        },
        "sampled": {
            "sampled": {"ttft": 0.3737, "dec": 21.78, "tps": 45.914, "pre": 2.708},
        },
    },
    "P4a_lanes4_mtp4": {
        "concurrency": {
            "c1": {"ttft": 0.6748, "dec": 26.416, "tps": 37.871, "pre": None, "agg": 34.536},
            "c2": {"ttft": 1.069, "dec": 28.746, "tps": 34.913, "pre": None, "agg": 59.524},
            "c4": {"ttft": 1.815, "dec": 32.506, "tps": 30.765, "pre": None, "agg": 96.989},
        },
    },
    "P4b_lanes8_mtp3": {
        "concurrency": {
            "c1": {"ttft": 0.6766, "dec": 26.089, "tps": 38.338, "pre": None, "agg": 34.914},
            "c2": {"ttft": 1.049, "dec": 28.861, "tps": 34.654, "pre": None, "agg": 60.89},
            "c4": {"ttft": 1.799, "dec": 33.154, "tps": 30.175, "pre": None, "agg": 94.246},
            "c8": {"ttft": 3.359, "dec": 41.592, "tps": 24.068, "pre": None, "agg": 138.732},
        },
    },
}
ACCEPT_EXPECTED = {
    "json": None,
    "chat": 39,
    "code": 32,
    "fact": 15,
    "list": 46,
    "math": 56,
    "multilingual": 9,
    "prose": 13,
    "refactor": 23,
    "sky": 19,
    "story": 6,
    "summary": 75,
}


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def pct(m, e):
    return "-" if m is None or not e else f"{(m / e - 1) * 100:+.1f}%"


def med(d, key):
    x = d.get(key) if isinstance(d, dict) else None
    return x.get("median") if isinstance(x, dict) else None


def natural(pt):
    digits = "".join(c for c in pt if c.isdigit())
    return (int(digits) if digits else 0, pt)


def table(step, suite, pts, with_expected):
    """pts: {point: {ok, n, ttft, dec, tps, pre, agg}} measured (or expected when with_expected is False)."""
    exp = EXPECTED.get(step, {}).get(suite, {})
    conc = suite == "concurrency"
    if with_expected:
        head = ["point", "ok", "TTFT s", "exp", "decode ms/tok", "exp", "tok/s", "exp", "d tok/s"]
        if suite == "prefill":
            head += ["prefill ms/tok", "exp"]
        if conc:
            head += ["agg tok/s", "exp", "d agg"]
    else:
        head = (
            ["point", "TTFT s", "decode ms/tok", "tok/s"]
            + (["prefill ms/tok"] if suite == "prefill" else [])
            + (["agg tok/s"] if conc else [])
        )
    out = [f"| {' | '.join(head)} |", f"|{'---|' * len(head)}"]
    for pt in sorted(pts, key=natural):
        m, e = pts[pt], exp.get(pt, {})
        if with_expected:
            row = [
                pt,
                m.get("ok", "-"),
                fmt(m.get("ttft"), 3),
                fmt(e.get("ttft"), 3),
                fmt(m.get("dec")),
                fmt(e.get("dec")),
                fmt(m.get("tps")),
                fmt(e.get("tps")),
                pct(m.get("tps"), e.get("tps")),
            ]
            if suite == "prefill":
                row += [fmt(m.get("pre"), 3), fmt(e.get("pre"), 3)]
            if conc:
                row += [fmt(m.get("agg")), fmt(e.get("agg")), pct(m.get("agg"), e.get("agg"))]
        else:
            row = [pt, fmt(m.get("ttft"), 3), fmt(m.get("dec")), fmt(m.get("tps"))]
            if suite == "prefill":
                row += [fmt(m.get("pre"), 3)]
            if conc:
                row += [fmt(m.get("agg"))]
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def load_suite(path):
    s = json.loads(path.read_text())
    res = {}
    for suite, pts in s.get("suites", {}).items():
        agg = pts.get("_aggregate", {})
        res[suite] = {}
        for pt, m in pts.items():
            if pt.startswith("_"):
                continue
            res[suite][pt] = {
                "ok": f"{m.get('ok')}/{m.get('requests')}",
                "ttft": med(m, "ttft_s"),
                "dec": med(m, "decode_ms_per_token"),
                "tps": med(m, "server_tps"),
                "pre": med(m, "prefill_ms_per_token"),
                "agg": (agg.get(pt, {}).get("aggregate_tok_s") or {}).get("median"),
            }
    return res


def acceptance(path):
    a = json.loads(path.read_text())
    print("## acceptance\n")
    print(f"gate (json 96/96): {'PASS' if a.get('gate_pass') else 'FAIL'}; continuation {a.get('continuation')}\n")
    print("| record | matched | first divergence | expected | tok/s |\n|---|---|---|---|---|")
    for p in a.get("prompts", []):
        name = p.get("prompt")
        d, e = p.get("divergence_index"), ACCEPT_EXPECTED.get(name)
        flag = "" if d == e else (" (earlier)" if d is not None and (e is None or d < e) else " (later)")
        print(
            f"| {name} | {p.get('matched_tokens')}/{p.get('compared_tokens')} | {d if d is not None else 'none'}{flag} "
            f"| {e if e is not None else 'none'} | {fmt(p.get('tokens_per_second'))} |"
        )
    print()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--print-expected" in sys.argv:
        for step in STEPS:
            print(f"### {step}\n")
            for suite in SUITES:
                if suite in EXPECTED.get(step, {}):
                    print(f"{suite} (medians)\n")
                    print(table(step, suite, EXPECTED[step][suite], False))
                    print()
        return 0
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(args[0])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    print(f"# Qwen3.8-Flash-Next QB2 results: {root}\n")
    for acc in sorted(root.glob("acceptance/acceptance.json")):
        acceptance(acc)
    found = 0
    for step in sorted(
        {p.parent.parent.name for p in root.glob("*/*/summary.json")},
        key=lambda s: (STEPS.index(s) if s in STEPS else 99, s),
    ):
        print(f"## {step}\n")
        for suite in SUITES:
            f = root / step / suite / "summary.json"
            if not f.is_file():
                continue
            found += 1
            for name, pts in load_suite(f).items():
                print(f"{name} (medians; exp = reference run)\n")
                print(table(step, name, pts, True))
                print()
    if not found:
        print("no */*/summary.json found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
