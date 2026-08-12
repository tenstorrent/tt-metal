# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive every headline figure the reports quote, from the committed artifacts.

Four consecutive review rounds of this stage found the same defect class and nothing
else: a number in `README.md`, `work_log.md`, `context_contract.json` or a source
docstring that no longer matched the CSV, log or junit it was taken from. The
existing `--check` scripts could not catch any of them --
`refresh_context_contract.py` regenerates `tests.*` and the PCC blocks but not the
`performance` block or any prose, and `summarize_pcc.py` only covers PCC.

So this closes the class rather than the instance: it recomputes each quoted figure
from its source of truth and fails on drift. It is deliberately *narrow* -- only figures that have a single mechanical source --
because a check that needs updating whenever the prose is reworded would be
abandoned.  It covers `README.md` and `context_contract.json`.  It does **not**
parse `work_log.md`: that file is chronological and deliberately keeps superseded
per-round snapshots, which are labelled as superseded rather than checked.

    python .../bench/check_reported_figures.py          # report
    python .../bench/check_reported_figures.py --check  # exit 1 on any drift

No device required; it reads committed artifacts only.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/optimized_decoder/
CONTRACT = ROOT.parent / "context_contract.json"
README = ROOT / "README.md"

#: Decode windows are captured over this many trace replays.
DECODE_REPLAYS = 8


def perf_rows(kind: str, window: str) -> list[dict]:
    path = ROOT / f"tracy/{kind}/{window}_perf_report.csv"
    with path.open() as handle:
        return list(csv.DictReader(handle))


def column(rows: list[dict], needle: str) -> str:
    return next(k for k in rows[0] if needle in k)


def device_us(kind: str, window: str, replays: int = 1) -> float:
    rows = perf_rows(kind, window)
    return sum(float(r[column(rows, "Device Time")]) for r in rows) / replays


def op_count(kind: str, window: str, replays: int = 1) -> int:
    return len(perf_rows(kind, window)) // replays


def max_gap_us(kind: str, window: str) -> float:
    rows = perf_rows(kind, window)
    gap = column(rows, "Gap")
    return max(float(r[gap] or 0) for r in rows)


def op_group_us(kind: str, window: str, needle: str, replays: int) -> float:
    rows = perf_rows(kind, window)
    code, time = column(rows, "OP Code"), column(rows, "Device Time")
    return sum(float(r[time]) for r in rows if needle in r[code]) / replays


def approx(actual: float, quoted: float, tol: float) -> bool:
    return abs(actual - quoted) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if any quoted figure has drifted")
    args = ap.parse_args()

    readme = README.read_text()
    contract = json.loads(CONTRACT.read_text())
    impl = (ROOT.parents[1] / "tt/optimized_decoder.py").read_text()
    failures: list[str] = []

    def expect(label: str, ok: bool, detail: str) -> None:
        print(f"{'ok  ' if ok else 'DRIFT'} {label}: {detail}")
        if not ok:
            failures.append(label)

    # ---- device times, against the contract's performance block -------------
    perf = contract["performance"]
    for kind in ("sliding", "full"):
        for window, key, replays in (
            ("prefill_128", f"128_{kind}", 1),
            ("prefill", f"8192_{kind}", 1),
            ("prefill_16384", f"16384_{kind}", 1),
        ):
            actual = device_us(kind, window) / 1000
            quoted = perf["prefill_ms_device"][key]
            expect(f"prefill device {key}", approx(actual, quoted, 0.01), f"{actual:.3f} ms vs {quoted}")
        for window, key in (("decode", f"{kind}@2048"), ("decode_131071", f"{kind}@131071")):
            actual = device_us(kind, window, DECODE_REPLAYS) / 1000
            quoted = perf["traced_decode_ms_per_token_device"][key]
            expect(f"decode device {key}", approx(actual, quoted, 0.002), f"{actual:.4f} ms vs {quoted}")

    # ---- op counts ----------------------------------------------------------
    ops = perf["device_ops_per_iteration"]
    for kind in ("sliding", "full"):
        for window, key, replays in (
            ("prefill_128", f"prefill_128_{kind}", 1),
            ("prefill", f"prefill_{kind}", 1),
            ("prefill_16384", f"prefill_16384_{kind}", 1),
            ("decode", f"decode_{kind}", DECODE_REPLAYS),
        ):
            actual = op_count(kind, window, replays)
            expect(f"op count {key}", actual == ops[key], f"{actual} vs {ops[key]}")

    # ---- figures quoted in the README prose ---------------------------------
    gap = max_gap_us("sliding", "prefill_16384")
    expect(
        "README worst sliding prefill op-to-op gap",
        re.search(rf"worst (?:single )?gap {gap:.3f} μs|is \*\*{gap:.3f} μs\*\*", readme) is not None,
        f"{gap:.3f} us quoted in the sentence that claims it",
    )

    for needle, pattern, tol in (
        ("LayerNorm", r"six norms ([\d.]+) μs", 0.1),
        ("BinaryNg", r"`BinaryNg` ([\d.]+) μs", 0.1),
        ("SdpaDecode", r"`SdpaDecode` ([\d.]+) μs", 0.1),
        ("PagedUpdateCache", r"dispatches are ([\d.]+) μs of a 1072 μs step", 0.05),
    ):
        actual = op_group_us("sliding", "decode", needle, DECODE_REPLAYS)
        match = re.search(pattern, readme)
        if match is None:
            expect(f"README {needle} figure", False, "pattern not found in README")
        else:
            quoted = float(match.group(1))
            expect(f"README {needle} μs", approx(actual, quoted, tol), f"{actual:.2f} vs {quoted}")

    # ---- test counts and PCC populations ------------------------------------
    suite = ET.parse(ROOT / "test_results.xml").getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total = int(suite.get("tests"))
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    expect("junit clean", failed == 0, f"{failed} failures/errors")
    expect("contract test total", contract["tests"]["total"] == total, f"{contract['tests']['total']} vs {total}")
    expect(f"README quotes {total} tests", f"{total} tests, {total} passed" in readme, "present")
    expect(
        f"README checklist quotes {total}/{total}",
        f"{total}/{total}, worst real-weight" in readme,
        "present",
    )

    real = contract["tests"]["real_weight_checks"]
    expect(f"README quotes {real} real-weight checks", f"all {real}\nreal-weight checks" in readme, "present")
    expect(
        "contract real_weights note carries no hard-coded count",
        not re.search(r"All \d+ real-weight checks", contract["tested"]["real_weights"]["note"]),
        "no literal count",
    )

    # ---- the precision frontier, against its single committed run -----------
    frontier = (ROOT / "logs/layer_ab_real_final.log").read_text()
    rows_real = re.findall(
        r"AB\[real\]\s+(\S+)\s+kind=(\S+)\s+prefill\d+=\s*([\d.]+) ms\s+traced_decode@\d+=\s*([\d.]+) "
        r"ms/token\s+prefill_pcc=([\d.]+) decode_pcc=([\d.]+)",
        frontier,
    )
    expect("precision frontier run has all 5 candidates x 2 kinds", len(rows_real) == 10, f"{len(rows_real)} rows")
    for cand, kind, prefill, decode, ppcc, dpcc in rows_real:
        for value, what in (
            (prefill, "prefill ms"),
            (decode, "decode ms"),
            (ppcc, "prefill pcc"),
            (dpcc, "decode pcc"),
        ):
            expect(
                f"frontier {cand}/{kind} {what}",
                value in readme,
                f"{value} present in README",
            )

    # ---- per-role DRAM bandwidth / utilisation / FLOPs, decode --------------
    drows = perf_rows("sliding", "decode")
    code = column(drows, "OP Code")
    bwk = next(k for k in drows[0] if k.strip() == "DRAM")
    pctk = column(drows, "DRAM %")
    flopk = column(drows, "FLOPs %")
    for shape in ("32 x 6656 x 4608", "32 x 6656 x 4096", "32 x 4096 x 6656"):
        sel = [r for r in drows if shape in r[code]]
        bw = sum(float(r[bwk].replace("GB/s", "")) for r in sel) / len(sel)
        pct = sum(float(r[pctk]) for r in sel) / len(sel)
        flops = sum(float(r[flopk]) for r in sel) / len(sel)
        expect(f"decode {shape} GB/s", f"{bw:.1f} GB/s" in readme, f"{bw:.1f} GB/s")
        expect(f"decode {shape} DRAM %", f"{pct:.1f} %" in readme, f"{pct:.1f} %")
        expect(f"decode {shape} FLOPs %", f"{flops:.1f} %" in readme, f"{flops:.1f} %")

    # ---- the fused_activation A/B, quoted in code and README ----------------
    ab = (ROOT / "logs/layer_ab_fused_activation.log").read_text()
    measured = {
        (m.group(1), m.group(2)): float(m.group(3))
        for m in re.finditer(r"AB\s+(\S+)\s+kind=(\S+)\s+.*?traced_decode@\d+=\s*([\d.]+)", ab)
    }
    expect("fused_activation A/B log parsed", len(measured) == 4, f"{len(measured)} rows (expected 4)")
    for (cand, kind), value in measured.items():
        where = "implementation docstring" if cand == "fused_act" else "README"
        expect(
            f"fused_activation A/B {cand}/{kind}",
            f"{value:.4f}" in impl and f"{value:.4f}" in readme,
            f"{value:.4f} present in both README and {where}",
        )

    # ---- watcher ------------------------------------------------------------
    watcher = contract["tested"]["watcher"]
    expect(
        "README quotes the watcher node-id count",
        f"over {watcher['tests']} node ids" in readme,
        f"{watcher['tests']} node ids",
    )

    print()
    if failures:
        print(f"{len(failures)} figure(s) drifted from the committed artifacts:", file=sys.stderr)
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
        return 1 if args.check else 0
    print(
        "all checked figures match the committed artifacts "
        "(README.md and context_contract.json; work_log.md is chronological and not parsed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
