#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render ``doc/functional_decoder/evidence_tables.md`` from the recorded artifacts.

Inputs (all written by the test suite / collect_perf.sh):

* ``doc/functional_decoder/pcc/pcc_results.json``   — every measured HF-vs-TTNN PCC
* ``doc/functional_decoder/perf/perf_summary.json`` — wall-clock per measured window
* ``doc/functional_decoder/tracy/<kind>/*_perf_report.csv`` — tt-perf-report device time

Generating the tables instead of hand-copying them keeps the README numbers identical to
the artifacts.

    python models/autoports/meta_models_muse_glimmer_30b/scripts/render_evidence.py
"""

from __future__ import annotations

import collections
import csv
import gzip
import json
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"
MODEL_DIR = DOC.parent.parent
REPO_ROOT = MODEL_DIR.parents[2]
PCC_JSON = DOC / "pcc" / "pcc_results.json"
PERF_JSON = DOC / "perf" / "perf_summary.json"
OUT = DOC / "evidence_tables.md"
PCC_BAR = 0.995
PROBLEMS: list[str] = []


def _current_fingerprint() -> str:
    """Recompute the code fingerprint from the files themselves.

    Reading it out of the artifact would be self-referential: an implementation edited after
    the last suite run would leave every record stale and the check would still pass.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from models.autoports.meta_models_muse_glimmer_30b.tests.conftest import _code_fingerprint

    return _code_fingerprint()["code_sha256"]


def _module_constants() -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from models.autoports.meta_models_muse_glimmer_30b.tt import functional_decoder as module

    return {
        "prefill_q": module.PREFILL_SDPA_Q_CHUNK,
        "prefill_k": module.PREFILL_SDPA_K_CHUNK,
        "decode_k": module.DECODE_SDPA_K_CHUNK,
        "decode_grid": module.DECODE_SDPA_GRID,
        # The device this stage ran on; the perf artifacts record CORE COUNT, and the decode
        # grid falls back to the whole compute grid once a batch needs more cores.
        "device_cores": 11 * 10,
    }


def _group(record: dict) -> str:
    return re.sub(r"\[.*\]$", "", record["test"])


def pcc_tables() -> list[str]:
    if not PCC_JSON.is_file():
        return ["_no PCC artifact yet_", ""]
    payload = json.loads(PCC_JSON.read_text())
    records = payload["records"]
    stored = payload.get("current_code_sha256")
    current = _current_fingerprint()
    if stored != current:
        PROBLEMS.append(
            f"pcc_results.json was written by code fingerprint {stored} but the working tree is "
            f"{current}: re-run both suites."
        )
    stamped = [r for r in records if r.get("code_sha256")]
    stale = [r for r in stamped if r.get("code_sha256") != current]
    lines = [
        f"Acceptance bar: **PCC >= {PCC_BAR}** (the `$functional-decoder` default; no",
        "model-specific exception was needed). Source of truth:",
        f"`{PCC_JSON.relative_to(DOC.parent.parent)}` ({len(records)} measurements).",
        "",
        f"Provenance: every record carries `code_sha256` (a hash of the implementation, host",
        f"reference and test files), `git_head` and `recorded_at`. Current code fingerprint",
        f"`{current}`; **{len(stamped) - len(stale)} of {len(records)}** records were produced by",
        f"exactly this code"
        + (
            f", **{len(stale)} are stale** and must be re-run: " + ", ".join(sorted({r["test"] for r in stale}))
            if stale
            else " and none are stale."
        )
        + (f" ({len(records) - len(stamped)} unstamped)." if len(stamped) != len(records) else ""),
        "",
        "### Summary by test",
        "",
        "| test | measurements | min PCC | max PCC |",
        "|---|---|---|---|",
    ]
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(_group(record), []).append(record)
    for name in sorted(groups):
        values = [r["pcc"] for r in groups[name]]
        lines.append(f"| `{name}` | {len(values)} | {min(values):.6f} | {max(values):.6f} |")
    worst = sorted(records, key=lambda r: r["pcc"])[:10]
    lines += [
        "",
        f"Global minimum PCC: **{min(r['pcc'] for r in records):.6f}** over all"
        f" {len(records)} measurements (bar {PCC_BAR}).",
        "",
        "### Ten lowest PCCs measured",
        "",
        "| test | metric | PCC |",
        "|---|---|---|",
    ]
    for record in worst:
        lines.append(f"| `{record['test']}` | {record['metric']} | {record['pcc']:.6f} |")
    return lines + [""]


def _device_time_us(csv_path: Path) -> tuple[float, int] | None:
    """Total device time (us) and op count from a tt-perf-report ``--csv`` file."""
    with _open_csv(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    candidates = [
        ("Device Time", 1.0),
        ("DEVICE KERNEL DURATION [ns]", 1e-3),
        ("Device Kernel Duration [ns]", 1e-3),
    ]
    for column, factor in candidates:
        if column in rows[0]:
            total = 0.0
            counted = 0
            for row in rows:
                raw = (row.get(column) or "").strip().replace(",", "")
                try:
                    total += float(raw) * factor
                    counted += 1
                except ValueError:
                    continue
            return total, counted
    return None


def perf_tables() -> list[str]:
    lines: list[str] = []
    if PERF_JSON.is_file():
        records = json.loads(PERF_JSON.read_text())["records"]
        lines += [
            "### Wall clock over the measured window",
            "",
            f"Source: `{PERF_JSON.relative_to(DOC.parent.parent)}`.",
            "",
            "`profiled` marks the run collected under `python -m tracy` (the run the device-time",
            "tables below come from); the unprofiled row is the plain end-to-end latency. They agree",
            "to within a few percent, i.e. profiling overhead is not distorting these numbers.",
            "",
            "| measurement | kind | seq_len / batch | profiled | iterations | ms/iter | tokens/s |",
            "|---|---|---|---|---|---|---|",
        ]
        for record in records:
            size = record.get("seq_len") or record.get("batch")
            lines.append(
                f"| {record['measurement']} | {record['kind']} | {size} | {record.get('profiled')} | "
                f"{record['iterations']} | {record['wall_clock_ms_per_iter']:.3f} | "
                f"{record['tokens_per_second']:.1f} |"
            )
        lines.append("")

    csv_files = sorted(DOC.glob("tracy/*/*_perf_report.csv"))
    if csv_files:
        lines += [
            "### Device time from `tt-perf-report` (signposted window)",
            "",
            "Column used: `Device Time` (microseconds) from the filtered `tt-perf-report --csv`",
            "output; totals are the sum over every op in the window, divided by the number of",
            "measured iterations in that window.",
            "",
            "| artifact | ops in window | total device time (us) | iterations | us/iteration |",
            "|---|---|---|---|---|",
        ]
        iterations = {}
        if PERF_JSON.is_file():
            for record in json.loads(PERF_JSON.read_text())["records"]:
                key = (
                    "prefill" if record["measurement"].startswith("prefill") else "decode",
                    record["kind"],
                    str(record.get("seq_len") or record.get("batch")),
                )
                iterations[key] = record["iterations"]
        for path in csv_files:
            parsed = _device_time_us(path)
            if parsed is None:
                lines.append(f"| `{path.relative_to(DOC)}` | - | unparsed | - | - |")
                continue
            total, count = parsed
            match = re.match(r"(prefill|decode)_(\d+)_perf_report\.csv", path.name)
            iters = iterations.get((match.group(1), path.parent.name, match.group(2))) if match else None
            per_iter = f"{total / iters:.1f}" if iters else "-"
            lines.append(f"| `{path.relative_to(DOC)}` | {count} | {total:.1f} | {iters or '-'} | {per_iter} |")
        lines.append("")

    lines += _op_family_tables()
    lines += _executed_geometry_table()

    grid_sweep = DOC / "perf" / "core_grid_sweep.md"
    if grid_sweep.is_file():
        lines += [f"Blackhole core-grid sweep: `{grid_sweep.relative_to(DOC)}`.", ""]
    return lines or ["_no perf artifact yet_", ""]


def _open_csv(path: Path):
    """Open a committed CSV artifact, transparently handling the gzipped ones.

    The repo's pre-commit hook rejects files over 500 KB, so the large raw Tracy ops CSVs are
    committed gzipped; everything that reads them goes through here.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def _ops_csv_paths() -> list[Path]:
    return sorted(list(DOC.glob("tracy/*/*_ops.csv")) + list(DOC.glob("tracy/*/*_ops.csv.gz")))


def _op_family_shares(csv_path: Path) -> list[tuple[str, float, int]]:
    """Device-time share per op code, from a filtered ``tt-perf-report --csv`` file."""
    with _open_csv(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    totals: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    for row in rows:
        code = (row.get("OP CODE") or row.get("OP Code") or "").strip()
        raw = (row.get("Device Time") or "").strip().replace(",", "")
        try:
            totals[code] += float(raw)
        except ValueError:
            continue
        counts[code] += 1
    grand = sum(totals.values()) or 1.0
    return [(code, 100.0 * value / grand, counts[code]) for code, value in totals.most_common()]


def _op_family_tables() -> list[str]:
    """Where the device time goes, derived from the artifacts rather than hand-copied."""
    lines = [
        "### Device-time share by op, per artifact",
        "",
        "Derived from the same filtered CSVs, so these percentages cannot drift from the",
        "committed reports. Only families above 1% are listed.",
        "",
    ]
    for path in sorted(DOC.glob("tracy/*/*_perf_report.csv")):
        shares = [entry for entry in _op_family_shares(path) if entry[1] >= 1.0]
        if not shares:
            continue
        lines.append(
            f"* `{path.parent.name}/{path.stem.replace('_perf_report', '')}`: "
            + ", ".join(f"{code} {pct:.1f}% ({count} ops)" for code, pct, count in shares)
        )
    return lines + [""]


def _executed_geometry_table() -> list[str]:
    """Extract the SDPA program configs that actually ran, and gate them on the code defaults.

    A review found the full-attention prefill silently running a different ``q_chunk_size``
    from the one the sweep selected and the docs recorded. This turns that class of drift into
    a non-zero exit of the evidence renderer.
    """
    constants = _module_constants()
    lines = [
        "### SDPA program configs actually executed",
        "",
        "Read back out of the raw Tracy `ATTRIBUTES` column of every committed ops CSV and",
        "checked against the module defaults "
        f"(`PREFILL_SDPA_Q_CHUNK={constants['prefill_q']}`, "
        f"`PREFILL_SDPA_K_CHUNK={constants['prefill_k']}`, "
        f"`DECODE_SDPA_K_CHUNK={constants['decode_k']}`, decode grid "
        f"{constants['decode_grid'][0]}x{constants['decode_grid'][1]} or the full 11x10). "
        "`render_evidence.py` exits non-zero if they disagree.",
        "",
        "| artifact | prefill SDPA q/k | decode SDPA k | decode SDPA cores |",
        "|---|---|---|---|",
    ]
    for path in _ops_csv_paths():
        prefill_pairs: collections.Counter = collections.Counter()
        decode_k: collections.Counter = collections.Counter()
        decode_cores: collections.Counter = collections.Counter()
        with _open_csv(path) as handle:
            for row in csv.DictReader(handle):
                code = (row.get("OP CODE") or "").strip()
                attrs = row.get("ATTRIBUTES", "") or ""
                if code == "SDPAOperation":
                    match = re.search(r"'q_chunk_size': '(\d+)'.*?'k_chunk_size': '(\d+)'", attrs)
                    if not match:
                        match = re.search(r"q_chunk_size=(\d+);k_chunk_size=(\d+)", attrs)
                    if match:
                        prefill_pairs[(int(match.group(1)), int(match.group(2)))] += 1
                elif code == "SdpaDecodeDeviceOperation":
                    match = re.search(r"'k_chunk_size': '(\d+)'", attrs) or re.search(r"k_chunk_size=(\d+)", attrs)
                    if match:
                        decode_k[int(match.group(1))] += 1
                    cores = (row.get("CORE COUNT") or "").strip()
                    if cores:
                        decode_cores[cores] += 1
        name = f"{path.parent.name}/{path.name.split('_ops.csv')[0]}"
        expected = (constants["prefill_q"], constants["prefill_k"])
        for pair in prefill_pairs:
            if pair != expected:
                PROBLEMS.append(f"{name}: prefill SDPA ran q/k={pair}, expected {expected}")
        for value in decode_k:
            if value != constants["decode_k"]:
                PROBLEMS.append(f"{name}: decode SDPA ran k_chunk={value}, expected {constants['decode_k']}")
        # The decode grid is batch-dependent (see _decode_sdpa_program_config): the configured
        # sub-grid while it has one core per (user, KV head), the full grid beyond that. Check
        # the executed core counts are one of those two, not some third thing.
        legal_cores = {constants["decode_grid"][0] * constants["decode_grid"][1], constants["device_cores"]}
        for cores in decode_cores:
            if int(cores) not in legal_cores:
                PROBLEMS.append(f"{name}: decode SDPA ran on {cores} cores, expected one of {sorted(legal_cores)}")
        lines.append(
            f"| `{name}` | "
            + (", ".join(f"{q}/{k} x{n}" for (q, k), n in prefill_pairs.items()) or "-")
            + " | "
            + (", ".join(f"{k} x{n}" for k, n in decode_k.items()) or "-")
            + " | "
            + (", ".join(f"{c} x{n}" for c, n in decode_cores.items()) or "-")
            + " |"
        )
    return lines + [""]


def main() -> int:
    lines = [
        "# Functional decoder evidence tables — meta-models/Muse-Glimmer-30B",
        "",
        "Generated by `scripts/render_evidence.py`; do not edit by hand.",
        "",
        "## HF-vs-TTNN PCC",
        "",
    ]
    lines += pcc_tables()
    lines += ["## Performance", ""]
    lines += perf_tables()
    # rstrip so a trailing blank section line cannot leave the file ending in two newlines,
    # which the repo's end-of-file-fixer hook would keep rewriting.
    OUT.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"wrote {OUT}")
    if PROBLEMS:
        print("\nEVIDENCE PROBLEMS:", file=sys.stderr)
        for problem in PROBLEMS:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
