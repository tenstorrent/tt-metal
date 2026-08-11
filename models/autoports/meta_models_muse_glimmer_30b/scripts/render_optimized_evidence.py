#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render ``doc/optimized_decoder/evidence_tables.md`` from the recorded artifacts.

Inputs (all written by the test suite / ``scripts/collect_optimized_perf.sh``):

* ``doc/optimized_decoder/pcc/pcc_results.json``            — every measured HF-vs-TTNN PCC
* ``doc/optimized_decoder/perf/perf_summary.json``          — wall clock per measured window
* ``doc/optimized_decoder/perf/candidates.json``            — the stage's candidate table
* ``doc/optimized_decoder/tracy/<kind>/*_perf_report.csv``  — tt-perf-report device time
* ``doc/functional_decoder/perf/perf_summary.json``         — the stage-01 baseline

Generating the tables instead of hand-copying them keeps the README numbers identical to
the artifacts, and lets the renderer *fail* when they drift: it exits non-zero if a PCC
record was produced by different code than the working tree, if a measured matmul row does
not carry the dtype/fidelity the selected precision policy claims (the `$optimize` skill's
OPT-013), or if an SDPA program config other than the configured one ran.

    python models/autoports/meta_models_muse_glimmer_30b/scripts/render_optimized_evidence.py
"""

from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc" / "optimized_decoder"
MODEL_DIR = DOC.parent.parent
REPO_ROOT = MODEL_DIR.parents[2]
PCC_JSON = DOC / "pcc" / "pcc_results.json"
PERF_JSON = DOC / "perf" / "perf_summary.json"
CANDIDATES_JSON = DOC / "perf" / "candidates.json"
ACCOUNTING_JSON = DOC / "perf" / "accounting.json"
BASELINE_JSON = DOC.parent / "functional_decoder" / "perf" / "perf_summary.json"
OUT = DOC / "evidence_tables.md"
PCC_BAR = 0.995
PROBLEMS: list[str] = []


def _current_fingerprint() -> str:
    """Recompute the optimized suite's code fingerprint from the files themselves."""
    sys.path.insert(0, str(REPO_ROOT))
    from models.autoports.meta_models_muse_glimmer_30b.tests.test_optimized_decoder import FINGERPRINTED_SOURCES

    digest = hashlib.sha256()
    for relative in FINGERPRINTED_SOURCES:
        digest.update((REPO_ROOT / relative).read_bytes())
    return digest.hexdigest()[:16]


def _module():
    sys.path.insert(0, str(REPO_ROOT))
    from models.autoports.meta_models_muse_glimmer_30b.tt import optimized_decoder as module

    return module


def _group(record: dict) -> str:
    return re.sub(r"\[.*\]$", "", record["test"])


def _open_csv(path: Path):
    """Open a committed CSV artifact, transparently handling the gzipped ones."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def _ops_csv_paths() -> list[Path]:
    return sorted(list(DOC.glob("tracy/*/*_ops.csv")) + list(DOC.glob("tracy/*/*_ops.csv.gz")))


# ------------------------------------------------------------------------------ PCC


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
            f"{current}: re-run the optimized suite."
        )
    stamped = [r for r in records if r.get("code_sha256")]
    stale = [r for r in stamped if r.get("code_sha256") != current]
    diagnostic = [r for r in records if r["threshold"] < PCC_BAR]
    gating = [r for r in records if r["threshold"] >= PCC_BAR]
    lines = [
        f"Acceptance bar: **PCC >= {PCC_BAR}**, unchanged from the functional stage. Source of",
        f"truth: `{PCC_JSON.relative_to(MODEL_DIR)}` ({len(records)} measurements,",
        f"{len(gating)} of them gating).",
        "",
        "Provenance: every record carries `code_sha256` (a hash of the optimized layer, the",
        "functional layer it inherits its correctness contracts from, the host reference and the",
        f"test files), `git_head` and `recorded_at`. Current fingerprint `{current}`;",
        f"**{len(stamped) - len(stale)} of {len(records)}** records were produced by exactly this"
        + (
            " code, **" + str(len(stale)) + " are stale**: " + ", ".join(sorted({r["test"] for r in stale}))
            if stale
            else " code and none are stale."
        ),
        "",
        f"{len(diagnostic)} records carry a *diagnostic* threshold below the bar. They all belong to",
        "`test_synthetic_weight_precision_discrepancy`, which measures the BFP4 policies on",
        "synthetic random weights on purpose; acceptance for the shipped policy comes from the",
        "real-weight tests. See the README section *Precision policy*.",
        "",
        "### Summary by test",
        "",
        "| test | measurements | min PCC | max PCC | bar |",
        "|---|---|---|---|---|",
    ]
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(_group(record), []).append(record)
    for name in sorted(groups):
        values = [r["pcc"] for r in groups[name]]
        bars = {r["threshold"] for r in groups[name]}
        lines.append(
            f"| `{name}` | {len(values)} | {min(values):.6f} | {max(values):.6f} | "
            f"{', '.join(f'{b:g}' for b in sorted(bars))} |"
        )
    worst_gating = sorted(gating, key=lambda r: r["pcc"])[:10]
    lines += [
        "",
        f"Global minimum over the {len(gating)} **gating** measurements: "
        f"**{min(r['pcc'] for r in gating):.6f}** (bar {PCC_BAR}).",
        "",
        "### Ten lowest gating PCCs measured",
        "",
        "| test | metric | PCC |",
        "|---|---|---|",
    ]
    for record in worst_gating:
        lines.append(f"| `{record['test']}` | {record['metric']} | {record['pcc']:.6f} |")

    real = [r for r in records if r.get("weights") == "real"]
    if real:
        lines += [
            "",
            "### Real-checkpoint-weight measurements (the acceptance evidence for BFP4)",
            "",
            "| test | metric | PCC |",
            "|---|---|---|",
        ]
        for record in sorted(real, key=lambda r: (r["test"], r["metric"])):
            lines.append(f"| `{record['test']}` | {record['metric']} | {record['pcc']:.6f} |")
    return lines + [""]


# ----------------------------------------------------------------------------- perf


def _device_time_us(csv_path: Path) -> tuple[float, int] | None:
    """Total device time (us) and op count from a tt-perf-report ``--csv`` file."""
    with _open_csv(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    for column, factor in (
        ("Device Time", 1.0),
        ("DEVICE KERNEL DURATION [ns]", 1e-3),
    ):
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


def _gap_us(csv_path: Path) -> float:
    with _open_csv(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    total = 0.0
    for row in rows:
        raw = (row.get("Op-to-Op Gap") or "").strip().replace(",", "")
        try:
            total += float(raw)
        except ValueError:
            continue
    return total


def _iteration_counts() -> dict:
    counts = {}
    if PERF_JSON.is_file():
        for record in json.loads(PERF_JSON.read_text())["records"]:
            key = (
                "prefill" if record["measurement"].startswith("prefill") else "decode",
                record["kind"],
                str(record.get("seq_len") or record.get("batch")),
                bool(record.get("profiled")),
            )
            counts[key] = record["iterations"]
    return counts


def before_after_table() -> list[str]:
    """Functional (stage 01) versus optimized (stage 02) warmed latency, same harness."""
    if not (PERF_JSON.is_file() and BASELINE_JSON.is_file()):
        return []

    def index(path: Path) -> dict:
        out = {}
        for record in json.loads(path.read_text())["records"]:
            if record.get("profiled"):
                continue
            key = (record["measurement"], record["kind"], record.get("seq_len"), record.get("batch"))
            out[key] = record
        return out

    before, after = index(BASELINE_JSON), index(PERF_JSON)
    lines = [
        "### Before / after: warmed prefill and traced warmed decode",
        "",
        "Both stages measured with the same harness shape (compile+warm, synchronize, signpost,",
        "measured window, synchronize, signpost) on one Blackhole chip, unprofiled runs only.",
        "Baseline: `doc/functional_decoder/perf/perf_summary.json`.",
        "",
        "| measurement | kind | seq_len / batch | functional ms | optimized ms | speedup |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(after, key=lambda k: (k[0], k[1], k[2] or 0, k[3] or 0)):
        if key not in before:
            continue
        size = key[2] or key[3]
        b_ms = before[key]["wall_clock_ms_per_iter"]
        a_ms = after[key]["wall_clock_ms_per_iter"]
        lines.append(f"| {key[0]} | {key[1]} | {size} | {b_ms:.3f} | {a_ms:.3f} | **{b_ms / a_ms:.2f}x** |")
    return lines + [""]


def perf_tables() -> list[str]:
    lines: list[str] = []
    lines += before_after_table()
    if PERF_JSON.is_file():
        records = json.loads(PERF_JSON.read_text())["records"]
        lines += [
            "### Wall clock over the measured window",
            "",
            f"Source: `{PERF_JSON.relative_to(MODEL_DIR)}`. `profiled` marks the run collected under",
            "`python -m tracy` (the run the device-time tables below come from); the unprofiled row",
            "is the plain end-to-end latency.",
            "",
            "| measurement | kind | policy | seq_len / batch | profiled | iters | ms/iter | tokens/s |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for record in records:
            size = record.get("seq_len") or record.get("batch")
            lines.append(
                f"| {record['measurement']} | {record['kind']} | {record.get('policy', '-')} | {size} | "
                f"{record.get('profiled')} | {record['iterations']} | "
                f"{record['wall_clock_ms_per_iter']:.3f} | {record['tokens_per_second']:.1f} |"
            )
        lines.append("")

    csv_files = sorted(DOC.glob("tracy/*/*_perf_report.csv"))
    if csv_files:
        iterations = _iteration_counts()
        lines += [
            "### Device time from `tt-perf-report` (signposted window)",
            "",
            "Column used: `Device Time` (microseconds) from the filtered `tt-perf-report --csv`",
            "output; totals are the sum over every op in the window, divided by the number of",
            "measured iterations. `op-to-op gap` is the summed dispatch gap over the same window -",
            "the host-side term that separates device time from end-to-end latency.",
            "",
            "| artifact | ops | device time (us) | iters | us/iter | gap us/iter |",
            "|---|---|---|---|---|---|",
        ]
        for path in csv_files:
            parsed = _device_time_us(path)
            if parsed is None:
                lines.append(f"| `{path.relative_to(DOC)}` | - | unparsed | - | - | - |")
                continue
            total, count = parsed
            match = re.match(r"(prefill|decode)_(\d+)_perf_report\.csv", path.name)
            iters = iterations.get((match.group(1), path.parent.name, match.group(2), True)) if match else None
            per_iter = f"{total / iters:.1f}" if iters else "-"
            gap = f"{_gap_us(path) / iters:.1f}" if iters else "-"
            lines.append(f"| `{path.relative_to(DOC)}` | {count} | {total:.1f} | {iters or '-'} | {per_iter} | {gap} |")
        lines.append("")

    lines += _op_family_tables()
    lines += _matmul_rows_table()
    lines += _executed_geometry_table()
    lines += _candidate_table()
    lines += _accounting_table()
    return lines or ["_no perf artifact yet_", ""]


def _op_family_shares(csv_path: Path) -> list[tuple[str, float, int]]:
    with _open_csv(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    totals: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    for row in rows:
        code = (row.get("OP CODE") or row.get("OP Code") or "").strip()
        # collapse the shape suffix so "MatmulDeviceOperation 32 x 6656 x 4608" groups
        code = code.split(" ")[0]
        raw = (row.get("Device Time") or "").strip().replace(",", "")
        try:
            totals[code] += float(raw)
        except ValueError:
            continue
        counts[code] += 1
    grand = sum(totals.values()) or 1.0
    return [(code, 100.0 * value / grand, counts[code]) for code, value in totals.most_common()]


def _op_family_tables() -> list[str]:
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


_EXPECTED_WEIGHT_DTYPE = {
    "6656 x 4608": "attn",  # QKV
    "6656 x 4096": "attn",  # attention gate
    "4096 x 6656": "attn",  # output projection
    "6656 x 19968": "mlp",  # SwiGLU gate / up
    "19968 x 6656": "mlp_down",  # SwiGLU down
}


def _matmul_rows_table() -> list[str]:
    """Prove the selected dtype/fidelity policy reached the measured matmuls (OPT-013).

    A policy object, a constructor default or a JSON summary is only intent; the check is
    the dtype and math fidelity the profiler recorded for each dominant row.
    """
    module = _module()
    policy = module.POLICIES[module.DEFAULT_POLICY]
    expected = {
        "attn": (str(policy.attn_weight_dtype).split(".")[-1], str(policy.attn_fidelity).split(".")[-1]),
        "mlp": (str(policy.mlp_weight_dtype).split(".")[-1], str(policy.mlp_fidelity).split(".")[-1]),
        "mlp_down": (
            str(policy.mlp_down_weight_dtype).split(".")[-1],
            str(policy.mlp_down_fidelity).split(".")[-1],
        ),
    }
    lines = [
        "### Dominant matmul rows: dtype, fidelity and geometry actually executed",
        "",
        f"Selected precision policy: **`{policy.name}`**. Every decode matmul row below is checked",
        "against it; `render_optimized_evidence.py` exits non-zero on a mismatch, so a stale weight",
        "cache or a helper default silently reverting BFP4 to BF16 cannot pass unnoticed.",
        "",
        "| artifact | matmul | cores | in1 dtype | fidelity | DRAM-sharded | in0 memory | in0_block_w | out subblock | device us | DRAM % | FLOPs % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    seen: set[tuple] = set()
    for path in sorted(DOC.glob("tracy/*/decode_*_perf_report.csv")):
        with _open_csv(path) as handle:
            rows = [
                r for r in csv.DictReader(handle) if (r.get("OP Code") or r.get("OP CODE") or "").startswith("Matmul")
            ]
        aggregated: dict[str, list[dict]] = {}
        for row in rows:
            aggregated.setdefault(row.get("OP Code") or row["OP CODE"], []).append(row)
        artifact = f"{path.parent.name}/{path.stem.replace('_perf_report', '')}"
        for code, group in aggregated.items():
            first = group[0]
            shape = code.replace("MatmulDeviceOperation ", "")
            key = shape.replace("32 x ", "").strip()
            times = [float(r["Device Time"]) for r in group]
            subblock = f"{first.get('Output Subblock H') or '-'}x{first.get('Output Subblock W') or '-'}"
            lines.append(
                f"| `{artifact}` | {shape} | {first['Cores']} | {first['Input 1 Datatype']} | "
                f"{first['Math Fidelity'].split()[0]} | {first['DRAM Sharded']} | "
                f"{first['Input 0 Memory'].replace('DEV_0_', '')} | {first['Inner Dim Block Size']} | "
                f"{subblock} | {sum(times) / len(times):.1f} | {float(first['DRAM %']):.1f} | "
                f"{float(first['FLOPs %']):.1f} |"
            )
            role = _EXPECTED_WEIGHT_DTYPE.get(key)
            if role and (artifact, code) not in seen:
                seen.add((artifact, code))
                want_dtype, want_fidelity = expected[role]
                got_dtype = first["Input 1 Datatype"].strip()
                got_fidelity = first["Math Fidelity"].split()[0].strip()
                if got_dtype != want_dtype:
                    PROBLEMS.append(
                        f"{artifact}: matmul {shape} ran with in1 dtype {got_dtype}, policy "
                        f"'{policy.name}' claims {want_dtype}"
                    )
                if got_fidelity != want_fidelity:
                    PROBLEMS.append(
                        f"{artifact}: matmul {shape} ran at {got_fidelity}, policy "
                        f"'{policy.name}' claims {want_fidelity}"
                    )
    return lines + [""]


def _executed_geometry_table() -> list[str]:
    """SDPA program configs that actually ran, gated on the module defaults."""
    module = _module()
    decode_grid = module.DECODE_SDPA_GRID
    lines = [
        "### SDPA program configs actually executed",
        "",
        "Read back out of the raw Tracy `ATTRIBUTES` column of every committed ops CSV and checked",
        f"against the module defaults (`PREFILL_SDPA_Q_CHUNK={module.PREFILL_SDPA_Q_CHUNK}`,",
        f"`PREFILL_SDPA_K_CHUNK={module.PREFILL_SDPA_K_CHUNK}`,",
        f"`DECODE_SDPA_K_CHUNK={module.DECODE_SDPA_K_CHUNK}`, decode grid "
        f"{decode_grid[0]}x{decode_grid[1]} or the full 11x10).",
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
                    if match:
                        prefill_pairs[(int(match.group(1)), int(match.group(2)))] += 1
                elif code == "SdpaDecodeDeviceOperation":
                    match = re.search(r"'k_chunk_size': '(\d+)'", attrs)
                    if match:
                        decode_k[int(match.group(1))] += 1
                    cores = (row.get("CORE COUNT") or "").strip()
                    if cores:
                        decode_cores[cores] += 1
        name = f"{path.parent.name}/{path.name.split('_ops.csv')[0]}"
        expected = (module.PREFILL_SDPA_Q_CHUNK, module.PREFILL_SDPA_K_CHUNK)
        for pair in prefill_pairs:
            if pair != expected:
                PROBLEMS.append(f"{name}: prefill SDPA ran q/k={pair}, expected {expected}")
        for value in decode_k:
            if value != module.DECODE_SDPA_K_CHUNK:
                PROBLEMS.append(f"{name}: decode SDPA ran k_chunk={value}, expected {module.DECODE_SDPA_K_CHUNK}")
        legal_cores = {decode_grid[0] * decode_grid[1], 110}
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


def _candidate_table() -> list[str]:
    if not CANDIDATES_JSON.is_file():
        return []
    payload = json.loads(CANDIDATES_JSON.read_text())
    lines = [
        "### Candidate table (traced warmed decode, batch 1, context 4096, sliding_rope)",
        "",
        f"Source: `{CANDIDATES_JSON.relative_to(MODEL_DIR)}`. Every candidate was measured in the",
        "same process on the same device with the same harness; the functional layer's 3.242 ms is",
        "the stage-01 baseline for the same workload.",
        "",
    ]
    for section in payload["sections"]:
        lines += [
            f"**{section['title']}**",
            "",
            "| candidate | decode ms | prefill 8192 ms | kept | note |",
            "|---|---|---|---|---|",
        ]
        for row in section["rows"]:
            decode = f"{row['decode_ms']:.4f}" if row.get("decode_ms") is not None else row.get("error", "-")
            prefill = f"{row['prefill_ms']:.2f}" if row.get("prefill_ms") is not None else "-"
            lines.append(
                f"| {row['label']} | {decode} | {prefill} | {'**yes**' if row.get('kept') else 'no'} | "
                f"{row.get('note', '')} |"
            )
        lines.append("")
    return lines


def _accounting_table() -> list[str]:
    if not ACCOUNTING_JSON.is_file():
        return []
    payload = json.loads(ACCOUNTING_JSON.read_text())
    lines = [
        "### Performance accounting (same run)",
        "",
        f"Source: `{ACCOUNTING_JSON.relative_to(MODEL_DIR)}`.",
        "",
        "| workload | roofline ms | device ms | end-to-end ms | device/roofline | e2e-device (host) ms |",
        "|---|---|---|---|---|---|",
    ]
    for entry in payload["workloads"]:
        lines.append(
            f"| {entry['name']} | {entry['roofline_ms']:.3f} | {entry['device_ms']:.3f} | "
            f"{entry['e2e_ms']:.3f} | {entry['device_ms'] / entry['roofline_ms']:.2f}x | "
            f"{entry['e2e_ms'] - entry['device_ms']:.3f} |"
        )
    lines += ["", "Named limitations:", ""]
    for item in payload.get("named_limitations", []):
        lines.append(f"* {item}")
    return lines + [""]


def main() -> int:
    lines = [
        "# Optimized decoder evidence tables — meta-models/Muse-Glimmer-30B",
        "",
        "Generated by `scripts/render_optimized_evidence.py`; do not edit by hand.",
        "",
        "## HF-vs-TTNN PCC",
        "",
    ]
    lines += pcc_tables()
    lines += ["## Performance", ""]
    lines += perf_tables()
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
