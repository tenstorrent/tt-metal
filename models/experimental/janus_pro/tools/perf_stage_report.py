# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Condense one tracy run of the vision tower into a committable per-stage report.

The raw ops_perf_results CSV is ~12 MB and cannot be checked in. This keeps what the
PERF.md change log is matched against: the per-op-code and per-matmul-shape breakdown of
a single trace replay, plus the kernel total.

Usage:
    python -m models.experimental.janus_pro.tools.perf_stage_report \
        <ops_perf_results.csv> --stage <slug> --sha <commit> [--note "..."]

Writes models/experimental/janus_pro/perf_reports/<slug>.md.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "perf_reports"
# The tower's first op. Replays are sliced here, and the first slice is dropped: its
# leading gap is the inter-replay trace turnaround, not tower work.
FIRST_OP_MARKER = "768 x 1024"


def _us(value):
    """tt-perf-report emits '12.3 us' / '1.2 ms' / '900 ns'; return microseconds."""
    match = re.match(r"([\d.]+)\s*(ns|μs|us|ms)?", str(value).strip())
    if not match:
        return 0.0
    amount, unit = float(match.group(1)), match.group(2)
    if unit == "ns":
        return amount / 1000
    if unit == "ms":
        return amount * 1000
    return amount


PROSE_HEADING = "## What this change was"


def _delta_against_previous(stage, kernel_ms):
    """Kernel-time change this stage caused, read from the report one number below it."""
    number = int(stage[:2])
    if number == 0:
        return None
    prev = next(REPORTS_DIR.glob(f"{number - 1:02d}-*.md"), None)
    if prev is None:
        return None
    match = re.search(r"kernel time \([^)]*\): \*\*([\d.]+) ms", prev.read_text())
    return kernel_ms - float(match.group(1)) if match else None


def previous_report(stage):
    """The report one stage below this one, or None for the baseline."""
    number = int(stage[:2])
    if number == 0:
        return None
    return next(REPORTS_DIR.glob(f"{number - 1:02d}-*.md"), None)


def op_code_table(text):
    """`{op: (instances, us_each)}` parsed out of a report's op-code table.

    Read from the rendered markdown rather than the source CSV, so a stage can be compared with its
    predecessor without both runs' 12 MB profiles being on hand.
    """
    section = re.search(r"## Kernel time by op code[^\n]*\n\n(\|[^\n]*\n)\|[^\n]*\n((?:\|[^\n]*\n)+)", text)
    if not section:
        return {}
    # Column positions come from the header: a report may predate the delta columns or carry them.
    header = [c.strip() for c in section.group(1).strip().strip("|").split("|")]
    try:
        i_n, i_us = header.index("inst"), header.index("us each")
    except ValueError:
        return {}
    rows = {}
    for line in section.group(2).strip().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) <= max(i_n, i_us) or cells[0].startswith("**"):
            continue
        try:
            rows[cells[0]] = (int(cells[i_n]), float(cells[i_us]))
        except ValueError:
            continue
    return rows


def _delta_cells(op, instances, us_each, before):
    """`(Δ inst, Δ us each)` against the previous stage, as display strings."""
    if before is None:
        return "", ""
    if op not in before:
        return "new", "new"
    prev_n, prev_us = before[op]
    return f"{instances - prev_n:+d}", f"{us_each - prev_us:+.2f}"


def matmul_table(text):
    """`{shape: [(fidelity, instances, us_each), ...]}` from a report's matmul table."""
    section = re.search(r"## Matmul instances by shape\n\n(\|[^\n]*\n)\|[^\n]*\n((?:\|[^\n]*\n)+)", text)
    if not section:
        return {}
    header = [c.strip() for c in section.group(1).strip().strip("|").split("|")]
    try:
        i_n, i_us, i_fid = header.index("inst"), header.index("us each"), header.index("fidelity")
    except ValueError:
        return {}
    rows = {}
    for line in section.group(2).strip().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) <= max(i_n, i_us, i_fid):
            continue
        try:
            rows.setdefault(cells[0], []).append((cells[i_fid], int(cells[i_n]), float(cells[i_us])))
        except ValueError:
            continue
    return rows


def matmul_deltas(shape, fidelity, instances, us_each, before):
    """`(Δ inst, Δ us each)` for one matmul row against the previous stage.

    The instance count has to agree, so a stage that only changes fidelity still compares against
    the same op. Where it does not agree the two rows are not the same set of instances -- one side
    is grouping c_fc together with the aligner's fc1, which share the 576x1024x4096 shape -- and
    there is no honest number to print, so the cells stay empty.
    """
    if before is None:
        return "", ""
    candidates = [c for c in before.get(shape, []) if c[1] == instances]
    if not candidates:
        return ("new", "new") if shape not in before else ("—", "—")
    # Same count at more than one fidelity is possible in principle; prefer the matching fidelity.
    _, prev_n, prev_us = next((c for c in candidates if c[0] == fidelity), candidates[0])
    return f"{instances - prev_n:+d}", f"{us_each - prev_us:+.1f}"


def _existing_prose(path):
    """The hand-written explanation, if this report already has one."""
    if not path.exists():
        return None
    text = path.read_text()
    if PROSE_HEADING not in text:
        return None
    body = text[text.index(PROSE_HEADING) :]
    end = body.find("\n## ", 1)
    return body[:end] if end != -1 else body


def _render(csv_path, stage, sha, note, single_pass):
    rendered = REPORTS_DIR / f"{stage}.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # A traced run brackets its replays in signposts. A single eager forward pass, which is
    # how the unoptimized baseline had to be measured, emits none.
    signposts = [] if single_pass else ["--start-signpost", "start", "--end-signpost", "stop"]
    # Alongside the interpreter running this, not whatever is on PATH: the venv's bin is not
    # exported in every shell the profiler is driven from.
    report_bin = Path(sys.executable).parent / "tt-perf-report"
    subprocess.run(
        [str(report_bin), *signposts, "--no-advice", "--no-summary", "--csv", str(rendered), str(csv_path)],
        check=True,
        capture_output=True,
    )
    frame = pd.read_csv(rendered)
    rendered.unlink()

    frame["dt"] = frame["Device Time"].map(_us)
    codes = frame["OP Code"].astype(str)

    if single_pass:
        one = frame.assign(family=codes.str.replace(r" \d+ x.*", "", regex=True))
        kernel_ms = frame["dt"].sum() / 1000
        ops_per_replay = len(frame)
        replay_note = "one eager forward pass, not traced"
    else:
        starts = [i for i, code in enumerate(codes) if FIRST_OP_MARKER in code]
        if len(starts) < 3:
            sys.exit(f"expected several replays, found {len(starts)} occurrences of {FIRST_OP_MARKER!r}")
        replays = [
            frame.iloc[starts[i] : (starts[i + 1] if i + 1 < len(starts) else len(frame))] for i in range(len(starts))
        ]
        kernel_ms = sum(r["dt"].sum() for r in replays[1:]) / (len(replays) - 1) / 1000
        ops_per_replay = starts[1] - starts[0]
        replay_note = f"mean of replays 2-{len(replays)}"
        one = replays[1].assign(family=codes.iloc[starts[1] : starts[2]].str.replace(r" \d+ x.*", "", regex=True))
    by_family = one.groupby("family").agg(n=("dt", "size"), ms=("dt", "sum")).sort_values("ms", ascending=False)
    by_family["ms"] /= 1000
    by_family["us_each"] = by_family["ms"] * 1000 / by_family["n"]
    by_family["pct"] = by_family["ms"] / by_family["ms"].sum() * 100

    matmuls = one[one["OP Code"].astype(str).str.contains("Matmul")].copy()
    for col, name in (("FLOPs %", "flops_pct"), ("DRAM %", "dram_pct"), ("Cores", "cores")):
        matmuls[name] = pd.to_numeric(matmuls[col].astype(str).str.rstrip(" %"), errors="coerce")
    matmuls["fidelity"] = matmuls["Math Fidelity"].astype(str).str.split().str[0]
    matmuls["shape"] = matmuls["OP Code"].astype(str)
    # Grouped by shape AND fidelity: the tower runs 576x1024x4096 both as the MLP's c_fc at LoFi and
    # as the aligner's fc1 at HiFi2, and averaging the two hides both.
    by_shape = matmuls.groupby(["shape", "fidelity"], as_index=False).agg(
        n=("dt", "size"),
        ms=("dt", "sum"),
        cores=("cores", "max"),
        flops_pct=("flops_pct", "mean"),
        dram_pct=("dram_pct", "mean"),
    )
    by_shape["us_each"] = by_shape["ms"] / by_shape["n"]
    by_shape["ms"] /= 1000
    by_shape = by_shape.sort_values("ms", ascending=False)

    out = REPORTS_DIR / f"{stage}.md"
    lines = [
        f"# Stage: {stage}",
        "",
        f"- source commit: `{sha}`",
        f"- kernel time ({replay_note}): **{kernel_ms:.3f} ms**",
    ]
    delta = _delta_against_previous(stage, kernel_ms)
    if delta is not None:
        lines.append(f"- change from the previous stage: **{delta:+.3f} ms**")
    lines.append(f"- device ops: **{ops_per_replay}**")
    if note:
        lines.append(f"- note: {note}")
    # Written by hand, not derived from the CSV, so regenerating a report must not discard it.
    prose = _existing_prose(out)
    if prose:
        lines += ["", prose.rstrip()]
    scope = "the pass" if single_pass else "one replay"
    lines += [
        "",
        f"## Kernel time by op code, {scope}",
        "",
    ]
    prev = previous_report(stage)
    before = op_code_table(prev.read_text()) if prev else None
    if before:
        lines += ["| Op | inst | Δ inst | us each | Δ us each | ms | % |", "|---|---:|---:|---:|---:|---:|---:|"]
    else:
        lines += ["| Op | inst | us each | ms | % |", "|---|---:|---:|---:|---:|"]
    for family, row in by_family.iterrows():
        d_n, d_us = _delta_cells(family, int(row.n), row.us_each, before)
        cells = [family, f"{int(row.n)}"]
        if before:
            cells += [d_n]
        cells += [f"{row.us_each:.2f}"]
        if before:
            cells += [d_us]
        cells += [f"{row.ms:.3f}", f"{row.pct:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    if before:
        gone = [op for op in before if op not in by_family.index]
        for op in gone:
            n, us = before[op]
            lines.append(f"| {op} | 0 | {-n:+d} | — | gone | 0.000 | 0.0 |")
    lines += [
        "",
        "## Matmul instances by shape",
        "",
    ]
    mm_before = matmul_table(prev.read_text()) if prev else None
    if mm_before:
        lines += [
            "| shape | inst | Δ inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    else:
        lines += [
            "| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    for _, row in by_shape.iterrows():
        shape = row["shape"].replace("MatmulDeviceOperation ", "")
        cells = [shape, f"{int(row.n)}"]
        if mm_before:
            d_n, d_us = matmul_deltas(shape, row.fidelity, int(row.n), row.us_each, mm_before)
            cells += [d_n, f"{row.us_each:.1f}", d_us]
        else:
            cells += [f"{row.us_each:.1f}"]
        cells += [
            f"{row.ms:.3f}",
            f"{int(row.cores)}",
            f"{row.flops_pct:.1f}",
            f"{row.dram_pct:.1f}",
            str(row.fidelity),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how",
        "well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why",
        "`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved",
        "14 points while the op's time did not change at all.",
    ]

    out.write_text("\n".join(lines) + "\n")
    print(f"{out}  kernel {kernel_ms:.3f} ms  ops {ops_per_replay}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="ops_perf_results_*.csv from a tracy run of the tower perf test")
    parser.add_argument("--stage", required=True, help="slug matching the PERF.md change-log row")
    parser.add_argument("--sha", required=True, help="commit the run measured")
    parser.add_argument("--note", default="", help="one line of context")
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="the run is one eager forward pass with no signposts (how the baseline was measured)",
    )
    args = parser.parse_args()
    _render(args.csv, args.stage, args.sha, args.note, args.single_pass)


if __name__ == "__main__":
    main()
