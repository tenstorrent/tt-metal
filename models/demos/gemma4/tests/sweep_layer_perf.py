# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep the per-chunk cost of a gemma4 global and sliding decoder layer, and file the reports.

What this answers: at chunk N of a 256k chunked prefill, what does one global decoder layer
cost and what does one sliding layer cost. The global layer attends the whole prefix, so its
ring gather grows with N; the sliding layer attends a 1024-token window, so it should stay
roughly flat past chunk 0. Those two curves are what size a 60-layer prefill, and neither is
visible in a whole-model timing.

It drives ``test_prefill_layer_perf_chunk_n`` and writes everything into one directory per
run.

**Two phases, because they want opposite things from the profiler.**

*timings* — one unprofiled run covering every chunk in order. Chunks 0..63 are replayed
contiguously, so each measured chunk attends over a prefix the run itself wrote: the cache
state is the real one. Fast (replays are single-layer, a few ms each) and it produces the
whole depth curve in one go. This is the phase to read the curve off.

*profile* — one profiled run per chunk, small enough that ``tt-perf-report`` gets a clean
per-op table. Batching many chunks into one process does not work: the device profiler
records every replay, and tt-metal's tracy wrapper allows a hard-coded 15s for the capture
tool to flush (``tools/tracy/__main__.py``), which a large run blows past and then reports
as the misleading "No profiling data could be captured".

These runs use ``GEMMA4_PERF_KV_FILL=random`` so cost stays flat in chunk index; replay fill
would make chunk 63 pay 126 extra profiled replays. That trade is checkable rather than
assumed — ``summary.csv`` puts the replay-filled timing beside the profiled one for the same
chunk, so every profiled cell is its own A/B.

Whatever the mode, the prefix has to be in the cache: measured at chunk 32, a zeroed cache
reads 17.95ms against 21.01ms with the prefix present, warm spreads non-overlapping. Ring
cost depends on the cache holding real data.

Layout::

    generated/gemma4_layer_perf/<run_id>/
      README.md                     what this is and how to re-slice any cell
      manifest.json                 every run: command, env, status, wall time
      timings.csv                   the depth curve — one row per (chunk, layer type)
      model_estimate.csv            10 x global + 50 x sliding, per chunk
      summary.csv                   timings joined with the profiled per-op totals
      chunk000/
        global.perf.txt             tt-perf-report table for this cell
        global.perf.csv             the same table as CSV
        global.json                 signposts, source CSV, measured ms
        sliding.perf.txt
        sliding.perf.csv
        sliding.json
      chunk001/
        ...
      raw/
        batch000_chunk000-015/
          ops_perf.csv[.gz]         the profiler CSV shared by this batch's cells
          pytest.log
          meta.json
      timings/
        pytest.log                  the unprofiled sweep's full log

Usage::

    source python_env/bin/activate
    export PYTHONPATH=$(pwd) TT_METAL_HOME=$(pwd) \\
           HF_HUB_OFFLINE=1 HF_HOME=/localdev/svuckovic/huggingface \\
           HF_MODEL=google/gemma-4-31B-it \\
           TT_CACHE_PATH=/localdev/svuckovic/huggingface/tt_cache/google--gemma-4-31B-it

    # everything: the depth curve, then profiled batches over all 64 chunks
    python -m models.demos.gemma4.tests.sweep_layer_perf

    # just the curve (minutes, no profiler)
    python -m models.demos.gemma4.tests.sweep_layer_perf --phase timings

    # per-op breakdowns for a contiguous span only
    python -m models.demos.gemma4.tests.sweep_layer_perf --phase profile --chunks 0:16

    # see what it would run
    python -m models.demos.gemma4.tests.sweep_layer_perf --dry-run
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TEST_FILE = "models/demos/gemma4/demo/text_demo_prefill.py"
TEST_NAME = "test_prefill_layer_perf_chunk_n"
PROFILER_REPORTS = Path("generated/profiler/reports")
DEFAULT_OUT_ROOT = Path("generated/gemma4_layer_perf")

# Must match PERF_CONTEXT_LEN / LONG_CONTEXT_CHUNK in the test, which are constants there.
# The chunk is tied to CP (window 1024 * CP 8) on the 8x4 mesh this sweep targets.
CHUNK = 8192
CONTEXT_LEN = 262144
N_CHUNKS = CONTEXT_LEN // CHUNK
# Match the test's warmup_iters / token_source params; both appear in the node id, so
# neither is optional.
WARMUP_ITERS = 5
TOKEN_SOURCE = "text"

# 31B's layer mix, for the whole-model extrapolation.
N_GLOBAL, N_SLIDING = 10, 50

TAGS = ("global", "sliding")

# The test has no timeout marker problem of its own (it carries @pytest.mark.timeout(7200)),
# but the profiled runs add tracy's post-processing on top, so leave the ceiling to the
# marker and don't second-guess it from here.
TRACY_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT": "20000",
}

# One machine-readable line per measured cell, emitted by the test:
#   [layer_perf_chunk] RESULT type=global chunk=7 ring_depth=7 kv_actual_global=28672
#   measured_ms=123.45 tok_s=33184 warm_best_ms=122.90 warm_worst_ms=125.10 noisy=0
#   signposts=gemma4-layer-global-chunk7-start,gemma4-layer-global-chunk7-stop
RESULT_RE = re.compile(r"\[layer_perf_chunk\] RESULT (?P<fields>.+)$")


def parse_results(log_text):
    """Extract the per-cell RESULT records the test logged, keyed by (tag, chunk index).

    Values are the raw key=value fields with numbers converted where they parse — the test
    owns the field list, so this stays generic rather than naming each one.
    """
    out = {}
    for line in log_text.splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        fields = {}
        for token in match.group("fields").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            for cast in (int, float):
                try:
                    fields[key] = cast(value)
                    break
                except ValueError:
                    continue
            else:
                fields[key] = value
        if "type" in fields and "chunk" in fields:
            out[(fields["type"], int(fields["chunk"]))] = fields
    return out


def signposts(tag, chunk_idx):
    """Must match ``_perf_signposts`` in the test module."""
    base = f"gemma4-layer-{tag}-chunk{chunk_idx}"
    return f"{base}-start", f"{base}-stop"


def node_id(chunk_param, type_param, mesh):
    """The test's pytest node id for one cell.

    pytest orders parametrize ids bottom-decorator-first, with the arch prepended by
    conftest. The test stacks chunk_idx (bottom), layer_type, chunk_size, context_len,
    warmup_iters, token_source, then mesh (top), so this suffix has to track that order --
    and the values have to match the test's params, which is what CHUNK / CONTEXT_LEN /
    WARMUP_ITERS / TOKEN_SOURCE above are for.
    """
    return (
        f"{TEST_FILE}::{TEST_NAME}[blackhole-{chunk_param}-{type_param}-sz{CHUNK}"
        f"-ctx_{CONTEXT_LEN // 1024}k-warm{WARMUP_ITERS}-{TOKEN_SOURCE}-{mesh}]"
    )


def new_report_dirs(before):
    """Report directories tracy created since ``before`` was snapshotted.

    Snapshot-and-diff rather than "the newest directory": another profiled run on the same
    box would make newest-wins attribute the wrong CSV to this chunk, and a silently
    mismatched CSV is worse than a missing one.
    """
    if not PROFILER_REPORTS.is_dir():
        return []
    return sorted(p for p in PROFILER_REPORTS.iterdir() if p.is_dir() and p not in before)


def find_ops_csv(report_dirs):
    for d in report_dirs:
        matches = sorted(d.glob("ops_perf_results_*.csv"))
        if matches:
            return matches[0]
    return None


def run_pytest(cmd, env, log_path, watch_reports):
    """Run a pytest/tracy command, save its log, and report any profiler dir it created."""
    before = set(PROFILER_REPORTS.iterdir()) if (watch_reports and PROFILER_REPORTS.is_dir()) else set()
    t0 = time.time()
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall_s = time.time() - t0
    log_text = completed.stdout + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text)
    created = new_report_dirs(before) if watch_reports else []
    return {
        "returncode": completed.returncode,
        "wall_s": round(wall_s, 1),
        "log": str(log_path),
        "log_text": log_text,
        "report_dirs": [str(p) for p in created],
        # str() only when a CSV was actually found: str(None) yields the string "None",
        # which slips past an `is None` check and then fails as a filename.
        "ops_csv_source": (lambda f: str(f) if f else None)(find_ops_csv(created)),
    }


# ── Phase 1: the depth curve, unprofiled ──────────────────────────────────────


def run_timings(chunk_idxs, args, out_root, env_base):
    """One unprofiled run over every requested chunk, in order.

    The whole curve goes through the ``chunkall`` param, which walks every depth inside a
    single model load and lets each measured chunk attend over a prefix that same run
    wrote. A subset has no such param, so it becomes one node id per chunk -- correct, but
    it pays a model load and a cache fill per chunk.
    """
    phase_dir = out_root / "timings"
    meta_path = phase_dir / "meta.json"
    log_path = phase_dir / "pytest.log"

    env = dict(env_base)
    # The unprofiled pass is the reference, so it pays for the exact prefix.
    env["GEMMA4_PERF_KV_FILL"] = "replay"
    # No profiler: TT_METAL_DEVICE_PROFILER is deliberately absent so replays stay cheap.
    env.pop("TT_METAL_DEVICE_PROFILER", None)

    whole_curve = chunk_idxs == list(range(N_CHUNKS))
    targets = (
        [node_id("chunkall", "both", args.mesh)]
        if whole_curve
        else [node_id(f"chunk{c}", "both", args.mesh) for c in chunk_idxs]
    )
    cmd = [sys.executable, "-m", "pytest", *targets, "-sv", "--timeout=0"]
    label = (
        f"timings: {len(chunk_idxs)} chunks ({chunk_idxs[0]}..{chunk_idxs[-1]}), both layer types, unprofiled"
        f"{'' if whole_curve else f' -- {len(targets)} separate model loads'}"
    )

    if meta_path.exists() and not args.force:
        existing = json.loads(meta_path.read_text())
        if existing.get("status") == "ok":
            print(f"  [skip] {label} — already complete (--force to redo)")
            return existing
    if args.dry_run:
        print(f"  [dry-run] {label}\n    {' '.join(cmd)}")
        return {"status": "dry-run", "cmd": cmd}

    print(f"  [run ] {label}")
    outcome = run_pytest(cmd, env, log_path, watch_reports=False)
    results = parse_results(outcome.pop("log_text"))
    record = {
        "phase": "timings",
        "cmd": cmd,
        "env_overrides": {"GEMMA4_PERF_KV_FILL": "replay"},
        "chunks": chunk_idxs,
        "results": {f"{tag}/chunk{idx}": fields for (tag, idx), fields in results.items()},
        **outcome,
    }
    record["status"] = "ok" if outcome["returncode"] == 0 and results else "failed"
    phase_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(record, indent=2))
    if record["status"] == "ok":
        print(f"  [ok  ] timings in {outcome['wall_s']:.0f}s — {len(results)} cells measured")
    else:
        print(f"  [FAIL] timings — rc={outcome['returncode']}, {len(results)} cells parsed " f"(see {log_path})")
    return record


# ── Phase 2: one profiled run per chunk ───────────────────────────────────────


def run_profile_chunk(chunk_idx, args, out_root, env_base):
    """Profile both layer types at one chunk depth, then slice each by signpost."""
    cell_dir = out_root / f"chunk{chunk_idx:03d}"
    meta_path = cell_dir / "meta.json"
    log_path = cell_dir / "pytest.log"
    label = f"profile chunk {chunk_idx} (global + sliding)"

    env = dict(env_base)
    env.update(TRACY_ENV)
    # Random fill keeps the cost flat in chunk index; replay fill would make chunk 63
    # cost 126 extra profiled replays. Cross-check a profiled cell against the
    # replay-filled timing for the same chunk in summary.csv.
    env["GEMMA4_PERF_KV_FILL"] = "random"

    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-p",
        "-r",
        "-v",
        "-m",
        "pytest",
        node_id(f"chunk{chunk_idx}", "both", args.mesh),
        "-sv",
    ]

    if meta_path.exists() and not args.force:
        existing = json.loads(meta_path.read_text())
        if existing.get("status") == "ok":
            print(f"  [skip] {label} -- already complete (--force to redo)")
            return existing
    if args.dry_run:
        print(f"  [dry-run] {label}\n    {' '.join(cmd)}")
        return {"status": "dry-run", "cmd": cmd, "chunk_idx": chunk_idx}

    print(f"  [run ] {label}")
    outcome = run_pytest(cmd, env, log_path, watch_reports=True)
    results = parse_results(outcome.pop("log_text"))
    record = {
        "phase": "profile",
        "chunk_idx": chunk_idx,
        "cmd": cmd,
        "env_overrides": {"GEMMA4_PERF_KV_FILL": "random", **TRACY_ENV},
        "results": {f"{tag}/chunk{idx}": fields for (tag, idx), fields in results.items()},
        **outcome,
    }

    source = outcome["ops_csv_source"]
    if source is None:
        record["status"] = "no-csv"
        cell_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(record, indent=2))
        print(f"  [FAIL] {label} -- rc={outcome['returncode']}, no ops_perf CSV (see {log_path})")
        return record

    cell_dir.mkdir(parents=True, exist_ok=True)
    dest = cell_dir / ("ops_perf.csv.gz" if args.gzip else "ops_perf.csv")
    if args.gzip:
        with open(source, "rb") as src, gzip.open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(source, dest)
    record["ops_csv"] = str(dest)

    cells = []
    for tag in TAGS:
        cells.append(slice_cell(dest, tag, chunk_idx, cell_dir, results.get((tag, chunk_idx))))
    record["cells"] = cells
    record["status"] = "ok" if outcome["returncode"] == 0 else "test-failed"
    meta_path.write_text(json.dumps(record, indent=2))
    if outcome["returncode"] != 0:
        print(f"  [warn] {label} -- pytest rc={outcome['returncode']} but a CSV exists; sliced anyway")
    else:
        print(f"  [ok  ] {label} in {outcome['wall_s']:.0f}s")
    return record


def slice_cell(ops_csv, tag, chunk_idx, cell_dir, result_fields):
    """Run tt-perf-report over one signposted region and file it next to the CSV."""
    start, stop = signposts(tag, chunk_idx)
    txt_path = cell_dir / f"{tag}.perf.txt"
    csv_path = cell_dir / f"{tag}.perf.csv"
    json_path = cell_dir / f"{tag}.json"

    # Two invocations on purpose: passing --csv sends the table to the file INSTEAD of
    # printing it, so a single run with --csv leaves a 9-line text report with the table
    # missing. The text run is the one a human reads (it keeps the op summary and the
    # roofline advice); the csv run is the one pandas reads.
    base = ["tt-perf-report", "--start-signpost", start, "--end-signpost", stop, "--no-color"]
    text_cmd = base + [str(ops_csv)]
    csv_cmd = base + ["--csv", str(csv_path), str(ops_csv)]
    text_run = subprocess.run(text_cmd, capture_output=True, text=True)
    txt_path.write_text(text_run.stdout + text_run.stderr)
    csv_run = subprocess.run(csv_cmd, capture_output=True, text=True)
    completed = text_run if text_run.returncode != 0 else csv_run

    cell = {
        "chunk_idx": chunk_idx,
        "layer_type": tag,
        "ring_depth": chunk_idx,
        "kv_actual_global": chunk_idx * CHUNK,
        "start_signpost": start,
        "end_signpost": stop,
        "ops_csv": str(ops_csv),
        "perf_report_txt": str(txt_path),
        "perf_report_csv": str(csv_path) if csv_path.exists() else None,
        "tt_perf_report_cmd": text_cmd,
        "tt_perf_report_csv_cmd": csv_cmd,
        "tt_perf_report_rc": completed.returncode,
        "measured": result_fields or None,
    }
    cell["device_time_us"] = device_total_us(cell["perf_report_csv"])
    json_path.write_text(json.dumps(cell, indent=2))
    if completed.returncode != 0:
        print(f"    [warn] tt-perf-report rc={completed.returncode} for {tag} chunk {chunk_idx} — see {txt_path.name}")
    return cell


def device_total_us(perf_csv):
    """Summed device time over the ops in the sliced region, in microseconds.

    tt-perf-report merges the 32 devices by default, so its table is already per-device
    rather than a mesh-wide sum — which is the number that means latency. Its ``Device Time``
    column is in microseconds; the raw profiler column (nanoseconds) is accepted as a
    fallback in case a future format drops the friendly one.

    This should land just under the measured wall-clock, the gap being host-side dispatch
    and the synchronize. At chunk 0 it came to 3718us of device time against a 3.87ms
    replay. Returns None when no duration column is recognized — the text report is the
    source of truth and this only feeds the summary table.
    """
    if perf_csv is None or not Path(perf_csv).exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_csv(perf_csv, low_memory=False)
    except Exception:
        return None
    for col in df.columns:
        if col.strip().lower() == "device time":
            return round(float(pd.to_numeric(df[col], errors="coerce").sum()), 1)
    for col in df.columns:
        if col.strip().upper().startswith("DEVICE KERNEL DURATION"):
            return round(float(pd.to_numeric(df[col], errors="coerce").sum()) / 1000.0, 1)
    return None


# ── Output tables ─────────────────────────────────────────────────────────────


def write_table(path, rows):
    if not rows:
        return None
    with open(path, "w", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_tables(timings, profiles, out_root):
    """timings.csv (the curve), model_estimate.csv, and summary.csv (curve + per-op totals)."""
    curve = []
    for key, fields in sorted(
        (timings or {}).items(), key=lambda kv: (int(kv[0].split("chunk")[-1]), kv[0].split("/")[0])
    ):
        tag = key.split("/")[0]
        curve.append(
            {
                "chunk_idx": fields.get("chunk"),
                "layer_type": tag,
                "ring_depth": fields.get("ring_depth"),
                "kv_actual_global": fields.get("kv_actual_global"),
                "measured_ms": fields.get("measured_ms"),
                "warm_best_ms": fields.get("warm_best_ms"),
                "warm_worst_ms": fields.get("warm_worst_ms"),
                "tok_s": fields.get("tok_s"),
                "noisy": fields.get("noisy"),
            }
        )
    write_table(out_root / "timings.csv", curve)

    by_chunk = {}
    for row in curve:
        by_chunk.setdefault(row["chunk_idx"], {})[row["layer_type"]] = row["measured_ms"]
    estimate = []
    for chunk_idx in sorted(k for k in by_chunk if k is not None):
        g, s = by_chunk[chunk_idx].get("global"), by_chunk[chunk_idx].get("sliding")
        if g is None or s is None:
            continue
        estimate.append(
            {
                "chunk_idx": chunk_idx,
                "global_ms": g,
                "sliding_ms": s,
                "est_60_layer_ms": round(N_GLOBAL * g + N_SLIDING * s, 2),
            }
        )
    write_table(out_root / "model_estimate.csv", estimate)

    # summary.csv joins the two phases so a profiled cell can be sanity-checked against the
    # curve it should agree with.
    profiled = {}
    for record in profiles:
        for cell in record.get("cells", []) or []:
            profiled[(cell["layer_type"], cell["chunk_idx"])] = cell
    summary = []
    keys = sorted(
        set((r["layer_type"], r["chunk_idx"]) for r in curve if r["chunk_idx"] is not None) | set(profiled),
        key=lambda k: (k[1], k[0]),
    )
    curve_by_key = {(r["layer_type"], r["chunk_idx"]): r for r in curve}
    for key in keys:
        tag, chunk_idx = key
        c = curve_by_key.get(key, {})
        p = profiled.get(key, {})
        p_measured = (p.get("measured") or {}).get("measured_ms")
        summary.append(
            {
                "chunk_idx": chunk_idx,
                "layer_type": tag,
                "ring_depth": chunk_idx,
                "kv_actual_global": chunk_idx * CHUNK,
                "timings_measured_ms": c.get("measured_ms"),
                "timings_noisy": c.get("noisy"),
                "profiled_measured_ms": p_measured,
                "device_time_us": p.get("device_time_us"),
                "perf_report_txt": p.get("perf_report_txt"),
                "ops_csv": p.get("ops_csv"),
                "start_signpost": signposts(tag, chunk_idx)[0],
                "end_signpost": signposts(tag, chunk_idx)[1],
            }
        )
    write_table(out_root / "summary.csv", summary)
    return curve, estimate, summary


README = """# gemma4 per-chunk decoder layer perf — run `{run_id}`

Per-chunk device cost of ONE global and ONE sliding decoder layer during a
{context_len}-token chunked prefill ({n_chunks} chunks of {chunk} tokens), measured by
`{test_name}` and filed by `models/demos/gemma4/tests/sweep_layer_perf.py`.

## Start here

- **`timings.csv`** — the depth curve. One row per (chunk, layer type) with the measured
  replay in ms and the warm spread around it. Produced by a single unprofiled run that
  replayed chunks in order, so every measured chunk attended over a prefix that same run
  wrote.
- **`model_estimate.csv`** — `{n_global} x global + {n_sliding} x sliding` per chunk, the 31B layer
  mix. Excludes embedding, head, and the inter-layer CCL a single-layer graph cannot
  contain, so read it as a floor on the 60-layer body.
- **`summary.csv`** — the curve joined with the profiled runs' per-op totals, so a
  profiled cell can be checked against the timing it should agree with.

## Per-op breakdowns

`chunk<NNN>/{{global,sliding}}.perf.txt` is the `tt-perf-report` table for that cell, with
`.perf.csv` the same table for pandas and `.json` the signposts and source CSV. Each
`chunk<NNN>/ops_perf.csv` is the profiler output of that chunk's own run.

Re-slice a cell by hand:

```
tt-perf-report --start-signpost gemma4-layer-global-chunk7-start \\
               --end-signpost   gemma4-layer-global-chunk7-stop \\
               chunk007/ops_perf.csv
```

Re-run one cell — the chunk index is a pytest param, so it is addressable directly:

```
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
  python -m tracy -p -r -v -m pytest \\
  '{test_file}::{test_name}[blackhole-chunk7-global-8x4]' -sv
```

## Reading the numbers

The two layer types are expected to diverge with depth, and that divergence is the reason
to measure them apart:

- **global** attends the whole prefix, so its ring gather grows with the chunk index.
- **sliding** attends a 1024-token window, so it should stay roughly flat past chunk 0.

Two caveats worth keeping in mind:

- Every measured chunk here attended over a **real** prefix, in both phases. That matters:
  a zeroed prefix reads 14.6% low on the global layer at chunk 32 (8.2% on sliding), with
  non-overlapping warm spreads, so ring cost depends on cache contents and not only on the
  `kv_actual_global` scalar. See `validation/VALIDATION.md`.
- Output *values* are still not meaningful, and nothing asserts on them beyond finiteness:
  a layer's true input is several layers deep and cannot be reproduced standalone.
- `noisy=1` means the measured replay fell outside the spread of its warm replays — the
  machine was busy, and that cell should be re-run before it is trusted.
"""


def parse_chunks(spec, parser):
    if spec == "all":
        return list(range(N_CHUNKS))
    if ":" in spec:
        lo, hi = spec.split(":", 1)
        idxs = list(range(int(lo or 0), int(hi or N_CHUNKS)))
    else:
        idxs = [int(c) for c in spec.split(",") if c.strip() != ""]
    bad = [c for c in idxs if not 0 <= c < N_CHUNKS]
    if bad:
        parser.error(f"chunk indices out of range for a {CONTEXT_LEN}-token context ({N_CHUNKS} chunks): {bad}")
    if not idxs:
        parser.error("no chunks selected")
    return idxs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--chunks",
        default="all",
        help="chunk indices: 'all' (default), a comma list '0,1,2,4', or a range '0:16' (end-exclusive)",
    )
    parser.add_argument(
        "--phase",
        choices=("both", "timings", "profile"),
        default="both",
        help="'timings' is the fast unprofiled depth curve, 'profile' the per-chunk per-op "
        "breakdowns, 'both' (default) runs the curve first",
    )
    parser.add_argument("--mesh", default="8x4", help="mesh id in the pytest node id (default 8x4)")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help=f"default {DEFAULT_OUT_ROOT}")
    parser.add_argument(
        "--run-id",
        default=None,
        help="output subdirectory name (default: a UTC timestamp). Pass the same --run-id to "
        "resume: completed runs are skipped and their saved results reused.",
    )
    parser.add_argument("--gzip", action="store_true", help="gzip the copied profiler CSVs")
    parser.add_argument("--force", action="store_true", help="re-run steps that already completed")
    parser.add_argument("--dry-run", action="store_true", help="print what would run and exit")
    args = parser.parse_args()

    chunk_idxs = parse_chunks(args.chunks, parser)

    # Every path here — the pytest node id, the profiler report dir — is relative to the repo
    # root, and tracy writes its reports relative to the cwd it was launched in.
    if not Path(TEST_FILE).is_file():
        parser.error(f"run this from the tt-metal repo root: {TEST_FILE} not found relative to {Path.cwd()}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    out_root = Path(args.out_root) / run_id
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    env_base = dict(os.environ)
    do_timings = args.phase in ("both", "timings")
    do_profile = args.phase in ("both", "profile")
    print(
        f"gemma4 layer perf sweep -> {out_root}\n"
        f"  chunks: {len(chunk_idxs)} ({chunk_idxs[0]}..{chunk_idxs[-1]})  cells: {2 * len(chunk_idxs)}\n"
        f"  phases: {'timings (1 run)' if do_timings else ''}"
        f"{' + ' if do_timings and do_profile else ''}"
        f"{f'profile ({len(chunk_idxs)} runs)' if do_profile else ''}"
    )

    manifest = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "context_len": CONTEXT_LEN,
        "chunk": CHUNK,
        "n_chunks": N_CHUNKS,
        "chunks_requested": chunk_idxs,
        "mesh": args.mesh,
        "phase": args.phase,
        "test": f"{TEST_FILE}::{TEST_NAME}",
        "timings": None,
        "profiles": [],
    }

    def flush():
        if args.dry_run:
            return
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        build_tables(
            (manifest.get("timings") or {}).get("results"),
            manifest["profiles"],
            out_root,
        )

    timings_record = None
    if do_timings:
        timings_record = run_timings(chunk_idxs, args, out_root, env_base)
        manifest["timings"] = timings_record
        flush()
    else:
        # Resuming into the profile phase only. The depth curve lives in the timings phase's
        # own meta.json, so reload it rather than leaving manifest["timings"] None -- otherwise
        # the next flush rewrites summary.csv with every timings column blank, quietly
        # destroying the join between the curve and the per-op totals.
        prior = out_root / "timings" / "meta.json"
        if prior.exists():
            try:
                timings_record = json.loads(prior.read_text())
                manifest["timings"] = timings_record
                print(f"  [load] reused {len(timings_record.get('results', {}))} timings cells from {prior}")
            except Exception as exc:
                print(f"  [warn] could not reload {prior}: {exc}")

    if do_profile:
        for chunk_idx in chunk_idxs:
            manifest["profiles"].append(run_profile_chunk(chunk_idx, args, out_root, env_base))
            # Flushed after every chunk, so an interrupted sweep still leaves usable tables.
            flush()

    if args.dry_run:
        print("dry run — nothing executed")
        return 0

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_root / "README.md").write_text(
        README.format(
            run_id=run_id,
            context_len=CONTEXT_LEN,
            n_chunks=N_CHUNKS,
            chunk=CHUNK,
            n_global=N_GLOBAL,
            n_sliding=N_SLIDING,
            test_file=TEST_FILE,
            test_name=TEST_NAME,
        )
    )

    curve, estimate, _summary = build_tables(
        (manifest.get("timings") or {}).get("results"), manifest["profiles"], out_root
    )
    if not curve and not any(p.get("cells") for p in manifest["profiles"]):
        print(f"\nnothing was measured — see the logs under {out_root}")
        return 1

    print(f"\n{out_root}")
    for name in ("timings.csv", "model_estimate.csv", "summary.csv", "README.md", "manifest.json"):
        if (out_root / name).exists():
            print(f"  {name}")

    if estimate:
        print("\nchunk  global_ms  sliding_ms  est_60_layer_ms")
        for row in estimate:
            print(
                f"{row['chunk_idx']:5d}  {row['global_ms']:9.2f}  {row['sliding_ms']:10.2f}  {row['est_60_layer_ms']:15.0f}"
            )
        first, last = estimate[0], estimate[-1]
        if last["chunk_idx"] != first["chunk_idx"]:
            print(
                f"\ndepth cost chunk {first['chunk_idx']} -> {last['chunk_idx']}: "
                f"global {first['global_ms']:.2f} -> {last['global_ms']:.2f}ms "
                f"({last['global_ms'] / first['global_ms']:.2f}x), "
                f"sliding {first['sliding_ms']:.2f} -> {last['sliding_ms']:.2f}ms "
                f"({last['sliding_ms'] / first['sliding_ms']:.2f}x)"
            )

    noisy = [r for r in curve if r.get("noisy")]
    if noisy:
        print(f"\n{len(noisy)} cell(s) flagged noisy — re-run before trusting them:")
        for row in noisy:
            print(f"  {row['layer_type']} chunk {row['chunk_idx']}")

    failures = [p for p in manifest["profiles"] if p.get("status") not in ("ok", "dry-run")]
    if failures:
        print(f"\n{len(failures)} profiled run(s) did not complete:")
        for record in failures:
            print(f"  chunk {record.get('chunk_idx')}: {record.get('status')} (see {record.get('log')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
