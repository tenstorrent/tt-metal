#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Run, parse and report the MiniMax-H3 AGMM fused-vs-unfused sweeps driven by the unit test.

The device work is `test_h3_agmm_sweep` in models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py:
one device session per (op, M, mode) that warms up + PCC-checks every combo, then trace-executes the valid combos
between Tracy signposts and writes a JSON sidecar (combo order, pcc, status). This tool

  run     runs those pytest cases under the Tracy device profiler, one subprocess each, then parses
  parse   joins one profiler ops log with its sidecar and appends rows to the results CSV
  report  prints the markdown tables: fused blockings, all-gather hyperparameters, unfused matmul blockings, and a
          per-shape fused vs all-gather + matmul summary

Modes: fused (all_gather_minimal_matmul_async, 8x8), ag (all_gather_async), mm8x8 / mm8x9 (the standalone matmul
on the AGMM grid / on the full grid the model's unfused branch uses). Combos are 5-tuples: matmul blockings
(M_block, K_block, N_block, sb_h, sb_w) or all-gather hyperparameters (workers_per_link, chunks_per_sync,
buffers_per_channel, 0, 0).

Examples
  python agmm_unit_sweep.py run --ops to_out --M 13664 --modes fused --combos '[[8,8,6,2,2]]'
  python agmm_unit_sweep.py run --ops to_qkv,to_out,ff1 --M 13664 --modes fused,ag,mm8x8,mm8x9
  python agmm_unit_sweep.py report --M 13664 --top 10

Durations are the device kernel duration averaged over the ring's devices, the convention of
models/tt_dit/utils/sweep_mm_block_sizes.py, so numbers here compare directly with its CSV.
Not a test; pytest leaves it alone.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shlex
import sys
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimax_h3_ops import AGMM_OPS, OPS_BY_NAME  # noqa: E402

TEST_FILE = "models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py"
TEST_NAME = "test_h3_agmm_sweep"
DEFAULT_CSV = "agmm_h3_sweep_results.csv"
# Unit-test device rows (the production mesh row of the H3 model on each galaxy) and the arch prefix pytest puts in
# front of every id on that arch.
DEVICE_CONFIG_ARCH = {"wh4x8links4_ring": "wormhole_b0", "bh4x8links2_ring": "blackhole"}
DEFAULT_DEVICE_CONFIG = {"wormhole_b0": "wh4x8links4_ring", "blackhole": "bh4x8links2_ring"}
ARCH_SHORT = {"wormhole_b0": "WH", "blackhole": "BH"}
# mm_ring: standalone matmul on the AGMM worker grid; mm_full: on the model's full matmul grid. The grid itself is
# in the CSV's core_grid column. The first Wormhole study wrote the modes as mm8x8 / mm8x9.
MODES = ("fused", "ag", "mm_ring", "mm_full")
MODE_ALIASES = {"mm8x8": "mm_ring", "mm8x9": "mm_full"}
SIDECAR_DIR = os.path.join("generated", "agmm_h3_sweep")
AG_PRODUCTION = (3, 16, 2)  # CCLManager.get_ag_hyperparams for > 512 rows
CSV_COLUMNS = [
    "device_config",
    "op",
    "M",
    "K",
    "N",
    "mode",
    "core_grid",
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "pcc",
    "rel_rmse",
    "device_kernel_duration_ns",
    "status",
    "subdir",
    "shipped",  # 1 when the combo is the blocking the model resolves for that grid (from the sweep's sidecar)
]


# ----------------------------------------------------------------------------------------------------------------
# naming
# ----------------------------------------------------------------------------------------------------------------
def subdir_name(device_config: str, op: str, M: int, mode: str) -> str:
    return f"agmm_h3_sweep_{device_config}_{op}_M{M}_{mode}"


def sidecar_path(subdir: str) -> str:
    return os.path.join(SIDECAR_DIR, f"{subdir}.json")


def node_id(arch: str, mode: str, M: int, op: str, device_config: str) -> str:
    # pytest composes the id from the innermost parametrize outwards: mode, M, op, device row.
    return f"{TEST_FILE}::{TEST_NAME}[{arch}-{mode}-M{M}-{op}-{device_config}]"


def detect_arch() -> str | None:
    """ttnn's arch name for the machine we are on; None when ttnn is not importable (host-only use)."""
    try:
        import ttnn

        return ttnn.get_arch_name()
    except Exception:
        return None


def production_unfused_blocking(spec) -> tuple[int, ...] | None:
    """Blocking the model's unfused branch would run on the full grid: AGMM_BLOCK_SIZES[(K, N)] + subblock (2, 2)."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "models",
        "transformers",
        "minimax_h3",
        "agmm_config.py",
    )
    if not os.path.exists(path):
        return None
    module_spec = importlib.util.spec_from_file_location("agmm_config", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules.setdefault("agmm_config", module)
    module_spec.loader.exec_module(module)
    blocks = module.AGMM_BLOCK_SIZES.get((spec.K, spec.N))
    return None if blocks is None else (*blocks, 2, 2)


# ----------------------------------------------------------------------------------------------------------------
# profiler log parsing (same rules as sweep_mm_block_sizes.parse_ops_log, without importing ttnn)
# ----------------------------------------------------------------------------------------------------------------
def parse_ops_log(subdir: str, expected_ops: int) -> list[float]:
    """Per-op device kernel durations (ns) between the start/stop signposts, in dispatch order.

    Rows are grouped by GLOBAL CALL COUNT and averaged over devices; trace replay gives every device its own call
    count, so when the count is a multiple of expected_ops the groups are chunked and averaged again."""
    import numpy as np
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    signposts = df[df["OP TYPE"] == "signpost"]
    starts = signposts[signposts["OP CODE"] == "start"]
    stops = signposts[signposts["OP CODE"] == "stop"]
    if starts.empty or stops.empty:
        print(f"  WARN: no start/stop signposts in {filename}", file=sys.stderr)
        return []
    df = df.iloc[starts.index[0] + 1 : stops.index[0]]
    df = df[df["OP TYPE"] != "signpost"]
    df = df[df["DEVICE KERNEL DURATION [ns]"] != "-"]
    if df.empty:
        return []
    df = df.copy()
    df["DEVICE KERNEL DURATION [ns]"] = df["DEVICE KERNEL DURATION [ns]"].astype(float)
    if "GLOBAL CALL COUNT" in df.columns:
        durations = df.groupby("GLOBAL CALL COUNT", sort=False)["DEVICE KERNEL DURATION [ns]"].mean().tolist()
    else:
        durations = df["DEVICE KERNEL DURATION [ns]"].tolist()
    if expected_ops and len(durations) > expected_ops and len(durations) % expected_ops == 0:
        per = len(durations) // expected_ops
        durations = np.array(durations).reshape(expected_ops, per).mean(axis=1).tolist()
    return durations


# ----------------------------------------------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------------------------------------------
def append_rows(csv_path: str, rows: list[dict]) -> None:
    """Append; an existing file keeps its own header (columns it lacks are dropped), so older CSVs stay appendable."""
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    fieldnames = CSV_COLUMNS
    if not new_file:
        with open(csv_path, newline="") as f:
            fieldnames = next(csv.reader(f))
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(csv_paths: list[str]) -> list[dict]:
    rows = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  WARN: {path} does not exist", file=sys.stderr)
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for key in ("M", "K", "N", "b0", "b1", "b2", "b3", "b4"):
                    row[key] = int(row[key])
                row["device_kernel_duration_ns"] = float(row["device_kernel_duration_ns"])
                row["pcc"] = float(row["pcc"]) if row["pcc"] not in ("", "None") else None
                row["rel_rmse"] = float(row["rel_rmse"]) if row["rel_rmse"] not in ("", "None") else None
                row["mode"] = MODE_ALIASES.get(row["mode"], row["mode"])
                row["shipped"] = {"1": True, "0": False}.get(row.get("shipped") or "", None)
                rows.append(row)
    return rows


def parse_one(subdir: str, sidecar: str, device_config: str, csv_path: str) -> list[dict]:
    """Join one profiler subdir with its sidecar; append rows; return them."""
    with open(sidecar) as f:
        side = json.load(f)
    measured = [tuple(c) for c in side["measured"]]
    durations = parse_ops_log(subdir, expected_ops=len(measured)) if measured else []
    if len(durations) != len(measured):
        print(
            f"  WARN {subdir}: {len(measured)} measured combos but {len(durations)} ops in the profiler log; "
            f"rows beyond the log are MISSING",
            file=sys.stderr,
        )
    by_combo = {tuple(r["combo"]): r for r in side["records"]}
    shipped = tuple(side.get("shipped_blocks") or ())
    rows = []
    for i, combo in enumerate(measured):
        rec = by_combo[combo]
        ok = i < len(durations)
        rows.append(
            {
                "device_config": device_config,
                "op": side["op"],
                "M": side["M"],
                "K": side["K"],
                "N": side["N"],
                "mode": side["mode"],
                "core_grid": side["grid"],
                **{f"b{j}": combo[j] for j in range(5)},
                "pcc": rec["pcc"],
                "rel_rmse": rec["rel_rmse"],
                "device_kernel_duration_ns": durations[i] if ok else -1,
                "status": "OK" if ok else "MISSING",
                "subdir": subdir,
                "shipped": int(tuple(combo) == shipped),
            }
        )
    for rec in side["records"]:
        if rec["status"] != "ok":
            combo = tuple(rec["combo"])
            rows.append(
                {
                    "device_config": device_config,
                    "op": side["op"],
                    "M": side["M"],
                    "K": side["K"],
                    "N": side["N"],
                    "mode": side["mode"],
                    "core_grid": side["grid"],
                    **{f"b{j}": combo[j] for j in range(5)},
                    "pcc": None,
                    "rel_rmse": None,
                    "device_kernel_duration_ns": -1,
                    "status": rec["status"],
                    "subdir": subdir,
                    "shipped": int(combo == shipped),
                }
            )
    append_rows(csv_path, rows)
    n_ok = sum(r["status"] == "OK" for r in rows)
    print(f"  {subdir}: {n_ok} OK rows, {len(rows) - n_ok} other, appended to {csv_path}")
    return rows


# ----------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------
def run_case(args, op: str, M: int, mode: str) -> None:
    subdir = subdir_name(args.device_config, op, M, mode)
    side = sidecar_path(subdir)
    os.makedirs(SIDECAR_DIR, exist_ok=True)
    if os.path.exists(side):
        os.remove(side)
    command = (
        f"pytest {shlex.quote(node_id(args.arch, mode, M, op, args.device_config))} "
        f"-x -s -p no:cacheprovider --timeout {args.timeout}"
    )
    env_overrides = {
        "H3_AGMM_SWEEP": "1",
        "H3_SWEEP_SIDECAR": side,
        "H3_SWEEP_MM_K_BLOCK_MAX": str(args.k_block_max),
        # Mid-run flushes are required so the worker can call ReadDeviceProfiler; the sweep itself only flushes
        # after warmup and after the measured pass (a Wormhole galaxy stalls on frequent mid-loop flushes).
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    }
    if args.combos:
        env_overrides["H3_SWEEP_EXPLICIT_COMBOS"] = args.combos
    if args.quiet:
        env_overrides["TT_LOGGER_LEVEL"] = "Error"
    print(f"\n=== {op} M={M} {mode} -> {subdir}")
    print("  " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env_overrides.items()) + " " + command)
    if args.dry_run:
        return
    saved = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    try:
        from tracy.process_model_log import run_device_profiler

        run_device_profiler(
            command,
            subdir,
            check_test_return_code=not args.keep_going,
            device_analysis_types=["device_kernel_duration"],
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if not os.path.exists(side):
        print(f"  WARN: no sidecar at {side}; the test did not reach the measured pass", file=sys.stderr)
        return
    parse_one(subdir, side, args.device_config, args.csv)


def cmd_run(args) -> None:
    ops = _select_ops(args.ops)
    for M in args.M:
        for op in ops:
            for mode in args.modes:
                run_case(args, op.name, M, mode)


def cmd_parse(args) -> None:
    for subdir in args.subdirs:
        parse_one(subdir, sidecar_path(subdir), args.device_config, args.csv)


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------
def _select_ops(names: str):
    if names in ("agmm", "all", ""):
        return list(AGMM_OPS)
    out = []
    for name in names.split(","):
        name = name.strip()
        if name not in OPS_BY_NAME:
            raise SystemExit(f"unknown op {name!r}; choose from {', '.join(OPS_BY_NAME)}")
        out.append(OPS_BY_NAME[name])
    return out


def _dedupe(rows: list[dict]) -> dict[tuple, dict]:
    """One row per (op, M, mode, grid, combo): the fastest OK run (the CSV is append-only)."""
    best = {}
    for r in rows:
        if r["status"] != "OK":
            continue
        key = (r["op"], r["M"], r["mode"], r["core_grid"], r["b0"], r["b1"], r["b2"], r["b3"], r["b4"])
        if key not in best or r["device_kernel_duration_ns"] < best[key]["device_kernel_duration_ns"]:
            best[key] = r
    return best


def is_shipped(r: dict, spec, mode: str) -> bool:
    """The blocking the model resolves for this row's grid: the sidecar's answer when the CSV carries it, else the
    Wormhole-era fallback (the registry's WH blocks for the AGMM grid, AGMM_BLOCK_SIZES for the full grid)."""
    if r.get("shipped") is not None:
        return r["shipped"]
    c = _combo(r)
    if mode == "mm_full":
        prod = production_unfused_blocking(spec)
        return prod is not None and c == prod
    return c == tuple(spec.blocks)


def _grid_of(group: list[dict]) -> str:
    return group[0]["core_grid"] if group else "?"


def _combo(r) -> tuple[int, ...]:
    return (r["b0"], r["b1"], r["b2"], r["b3"], r["b4"])


def _us(ns: float) -> str:
    return f"{ns / 1000:.1f}"


def _fmt_pcc(r) -> str:
    return "" if r["pcc"] is None else f"{r['pcc']:.6f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _ranked(rows: list[dict], top: int, marker_of) -> list[list[str]]:
    rows = sorted(rows, key=lambda r: r["device_kernel_duration_ns"])
    if not rows:
        return []
    best = rows[0]["device_kernel_duration_ns"]
    out = []
    for i, r in enumerate(rows):
        if top and i >= top and not marker_of(r):
            continue
        d = r["device_kernel_duration_ns"]
        out.append(
            [
                str(i + 1),
                ", ".join(str(v) for v in _combo(r)[: 3 if r["mode"] == "ag" else 5]),
                _us(d),
                f"+{100 * (d / best - 1):.1f}%" if i else "best",
                _fmt_pcc(r),
                marker_of(r),
            ]
        )
    return out


def cmd_report(args) -> None:
    rows = [r for r in read_rows(args.csv_paths) if r["device_config"] == args.device_config]
    if args.M:
        rows = [r for r in rows if r["M"] in args.M]
    ops = _select_ops(args.ops)
    best = _dedupe(rows)
    by_group = defaultdict(list)
    for r in best.values():
        by_group[(r["op"], r["M"], r["mode"])].append(r)
    Ms = sorted({r["M"] for r in best.values()})
    out = []
    p = out.append
    arch = ARCH_SHORT.get(DEVICE_CONFIG_ARCH.get(args.device_config, ""), "")
    p(f"# MiniMax-H3 AGMM sweeps on `{args.device_config}` ({arch})\n")
    p(
        "Device kernel duration, mean over the 4 ring devices, one traced execution per combo "
        "(`test_h3_agmm_sweep`). Blockings are (M_block, K_block, N_block, sb_h, sb_w); all-gather rows are "
        "(workers_per_link, chunks_per_sync, buffers_per_channel).\n"
    )

    p("## 1. Fused `all_gather_minimal_matmul_async` on the AGMM worker grid, fastest blocking first\n")
    for M in Ms:
        for spec in ops:
            group = by_group.get((spec.name, M, "fused"), [])
            if not group:
                continue
            p(f"### {spec.name}  M={M}  K={spec.K}  N={spec.N}  grid {_grid_of(group)}  ({spec.fusion})\n")
            p(
                _table(
                    ["rank", "blocking", "us", "vs best", "pcc", ""],
                    _ranked(group, args.top, lambda r, s=spec: "shipped" if is_shipped(r, s, "fused") else ""),
                )
            )
            p("")

    p("## 2a. Unfused step 1: `all_gather_async` of the activation, fastest hyperparameters first\n")
    for M in Ms:
        for spec in ops:
            group = by_group.get((spec.name, M, "ag"), [])
            if not group:
                continue
            p(f"### {spec.name}  M={M}  gathers ({M}, {spec.K_local}) -> ({M}, {spec.K}) per device\n")
            p(
                _table(
                    ["rank", "workers, chunks_per_sync, buffers", "us", "vs best", "pcc", ""],
                    _ranked(group, args.top, lambda r: "production" if _combo(r)[:3] == AG_PRODUCTION else ""),
                )
            )
            p("")

    p("## 2b. Unfused step 2: standalone matmul on the gathered activation, fastest blocking first\n")
    for M in Ms:
        for spec in ops:
            for mode in ("mm_full", "mm_ring"):
                group = by_group.get((spec.name, M, mode), [])
                if not group:
                    continue
                which = "the model's full matmul grid" if mode == "mm_full" else "the AGMM worker grid"
                p(f"### {spec.name}  M={M}  {mode}: {which} {_grid_of(group)}  ({spec.fusion})\n")

                def marker(r, mode=mode, spec=spec):
                    if not is_shipped(r, spec, mode):
                        return ""
                    return "model's unfused blocking" if mode == "mm_full" else "fused op's shipped blocking"

                p(_table(["rank", "blocking", "us", "vs best", "pcc", ""], _ranked(group, args.top, marker)))
                p("")

    p("## 3. Fused vs unfused, best of each\n")
    summary = []
    grids = {"full": set(), "ring": set()}
    for M in Ms:
        for spec in ops:

            def best_of(mode):
                group = by_group.get((spec.name, M, mode), [])
                return min(group, key=lambda r: r["device_kernel_duration_ns"]) if group else None

            f, a, m9, m8 = best_of("fused"), best_of("ag"), best_of("mm_full"), best_of("mm_ring")
            if not any((f, a, m9, m8)):
                continue
            grids["full"].add(_grid_of(by_group.get((spec.name, M, "mm_full"), [])))
            grids["ring"].add(_grid_of(by_group.get((spec.name, M, "mm_ring"), [])))

            def cell(r, n=5):
                if r is None:
                    return "n/a"
                return f"{_us(r['device_kernel_duration_ns'])} ({', '.join(map(str, _combo(r)[:n]))})"

            def total(*parts):
                if any(x is None for x in parts):
                    return None
                return sum(x["device_kernel_duration_ns"] for x in parts)

            u9, u8 = total(a, m9), total(a, m8)

            def delta(u):
                if f is None or u is None:
                    return "n/a"
                d = f["device_kernel_duration_ns"] - u
                return f"{d / 1000:+.1f} us ({100 * d / u:+.1f}%)"

            summary.append(
                [
                    spec.name,
                    str(M),
                    cell(f),
                    cell(a, 3),
                    cell(m9),
                    cell(m8),
                    "n/a" if u9 is None else _us(u9),
                    "n/a" if u8 is None else _us(u8),
                    delta(u9),
                ]
            )
    p(
        _table(
            [
                "op",
                "M",
                "best fused (blocking)",
                "best all-gather (params)",
                f"best matmul full grid ({'/'.join(sorted(grids['full'] - {'?'}) or ['-'])})",
                f"best matmul ring grid ({'/'.join(sorted(grids['ring'] - {'?'}) or ['-'])})",
                "AG + MM full",
                "AG + MM ring",
                "fused minus (AG + MM full)",
            ],
            summary,
        )
    )
    p("")
    p(
        "The unfused sums add two device kernel durations and omit the host dispatch gap between the two ops. "
        "The matmul rows were measured on an activation gathered with the production all-gather settings "
        f"{AG_PRODUCTION}."
    )
    text = "\n".join(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"report written to {args.out}")
    else:
        print(text)


# ----------------------------------------------------------------------------------------------------------------
# roofline: best measured time per (op, M, mode) against compute / DRAM / fabric bounds
# ----------------------------------------------------------------------------------------------------------------
def cmd_roofline(args) -> None:
    """Compare the best blocking of every (op, M, mode) with the roofline of the grid it ran on.

    Constants come from transformer_roofline.py's Arch for the row's architecture (WH or BH); formulas: compute =
    2*M*K*N / (cores * peak_per_core) with cores = the row's core_grid; DRAM = 2 B * (M*K + K*N) / BW (the gathered
    activation read once, the resident weight read once); fabric = (R-1) * shard / (2L) / link_bw with
    shard = 2 B * M * K/R, applied to the fused op and to the standalone all-gather."""
    import transformer_roofline as tr

    arch = tr.BH if DEVICE_CONFIG_ARCH.get(args.device_config) == "blackhole" else tr.WH
    rows = [r for r in read_rows(args.csv_paths) if r["device_config"] == args.device_config]
    if args.M:
        rows = [r for r in rows if r["M"] in args.M]
    best = _dedupe(rows)
    by_group = defaultdict(list)
    for r in best.values():
        by_group[(r["op"], r["M"], r["mode"])].append(r)
    Ms = sorted({r["M"] for r in best.values()})
    ops = _select_ops(args.ops)
    R, L = arch.ring_size, arch.num_links
    per_core = arch.peak_flops(args.fidelity, 1)
    out = []
    p = out.append
    p(f"# Best measured time vs roofline, `{args.device_config}` ({arch.short}), {args.fidelity}\n")
    p(
        f"Peak {per_core / 1e12:.3f} TFLOP/s per core ({args.fidelity}, {arch.clock_hz / 1e9:.1f} GHz); "
        f"DRAM {arch.dram_bw / 1e9:.0f} GB/s; fabric {L} links x {arch.link_bw / 1e9:.1f} GB/s, ring of {R}. "
        "util = compute roofline / measured; the limiter is the largest of the three bounds.\n"
    )
    header = [
        "op",
        "M",
        "mode",
        "cores",
        "best blocking",
        "measured us",
        "compute us",
        "DRAM us",
        "fabric us",
        "limiter",
        "attained TFLOP/s",
        "compute util",
        "measured / ideal",
    ]
    table = []
    for M in Ms:
        for spec in ops:
            flops = 2.0 * M * spec.K * spec.N
            bytes_dram = 2.0 * (M * spec.K + spec.K * spec.N)
            shard = 2.0 * M * spec.K / R
            t_fabric_s = (R - 1) * shard / (2 * L) / arch.link_bw
            for mode in ("fused", "mm_ring", "mm_full", "ag"):
                group = by_group.get((spec.name, M, mode), [])
                if not group:
                    continue
                r = min(group, key=lambda x: x["device_kernel_duration_ns"])
                meas = r["device_kernel_duration_ns"] / 1e3
                if mode == "ag":
                    table.append(
                        [
                            spec.name,
                            str(M),
                            mode,
                            "-",
                            ", ".join(map(str, _combo(r)[:3])),
                            f"{meas:.1f}",
                            "-",
                            "-",
                            f"{t_fabric_s * 1e6:.1f}",
                            "fabric",
                            "-",
                            "-",
                            f"{meas / (t_fabric_s * 1e6):.2f}x",
                        ]
                    )
                    continue
                gx, gy = (int(v) for v in r["core_grid"].split("x"))
                cores = gx * gy
                t_compute = flops / arch.peak_flops(args.fidelity, cores) * 1e6
                t_dram = bytes_dram / arch.dram_bw * 1e6
                t_fab = t_fabric_s * 1e6 if mode == "fused" else 0.0
                bounds = {"compute": t_compute, "DRAM": t_dram, "fabric": t_fab}
                limiter = max(bounds, key=bounds.get)
                table.append(
                    [
                        spec.name,
                        str(M),
                        mode,
                        str(cores),
                        ", ".join(map(str, _combo(r))),
                        f"{meas:.1f}",
                        f"{t_compute:.1f}",
                        f"{t_dram:.1f}",
                        f"{t_fab:.1f}" if t_fab else "-",
                        limiter,
                        f"{flops / meas / 1e6:.1f}",
                        f"{100 * t_compute / meas:.0f}%",
                        f"{meas / bounds[limiter]:.2f}x",
                    ]
                )
    p(_table(header, table))
    text = "\n".join(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"roofline table written to {args.out}")
    else:
        print(text)


# ----------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="results CSV (append-only)")
    ap.add_argument(
        "--device-config",
        default=None,
        choices=sorted(DEVICE_CONFIG_ARCH),
        help="unit-test device row id (default: the production row of --arch / this machine / the CSV's only row)",
    )
    ap.add_argument(
        "--arch",
        default=None,
        choices=sorted(ARCH_SHORT),
        help="ttnn arch name, the prefix of every pytest id (default: from --device-config, else this machine)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run cases under the device profiler and parse them")
    r.add_argument("--ops", default="agmm", help="agmm | comma list of to_qkv,to_out,ff1")
    r.add_argument("--M", type=int, nargs="+", default=[13664], help="rows per device")
    r.add_argument("--modes", default=",".join(MODES), help=f"comma list of {','.join(MODES)}")
    r.add_argument("--combos", default="", help="JSON list of 5-tuples to measure instead of the generated list")
    r.add_argument("--k-block-max", type=int, default=16, help="K_block cap for the unfused matmul sweeps")
    r.add_argument("--timeout", type=int, default=90000, help="pytest --timeout for one case")
    r.add_argument("--keep-going", action="store_true", help="parse even when the pytest case failed")
    r.add_argument("--quiet", action="store_true", help="TT_LOGGER_LEVEL=Error in the worker")
    r.add_argument("--dry-run", action="store_true", help="print the commands only")
    r.set_defaults(func=cmd_run)

    pa = sub.add_parser("parse", help="join existing profiler subdirs with their sidecars")
    pa.add_argument("subdirs", nargs="+")
    pa.set_defaults(func=cmd_parse)

    rp = sub.add_parser("report", help="markdown tables from the CSV")
    rp.add_argument("--csv-paths", nargs="*", default=None, help="CSV files to read (default: --csv)")
    rp.add_argument("--ops", default="agmm")
    rp.add_argument("--M", type=int, nargs="*", default=None)
    rp.add_argument("--top", type=int, default=10, help="rows per table (0 = all); marked rows always show")
    rp.add_argument("--out", default="", help="write the markdown here instead of stdout")
    rp.set_defaults(func=cmd_report)

    rl = sub.add_parser("roofline", help="best measured time per (op, M, mode) against compute/DRAM/fabric bounds")
    rl.add_argument("--csv-paths", nargs="*", default=None, help="CSV files to read (default: --csv)")
    rl.add_argument("--ops", default="agmm")
    rl.add_argument("--M", type=int, nargs="*", default=None)
    rl.add_argument("--fidelity", default="HiFi2", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"])
    rl.add_argument("--out", default="")
    rl.set_defaults(func=cmd_roofline)

    args = ap.parse_args()
    if args.cmd in ("report", "roofline") and not args.csv_paths:
        args.csv_paths = [args.csv]
    _resolve_arch_and_row(args)
    if args.cmd == "run":
        args.modes = [MODE_ALIASES.get(m.strip(), m.strip()) for m in args.modes.split(",") if m.strip()]
        bad = [m for m in args.modes if m not in MODES]
        if bad:
            raise SystemExit(f"unknown modes {bad}; choose from {MODES} (aliases {MODE_ALIASES})")
    args.func(args)


def _resolve_arch_and_row(args) -> None:
    """Fill args.arch / args.device_config from each other, the machine (run), or the CSV (report, roofline)."""
    if args.device_config and not args.arch:
        args.arch = DEVICE_CONFIG_ARCH[args.device_config]
    if not args.device_config and args.cmd in ("report", "roofline"):
        present = sorted({r["device_config"] for r in read_rows(args.csv_paths)})
        if args.arch:
            present = [d for d in present if DEVICE_CONFIG_ARCH.get(d) == args.arch]
        if len(present) == 1:
            args.device_config = present[0]
        elif not present:
            raise SystemExit("no rows in the CSV; pass --device-config")
        else:
            raise SystemExit(f"several device rows in the CSV {present}; pass --device-config")
        args.arch = DEVICE_CONFIG_ARCH[args.device_config]
    if not args.arch:
        args.arch = detect_arch()
        if args.arch not in ARCH_SHORT:
            raise SystemExit(f"cannot tell the arch ({args.arch!r}); pass --arch or --device-config")
    if not args.device_config:
        args.device_config = DEFAULT_DEVICE_CONFIG[args.arch]


if __name__ == "__main__":
    main()
