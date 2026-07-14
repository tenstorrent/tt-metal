# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape-list-driven AGMM/MM block-size sweep runner + perf history tracker.

Reads a list of shapes from a spec file (agmm/sweep_shapes.json), and for each
shape runs the full block-size sweep on device (reusing the machinery in
models/tt_dit/utils/sweep_mm_block_sizes.py via its MM_SWEEP_SHAPE_JSON entry
point), extracts the best (min-duration) blocking, and appends one row to an
append-only history CSV. The goal: keep a growing list of important shapes and
always know the current best perf on latest main, tracked over time/commits.

Outputs
-------
  agmm/sweep_history.csv        append-only, one row per (shape, run). Long
                                format keyed by (shape_id, git_commit) — filter
                                by shape_id to see the trend across commits.
  agmm/sweep_latest.csv         regenerated each run: most recent best per shape.
  agmm/sweeps/<commit>_<ts>/    full per-combo sweep CSV per shape (drill-down).

Two modes (block-size search vs. tracking)
------------------------------------------
Warmup JIT-compiles every candidate blocking (~2s each; a shape has 150-270
candidates => 5-8 min/shape), so an exhaustive sweep of every shape is ~1 hr of
device time. For routine per-commit tracking you don't need to re-search:

  --mode full    exhaustive block sweep; finds + records the best blocking.
                 Run occasionally to (re)establish the optimum per shape.
  --mode track   (default) measure ONLY each shape's last-known-best blocking
                 from history (~seconds/shape). Shapes with no prior best fall
                 back to a full sweep. This is the routine "how's main doing" run.

Usage
-----
  # routine tracking on latest main (fast: best-blocking only)
  python agmm/run_sweeps.py

  # re-establish the best blocking per shape (slow: full grid)
  python agmm/run_sweeps.py --mode full

  # subset by id or tag
  python agmm/run_sweeps.py --ids s2_compute_n1024 --mode full
  python agmm/run_sweeps.py --tags stage2

  # seed the spec file from the extracted instances, then exit
  python agmm/run_sweeps.py --seed-from-instances

Run from the repo root inside the tt-metal python env (needs ttnn + tracy), e.g.
  TT_METAL_HOME=$PWD PYTHONPATH=$PWD python_env/bin/python agmm/run_sweeps.py
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

# Repo root = parent of this file's dir (agmm/). Make sibling + model imports work
# regardless of cwd, and default all paths relative to the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)  # roofline_lib
sys.path.insert(0, _REPO_ROOT)  # models.*, tools.*

from roofline_lib import compute_roofline  # noqa: E402

DEFAULT_SPEC = os.path.join(_THIS_DIR, "sweep_shapes.json")
DEFAULT_HISTORY = os.path.join(_THIS_DIR, "sweep_history.csv")
DEFAULT_LATEST = os.path.join(_THIS_DIR, "sweep_latest.csv")
DEFAULT_ARCHIVE = os.path.join(_THIS_DIR, "sweeps")

WORKER = "models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep_worker_external"

HISTORY_COLUMNS = [
    "timestamp",
    "git_commit",
    "git_branch",
    "dirty",
    "mode",
    "shape_id",
    "device_config",
    "op_type",
    "M",
    "K",
    "N",
    "grid",
    "fusion_summary",
    "best_M_block",
    "best_K_block",
    "best_N_block",
    "best_sb_h",
    "best_sb_w",
    "best_duration_ns",
    "best_duration_us",
    "ideal_us",
    "limiter",
    "speedup",
    "n_measured",
    "n_skipped",
    "status",
    "sweep_csv_path",
]

PER_COMBO_COLUMNS = [
    "shape_id",
    "device_config",
    "op_type",
    "M",
    "K",
    "N",
    "grid",
    "M_block",
    "K_block",
    "N_block",
    "subblock_h",
    "subblock_w",
    "device_kernel_duration_ns",
    "status",
]


# ============================================================================
# git metadata
# ============================================================================


def git_metadata():
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    commit = _git("rev-parse", "--short", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return commit, branch, dirty


# ============================================================================
# spec loading / filtering / seeding
# ============================================================================


def fusion_summary(fusion):
    """Compact human-readable fusion label for the history row."""
    fusion = fusion or {}
    parts = []
    if fusion.get("chunks", 1) > 1:
        parts.append(f"chunks={fusion['chunks']}")
    if fusion.get("use_addcmul"):
        parts.append("addcmul")
    if fusion.get("use_matmul_split"):
        parts.append("mm_split")
    if fusion.get("fused_activation"):
        parts.append(str(fusion["fused_activation"]))
    if fusion.get("math_approx_mode"):
        parts.append("approx")
    return ",".join(parts) if parts else "-"


def load_spec(path):
    with open(path) as f:
        shapes = json.load(f)
    ids = [s["id"] for s in shapes]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate shape ids in {path}: {sorted(dupes)} (ids must be unique)")
    return shapes


def filter_shapes(shapes, ids, tags):
    if ids:
        idset = set(ids)
        shapes = [s for s in shapes if s["id"] in idset]
        missing = idset - {s["id"] for s in shapes}
        if missing:
            raise ValueError(f"--ids not found in spec: {sorted(missing)}")
    if tags:
        tagset = set(tags)
        shapes = [s for s in shapes if tagset & set(s.get("tags", []))]
    return shapes


def seed_from_instances(instances_path, out_path):
    """Generate a spec file of unique shapes from an agmm_instances.json extraction."""
    with open(instances_path) as f:
        instances = json.load(f)
    seen = {}
    for r in instances:
        gx, gy = (int(x) for x in r["grid"].split("-"))
        key = (r["M"], r["K_gathered"], r["N"], r["chunks"], bool(r["has_addcmul"]), (gx, gy))
        seen.setdefault(key, r)
    shapes = []
    for (M, K, N, chunks, addcmul, grid), r in sorted(seen.items()):
        fusion = {}
        if chunks > 1:
            fusion["chunks"] = chunks
            fusion["math_approx_mode"] = True
        if addcmul:
            fusion["use_addcmul"] = True
            fusion["scalar"] = 1.0
            fusion["math_approx_mode"] = True
        fus = fusion_summary(fusion).replace(",", "_").replace("=", "")
        shapes.append(
            {
                "id": f"m{M}_k{K}_n{N}" + (f"_{fus}" if fus != "-" else ""),
                "op_type": "agmm",
                "device_config": "bh_4x8",
                "M": M,
                "K": K,
                "N": N,
                "grid": list(grid),
                "dtype": "bfloat16",
                "math_fidelity": r["math_fidelity"],
                "fusion": fusion,
                "tags": [],
                "notes": "",
            }
        )
    with open(out_path, "w") as f:
        json.dump(shapes, f, indent=2)
    print(f"Wrote {len(shapes)} unique shapes to {out_path}")


# ============================================================================
# per-shape sweep
# ============================================================================


def last_known_best(history_path, shape_id):
    """Most recent OK best-blocking for a shape from history, or None.

    Returns (M_block, K_block, N_block, sb_h, sb_w) or None if never swept.
    """
    if not os.path.exists(history_path):
        return None
    best = None
    with open(history_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("shape_id") == shape_id and row.get("status") == "OK" and row.get("best_M_block"):
                best = row  # later rows overwrite → most recent wins
    if best is None:
        return None
    return (
        int(best["best_M_block"]),
        int(best["best_K_block"]),
        int(best["best_N_block"]),
        int(best["best_sb_h"]),
        int(best["best_sb_w"]),
    )


def run_one_shape(shape, timestamp, archive_run_dir, explicit_combo=None):
    """Run a block sweep for one shape; return (result_dict, all_combo_rows).

    If explicit_combo is given (track mode), measure only that single
    (M,K,N,sb_h,sb_w) blocking instead of the full grid. result_dict holds the
    best blocking + metadata for the history row; status is "OK" on success or
    an error tag otherwise.
    """
    from tracy.process_model_log import run_device_profiler
    from models.tt_dit.utils.sweep_mm_block_sizes import parse_ops_log

    shape_id = shape["id"]
    device_config = shape.get("device_config", "bh_4x8")
    op_type = shape["op_type"]
    M, K, N = shape["M"], shape["K"], shape["N"]
    grid = shape["grid"]
    grid_str = f"{grid[0]}x{grid[1]}"

    subdir = f"agmm_sweep_{shape_id}"
    combos_file = os.path.join(_THIS_DIR, f".valid_combos_{shape_id}.json")

    base = {
        "shape_id": shape_id,
        "device_config": device_config,
        "op_type": op_type,
        "M": M,
        "K": K,
        "N": N,
        "grid": grid_str,
        "fusion_summary": fusion_summary(shape.get("fusion")),
    }

    # Pass the whole record to the worker; it reads M/K/N/grid/op_type/fusion/etc.
    env_saved = {}

    def _setenv(k, v):
        env_saved[k] = os.environ.get(k)
        os.environ[k] = v

    _setenv("MM_SWEEP_SHAPE_JSON", json.dumps(shape))
    _setenv("MM_SWEEP_VALID_COMBOS_FILE", combos_file)
    _setenv("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    _setenv("TT_LOGGER_LEVEL", "Error")
    if explicit_combo is not None:
        _setenv("MM_SWEEP_EXPLICIT_COMBOS", json.dumps([list(explicit_combo)]))

    command = f"pytest {WORKER} -x -s"

    try:
        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
        except Exception as e:
            print(f"  [{shape_id}] sweep FAILED: {str(e)[:300]}", flush=True)
            return {**base, "status": "SWEEP_FAILED"}, []

        if not os.path.exists(combos_file):
            print(f"  [{shape_id}] no valid_combos file (worker skipped or died)", flush=True)
            return {**base, "status": "NO_COMBOS"}, []
        with open(combos_file) as f:
            valid_combos = [tuple(c) for c in json.load(f)]

        durations = parse_ops_log(subdir, expected_ops=len(valid_combos))
        if not durations:
            return {**base, "status": "NO_TIMINGS", "n_measured": len(valid_combos)}, []
        if len(durations) != len(valid_combos):
            print(
                f"  [{shape_id}] WARN: {len(valid_combos)} combos but {len(durations)} timings; "
                f"aligning to min length",
                flush=True,
            )

        n = min(len(durations), len(valid_combos))
        combo_rows = []
        for i in range(len(valid_combos)):
            m_blk, k_blk, n_blk, sb_h, sb_w = valid_combos[i]
            dur = durations[i] if i < n else -1
            combo_rows.append(
                {
                    **{c: base[c] for c in ("shape_id", "device_config", "op_type", "M", "K", "N", "grid")},
                    "M_block": m_blk,
                    "K_block": k_blk,
                    "N_block": n_blk,
                    "subblock_h": sb_h,
                    "subblock_w": sb_w,
                    "device_kernel_duration_ns": f"{dur:.0f}",
                    "status": "OK" if i < n else "MISSING",
                }
            )

        # Best = min duration among measured combos.
        measured = [(valid_combos[i], durations[i]) for i in range(n)]
        (best_combo, best_ns) = min(measured, key=lambda x: x[1])
        bm, bk, bn, bsh, bsw = best_combo

        result = {
            **base,
            "best_M_block": bm,
            "best_K_block": bk,
            "best_N_block": bn,
            "best_sb_h": bsh,
            "best_sb_w": bsw,
            "best_duration_ns": f"{best_ns:.0f}",
            "best_duration_us": f"{best_ns / 1e3:.2f}",
            "n_measured": n,
            "status": "OK",
        }

        # Roofline join (agmm only; K is the gathered K).
        if op_type == "agmm":
            ring_size = _ring_size_for(device_config)
            rl = compute_roofline(
                M,
                K,
                N,
                ring_size=ring_size,
                num_links=_num_links_for(device_config),
                grid=tuple(grid),
                math_fidelity=shape.get("math_fidelity", "HiFi2"),
                time_us=best_ns / 1e3,
            )
            result["ideal_us"] = f"{rl['ideal_us']:.2f}"
            result["limiter"] = rl["limiter"]
            result["speedup"] = f"{rl['speedup']:.2f}"

        # Archive per-combo detail.
        os.makedirs(archive_run_dir, exist_ok=True)
        sweep_csv = os.path.join(archive_run_dir, f"{shape_id}.csv")
        with open(sweep_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PER_COMBO_COLUMNS)
            w.writeheader()
            w.writerows(combo_rows)
        result["sweep_csv_path"] = os.path.relpath(sweep_csv, _REPO_ROOT)

        print(
            f"  [{shape_id}] BEST M={bm} K={bk} N={bn} sb=({bsh},{bsw}) -> {best_ns/1e3:.2f} us"
            + (f"  (ideal {result.get('ideal_us')} us, {result.get('speedup')}x)" if op_type == "agmm" else ""),
            flush=True,
        )
        return result, combo_rows

    finally:
        for k, v in env_saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if os.path.exists(combos_file):
            os.remove(combos_file)


def _ring_size_for(device_config):
    from models.tt_dit.utils.sweep_mm_block_sizes import DEVICE_CONFIGS

    cfg = DEVICE_CONFIGS[device_config]
    return cfg["mesh_shape"][cfg["cluster_axis"]]


def _num_links_for(device_config):
    from models.tt_dit.utils.sweep_mm_block_sizes import DEVICE_CONFIGS

    return DEVICE_CONFIGS[device_config]["num_links"]


# ============================================================================
# history / latest output
# ============================================================================


def append_history(history_path, rows):
    exists = os.path.exists(history_path)
    with open(history_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def regenerate_latest(history_path, latest_path):
    """Most recent OK row per shape_id (last write wins), sorted by shape_id."""
    if not os.path.exists(history_path):
        return
    latest = {}
    with open(history_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "OK":
                latest[row["shape_id"]] = row  # later rows overwrite earlier
    with open(latest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for sid in sorted(latest):
            w.writerow(latest[sid])


# ============================================================================
# main
# ============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", default=DEFAULT_SPEC, help=f"Spec file (default: {DEFAULT_SPEC})")
    p.add_argument(
        "--mode",
        choices=["full", "track"],
        default="track",
        help="full: exhaustive block sweep (finds best, ~minutes/shape). "
        "track (default): measure only each shape's last-known-best blocking from history "
        "(~seconds/shape); falls back to a full sweep for shapes with no prior best.",
    )
    p.add_argument("--ids", nargs="+", help="Only sweep these shape ids")
    p.add_argument("--tags", nargs="+", help="Only sweep shapes with any of these tags")
    p.add_argument("--history", default=DEFAULT_HISTORY, help=f"Append-only history CSV (default: {DEFAULT_HISTORY})")
    p.add_argument(
        "--latest", default=DEFAULT_LATEST, help=f"Latest-per-shape snapshot CSV (default: {DEFAULT_LATEST})"
    )
    p.add_argument(
        "--archive-dir", default=DEFAULT_ARCHIVE, help=f"Per-run per-combo archive root (default: {DEFAULT_ARCHIVE})"
    )
    p.add_argument(
        "--seed-from-instances",
        nargs="?",
        const=os.path.join(_THIS_DIR, "agmm_instances.json"),
        help="Generate the spec file from an agmm_instances.json extraction, then exit",
    )
    args = p.parse_args()

    if args.seed_from_instances:
        seed_from_instances(args.seed_from_instances, args.shapes)
        return

    shapes = filter_shapes(load_spec(args.shapes), args.ids, args.tags)
    if not shapes:
        print("No shapes selected.")
        return

    commit, branch, dirty = git_metadata()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_run_dir = os.path.join(args.archive_dir, f"{commit or 'nocommit'}_{timestamp}")

    print(f"Sweeping {len(shapes)} shape(s) [mode={args.mode}] @ {commit}{'-dirty' if dirty else ''} ({branch})")
    print(f"History: {args.history}  |  Archive: {archive_run_dir}")

    common = {"timestamp": timestamp, "git_commit": commit, "git_branch": branch, "dirty": dirty}
    history_rows = []
    for i, shape in enumerate(shapes, 1):
        explicit_combo = None
        row_mode = args.mode
        if args.mode == "track":
            explicit_combo = last_known_best(args.history, shape["id"])
            if explicit_combo is None:
                print(f"  [{shape['id']}] no prior best in history — falling back to full sweep", flush=True)
                row_mode = "full"
        print(
            f"\n[{i}/{len(shapes)}] {shape['id']} "
            f"({'best-only ' + str(explicit_combo) if explicit_combo else 'full grid'})",
            flush=True,
        )
        result, _ = run_one_shape(shape, timestamp, archive_run_dir, explicit_combo=explicit_combo)
        history_rows.append({**common, "mode": row_mode, **result})
        # Persist incrementally so a crash mid-run doesn't lose completed shapes.
        append_history(args.history, [history_rows[-1]])

    regenerate_latest(args.history, args.latest)

    ok = sum(1 for r in history_rows if r.get("status") == "OK")
    print(f"\n{'='*60}\nDone: {ok}/{len(history_rows)} shapes OK. Latest snapshot -> {args.latest}")


if __name__ == "__main__":
    main()
