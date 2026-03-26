#!/usr/bin/env python3
"""
Sweep ISL (input sequence length) × batch size for GLM-4.7-Flash on TT.

Uses a short base prompt and --simulate-context-len to replicate to target ISL.
Collects: per-user TPS, aggregate TPS, TTFT (time to first token).
Outputs: CSV results, markdown table, and matplotlib graphs.

Usage (from tt-metal repo root with venv activated):
  python models/demos/glm4_moe_lite/scripts/run_sweep_isl_batch.py [--dry-run] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

# ISL (input sequence length) in tokens - replicate prompt to reach these
ISL_VALUES = [2000, 4000, 8000, 16000, 32000, 64000, 128000]
BATCH_SIZES = [1, 2, 4, 8, 16, 20, 24, 28, 30, 32]

# Short prompt; script will use --simulate-context-len to repeat to target ISL
BASE_PROMPT = "Summarize the following document. "
MAX_NEW_TOKENS = 128
MESH_ROWS = 4
MESH_COLS = 8
PREFILL_CHUNK_SIZE = 32768
SCRIPT_PATH = "models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py"


def run_one(isl: int, batch_size: int, repo_root: Path, dry_run: bool, timeout_s: int) -> dict:
    """Run one (ISL, batch_size) combination; return metrics or error."""
    min_cache = isl + MAX_NEW_TOKENS
    cmd = [
        sys.executable,
        str(repo_root / SCRIPT_PATH),
        "--prompt",
        BASE_PROMPT,
        "--simulate-context-len",
        str(isl),
        "--min-cache-tokens",
        str(min_cache),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--batch-size",
        str(batch_size),
        "--mesh-rows",
        str(MESH_ROWS),
        "--mesh-cols",
        str(MESH_COLS),
        "--kv-cache-dtype",
        "bf8",
        "--phase",
        "both",
        "--enable-trace",
        "--trace-mode",
        "sampling",
    ]
    env = os.environ.copy()
    env["GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES"] = "1"
    env["GLM4_MOE_LITE_FUSE_QKV_A"] = "1"
    env["GLM4_MOE_LITE_FUSE_SHARED_GATE_UP"] = "1"
    env["GLM4_MOE_LITE_BATCHED_PREFILL"] = "1"
    env["GLM4_MOE_LITE_DECODE_L1_ACT"] = "1"
    env["GLM4_MOE_LITE_EP_L1"] = "1"
    env["GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE"] = str(PREFILL_CHUNK_SIZE)
    # env["GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP"] = "1"
    env["GLM4_MOE_LITE_BATCHED_PREFILL"] = "1"
    env["GLM4_MOE_LITE_TP"] = "1"
    env["GLM4_MOE_LITE_EXPERTS_TT_DTYPE"] = "bf4"
    env["GLM4_MOE_LITE_CCL_NUM_LINKS"] = "4"
    env["GLM4_MOE_LITE_CCL_TOPOLOGY"] = "ring"
    env["GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE"] = "1"
    env["GLM4_MOE_LITE_SKIP_TYPECAST"] = "1"

    result = {
        "isl": isl,
        "batch_size": batch_size,
        "prefill_s": None,
        "agg_tok_s": None,
        "per_user_tok_s": None,
        "ttft_ms": None,
        "first_token_decode_ms": None,
        "decode_mean_ms": None,
        "decode_min_ms": None,
        "decode_max_ms": None,
        "status": "pending",
        "oom_detail": None,
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            result["status"] = f"exit_{proc.returncode}"
            combined = stderr + stdout
            if "OOM" in combined or "out of memory" in combined.lower() or "FATAL" in combined:
                result["status"] = "OOM"
                for line in combined.splitlines():
                    ll = line.strip()
                    if any(k in ll for k in ("OOM", "Out of Memory", "out of memory", "Allocat", "FATAL")):
                        if len(ll) > 10:
                            result["oom_detail"] = ll[:200]
                            break
            return result

        # Parse: prefill_s=104.037 decode_tok_s=0.1225 tok_s=8.17
        m = re.search(r"prefill_s=([\d.]+)\s+decode_tok_s=[\d.]+\s+tok_s=([\d.]+)", stdout)
        if m:
            result["prefill_s"] = float(m.group(1))
            result["agg_tok_s"] = float(m.group(2))
            result["per_user_tok_s"] = result["agg_tok_s"] / batch_size if batch_size else result["agg_tok_s"]

        # Steady-state (subsequent tokens only): subsequent:   mean=  XXX  min=  YYY  max=  ZZZ
        m_sub = re.search(r"subsequent:\s+mean=\s*([\d.]+)\s+min=\s*([\d.]+)\s+max=\s*([\d.]+)", stdout)
        if m_sub and batch_size:
            steady_ms = float(m_sub.group(1))
            result["decode_mean_ms"] = steady_ms
            result["decode_min_ms"] = float(m_sub.group(2))
            result["decode_max_ms"] = float(m_sub.group(3))
            if steady_ms > 0:
                per_user = 1000.0 / steady_ms
                result["per_user_steady_tok_s"] = per_user
                result["steady_state_tok_s"] = batch_size * per_user
                result["agg_tok_s"] = result["steady_state_tok_s"]
                result["per_user_tok_s"] = result["per_user_steady_tok_s"]

        # Parse: first token:  XXXXX.X ms
        m_ft = re.search(r"first token:\s+([\d.]+)\s+ms", stdout)
        if m_ft:
            result["first_token_decode_ms"] = float(m_ft.group(1))
            # TTFT from user perspective: prefill + first decode token
            if result["prefill_s"] is not None:
                result["ttft_ms"] = result["prefill_s"] * 1000 + result["first_token_decode_ms"]

        if result["prefill_s"] is not None or result["agg_tok_s"] is not None:
            result["status"] = "ok"
        else:
            result["status"] = "parse_fail"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = f"error_{type(e).__name__}"

    return result


def load_existing_csv(csv_path: Path) -> dict[tuple[int, int], dict]:
    """Load existing sweep_results.csv; return dict (isl, batch_size) -> row (with numeric fields parsed)."""
    if not csv_path.is_file():
        return {}
    by_key = {}
    numeric_keys = {
        "prefill_s",
        "agg_tok_s",
        "per_user_tok_s",
        "steady_state_tok_s",
        "per_user_steady_tok_s",
        "ttft_ms",
        "first_token_decode_ms",
        "decode_mean_ms",
        "decode_min_ms",
        "decode_max_ms",
    }
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            isl = int(row["isl"])
            batch = int(row["batch_size"])
            for k in numeric_keys:
                v = row.get(k, "")
                row[k] = float(v) if v and v.strip() else None
            by_key[(isl, batch)] = row
    return by_key


def write_csv(results: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "isl",
                "batch_size",
                "prefill_s",
                "agg_tok_s",
                "per_user_tok_s",
                "steady_state_tok_s",
                "per_user_steady_tok_s",
                "ttft_ms",
                "first_token_decode_ms",
                "decode_mean_ms",
                "decode_min_ms",
                "decode_max_ms",
                "status",
                "oom_detail",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(results)


def write_table_md(results: list[dict], out_path: Path, all_results: list[dict] | None = None) -> None:
    """Write markdown tables: per-user TPS, aggregate TPS, TTFT (ms), decode latency, prefill time."""
    if all_results is None:
        all_results = results
    ok = [r for r in results if r["status"] == "ok"]
    isls = sorted({r["isl"] for r in ok})
    batches = sorted({r["batch_size"] for r in ok})
    if not isls or not batches:
        with open(out_path, "w") as f:
            f.write("No successful runs to tabulate.\n")
        return

    def get(r: dict, key: str):
        v = r.get(key)
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    by_key = {(r["isl"], r["batch_size"]): r for r in ok}

    # Prefer steady-state per-user TPS when available (matches short-context ~8–9 tok/s)
    def per_user_tps(r):
        if not r:
            return "OOM/fail"
        return (
            get(r, "per_user_steady_tok_s") if r.get("per_user_steady_tok_s") is not None else get(r, "per_user_tok_s")
        )

    lines = [
        "# GLM-4.7-Flash sweep: ISL × batch",
        "",
        "## Per-user TPS (tokens/sec per sequence; steady-state when available)",
        "",
        "| ISL \\ batch | " + " | ".join(str(b) for b in batches) + " |",
        "|" + "---|" * (len(batches) + 1) + "|",
    ]
    for isl in isls:
        row = [str(isl)]
        for b in batches:
            r = by_key.get((isl, b))
            row.append(per_user_tps(r))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Aggregate TPS (total tokens/sec)")
    lines.append("")
    lines.append("| ISL \\ batch | " + " | ".join(str(b) for b in batches) + " |")
    lines.append("|" + "---|" * (len(batches) + 1) + "|")
    for isl in isls:
        row = [str(isl)]
        for b in batches:
            r = by_key.get((isl, b))
            row.append(get(r, "agg_tok_s") if r else "OOM/fail")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## TTFT (time to first token, ms)")
    lines.append("")
    lines.append("| ISL \\ batch | " + " | ".join(str(b) for b in batches) + " |")
    lines.append("|" + "---|" * (len(batches) + 1) + "|")
    for isl in isls:
        row = [str(isl)]
        for b in batches:
            r = by_key.get((isl, b))
            row.append(get(r, "ttft_ms") if r else "OOM/fail")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Decode latency mean (ms) — ISL × batch")
    lines.append("")
    lines.append("| ISL \\ batch | " + " | ".join(str(b) for b in batches) + " |")
    lines.append("|" + "---|" * (len(batches) + 1) + "|")
    for isl in isls:
        row = [str(isl)]
        for b in batches:
            r = by_key.get((isl, b))
            row.append(get(r, "decode_mean_ms") if r else "OOM/fail")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Prefill time (s) — ISL × batch")
    lines.append("")
    lines.append("| ISL \\ batch | " + " | ".join(str(b) for b in batches) + " |")
    lines.append("|" + "---|" * (len(batches) + 1) + "|")
    for isl in isls:
        row = [str(isl)]
        for b in batches:
            r = by_key.get((isl, b))
            row.append(get(r, "prefill_s") if r else "OOM/fail")
        lines.append("| " + " | ".join(row) + " |")

    # OOM detail table — only for failed runs
    oom_runs = [
        r for r in all_results if r.get("status") in ("OOM", "exit_1") or (r.get("status", "").startswith("exit_"))
    ]
    if oom_runs:
        lines.append("")
        lines.append("## OOM/Failure Details")
        lines.append("")
        lines.append("| ISL | Batch | Status | Detail |")
        lines.append("|---|---|---|---|")
        for r in sorted(oom_runs, key=lambda x: (x["isl"], x["batch_size"])):
            detail = r.get("oom_detail") or "—"
            lines.append(f"| {r['isl']} | {r['batch_size']} | {r['status']} | {detail} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def plot_graphs(results: list[dict], out_dir: Path) -> None:
    """Generate matplotlib figures: heatmaps and line plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    ok = [r for r in results if r["status"] == "ok"]
    isls = sorted({r["isl"] for r in ok})
    batches = sorted({r["batch_size"] for r in ok})
    if not isls or not batches:
        return

    by_key = {(r["isl"], r["batch_size"]): r for r in ok}

    def to_matrix(key) -> np.ndarray:
        # key: str (row key) or callable(r) -> value
        M = np.full((len(isls), len(batches)), np.nan)
        for i, isl in enumerate(isls):
            for j, b in enumerate(batches):
                r = by_key.get((isl, b))
                if r:
                    v = key(r) if callable(key) else r.get(key)
                    if v is not None and (not isinstance(v, float) or not np.isnan(v)):
                        M[i, j] = v
        return M

    def per_user_tps_val(r):
        return r.get("per_user_steady_tok_s") if r.get("per_user_steady_tok_s") is not None else r.get("per_user_tok_s")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Per-user TPS heatmap (steady-state when available)
    M = to_matrix(per_user_tps_val)
    im0 = axes[0].imshow(M, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(batches)))
    axes[0].set_xticklabels(batches)
    axes[0].set_yticks(range(len(isls)))
    axes[0].set_yticklabels([str(x) for x in isls])
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("ISL (tokens)")
    axes[0].set_title("Per-user TPS (tok/s)")
    plt.colorbar(im0, ax=axes[0])

    # Aggregate TPS heatmap
    M2 = to_matrix("agg_tok_s")
    im1 = axes[1].imshow(M2, aspect="auto", cmap="plasma")
    axes[1].set_xticks(range(len(batches)))
    axes[1].set_xticklabels(batches)
    axes[1].set_yticks(range(len(isls)))
    axes[1].set_yticklabels([str(x) for x in isls])
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("ISL (tokens)")
    axes[1].set_title("Aggregate TPS (tok/s)")
    plt.colorbar(im1, ax=axes[1])

    # TTFT heatmap (ms)
    M3 = to_matrix("ttft_ms")
    im2 = axes[2].imshow(M3, aspect="auto", cmap="inferno_r")
    axes[2].set_xticks(range(len(batches)))
    axes[2].set_xticklabels(batches)
    axes[2].set_yticks(range(len(isls)))
    axes[2].set_yticklabels([str(x) for x in isls])
    axes[2].set_xlabel("Batch size")
    axes[2].set_ylabel("ISL (tokens)")
    axes[2].set_title("TTFT (ms)")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    fig.savefig(out_dir / "sweep_heatmaps.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Line plot: per-user TPS vs batch for each ISL (steady-state when available)
    fig2, ax = plt.subplots(figsize=(10, 6))
    for isl in isls:
        xs = []
        ys = []
        for b in batches:
            r = by_key.get((isl, b))
            v = per_user_tps_val(r) if r else None
            if r and v is not None:
                xs.append(b)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"ISL={isl}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Per-user TPS (tok/s)")
    ax.set_title("Per-user TPS vs batch size by ISL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.savefig(out_dir / "sweep_per_user_tps_vs_batch.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Line plot: aggregate TPS vs batch for each ISL
    fig3, ax = plt.subplots(figsize=(10, 6))
    for isl in isls:
        xs = []
        ys = []
        for b in batches:
            r = by_key.get((isl, b))
            if r and r.get("agg_tok_s") is not None:
                xs.append(b)
                ys.append(r["agg_tok_s"])
        if xs:
            ax.plot(xs, ys, marker="s", label=f"ISL={isl}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Aggregate TPS (tok/s)")
    ax.set_title("Aggregate TPS vs batch size by ISL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig3.savefig(out_dir / "sweep_agg_tps_vs_batch.png", dpi=120, bbox_inches="tight")
    plt.close()

    # TTFT vs batch for each ISL
    fig4, ax = plt.subplots(figsize=(10, 6))
    for isl in isls:
        xs = []
        ys = []
        for b in batches:
            r = by_key.get((isl, b))
            if r and r.get("ttft_ms") is not None:
                xs.append(b)
                ys.append(r["ttft_ms"])
        if xs:
            ax.plot(xs, ys, marker="^", label=f"ISL={isl}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to first token vs batch size by ISL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig4.savefig(out_dir / "sweep_ttft_vs_batch.png", dpi=120, bbox_inches="tight")
    plt.close()


def recover_from_log(log_path: Path, isls: list[int], batches: list[int], out_dir: Path) -> list[dict]:
    """Parse sweep_log.txt to recover results (e.g. after process was killed). Returns results list."""
    lines = log_path.read_text().splitlines()
    results = []
    run_re = re.compile(r"\[\d+/\d+\]\s+ISL=(\d+)\s+batch=(\d+)\s+\.\.\.")
    for i, line in enumerate(lines):
        m = run_re.search(line)
        if not m:
            continue
        isl = int(m.group(1))
        batch_size = int(m.group(2))
        # Only use the very next line as the result line (avoids grabbing a later run's metrics)
        result_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        r = {
            "isl": isl,
            "batch_size": batch_size,
            "prefill_s": None,
            "agg_tok_s": None,
            "per_user_tok_s": None,
            "ttft_ms": None,
            "first_token_decode_ms": None,
            "status": "missing",
        }
        if "prefill_s=" in result_line and "tok_s=" in result_line:
            pm = re.search(r"prefill_s=([\d.]+).*?tok_s=([\d.]+)", result_line)
            if pm:
                r["prefill_s"] = float(pm.group(1))
                r["agg_tok_s"] = float(pm.group(2))
                r["per_user_tok_s"] = r["agg_tok_s"] / batch_size if batch_size else r["agg_tok_s"]
            ftm = re.search(r"ttft_ms=([\d.]+)", result_line)
            if ftm:
                r["ttft_ms"] = float(ftm.group(1))
            r["status"] = "ok"
        elif "status=OOM" in result_line:
            r["status"] = "OOM"
        elif "status=exit_" in result_line:
            em = re.search(r"status=(exit_\d+)", result_line)
            r["status"] = em.group(1) if em else "exit_1"
        elif "status=timeout" in result_line:
            r["status"] = "timeout"
        else:
            r["status"] = "incomplete"
        results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep ISL × batch for GLM-4.7-Flash")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only, do not run")
    ap.add_argument(
        "--recover-from-log",
        type=Path,
        metavar="PATH",
        help="Recover results from a sweep log file and write CSV/table/graphs (no device runs)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/demos/glm4_moe_lite/experiments/sweep_isl_batch"),
        help="Output directory for CSV, table, and graphs",
    )
    ap.add_argument(
        "--start-from",
        type=int,
        metavar="N",
        default=1,
        help="1-based run index to start from; loads existing sweep_results.csv and skips runs before N (default 1)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per run in seconds (default 600)",
    )
    ap.add_argument(
        "--isl",
        type=int,
        nargs="+",
        default=ISL_VALUES,
        help="ISL values (default: 1k 4k 8k 16k 20k)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=BATCH_SIZES,
        help="Batch sizes to sweep",
    )
    args = ap.parse_args()

    if args.recover_from_log is not None:
        log_path = args.recover_from_log
        if not log_path.is_file():
            print(f"Log file not found: {log_path}", file=sys.stderr)
            return 1
        args.out_dir.mkdir(parents=True, exist_ok=True)
        isls = sorted(args.isl)
        batches = sorted(args.batch)
        results = recover_from_log(log_path, isls, batches, args.out_dir)
        write_csv(results, args.out_dir / "sweep_results.csv")
        write_table_md(results, args.out_dir / "sweep_table.md", all_results=results)
        plot_graphs(results, args.out_dir)
        n_ok = sum(1 for r in results if r["status"] == "ok")
        print(f"Recovered {len(results)} runs ({n_ok} ok) from {log_path}", flush=True)
        print(f"Wrote {args.out_dir / 'sweep_results.csv'}, sweep_table.md, and graphs", flush=True)
        return 0

    # Repo root: directory that contains SCRIPT_PATH (e.g. tt-metal)
    repo_root = Path(__file__).resolve().parent
    for _ in range(6):
        if (repo_root / SCRIPT_PATH).exists():
            break
        repo_root = repo_root.parent
    else:
        repo_root = Path.cwd()
    if not (repo_root / SCRIPT_PATH).exists():
        print(f"Script not found: {repo_root / SCRIPT_PATH}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    isls = sorted(args.isl)
    batches = sorted(args.batch)
    total = len(isls) * len(batches)
    start_from = max(1, int(args.start_from))
    csv_path = args.out_dir / "sweep_results.csv"
    existing = load_existing_csv(csv_path) if start_from > 1 else {}
    if start_from > 1:
        print(f"Resume: start-from={start_from}, loaded {len(existing)} existing rows from {csv_path}", flush=True)
    print(f"Sweep: ISL={isls}, batch={batches} -> {total} runs (dry_run={args.dry_run})", flush=True)

    results = []
    for i, isl in enumerate(isls):
        for j, batch_size in enumerate(batches):
            idx = i * len(batches) + j + 1
            if idx < start_from:
                # Use existing result for this (isl, batch) or placeholder
                r = existing.get((isl, batch_size))
                if r is None:
                    r = {
                        "isl": isl,
                        "batch_size": batch_size,
                        "prefill_s": None,
                        "agg_tok_s": None,
                        "per_user_tok_s": None,
                        "ttft_ms": None,
                        "first_token_decode_ms": None,
                        "status": "skipped",
                    }
                results.append(r)
                continue
            print(f"[{idx}/{total}] ISL={isl} batch={batch_size} ...", flush=True)
            r = run_one(isl, batch_size, repo_root, args.dry_run, args.timeout)
            results.append(r)
            # Write CSV after each run so partial results survive if the process is killed
            write_csv(results, csv_path)
            if not args.dry_run and r["status"] == "ok":
                pu = (
                    r.get("per_user_steady_tok_s")
                    if r.get("per_user_steady_tok_s") is not None
                    else r["per_user_tok_s"]
                )
                decode_info = ""
                if r.get("decode_mean_ms") is not None:
                    decode_info = f" decode_ms(mean={r['decode_mean_ms']:.1f} min={r['decode_min_ms']:.1f} max={r['decode_max_ms']:.1f})"
                print(
                    f"    prefill_s={r['prefill_s']:.1f} agg_tok_s={r['agg_tok_s']:.2f} "
                    f"per_user_tok_s={pu:.2f} ttft_ms={r['ttft_ms']:.0f}{decode_info}",
                    flush=True,
                )
            else:
                print(f"    status={r['status']}", flush=True)

    print(f"Wrote {csv_path}", flush=True)

    write_table_md(results, args.out_dir / "sweep_table.md", all_results=results)
    print(f"Wrote {args.out_dir / 'sweep_table.md'}", flush=True)

    if not args.dry_run:
        plot_graphs(results, args.out_dir)
        print(f"Wrote graphs to {args.out_dir}", flush=True)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Done: {n_ok}/{total} ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
