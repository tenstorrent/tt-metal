#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Sweep ISL (input sequence length) × batch size for GLM-4.7 REAP MoE on TT (glm4_moe).

Invokes ``debug_run_full_tt_greedy.py`` with ``--simulate-context-len`` (same idea as
glm4_moe_lite). Sets **GLM4_MOE_*** environment knobs (not GLM4_MOE_LITE_*).

Collects: per-user TPS, aggregate TPS, TTFT, decode latency (when available).
Outputs: CSV, markdown tables, matplotlib graphs.

Usage (from tt-metal repo root, PYTHONPATH=repo root, venv activated):

  python models/demos/glm4_moe/scripts/run_sweep_isl_batch.py [--dry-run] [--out-dir DIR]

Override tuning by exporting GLM4_MOE_* before running; this script overwrites a small
default set (see ``_glm4_moe_sweep_env``) — edit that function or export after if you
need different precedence.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

# Default sweep grids (override with --isl / --batch)
ISL_VALUES = [2000, 4000, 8000, 16000, 32000, 64000, 128000]
BATCH_SIZES = [1, 2, 4, 8, 16, 20, 24, 28, 30, 32]

BASE_PROMPT = "Summarize the following document. "
MAX_NEW_TOKENS = 2
# Galaxy TG-style default (TP×EP mesh); override with --mesh-rows / --mesh-cols
MESH_ROWS = 8
MESH_COLS = 4
PREFILL_CHUNK_SIZE = 32768
SCRIPT_PATH = "models/demos/glm4_moe/scripts/debug_run_full_tt_greedy.py"
DEFAULT_MODEL_ID = "cerebras/GLM-4.7-REAP-218B-A32B"


def _glm4_moe_sweep_env(
    *,
    prefill_chunk: int,
    ccl_num_links: int | None = None,
    ccl_topology: str | None = None,
) -> dict[str, str]:
    """
    Defaults aligned with common REAP MoE bring-up (trace-safe reduces, EP L1, BF4 experts).

    CCL multi-link / topology (optional): pass ``ccl_num_links`` / ``ccl_topology`` or set
    ``GLM4_MOE_CCL_NUM_LINKS``, ``GLM4_MOE_CCL_TOPOLOGY``, or per-axis
    ``GLM4_MOE_CCL_NUM_LINKS_AXIS0`` / ``_AXIS1`` in the environment before running.
    Default stack behavior remains 1 link + linear if unset.
    """
    d: dict[str, str] = {
        # Trace / all-reduce: avoid host reads during traced decode
        "GLM4_MOE_REDUCE_IMPL": "native",
        "GLM4_MOE_EP_REDUCE_DEVICE": "1",
        # MoE / prefill
        "GLM4_MOE_EP_L1": "1",
        "GLM4_MOE_PREFILL_CHUNK_SIZE": str(prefill_chunk),
        # Memory / expert weights (matches many lite sweeps using bf4 experts)
        "GLM4_MOE_EXPERTS_TT_DTYPE": "bf4",
    }
    if ccl_num_links is not None:
        d["GLM4_MOE_CCL_NUM_LINKS"] = str(max(1, int(ccl_num_links)))
    if ccl_topology is not None:
        d["GLM4_MOE_CCL_TOPOLOGY"] = str(ccl_topology)
    return d


def _print_child_output_tail(
    *,
    isl: int,
    batch_size: int,
    stdout: str,
    stderr: str,
    max_lines: int,
) -> None:
    """Echo captured child output so device/busy errors match an interactive greedy run."""
    block = f"--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}\n"
    lines = block.splitlines()
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    print(
        f"\n[child log ISL={isl} batch={batch_size}] last {len(tail)} line(s):\n" + "\n".join(tail),
        flush=True,
        file=sys.stderr,
    )


def run_one(
    isl: int,
    batch_size: int,
    repo_root: Path,
    dry_run: bool,
    timeout_s: int,
    *,
    mesh_rows: int,
    mesh_cols: int,
    model_id: str,
    max_new_tokens: int,
    max_batch_size: int,
    prefill_chunk: int,
    ccl_num_links: int | None,
    ccl_topology: str | None,
    verbose_child_output: bool,
    child_output_tail_lines: int,
) -> dict:
    """Run one (ISL, batch_size) combination; return metrics or error."""
    min_cache = isl + max_new_tokens
    cmd = [
        sys.executable,
        str(repo_root / SCRIPT_PATH),
        "--model-id",
        model_id,
        "--prompt",
        BASE_PROMPT,
        "--simulate-context-len",
        str(isl),
        "--min-cache-tokens",
        str(min_cache),
        "--max-new-tokens",
        str(max_new_tokens),
        "--batch-size",
        str(batch_size),
        "--max-batch-size",
        str(max_batch_size),
        "--mesh-rows",
        str(mesh_rows),
        "--mesh-cols",
        str(mesh_cols),
        "--kv-cache-dtype",
        "bf8",
        "--enable-trace",
        "--trace-mode",
        "sampling",
    ]
    env = os.environ.copy()
    env.update(
        _glm4_moe_sweep_env(
            prefill_chunk=prefill_chunk,
            ccl_num_links=ccl_num_links,
            ccl_topology=ccl_topology,
        )
    )
    env.setdefault("HF_MODEL", model_id)
    env.setdefault("GLM4_MOE_HF_MODEL", model_id)

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
            if verbose_child_output:
                _print_child_output_tail(
                    isl=isl,
                    batch_size=batch_size,
                    stdout=stdout,
                    stderr=stderr,
                    max_lines=int(child_output_tail_lines),
                )
            if "OOM" in combined or "out of memory" in combined.lower() or "FATAL" in combined:
                result["status"] = "OOM"
                for line in combined.splitlines():
                    ll = line.strip()
                    if any(k in ll for k in ("OOM", "Out of Memory", "out of memory", "Allocat", "FATAL")):
                        if len(ll) > 10:
                            result["oom_detail"] = ll[:200]
                            break
            return result

        m = re.search(r"prefill_s=([\d.]+)\s+decode_tok_s=[\d.]+\s+tok_s=([\d.]+)", stdout)
        if m:
            result["prefill_s"] = float(m.group(1))
            result["agg_tok_s"] = float(m.group(2))
            result["per_user_tok_s"] = result["agg_tok_s"] / batch_size if batch_size else result["agg_tok_s"]

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

        m_ft = re.search(r"first token:\s+([\d.]+)\s+ms", stdout)
        if m_ft:
            result["first_token_decode_ms"] = float(m_ft.group(1))
            if result["prefill_s"] is not None:
                result["ttft_ms"] = result["prefill_s"] * 1000 + result["first_token_decode_ms"]

        m_ttft = re.search(r"ttft_ms=([\d.]+)", stdout)
        if m_ttft and result["ttft_ms"] is None:
            result["ttft_ms"] = float(m_ttft.group(1))

        if result["prefill_s"] is not None or result["agg_tok_s"] is not None:
            result["status"] = "ok"
        else:
            result["status"] = "parse_fail"
            if verbose_child_output:
                _print_child_output_tail(
                    isl=isl,
                    batch_size=batch_size,
                    stdout=stdout,
                    stderr=stderr,
                    max_lines=int(child_output_tail_lines),
                )

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = f"error_{type(e).__name__}"

    return result


def load_existing_csv(csv_path: Path) -> dict[tuple[int, int], dict]:
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
                row[k] = float(v) if v and str(v).strip() else None
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

    def per_user_tps(r):
        if not r:
            return "OOM/fail"
        return (
            get(r, "per_user_steady_tok_s") if r.get("per_user_steady_tok_s") is not None else get(r, "per_user_tok_s")
        )

    lines = [
        "# GLM-4.7-REAP (glm4_moe) sweep: ISL × batch",
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

    oom_runs = [
        r for r in all_results if r.get("status") in ("OOM", "exit_1") or (str(r.get("status", "")).startswith("exit_"))
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

    def to_matrix(key):
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


def recover_from_log(log_path: Path) -> list[dict]:
    lines = log_path.read_text().splitlines()
    results = []
    run_re = re.compile(r"\[\d+/\d+\]\s+ISL=(\d+)\s+batch=(\d+)\s+\.\.\.")
    for i, line in enumerate(lines):
        m = run_re.search(line)
        if not m:
            continue
        isl = int(m.group(1))
        batch_size = int(m.group(2))
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
    ap = argparse.ArgumentParser(description="Sweep ISL × batch for GLM-4.7-REAP (glm4_moe)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only, do not run")
    ap.add_argument(
        "--recover-from-log",
        type=Path,
        metavar="PATH",
        help="Recover results from a sweep log file and write CSV/table/graphs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/demos/glm4_moe/experiments/sweep_isl_batch"),
        help="Output directory for CSV, table, and graphs",
    )
    ap.add_argument("--start-from", type=int, metavar="N", default=1, help="1-based run index to resume from")
    ap.add_argument("--timeout", type=int, default=600, help="Timeout per run in seconds")
    ap.add_argument("--isl", type=int, nargs="+", default=ISL_VALUES, help="ISL values (tokens)")
    ap.add_argument("--batch", type=int, nargs="+", default=BATCH_SIZES, help="Batch sizes")
    ap.add_argument("--mesh-rows", type=int, default=MESH_ROWS)
    ap.add_argument("--mesh-cols", type=int, default=MESH_COLS)
    ap.add_argument("--model-id", type=str, default=os.environ.get("HF_MODEL") or DEFAULT_MODEL_ID)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Forwarded to debug script / Glm4MoeTT.create (default: max of --batch, at least 32)",
    )
    ap.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=PREFILL_CHUNK_SIZE,
        help="Sets GLM4_MOE_PREFILL_CHUNK_SIZE in the child environment",
    )
    ap.add_argument(
        "--ccl-num-links",
        type=int,
        default=None,
        metavar="N",
        help="Sets GLM4_MOE_CCL_NUM_LINKS=N in the child (omit for default 1). Per-axis: export GLM4_MOE_CCL_NUM_LINKS_AXIS0 / _AXIS1.",
    )
    ap.add_argument(
        "--ccl-topology",
        choices=["linear", "ring"],
        default=None,
        help="Sets GLM4_MOE_CCL_TOPOLOGY in the child (omit for default linear).",
    )
    ap.add_argument(
        "--verbose-child-output",
        action="store_true",
        help="On child failure or parse_fail, print last lines of its stdout+stderr (see device busy / TT errors).",
    )
    ap.add_argument(
        "--child-output-tail-lines",
        type=int,
        default=80,
        metavar="N",
        help="With --verbose-child-output, how many trailing lines to print (default 80).",
    )
    args = ap.parse_args()

    if args.recover_from_log is not None:
        log_path = args.recover_from_log
        if not log_path.is_file():
            print(f"Log file not found: {log_path}", file=sys.stderr)
            return 1
        args.out_dir.mkdir(parents=True, exist_ok=True)
        results = recover_from_log(log_path)
        write_csv(results, args.out_dir / "sweep_results.csv")
        write_table_md(results, args.out_dir / "sweep_table.md", all_results=results)
        plot_graphs(results, args.out_dir)
        n_ok = sum(1 for r in results if r["status"] == "ok")
        print(f"Recovered {len(results)} runs ({n_ok} ok) from {log_path}", flush=True)
        return 0

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

    max_batch = args.max_batch_size
    if max_batch is None:
        max_batch = max(32, max(args.batch))

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
            r = run_one(
                isl,
                batch_size,
                repo_root,
                args.dry_run,
                args.timeout,
                mesh_rows=int(args.mesh_rows),
                mesh_cols=int(args.mesh_cols),
                model_id=str(args.model_id),
                max_new_tokens=int(args.max_new_tokens),
                max_batch_size=int(max_batch),
                prefill_chunk=int(args.prefill_chunk_size),
                ccl_num_links=args.ccl_num_links,
                ccl_topology=args.ccl_topology,
                verbose_child_output=bool(args.verbose_child_output),
                child_output_tail_lines=int(args.child_output_tail_lines),
            )
            results.append(r)
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
                ttft_s = f" ttft_ms={r['ttft_ms']:.0f}" if r.get("ttft_ms") is not None else ""
                # Include tok_s= for recover_from_log (same aggregate as agg_tok_s).
                print(
                    f"    prefill_s={r['prefill_s']:.1f} tok_s={r['agg_tok_s']:.2f} agg_tok_s={r['agg_tok_s']:.2f} "
                    f"per_user_tok_s={pu:.2f}{ttft_s}{decode_info}",
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
