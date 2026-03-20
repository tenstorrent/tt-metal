#!/usr/bin/env python3
"""
Sweep ISL (input sequence length) x batch size for GLM-4.7-Flash on TT
using pre-tokenized JSON input files.

Uses input_data_prefill_{isl}.json from sample_prompts/ to skip tokenization.
Falls back to --simulate-context-len if a JSON file is not found for a given ISL.

Generate the JSON files first:
  python models/demos/glm4_moe_lite/scripts/generate_input_jsons.py

Then run:
  python models/demos/glm4_moe_lite/scripts/run_sweep_isl_json.py [--dry-run] [--out-dir DIR]

Outputs: CSV results, markdown table, and matplotlib graphs (same format as run_sweep_isl_batch.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ISL_VALUES = [128, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
BATCH_SIZES = [1]

FALLBACK_PROMPT = "Summarize the following document. "
MAX_NEW_TOKENS = 128
MESH_ROWS = 1
MESH_COLS = 8
PREFILL_CHUNK_SIZE = 32768
SCRIPT_PATH = "models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py"
SAMPLE_PROMPTS_DIR = "models/demos/glm4_moe_lite/sample_prompts"


def _find_json_for_isl(isl: int, repo_root: Path) -> Path | None:
    """Return the pre-tokenized JSON path if it exists, else None."""
    p = repo_root / SAMPLE_PROMPTS_DIR / f"input_data_prefill_{isl}.json"
    return p if p.is_file() else None


def _build_env() -> dict:
    env = os.environ.copy()
    env["TT_METAL_GTEST_ETH_DISPATCH"] = "1"
    env["GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES"] = "1"
    env["GLM4_MOE_LITE_FUSE_QKV_A"] = "1"
    env["GLM4_MOE_LITE_FUSE_SHARED_GATE_UP"] = "1"
    env["GLM4_MOE_LITE_DECODE_L1_ACT"] = "1"
    env["GLM4_MOE_LITE_EP_L1"] = "1"
    env["GLM4_MOE_LITE_EXPLICIT_PROG_CFG"] = "1"
    env["GLM4_MOE_LITE_TP"] = "1"
    env["GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE"] = str(PREFILL_CHUNK_SIZE)
    return env


def run_one(isl: int, batch_size: int, repo_root: Path, dry_run: bool, timeout_s: int) -> dict:
    """Run one (ISL, batch_size) combination; return metrics or error."""
    min_cache = isl + MAX_NEW_TOKENS

    json_path = _find_json_for_isl(isl, repo_root)

    cmd = [
        sys.executable,
        str(repo_root / SCRIPT_PATH),
    ]
    if json_path is not None:
        cmd += ["--input-tokens-json", str(json_path)]
    else:
        cmd += ["--prompt", FALLBACK_PROMPT, "--simulate-context-len", str(isl)]

    cmd += [
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
        "--warmup",
    ]

    result = {
        "isl": isl,
        "batch_size": batch_size,
        "input_mode": "json" if json_path else "simulate",
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
        print(f"    [dry-run] {'JSON' if json_path else 'fallback'}: {' '.join(cmd[-6:])}", flush=True)
        return result

    env = _build_env()
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
            last_lines = [l for l in combined.splitlines() if l.strip()][-10:]
            print(f"    [ERROR rc={proc.returncode}] last output:", flush=True)
            for l in last_lines:
                print(f"      {l}", flush=True)
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
                # Per-user: one token per sequence per wall-clock step; aggregate: all B in parallel
                per_user = 1000.0 / steady_ms
                result["per_user_steady_tok_s"] = per_user
                result["steady_state_tok_s"] = batch_size * per_user
                # Runner's printed tok_s counts only one stream's tokens in the numerator — use physics here
                result["agg_tok_s"] = result["steady_state_tok_s"]
                result["per_user_tok_s"] = result["per_user_steady_tok_s"]

        m_ft = re.search(r"first token:\s+([\d.]+)\s+ms", stdout)
        if m_ft:
            result["first_token_decode_ms"] = float(m_ft.group(1))
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


# ---------------------------------------------------------------------------
# CSV / Markdown / Plotting (same format as run_sweep_isl_batch.py)
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "isl",
    "batch_size",
    "input_mode",
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
]

NUMERIC_KEYS = {
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


def load_existing_csv(csv_path: Path) -> dict[tuple[int, int], dict]:
    if not csv_path.is_file():
        return {}
    by_key = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            isl = int(row["isl"])
            batch = int(row["batch_size"])
            for k in NUMERIC_KEYS:
                v = row.get(k, "")
                row[k] = float(v) if v and v.strip() else None
            by_key[(isl, batch)] = row
    return by_key


def write_csv(results: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


def _fmt(r: dict, key: str) -> str:
    v = r.get(key)
    if v is None:
        return "\u2014"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _per_user_tps(r: dict | None) -> str:
    if not r:
        return "OOM/fail"
    if r.get("per_user_steady_tok_s") is not None:
        return _fmt(r, "per_user_steady_tok_s")
    return _fmt(r, "per_user_tok_s")


def write_table_md(results: list[dict], out_path: Path, all_results: list[dict] | None = None) -> None:
    if all_results is None:
        all_results = results
    ok = [r for r in results if r["status"] == "ok"]
    isls = sorted({r["isl"] for r in ok})
    batches = sorted({r["batch_size"] for r in ok})
    if not isls or not batches:
        out_path.write_text("No successful runs to tabulate.\n")
        return

    by_key = {(r["isl"], r["batch_size"]): r for r in ok}

    def _table(title: str, value_fn):
        lines = [
            f"## {title}",
            "",
            "| ISL \\ batch | " + " | ".join(str(b) for b in batches) + " |",
            "|" + "---|" * (len(batches) + 1) + "|",
        ]
        for isl in isls:
            row = [str(isl)]
            for b in batches:
                r = by_key.get((isl, b))
                row.append(value_fn(r) if r else "OOM/fail")
            lines.append("| " + " | ".join(row) + " |")
        return lines

    sections = [
        "# GLM-4.7-Flash sweep (JSON inputs): ISL x batch",
        "",
    ]
    sections += _table("Per-user TPS (tokens/sec per sequence; steady-state when available)", _per_user_tps) + [""]
    sections += _table("Aggregate TPS (total tokens/sec)", lambda r: _fmt(r, "agg_tok_s")) + [""]
    sections += _table("TTFT (time to first token, ms)", lambda r: _fmt(r, "ttft_ms")) + [""]
    sections += _table("Decode latency mean (ms)", lambda r: _fmt(r, "decode_mean_ms")) + [""]
    sections += _table("Prefill time (s)", lambda r: _fmt(r, "prefill_s"))

    oom_runs = [r for r in all_results if r.get("status") in ("OOM",) or (r.get("status", "").startswith("exit_"))]
    if oom_runs:
        sections += ["", "## OOM/Failure Details", "", "| ISL | Batch | Status | Detail |", "|---|---|---|---|"]
        for r in sorted(oom_runs, key=lambda x: (x["isl"], x["batch_size"])):
            detail = r.get("oom_detail") or "\u2014"
            sections.append(f"| {r['isl']} | {r['batch_size']} | {r['status']} | {detail} |")

    out_path.write_text("\n".join(sections))


def plot_graphs(results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available, skipping graphs", flush=True)
        return

    ok = [r for r in results if r["status"] == "ok"]
    isls = sorted({r["isl"] for r in ok})
    batches = sorted({r["batch_size"] for r in ok})
    if not isls or not batches:
        return

    by_key = {(r["isl"], r["batch_size"]): r for r in ok}

    def _matrix(key_fn) -> np.ndarray:
        M = np.full((len(isls), len(batches)), np.nan)
        for i, isl in enumerate(isls):
            for j, b in enumerate(batches):
                r = by_key.get((isl, b))
                if r:
                    v = key_fn(r)
                    if v is not None:
                        M[i, j] = v
        return M

    def _pu_val(r):
        return r.get("per_user_steady_tok_s") if r.get("per_user_steady_tok_s") is not None else r.get("per_user_tok_s")

    # --- heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, title, fn, cmap in [
        (axes[0], "Per-user TPS (tok/s)", _pu_val, "viridis"),
        (axes[1], "Aggregate TPS (tok/s)", lambda r: r.get("agg_tok_s"), "plasma"),
        (axes[2], "TTFT (ms)", lambda r: r.get("ttft_ms"), "inferno_r"),
    ]:
        M = _matrix(fn)
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels(batches)
        ax.set_yticks(range(len(isls)))
        ax.set_yticklabels([str(x) for x in isls])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("ISL (tokens)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_dir / "sweep_heatmaps.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- line plots ---
    def _lineplot(filename, ylabel, title, key_fn, marker):
        fig_l, ax = plt.subplots(figsize=(10, 6))
        for isl in isls:
            xs, ys = [], []
            for b in batches:
                r = by_key.get((isl, b))
                v = key_fn(r) if r else None
                if v is not None:
                    xs.append(b)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker=marker, label=f"ISL={isl}")
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_l.savefig(out_dir / filename, dpi=120, bbox_inches="tight")
        plt.close()

    _lineplot(
        "sweep_per_user_tps_vs_batch.png", "Per-user TPS (tok/s)", "Per-user TPS vs batch size by ISL", _pu_val, "o"
    )
    _lineplot(
        "sweep_agg_tps_vs_batch.png",
        "Aggregate TPS (tok/s)",
        "Aggregate TPS vs batch size by ISL",
        lambda r: r.get("agg_tok_s"),
        "s",
    )
    _lineplot(
        "sweep_ttft_vs_batch.png",
        "TTFT (ms)",
        "Time to first token vs batch size by ISL",
        lambda r: r.get("ttft_ms"),
        "^",
    )


# ---------------------------------------------------------------------------
# Log recovery (same approach as run_sweep_isl_batch.py)
# ---------------------------------------------------------------------------


def recover_from_log(log_path: Path) -> list[dict]:
    lines = log_path.read_text().splitlines()
    results = []
    run_re = re.compile(r"\[\d+/\d+\]\s+ISL=(\d+)\s+batch=(\d+)\s+\.\.\.")
    for i, line in enumerate(lines):
        m = run_re.search(line)
        if not m:
            continue
        isl, batch_size = int(m.group(1)), int(m.group(2))
        result_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        r: dict = {
            "isl": isl,
            "batch_size": batch_size,
            "input_mode": "unknown",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep ISL x batch for GLM-4.7-Flash using pre-tokenized JSON inputs")
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
        default=Path("models/demos/glm4_moe_lite/experiments/sweep_isl_json"),
        help="Output directory for CSV, table, and graphs",
    )
    ap.add_argument(
        "--start-from",
        type=int,
        metavar="N",
        default=1,
        help="1-based run index to resume from (loads existing CSV and skips earlier runs)",
    )
    ap.add_argument("--timeout", type=int, default=600, help="Timeout per run in seconds (default 600)")
    ap.add_argument("--isl", type=int, nargs="+", default=ISL_VALUES, help="ISL values to sweep")
    ap.add_argument("--batch", type=int, nargs="+", default=BATCH_SIZES, help="Batch sizes to sweep")
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    isls = sorted(args.isl)
    batches = sorted(args.batch)
    total = len(isls) * len(batches)
    start_from = max(1, int(args.start_from))
    csv_path = args.out_dir / "sweep_results.csv"
    existing = load_existing_csv(csv_path) if start_from > 1 else {}
    if start_from > 1:
        print(f"Resume: start-from={start_from}, loaded {len(existing)} existing rows from {csv_path}", flush=True)

    # Report which ISLs have JSON files available
    json_available = {isl: _find_json_for_isl(isl, repo_root) is not None for isl in isls}
    json_isls = [isl for isl, avail in json_available.items() if avail]
    sim_isls = [isl for isl, avail in json_available.items() if not avail]
    print(f"Sweep: ISL={isls}, batch={batches} -> {total} runs (dry_run={args.dry_run})", flush=True)
    if json_isls:
        print(f"  JSON inputs available for ISL: {json_isls}", flush=True)
    if sim_isls:
        print(f"  Falling back to --simulate-context-len for ISL: {sim_isls}", flush=True)

    results: list[dict] = []
    for i, isl in enumerate(isls):
        for j, batch_size in enumerate(batches):
            idx = i * len(batches) + j + 1
            if idx < start_from:
                r = existing.get((isl, batch_size))
                if r is None:
                    r = {
                        "isl": isl,
                        "batch_size": batch_size,
                        "input_mode": "skipped",
                        "prefill_s": None,
                        "agg_tok_s": None,
                        "per_user_tok_s": None,
                        "ttft_ms": None,
                        "first_token_decode_ms": None,
                        "status": "skipped",
                    }
                results.append(r)
                continue

            mode_tag = "JSON" if json_available.get(isl) else "sim"
            print(f"[{idx}/{total}] ISL={isl} batch={batch_size} ({mode_tag}) ...", flush=True)
            r = run_one(isl, batch_size, repo_root, args.dry_run, args.timeout)
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
                    decode_info = (
                        f" decode_ms(mean={r['decode_mean_ms']:.1f}"
                        f" min={r['decode_min_ms']:.1f}"
                        f" max={r['decode_max_ms']:.1f})"
                    )
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
