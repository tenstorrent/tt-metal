# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep demo_ttnn.py over --max_new_tokens values and report TTFT, decode tok/s,
and prefill tok/s for each configuration.

Each config runs in a fresh subprocess to avoid device-state contamination.
Metrics are read from the per-run meta JSON written by demo_ttnn.py.

Usage (from tt-metal root):
    python models/experimental/vibevoice/tests/perf/sweep_max_new_tokens.py
    python models/experimental/vibevoice/tests/perf/sweep_max_new_tokens.py --demo 4p_climate_45min
    python models/experimental/vibevoice/tests/perf/sweep_max_new_tokens.py --tokens 32 64 128 256
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEMO_TTNN = Path(__file__).resolve().parents[2] / "demo" / "demo.py"
DEFAULT_DEMO = "4p_climate_45min"
DEFAULT_TOKENS = [32, 64, 128, 256]
SWEEP_OUT_BASE = Path("/tmp/vv_sweep")


def _run_one(demo: str, max_new_tokens: int, sweep_dir: Path) -> dict:
    """Run demo_ttnn.py for one config; return metrics dict from the meta JSON."""
    out_dir = sweep_dir / str(max_new_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(DEMO_TTNN),
        "--demo",
        demo,
        "--max_new_tokens",
        str(max_new_tokens),
        "--output_dir",
        str(out_dir),
    ]

    print(f"\n{'=' * 60}", flush=True)
    print(f"[sweep] max_new_tokens={max_new_tokens}  demo={demo}", flush=True)
    print(f"[sweep] cmd: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, text=True)
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[sweep] ERROR: demo_ttnn.py exited {result.returncode} for max_new_tokens={max_new_tokens}", flush=True)
        return {"max_new_tokens": max_new_tokens, "error": f"exit {result.returncode}"}

    meta_path = out_dir / demo / f"{demo}_meta.json"
    if not meta_path.is_file():
        print(f"[sweep] ERROR: meta JSON not found at {meta_path}", flush=True)
        return {"max_new_tokens": max_new_tokens, "error": "meta missing"}

    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "max_new_tokens": max_new_tokens,
        "prefill_tokens": meta.get("prefill_tokens"),
        "ar_tokens_generated": meta.get("ar_tokens_generated"),
        "ttft_s": meta.get("ttft_s"),
        "decode_wall_s": meta.get("decode_wall_s"),
        "decode_toks_per_s": meta.get("decode_toks_per_s"),
        "prefill_toks_per_s": meta.get("prefill_toks_per_s"),
        "generate_wall_s": meta.get("generate_wall_s"),
        "tt_wav": meta.get("tt_wav"),
        "subprocess_wall_s": round(wall, 1),
    }


def _print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 80}", flush=True)
    print("[sweep] SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)

    col_w = [16, 16, 12, 10, 14, 16, 16, 14]
    headers = [
        "max_new_tokens",
        "prefill_tokens",
        "ar_tokens",
        "TTFT (s)",
        "decode (s)",
        "decode (tok/s)",
        "prefill (tok/s)",
        "wall (s)",
    ]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line, flush=True)
    print("-" * sum(col_w), flush=True)

    for row in rows:
        if "error" in row:
            print(f"{str(row['max_new_tokens']).ljust(col_w[0])}{''.ljust(col_w[1])}ERROR: {row['error']}", flush=True)
            continue

        def _fmt(val, fmt=".2f"):
            return "-" if val is None else format(val, fmt)

        cols = [
            str(row["max_new_tokens"]),
            str(row.get("prefill_tokens") or "-"),
            str(row.get("ar_tokens_generated") or "-"),
            _fmt(row.get("ttft_s")),
            _fmt(row.get("decode_wall_s")),
            _fmt(row.get("decode_toks_per_s")),
            _fmt(row.get("prefill_toks_per_s"), ".0f"),
            _fmt(row.get("subprocess_wall_s")),
        ]
        print("".join(c.ljust(w) for c, w in zip(cols, col_w)), flush=True)

    print(f"{'=' * 80}", flush=True)

    # JSON dump for easy post-processing
    json_out = SWEEP_OUT_BASE / "sweep_results.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"[sweep] full results → {json_out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep demo_ttnn.py over --max_new_tokens values")
    ap.add_argument("--demo", default=DEFAULT_DEMO)
    ap.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=DEFAULT_TOKENS,
        metavar="N",
        help="max_new_tokens values to sweep (default: 32 64 128 256)",
    )
    args = ap.parse_args()

    sweep_dir = SWEEP_OUT_BASE / args.demo
    rows: list[dict] = []

    for max_new_tokens in args.tokens:
        row = _run_one(args.demo, max_new_tokens, sweep_dir)
        rows.append(row)
        # Print incremental result after each run
        if "error" not in row:
            print(
                f"[sweep] done max_new_tokens={max_new_tokens}: "
                f"TTFT={row['ttft_s']}s  "
                f"decode={row['decode_toks_per_s']} tok/s  "
                f"prefill={row['prefill_toks_per_s']} tok/s",
                flush=True,
            )

    _print_summary(rows)
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
