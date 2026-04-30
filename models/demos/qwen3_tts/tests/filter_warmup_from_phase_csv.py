#!/usr/bin/env python3
"""
Post-process the tracy ops_perf_results CSV from profile_single_layer.py runs.

profile_single_layer.py with --mode both runs:
    prefill (warmup * N + measure * 1)  then  decode (warmup * N + measure * 1)

The qwen3_tts model uses the prefill SDPA kernel for both modes (no SdpaDecode op),
so parse_profiler_report.py's default first-SdpaDecode split heuristic does NOT
apply. Instead we split by INPUT_0 logical Y dimension:
    prefill ops have Y == prefill_seq_len (e.g., 128)
    decode  ops have Y == 1
Some ops (weight loads, host ops) may have other Y values; we use a sliding
majority vote to find the prefill→decode transition.

Within each phase, all (warmup + measure) iterations emit the same op sequence,
so the last `measure / (warmup + measure)` fraction of rows = the measurement
iteration ops.

Outputs per-op CSVs preserving every column from the raw report so you have full
device-kernel detail (DEVICE KERNEL DURATION, CORE COUNT, shapes, etc.).

Usage:
    python filter_warmup_from_phase_csv.py <raw_ops_perf_csv> \\
        --prefill-seq-len 128 --warmup 2 --measure 1 --output-dir perf_out/
"""

import argparse
import csv
from pathlib import Path


def _y_pad(row: dict) -> int:
    """Extract the logical Y dim. Tracy emits 'padded[logical]' (e.g. '32[1]')."""
    raw = (row.get("INPUT_0_Y_PAD[LOGICAL]", "") or "").strip()
    if not raw:
        return -1
    if "[" in raw and "]" in raw:
        inner = raw[raw.index("[") + 1 : raw.index("]")]
        try:
            return int(inner)
        except ValueError:
            pass
    try:
        return int(raw)
    except ValueError:
        return -1


def partition_phases(
    rows: list[dict], prefill_seq_len: int, decode_seq_len: int = 1
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows into (setup, prefill, decode) by INPUT_0 logical Y dim.

    setup   = leading rows whose Y matches neither prefill nor decode
              (one-time weight tilizes, etc. — emitted only on first call).
    prefill = consecutive rows with Y == prefill_seq_len (after setup).
    decode  = consecutive rows with Y == decode_seq_len (after prefill).

    This relies on profile_single_layer.py running prefill BEFORE decode and
    using a prefill seq_len distinct from the decode seq_len (default 1).
    """
    n = len(rows)
    i = 0
    # 1. Skip setup
    while i < n:
        y = _y_pad(rows[i])
        if y == prefill_seq_len or y == decode_seq_len:
            break
        i += 1
    setup = rows[:i]
    # 2. Collect prefill
    prefill_start = i
    while i < n:
        y = _y_pad(rows[i])
        if y == prefill_seq_len:
            i += 1
        elif y == decode_seq_len:
            break
        else:
            # Stray non-phase op inside prefill (rare). Keep walking until decode.
            i += 1
    prefill = rows[prefill_start:i]
    # 3. Decode = remainder
    decode = rows[i:]
    return setup, prefill, decode


def trim_warmup(rows: list[dict], warmup: int, measure: int, label: str) -> list[dict]:
    if not rows:
        print(f"  {label}: empty")
        return rows
    total_iters = warmup + measure
    if len(rows) % total_iters != 0:
        print(
            f"  WARNING [{label}]: {len(rows)} rows is not divisible by "
            f"warmup+measure={total_iters} — ops/iter may not be constant. "
            f"Using floor division."
        )
    ops_per_iter = len(rows) // total_iters
    keep = ops_per_iter * measure
    return rows[-keep:] if keep > 0 else []


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict], label: str) -> None:
    total_ns = 0.0
    for r in rows:
        try:
            total_ns += float(r.get("DEVICE KERNEL DURATION [ns]", 0) or 0)
        except ValueError:
            pass
    print(f"  {label}: {len(rows)} ops, total DEVICE KERNEL DURATION = {total_ns/1000:.2f} us")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv", type=Path)
    ap.add_argument("--prefill-seq-len", type=int, default=128)
    ap.add_argument("--decode-seq-len", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measure", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=Path("perf_out"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.raw_csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"Read {len(rows)} ops from {args.raw_csv}")

    setup, prefill_all, decode_all = partition_phases(rows, args.prefill_seq_len, args.decode_seq_len)
    print(
        f"Setup (one-time weight tilizes etc.): {len(setup)} ops dropped; "
        f"prefill={len(prefill_all)} ops; decode={len(decode_all)} ops"
    )

    print("\nDropping warmup iterations (warmup={}, measure={})...".format(args.warmup, args.measure))
    prefill_meas = trim_warmup(prefill_all, args.warmup, args.measure, "prefill")
    decode_meas = trim_warmup(decode_all, args.warmup, args.measure, "decode")

    pref_path = args.output_dir / "prefill_one_layer.csv"
    dec_path = args.output_dir / "decode_one_layer.csv"
    write_csv(prefill_meas, pref_path, fieldnames)
    write_csv(decode_meas, dec_path, fieldnames)

    print("\nSummary (1-layer measurement iter, all warmup dropped):")
    summarize(prefill_meas, f"prefill -> {pref_path}")
    summarize(decode_meas, f"decode  -> {dec_path}")


if __name__ == "__main__":
    main()
