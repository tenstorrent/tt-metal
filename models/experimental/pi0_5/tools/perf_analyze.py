#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile -> bucket -> suggest-lever analyzer for pi0.5 perf runs.

Implements the playbook recipe at PERF_PLAYBOOKS/09_PROFILING_AND_OP_ANALYSIS.md
sections 3-5 plus the per-bucket lever suggestions from sections 1-8 of files
02..06. Reads a tracy-generated ops_perf_results_*.csv and emits:

  1. Total device kernel time (the headline metric)
  2. Op-family rollup (by OP CODE; both top-by-time AND top-by-count)
  3. Matmul subgroup breakdown by (M, K, N) -> "which matmul is which"
  4. Suggested levers per top bucket, citing the playbook section

This replaces the "manually stare at the CSV" workflow with one command.
Targeted at the pi0.5 model but the bucketing logic is model-agnostic.

Usage:
    python -m models.experimental.pi0_5.tools.perf_analyze \\
        generated/profiler/reports/<TIMESTAMP>/ops_perf_results_<TIMESTAMP>.csv

Optional flags:
    --top N            top-N buckets to show in each section (default 10)
    --target-ms FLOAT  total kernel target; prints gap (default 50.0)
    --shape OP_CODE    drill into one op family's shape breakdown
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Playbook-derived per-bucket levers. Keep concise: file:§ + lever sentence.
LEVERS: Dict[str, List[str]] = {
    "MatmulDeviceOperation": [
        "05 §5: re-sweep subblock with `fp32_dest_acc_en=False` (cap 4->8). Wide-N (FF1, attn-out) wants (1,8); wide-K (FF2, o_proj) wants (8,1).",
        "05 §6: fidelity walk HiFi2->LoFi for bf8b weights (BGE-M3 saw -6 ms FF1, -2 ms FF2).",
        "05 §9: full sweep -> PCC gate -> in-model validate top 2-3.",
        "03 §3: tune in0_block_w to divisors of K_tiles; prefer 4 or 8.",
    ],
    "SDPAOperation": [
        "04 §5: audit Q/K/V dtype path for hidden typecasts (BGE trap: -13.7 ms).",
        "04 §3: pick q_chunk/k_chunk per shape; prefer single-chunk K when Skv fits.",
        "04 §6: exp_approx_mode=False is faster on BH at S=512.",
        "04 §7: fp32_dest_acc_en=True for SDPA (softmax sum needs fp32).",
    ],
    "LayerNormDeviceOperation": [
        "02 §4: HiFi2 + fp32_dest_acc_en=True (reduction op, depth-compounding error).",
        "02 §6: fuse residual via residual_input_tensor (eliminates 1 BinaryNg per norm).",
        "02 §5: bf8b activation through sharded reductions compounds PCC; use bf16 inputs.",
        "02 §7: chain into next op's shard config (avoid I->S/S->I round trips).",
    ],
    "RotaryEmbeddingDeviceOperation": [
        "08 §6: fused-QK-RoPE is decode-only upstream (TT_FATAL on seq>1). Kernel-floor reached at our shape.",
    ],
    "NlpCreateHeadsDeviceOperation": [
        "03 §5: MQA work-dist heuristic caps at 2 cores when num_kv_heads=1; upstream-blocked.",
    ],
    "BinaryNgDeviceOperation": [
        "06 §1: fold residual ADD into upstream norm (residual_input_tensor).",
        "06 §2: fold activation MUL/ADD into upstream matmul (fused_activation).",
        "06 §4: collapse adjacent unaries via ttnn.unary_chain.",
        "09 §4(b): big count + tiny us/call = canonical fusion target.",
    ],
    "ConcatDeviceOperation": [
        "08 §3: replace KV-cache concat with ttnn.experimental.paged_update_cache (in-place; eliminates the concat).",
    ],
    "ShardedToInterleavedDeviceOperation": [
        "06 §3: fuse dtype cast INTO the reshard via output_dtype=.",
        "06 §7: cross-block sharded handoff eliminates the I->S round-trip at block boundaries.",
    ],
    "InterleavedToShardedDeviceOperation": [
        "06 §3: fuse dtype cast INTO the reshard via output_dtype=.",
        "06 §7: cross-block sharded handoff eliminates the I->S round-trip at block boundaries.",
    ],
    "TypecastDeviceOperation": [
        "06 §3: fuse cast into adjacent reshard/matmul/norm via output_dtype= (eliminate the standalone op).",
        "04 §5: Q/K/V typecast before SDPA is the BGE-M3 -13.7 ms trap; remove it.",
    ],
    "NLPConcatHeadsDeviceOperation": [
        "03 §5b: encoder paths often emit this between SDPA and o_proj; consider folding into the o_proj layout.",
    ],
}


def _kdur_ns(row: dict) -> float:
    """Device kernel duration in ns; 0 if missing."""
    try:
        return float(row.get("DEVICE KERNEL DURATION [ns]", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _shape_key(row: dict) -> Tuple[str, str, str]:
    """(M, K, N) from INPUT_0_Y_PAD, INPUT_0_X_PAD, INPUT_1_X_PAD."""
    return (
        row.get("INPUT_0_Y_PAD[LOGICAL]", ""),
        row.get("INPUT_0_X_PAD[LOGICAL]", ""),
        row.get("INPUT_1_X_PAD[LOGICAL]", ""),
    )


def _strip_pad(s: str) -> str:
    """'512[512]' -> '512'."""
    return s.split("[")[0] if "[" in s else s


def _bucket_by_opcode(rows: List[dict]) -> Dict[str, Tuple[int, float]]:
    """{op_code: (count, total_ns)}."""
    out: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])
    for r in rows:
        code = r.get("OP CODE", "?")
        out[code][0] += 1
        out[code][1] += _kdur_ns(r)
    return {k: (v[0], v[1]) for k, v in out.items()}


def _bucket_matmul_by_shape(rows: List[dict]) -> Dict[Tuple[str, str, str], Tuple[int, float, List[str]]]:
    """{(M, K, N): (count, total_ns, cores_list)}."""
    out: Dict[Tuple[str, str, str], List] = defaultdict(lambda: [0, 0.0, []])
    for r in rows:
        if r.get("OP CODE") != "MatmulDeviceOperation":
            continue
        k = _shape_key(r)
        out[k][0] += 1
        out[k][1] += _kdur_ns(r)
        cc = r.get("CORE COUNT", "")
        if cc and cc not in out[k][2]:
            out[k][2].append(cc)
    return {k: (v[0], v[1], v[2]) for k, v in out.items()}


def _print_section(title: str):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv_path", type=Path, help="Path to ops_perf_results_*.csv")
    ap.add_argument("--top", type=int, default=10, help="Top-N buckets to show (default 10)")
    ap.add_argument("--target-ms", type=float, default=50.0, help="Total kernel target (default 50.0)")
    ap.add_argument("--shape", type=str, default=None, help="Drill into one OP CODE's shape breakdown")
    ap.add_argument(
        "--neighbors",
        type=str,
        default=None,
        help="OP CODE to neighbor-analyze (per 09 §5). Prints predecessor/successor "
        "patterns and fold-candidate buckets. E.g. --neighbors BinaryNgDeviceOperation",
    )
    args = ap.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: {args.csv_path} not found", file=sys.stderr)
        sys.exit(1)

    with args.csv_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("ERROR: CSV is empty", file=sys.stderr)
        sys.exit(1)

    total_ns = sum(_kdur_ns(r) for r in rows)
    total_ms = total_ns / 1e6

    _print_section(f"HEADLINE — {args.csv_path.name}")
    print(f"  Total ops:                {len(rows)}")
    print(f"  Total device kernel:      {total_ms:.3f} ms")
    print(f"  Target:                   <{args.target_ms:.1f} ms")
    print(f"  Gap to target:            {total_ms - args.target_ms:+.3f} ms")

    # === Section 1: op-family rollup, BOTH rankings ===
    opcode_buckets = _bucket_by_opcode(rows)

    _print_section(f"TOP-{args.top} BY DEVICE KERNEL TIME (per 09 §3)")
    print(f"  {'rank':<4}  {'op_code':<40}  {'calls':>5}  {'total ms':>9}  {'µs/call':>8}")
    by_time = sorted(opcode_buckets.items(), key=lambda kv: -kv[1][1])
    for i, (code, (n, t)) in enumerate(by_time[: args.top], 1):
        avg = t / n if n else 0
        pct = (t / total_ns * 100) if total_ns else 0
        print(f"  {i:<4}  {code:<40}  {n:>5}  {t/1e6:>9.3f}  {avg/1e3:>8.2f}  ({pct:>4.1f}%)")

    _print_section(f"TOP-{args.top} BY CALL COUNT (per 09 §4(b))")
    print(f"  Big count + tiny µs/call = canonical fusion target (06 §4).")
    print(f"  {'rank':<4}  {'op_code':<40}  {'calls':>5}  {'total ms':>9}  {'µs/call':>8}")
    by_count = sorted(opcode_buckets.items(), key=lambda kv: -kv[1][0])
    for i, (code, (n, t)) in enumerate(by_count[: args.top], 1):
        avg = t / n if n else 0
        print(f"  {i:<4}  {code:<40}  {n:>5}  {t/1e6:>9.3f}  {avg/1e3:>8.2f}")

    # === Section 2: matmul subgroup breakdown (per 09 §5) ===
    mm_buckets = _bucket_matmul_by_shape(rows)
    if mm_buckets:
        _print_section(f"MATMUL SHAPES (per 09 §5)")
        print(f"  {'M':>5}  {'K':>6}  {'N':>6}  {'calls':>5}  {'total ms':>9}  {'µs/call':>8}  {'cores':<15}")
        by_mm_time = sorted(mm_buckets.items(), key=lambda kv: -kv[1][1])
        for (m, k, n), (cnt, t, cores) in by_mm_time[: args.top]:
            avg = t / cnt if cnt else 0
            print(
                f"  {_strip_pad(m):>5}  {_strip_pad(k):>6}  {_strip_pad(n):>6}  {cnt:>5}  {t/1e6:>9.3f}  {avg/1e3:>8.2f}  {','.join(cores[:3]):<15}"
            )

    # === Section 3: shape drill-down (per 09 §5) ===
    if args.shape:
        target_rows = [r for r in rows if r.get("OP CODE") == args.shape]
        if not target_rows:
            print(f"\nNo rows for OP CODE = {args.shape}")
        else:
            shape_groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for r in target_rows:
                y = r.get("INPUT_0_Y_PAD[LOGICAL]", "")
                x = r.get("INPUT_0_X_PAD[LOGICAL]", "")
                shape_groups[(y, x)].append(_kdur_ns(r))
            _print_section(f"DRILL: {args.shape} by (input Y, X)")
            print(f"  {'Y':>10}  {'X':>10}  {'calls':>5}  {'total ms':>9}  {'µs/call':>8}")
            for (y, x), kdurs in sorted(shape_groups.items(), key=lambda kv: -sum(kv[1])):
                t = sum(kdurs)
                print(
                    f"  {_strip_pad(y):>10}  {_strip_pad(x):>10}  {len(kdurs):>5}  {t/1e6:>9.3f}  {t/len(kdurs)/1e3:>8.2f}"
                )

    # === Section 4: suggested levers per top bucket ===
    _print_section(f"SUGGESTED LEVERS (per playbook PERF_PLAYBOOKS/)")
    for i, (code, (n, t)) in enumerate(by_time[: min(5, args.top)], 1):
        levers = LEVERS.get(code)
        if not levers:
            continue
        print(f"\n  #{i} {code} — {t/1e6:.3f} ms ({n} calls, {t/n/1e3:.2f} µs/call):")
        for lev in levers:
            print(f"     • {lev}")

    # === Section 4.5: neighbor analysis (per 09 §5) ===
    if args.neighbors:
        _print_section(f"NEIGHBOR ANALYSIS: {args.neighbors} (per 09 §5)")

        # Sort by GLOBAL CALL COUNT to get execution order
        def _call_count(r):
            try:
                return int(r.get("GLOBAL CALL COUNT", 0) or 0)
            except (TypeError, ValueError):
                return 0

        ordered = sorted(rows, key=_call_count)
        # For each target call, record (prev_opcode, next_opcode, shape, ns)
        from collections import Counter as _C

        per_shape: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = defaultdict(list)
        for i, r in enumerate(ordered):
            if r.get("OP CODE") != args.neighbors:
                continue
            y = _strip_pad(r.get("INPUT_0_Y_PAD[LOGICAL]", ""))
            x = _strip_pad(r.get("INPUT_0_X_PAD[LOGICAL]", ""))
            prev = ordered[i - 1].get("OP CODE", "—") if i > 0 else "—"
            nxt = ordered[i + 1].get("OP CODE", "—") if i + 1 < len(ordered) else "—"
            per_shape[(y, x)].append((prev, nxt, _kdur_ns(r)))

        if not per_shape:
            print(f"  No rows for OP CODE = {args.neighbors}")
        else:
            total_t = sum(sum(c[2] for c in calls) for calls in per_shape.values())
            total_n = sum(len(calls) for calls in per_shape.values())
            print(f"  Total: {total_n} calls, {total_t/1e6:.3f} ms\n")
            print(
                f"  {'Y':>6} {'X':>6}  {'calls':>5}  {'total ms':>9}  {'µs/call':>8}  {'top predecessors':<40}  {'top successors':<40}"
            )
            print("  " + "-" * 124)
            for (y, x), calls in sorted(per_shape.items(), key=lambda kv: -sum(c[2] for c in kv[1])):
                n = len(calls)
                t = sum(c[2] for c in calls)
                prev_ctr = _C(c[0] for c in calls).most_common(2)
                next_ctr = _C(c[1] for c in calls).most_common(2)
                preds = ", ".join(f"{p[:24]}×{c}" for p, c in prev_ctr)
                succs = ", ".join(f"{p[:24]}×{c}" for p, c in next_ctr)
                print(f"  {y:>6} {x:>6}  {n:>5}  {t/1e6:>9.3f}  {t/n/1e3:>8.2f}  {preds:<40}  {succs:<40}")

            # Pattern classification — only meaningful for BinaryNg, but no harm
            # printing for others.
            print("\n  Fold pattern hits (per 06):")
            ln_prev = mm_prev = bg_prev = ln_next = 0
            for calls in per_shape.values():
                for prev, nxt, _ in calls:
                    if prev == "LayerNormDeviceOperation":
                        ln_prev += 1
                    elif prev == "MatmulDeviceOperation":
                        mm_prev += 1
                    elif prev == args.neighbors:
                        bg_prev += 1
                    if nxt == "LayerNormDeviceOperation":
                        ln_next += 1
            print(f"    LN → {args.neighbors[:14]}    : {ln_prev:>5} calls  (post-norm; usually fine)")
            print(
                f"    {args.neighbors[:14]} → LN    : {ln_next:>5} calls  (UNFUSED residual ADD candidates — 06 §1 / 02 §6)"
            )
            print(f"    Matmul → {args.neighbors[:14]}: {mm_prev:>5} calls  (fused_activation candidates — 06 §2)")
            print(
                f"    {args.neighbors[:8]} → {args.neighbors[:8]}: {bg_prev:>5} calls  (unary_chain candidates — 06 §4)"
            )

    # === Section 5: sanity checks (per 09 §7) ===
    _print_section(f"SANITY CHECKS (per 09 §7)")
    n_mm = opcode_buckets.get("MatmulDeviceOperation", (0, 0))[0]
    n_sdpa = opcode_buckets.get("SDPAOperation", (0, 0))[0]
    n_ln = opcode_buckets.get("LayerNormDeviceOperation", (0, 0))[0]
    print(f"  Matmuls: {n_mm:>5}   SDPA: {n_sdpa:>4}   LayerNorms: {n_ln:>4}")
    if n_sdpa > 0:
        ratio_mm = n_mm / n_sdpa
        ratio_ln = n_ln / n_sdpa
        print(f"  Matmul/SDPA ratio: {ratio_mm:.1f}× (LLM expects ~4×, encoder ~5-6×)")
        print(f"  LN/SDPA ratio:     {ratio_ln:.1f}× (expects ~2×; >3× hints at unfused residual adds)")

    print()


if __name__ == "__main__":
    main()
