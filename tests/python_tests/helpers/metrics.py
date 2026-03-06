# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance metrics calculation from hardware counter data.

Metrics calculated:
- Unpacker Write Efficiency: Measures how efficiently unpackers write data
  * SRCA_WRITE / UNPACK0_BUSY_THREAD0: Fraction of unpacker0 busy cycles spent writing data
  * SRCB_WRITE / UNPACK1_BUSY_THREAD0: Fraction of unpacker1 busy cycles spent writing data
  * Higher ratio (closer to 1.0) = more efficient, less stalling
  * Lower ratio = unpacker busy but not writing (stalled on dependencies)

- Packer Efficiency: Measures how efficiently packer uses destination data
  * PACKER_DEST_READ_AVAILABLE / PACKER_BUSY: Fraction of packer busy time with valid dest data
  * Higher ratio (closer to 1.0) = packer has data available, less dest stalling
  * Lower ratio = packer busy but waiting for destination to become valid
  * Only valid when using HW dvalid-based synchronization (not STALLWAIT)

- FPU Execution Efficiency: Measures how efficiently FPU executes available instructions
  * FPU_INSTRUCTION / FPU_INSTRN_AVAILABLE_1: Fraction of cycles with FPU work available where FPU executes
  * Higher ratio (closer to 1.0) = FPU executes whenever work is available (efficient)
  * Lower ratio = FPU instructions available but not executing (stalled in pipeline)

- Math Pipeline Utilization (EXPERIMENTAL): Measures math pipeline instruction flow efficiency
  * MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE: Fraction of available math instructions that start execution
  * Higher ratio (closer to 1.0) = math pipeline efficiently moves instructions (no pipe stalls)
  * Lower ratio = instructions available in pipe but not starting (pipeline stalled)

- Math-to-Pack Handoff Efficiency (EXPERIMENTAL): Measures pipeline balance between math and pack stages
  * AVAILABLE_MATH / PACKER_BUSY: Fraction of packer busy time where math has output ready
  * Higher ratio (closer to 1.0) = math keeps up with packer, good pipeline balance
  * Lower ratio = packer busy but waiting for math output (math is bottleneck)

- Unpacker-to-Math Data Flow (EXPERIMENTAL): Measures unpacker write availability during busy cycles
  * SRCA_WRITE_AVAILABLE / UNPACK0_BUSY_THREAD0: Fraction of unpacker0 busy time with srcA buffer space
  * SRCB_WRITE_AVAILABLE / UNPACK1_BUSY_THREAD0: Fraction of unpacker1 busy time with srcB buffer space
  * Higher ratio (closer to 1.0) = unpacker can write when busy, no backpressure from math
  * Lower ratio = unpacker busy but srcA/B full, stalled by math not consuming data fast enough
"""

import pandas as pd
from helpers.test_config import TestConfig


def _sum_count(df: pd.DataFrame, bank: str, counter_name: str) -> int:
    """Sum count for a specific counter across all threads."""
    mask = (df["bank"] == bank) & (df["counter_name"] == counter_name)
    return int(df.loc[mask, "count"].sum())


def _avg_count(df: pd.DataFrame, bank: str, counter_name: str) -> float:
    """Average count for a specific counter across all threads."""
    mask = (df["bank"] == bank) & (df["counter_name"] == counter_name)
    result = df.loc[mask, "count"]
    return float(result.mean()) if len(result) > 0 else 0.0


def _max_cycles(df: pd.DataFrame, bank: str) -> int:
    """Get max cycles for a specific bank across all threads."""
    bank_df = df[df["bank"] == bank]
    if bank_df.empty:
        return 0
    return int(bank_df["cycles"].max())


def _safe_div(numerator: float, denominator: float) -> float | None:
    """Safe division returning None if denominator is 0."""
    return (numerator / denominator) if denominator > 0 else None


def _pct(value: float | None) -> float | None:
    """Convert ratio to percentage."""
    return (value * 100.0) if value is not None else None


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute performance metrics from counter DataFrame.

    Args:
        df: DataFrame from read_counters() with columns:
            thread, bank, counter_name, counter_id, cycles, count, l1_mux

    Returns:
        Dictionary of computed metrics.
    """
    if df.empty:
        return {}

    # === Unpacker Write Efficiency ===
    # Ratio of write cycles to busy cycles - measures how efficiently unpacker uses busy time
    # SRCA_WRITE: Cycles where unpacker0 wrote data to srcA
    # UNPACK0_BUSY: Cycles where unpacker0 was busy processing thread 0 instructions
    # Higher ratio (closer to 1.0) means unpacker spends most busy time writing (efficient)
    # Lower ratio means unpacker is busy but not writing (stalled/waiting)
    srca_write = _avg_count(df, "TDMA_UNPACK", "SRCA_WRITE")
    srcb_write = _avg_count(df, "TDMA_UNPACK", "SRCB_WRITE")
    unpack0_busy = _avg_count(df, "TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    unpack1_busy = _avg_count(df, "TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")

    unpack0_efficiency = _safe_div(srca_write, unpack0_busy)
    unpack1_efficiency = _safe_div(srcb_write, unpack1_busy)

    # Combined unpacker efficiency (average of both)
    if unpack0_efficiency is not None and unpack1_efficiency is not None:
        unpack_efficiency = (unpack0_efficiency + unpack1_efficiency) / 2.0
    elif unpack0_efficiency is not None:
        unpack_efficiency = unpack0_efficiency
    elif unpack1_efficiency is not None:
        unpack_efficiency = unpack1_efficiency
    else:
        unpack_efficiency = None

    # === Packer Efficiency ===
    # Ratio of destination-available time to packer busy time
    # PACKER_DEST_READ_AVAILABLE: Cycles where packer could read from destination (data valid)
    # PACKER_BUSY: Cycles where packer pipeline has active instructions
    # Higher ratio (closer to 1.0) means packer has data available when busy (efficient)
    # Lower ratio means packer is busy but waiting for destination data (dest stalling)
    # NOTE: Only valid with HW dvalid synchronization, not with STALLWAIT
    packer_dest_available = _avg_count(df, "TDMA_PACK", "PACKER_DEST_READ_AVAILABLE")
    packer_busy = _avg_count(df, "TDMA_PACK", "PACKER_BUSY")
    pack_efficiency = _safe_div(packer_dest_available, packer_busy)

    # === FPU Execution Efficiency ===
    # Ratio of actual FPU executions to cycles where FPU instructions were available
    # FPU_INSTRUCTION: Actual FPU instructions executed by the math engine
    # FPU_INSTRN_AVAILABLE_1: Cycles where an FPU instruction from MATH thread could start
    # Higher ratio (closer to 1.0) means FPU executes whenever FPU work is available (efficient)
    # Lower ratio means FPU instructions are available but not executing (pipeline stalls)
    fpu_instruction = _avg_count(df, "FPU", "FPU_INSTRUCTION")
    fpu_instrn_available = _avg_count(df, "INSTRN_THREAD", "FPU_INSTRN_AVAILABLE_1")
    fpu_efficiency = _safe_div(fpu_instruction, fpu_instrn_available)

    # === Math Pipeline Utilization (EXPERIMENTAL) ===
    # Ratio of math instructions that started execution to those available in the pipe
    # MATH_INSTRN_STARTED: Instructions read from the math instruction pipe stage and started
    # MATH_INSTRN_AVAILABLE: Cycles where a math instruction was present in the pipe stage
    # Higher ratio (closer to 1.0) means math pipeline efficiently moves instructions (no pipe stalls)
    # Lower ratio means instructions are in pipe but not starting (pipeline stalled)
    math_instrn_started = _avg_count(df, "TDMA_UNPACK", "MATH_INSTRN_STARTED")
    math_instrn_available = _avg_count(df, "TDMA_UNPACK", "MATH_INSTRN_AVAILABLE")
    math_pipeline_util = _safe_div(math_instrn_started, math_instrn_available)

    # === Math-to-Pack Handoff Efficiency (EXPERIMENTAL) ===
    # Ratio of cycles where math output is available to packer busy cycles
    # AVAILABLE_MATH: Cycles where a math instruction was available to the packer
    # PACKER_BUSY: Cycles where packer pipeline has active instructions
    # Higher ratio (closer to 1.0) means math keeps up with packer demand (good pipeline balance)
    # Lower ratio means packer is busy but math output isn't ready (math is bottleneck)
    available_math = _avg_count(df, "TDMA_PACK", "AVAILABLE_MATH")
    math_to_pack_efficiency = _safe_div(available_math, packer_busy)

    # === Unpacker-to-Math Data Flow (EXPERIMENTAL) ===
    # Ratio of cycles where unpacker can write to srcA/srcB buffer vs unpacker busy cycles
    # SRCA_WRITE_AVAILABLE: Cycles where srcA buffer has space for unpacker to write
    # SRCB_WRITE_AVAILABLE: Cycles where srcB buffer has space for unpacker to write
    # UNPACK0/1_BUSY: Cycles where unpacker is actively processing
    # Higher ratio (closer to 1.0) means unpacker can write when busy, no math backpressure
    # Lower ratio means unpacker is busy but buffers are full (math not consuming fast enough)
    srca_write_available = _avg_count(df, "TDMA_UNPACK", "SRCA_WRITE_AVAILABLE")
    srcb_write_available = _avg_count(df, "TDMA_UNPACK", "SRCB_WRITE_AVAILABLE")
    unpack_to_math_flow0 = _safe_div(srca_write_available, unpack0_busy)
    unpack_to_math_flow1 = _safe_div(srcb_write_available, unpack1_busy)

    # Combined unpacker-to-math data flow (average of both)
    if unpack_to_math_flow0 is not None and unpack_to_math_flow1 is not None:
        unpack_to_math_flow = (unpack_to_math_flow0 + unpack_to_math_flow1) / 2.0
    elif unpack_to_math_flow0 is not None:
        unpack_to_math_flow = unpack_to_math_flow0
    elif unpack_to_math_flow1 is not None:
        unpack_to_math_flow = unpack_to_math_flow1
    else:
        unpack_to_math_flow = None

    return {
        # Raw counts
        "srca_write_count": srca_write,
        "srcb_write_count": srcb_write,
        "unpack0_busy_count": unpack0_busy,
        "unpack1_busy_count": unpack1_busy,
        "packer_dest_available_count": packer_dest_available,
        "packer_busy_count": packer_busy,
        "fpu_instruction_count": fpu_instruction,
        "fpu_instrn_available_count": fpu_instrn_available,
        "math_instrn_started_count": math_instrn_started,
        "math_instrn_available_count": math_instrn_available,
        "available_math_count": available_math,
        "srca_write_available_count": srca_write_available,
        "srcb_write_available_count": srcb_write_available,
        # Efficiency ratios (0.0 - 1.0)
        "unpack0_efficiency": unpack0_efficiency,
        "unpack1_efficiency": unpack1_efficiency,
        "unpack_efficiency": unpack_efficiency,
        "pack_efficiency": pack_efficiency,
        "fpu_efficiency": fpu_efficiency,
        "math_pipeline_util": math_pipeline_util,
        "math_to_pack_efficiency": math_to_pack_efficiency,
        "unpack_to_math_flow0": unpack_to_math_flow0,
        "unpack_to_math_flow1": unpack_to_math_flow1,
        "unpack_to_math_flow": unpack_to_math_flow,
        # Efficiency percentages (0.0 - 100.0)
        "unpack0_efficiency_pct": _pct(unpack0_efficiency),
        "unpack1_efficiency_pct": _pct(unpack1_efficiency),
        "unpack_efficiency_pct": _pct(unpack_efficiency),
        "pack_efficiency_pct": _pct(pack_efficiency),
        "fpu_efficiency_pct": _pct(fpu_efficiency),
        "math_pipeline_util_pct": _pct(math_pipeline_util),
        "math_to_pack_efficiency_pct": _pct(math_to_pack_efficiency),
        "unpack_to_math_flow0_pct": _pct(unpack_to_math_flow0),
        "unpack_to_math_flow1_pct": _pct(unpack_to_math_flow1),
        "unpack_to_math_flow_pct": _pct(unpack_to_math_flow),
    }


def print_metrics(results: pd.DataFrame) -> None:
    """Print performance metrics to console."""
    metrics = compute_metrics(results)

    if not metrics:
        print("No metrics to display.")
        return

    def fmt(value, decimals=2):
        """Format a value, returning 'N/A' for None."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    print(f"\n{'─' * 70}")
    print("  UNPACKER WRITE EFFICIENCY")
    print(f"{'─' * 70}")
    print("  Measures the fraction of unpacker busy cycles spent writing data.")
    print("  Higher ratio (→1.0) = efficient, unpacker writes when busy")
    print("  Lower ratio (→0.0) = inefficient, unpacker busy but stalled/waiting")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Writes':>12} {'Busy':>12} {'Efficiency':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Unpacker0 (SRCA):':<30} {metrics['srca_write_count']:>12.1f} {metrics['unpack0_busy_count']:>12.1f} {fmt(metrics['unpack0_efficiency']):>12}"
    )
    print(
        f"  {'Unpacker1 (SRCB):':<30} {metrics['srcb_write_count']:>12.1f} {metrics['unpack1_busy_count']:>12.1f} {fmt(metrics['unpack1_efficiency']):>12}"
    )
    print(
        f"  {'Combined Average:':<30} {'':<12} {'':<12} {fmt(metrics['unpack_efficiency']):>12}"
    )

    print(f"\n{'─' * 70}")
    print("  PACKER EFFICIENCY")
    print(f"{'─' * 70}")
    print("  Measures the fraction of packer busy cycles with valid dest data.")
    print("  Higher ratio (→1.0) = efficient, packer has data when busy")
    print("  Lower ratio (→0.0) = inefficient, packer stalled waiting for dest")
    print("  NOTE: Only valid with HW dvalid sync (not STALLWAIT)")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Dest Avail':>12} {'Busy':>12} {'Efficiency':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Packer:':<30} {metrics['packer_dest_available_count']:>12.1f} {metrics['packer_busy_count']:>12.1f} {fmt(metrics['pack_efficiency']):>12}"
    )

    print(f"\n{'─' * 70}")
    print("  FPU EXECUTION EFFICIENCY")
    print(f"{'─' * 70}")
    print("  Measures the fraction of FPU-available cycles where FPU executes.")
    print("  Higher ratio (→1.0) = efficient, FPU executes when work available")
    print("  Lower ratio (→0.0) = inefficient, FPU stalled despite available work")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'FPU Instr':>12} {'Available':>12} {'Efficiency':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Math FPU:':<30} {metrics['fpu_instruction_count']:>12.1f} {metrics['fpu_instrn_available_count']:>12.1f} {fmt(metrics['fpu_efficiency']):>12}"
    )

    print(f"\n{'─' * 70}")
    print("  MATH PIPELINE UTILIZATION (EXPERIMENTAL)")
    print(f"{'─' * 70}")
    print("  Measures math pipeline instruction flow efficiency.")
    print("  Higher ratio (→1.0) = efficient, pipeline moves instructions smoothly")
    print("  Lower ratio (→0.0) = inefficient, pipeline stalls despite available work")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Started':>12} {'Available':>12} {'Utilization':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Math Pipeline:':<30} {metrics['math_instrn_started_count']:>12.1f} {metrics['math_instrn_available_count']:>12.1f} {fmt(metrics['math_pipeline_util']):>12}"
    )

    print(f"\n{'─' * 70}")
    print("  MATH-TO-PACK HANDOFF EFFICIENCY (EXPERIMENTAL)")
    print(f"{'─' * 70}")
    print("  Measures pipeline balance between math and pack stages.")
    print("  Higher ratio (→1.0) = efficient, math keeps up with packer demand")
    print("  Lower ratio (→0.0) = inefficient, packer waits for math output")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Math Avail':>12} {'Pack Busy':>12} {'Efficiency':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Math→Pack:':<30} {metrics['available_math_count']:>12.1f} {metrics['packer_busy_count']:>12.1f} {fmt(metrics['math_to_pack_efficiency']):>12}"
    )

    print(f"\n{'─' * 70}")
    print("  UNPACKER-TO-MATH DATA FLOW (EXPERIMENTAL)")
    print(f"{'─' * 70}")
    print("  Measures unpacker write availability during busy cycles.")
    print("  Higher ratio (→1.0) = efficient, unpacker can write (no backpressure)")
    print("  Lower ratio (→0.0) = inefficient, buffers full (math not consuming)")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Buf Avail':>12} {'Busy':>12} {'Data Flow':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Unpacker0→Math (srcA):':<30} {metrics['srca_write_available_count']:>12.1f} {metrics['unpack0_busy_count']:>12.1f} {fmt(metrics['unpack_to_math_flow0']):>12}"
    )
    print(
        f"  {'Unpacker1→Math (srcB):':<30} {metrics['srcb_write_available_count']:>12.1f} {metrics['unpack1_busy_count']:>12.1f} {fmt(metrics['unpack_to_math_flow1']):>12}"
    )
    print(
        f"  {'Combined Average:':<30} {'':<12} {'':<12} {fmt(metrics['unpack_to_math_flow']):>12}"
    )

    print("\n" + "=" * 70 + "\n")


def export_metrics(
    results: pd.DataFrame,
    filename: str,
    test_params: dict = None,
    worker_id: str = "gw0",
) -> None:
    """Export metrics to CSV file in perf_data directory."""
    perf_dir = TestConfig.LLK_ROOT / "perf_data"
    perf_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(results)

    if not metrics:
        return

    if test_params:
        metrics.update(test_params)

    df = pd.DataFrame([metrics])
    output_path = perf_dir / f"{filename}.{worker_id}.csv"

    if output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(output_path, index=False)
