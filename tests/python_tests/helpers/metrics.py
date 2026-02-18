# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance metrics calculation from hardware counter data.

Metrics calculated:
- FPU Utilization: FPU_INSTRUCTION / cycles
- SFPU Utilization: SFPU_INSTRUCTION / cycles
- Math Utilization: FPU_OR_SFPU_INSTRN / cycles (combined FPU+SFPU)
- Unpacker0 Utilization: SRCA_WRITE / UNPACK0_BUSY_THREAD0
- Unpacker1 Utilization: SRCB_WRITE / UNPACK1_BUSY_THREAD0
- Packer Utilization: PACKER_BUSY / cycles
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

    # Get max cycle counts from each bank (wall clock)
    cycles_fpu = _max_cycles(df, "FPU")
    cycles_unpack = _max_cycles(df, "TDMA_UNPACK")
    cycles_pack = _max_cycles(df, "TDMA_PACK")

    # Use max cycles as wall clock reference
    wall_cycles = max(cycles_fpu, cycles_unpack, cycles_pack, 1)

    # === FPU/SFPU/Math Utilization ===
    # Average instruction counts across threads / cycles
    fpu_count = _avg_count(df, "FPU", "FPU_INSTRUCTION")
    sfpu_count = _avg_count(df, "FPU", "SFPU_INSTRUCTION")
    math_count = _avg_count(df, "FPU", "FPU_OR_SFPU_INSTRN")

    fpu_util = _safe_div(fpu_count, cycles_fpu)
    sfpu_util = _safe_div(sfpu_count, cycles_fpu)
    math_util = _safe_div(math_count, cycles_fpu)

    # === Unpacker Utilization ===
    # Average writes and busy cycles across all threads, then compute ratio
    srca_write = _avg_count(df, "TDMA_UNPACK", "SRCA_WRITE")
    srcb_write = _avg_count(df, "TDMA_UNPACK", "SRCB_WRITE")
    unpack0_busy = _avg_count(df, "TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    unpack1_busy = _avg_count(df, "TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")

    unpack0_util = _safe_div(srca_write, unpack0_busy)
    unpack1_util = _safe_div(srcb_write, unpack1_busy)

    # Combined unpacker utilization (average of both)
    if unpack0_util is not None and unpack1_util is not None:
        unpack_util = (unpack0_util + unpack1_util) / 2.0
    elif unpack0_util is not None:
        unpack_util = unpack0_util
    elif unpack1_util is not None:
        unpack_util = unpack1_util
    else:
        unpack_util = None

    # === Packer Utilization ===
    # Average PACKER_BUSY / cycles
    packer_busy = _avg_count(df, "TDMA_PACK", "PACKER_BUSY")
    pack_util = _safe_div(packer_busy, cycles_pack)

    return {
        # Cycle counts
        "wall_cycles": wall_cycles,
        "fpu_cycles": cycles_fpu,
        "unpack_cycles": cycles_unpack,
        "pack_cycles": cycles_pack,
        # Raw counts
        "fpu_count": fpu_count,
        "sfpu_count": sfpu_count,
        "math_count": math_count,
        "srca_write_count": srca_write,
        "srcb_write_count": srcb_write,
        "unpack0_busy_count": unpack0_busy,
        "unpack1_busy_count": unpack1_busy,
        "packer_busy_count": packer_busy,
        # Utilization ratios (0.0 - 1.0+)
        "fpu_util": fpu_util,
        "sfpu_util": sfpu_util,
        "math_util": math_util,
        "unpack0_util": unpack0_util,
        "unpack1_util": unpack1_util,
        "unpack_util": unpack_util,
        "pack_util": pack_util,
        # Utilization percentages (0.0 - 100.0+)
        "fpu_util_pct": _pct(fpu_util),
        "sfpu_util_pct": _pct(sfpu_util),
        "math_util_pct": _pct(math_util),
        "unpack0_util_pct": _pct(unpack0_util),
        "unpack1_util_pct": _pct(unpack1_util),
        "unpack_util_pct": _pct(unpack_util),
        "pack_util_pct": _pct(pack_util),
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
    print("  CYCLE COUNTS")
    print(f"{'─' * 70}")
    print(f"  {'Wall Cycles:':<30} {metrics['wall_cycles']:>15,}")
    print(f"  {'FPU Bank Cycles:':<30} {metrics['fpu_cycles']:>15,}")
    print(f"  {'Unpack Bank Cycles:':<30} {metrics['unpack_cycles']:>15,}")
    print(f"  {'Pack Bank Cycles:':<30} {metrics['pack_cycles']:>15,}")

    print(f"\n{'─' * 70}")
    print("  COMPUTE UTILIZATION (instructions / cycles)")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Count':>12} {'Util %':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12}")
    print(
        f"  {'FPU Utilization:':<30} {metrics['fpu_count']:>12.1f} {fmt(metrics['fpu_util_pct']):>11}%"
    )
    print(
        f"  {'SFPU Utilization:':<30} {metrics['sfpu_count']:>12.1f} {fmt(metrics['sfpu_util_pct']):>11}%"
    )
    print(
        f"  {'Math Utilization (FPU+SFPU):':<30} {metrics['math_count']:>12.1f} {fmt(metrics['math_util_pct']):>11}%"
    )

    print(f"\n{'─' * 70}")
    print("  UNPACKER UTILIZATION (writes / busy cycles)")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Writes':>12} {'Busy':>12} {'Ratio':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Unpacker0 (SRCA):':<30} {metrics['srca_write_count']:>12.1f} {metrics['unpack0_busy_count']:>12.1f} {fmt(metrics['unpack0_util']):>12}"
    )
    print(
        f"  {'Unpacker1 (SRCB):':<30} {metrics['srcb_write_count']:>12.1f} {metrics['unpack1_busy_count']:>12.1f} {fmt(metrics['unpack1_util']):>12}"
    )
    print(
        f"  {'Combined Unpacker:':<30} {'':<12} {'':<12} {fmt(metrics['unpack_util']):>12}"
    )

    print(f"\n{'─' * 70}")
    print("  PACKER UTILIZATION (busy / cycles)")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<30} {'Busy':>12} {'Util %':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Packer:':<30} {metrics['packer_busy_count']:>12.1f} {fmt(metrics['pack_util_pct']):>11}%"
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
