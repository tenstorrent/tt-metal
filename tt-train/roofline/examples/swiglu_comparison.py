#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU MLP implementation comparison for roofline analysis.

This script compares the performance of different SwiGLU MLP implementations:
1. MockLlamaMLP - Non-fused (separate matmuls + elementwise ops)
2. MockLlamaMLPFused (ROW_MCAST) - Fused with row multicast (weights read multiple times)
3. MockLlamaMLPFused (MCAST) - Fused with optimal multicast (everything read once)

Uses Llama model shapes: TinyLlama, 1B, 8B, and 405B.

Run from tt-train directory:
    python3 -m roofline.examples.swiglu_comparison
    python3 -m roofline.examples.swiglu_comparison --hardware p100
    python3 -m roofline.examples.swiglu_comparison --hardware n300 --batch 4 --seq 2048
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass
class LlamaMLPConfig:
    """Configuration for Llama MLP layer."""

    name: str
    embedding_size: int
    intermediate_dim: int
    description: str


# Llama model configurations (MLP dimensions)
LLAMA_CONFIGS: Dict[str, LlamaMLPConfig] = {
    "tinyllama": LlamaMLPConfig(
        name="TinyLlama 1.1B",
        embedding_size=2048,
        intermediate_dim=5632,  # Computed: int(4 * 2048 * 2/3) rounded to 256
        description="TinyLlama 1.1B (22 layers)",
    ),
    "llama-1b": LlamaMLPConfig(
        name="Llama 3.2 1B",
        embedding_size=2048,
        intermediate_dim=8192,
        description="Llama 3.2 1B (16 layers)",
    ),
    "llama-8b": LlamaMLPConfig(
        name="Llama 3.1 8B",
        embedding_size=4096,
        intermediate_dim=14336,
        description="Llama 3.1 8B (32 layers)",
    ),
    "llama-405b": LlamaMLPConfig(
        name="Llama 3.1 405B",
        embedding_size=16384,
        intermediate_dim=53248,
        description="Llama 3.1 405B (126 layers)",
    ),
}


def run_comparison(
    hardware: str = "n150",
    batch_size: int = 1,
    seq_len: int = 2048,
    output_plot: str = "swiglu_roofline.png",
):
    """Run SwiGLU MLP comparison and generate roofline plot."""
    from roofline import (
        MockTensor,
        RooflineContext,
        WORMHOLE_N150,
        WORMHOLE_N300,
        BLACKHOLE_P100,
        BLACKHOLE_P150,
        DataType,
        MathFidelity,
    )
    from roofline.modules import MockLlamaMLP, MockLlamaMLPFused
    from roofline.operations.swiglu_fused import SwiGLUFusedImpl

    # Hardware mapping
    hardware_map = {
        "n150": WORMHOLE_N150,
        "n300": WORMHOLE_N300,
        "p100": BLACKHOLE_P100,
        "p150": BLACKHOLE_P150,
    }

    if hardware not in hardware_map:
        print(f"Unknown hardware: {hardware}")
        print(f"Available hardware: {', '.join(hardware_map.keys())}")
        return

    hw_spec = hardware_map[hardware]

    print("=" * 90)
    print("SWIGLU MLP IMPLEMENTATION COMPARISON (Forward Pass)")
    print("=" * 90)
    print()
    print(f"Hardware: {hw_spec.name}")
    print(f"  Cores:       {hw_spec.tensix_cores_per_chip}")
    print(f"  Clock:       {hw_spec.clock_ghz} GHz")
    print(f"  DRAM BW:     {hw_spec.dram_bw_gb_s} GB/s")
    print(f"  Peak (HiFi4): {hw_spec.tflops_per_chip(MathFidelity.HiFi4):.1f} TFLOPs")
    print()
    print(f"Batch Configuration:")
    print(f"  batch_size:  {batch_size}")
    print(f"  seq_len:     {seq_len}")
    print()

    # Results storage for plotting
    all_results: List[Dict] = []

    # Table header
    print("-" * 130)
    print(
        f"{'Model':<15} {'Implementation':<25} {'Compute(ms)':<12} {'Memory(ms)':<12} "
        f"{'Total(ms)':<12} {'Speedup':<10} {'Bottleneck':<12} {'AI(F/B)':<10}"
    )
    print("-" * 130)

    for config_name, config in LLAMA_CONFIGS.items():
        # Create input tensor
        x = MockTensor(
            (batch_size, 1, seq_len, config.embedding_size),
            dtype=DataType.BFLOAT16,
            requires_grad=True,
        )

        # Results for this model
        model_results = {}

        # 1. Non-fused MockLlamaMLP
        mlp_nonfused = MockLlamaMLP(
            embedding_size=config.embedding_size,
            intermediate_dim=config.intermediate_dim,
        )
        ctx_nonfused = RooflineContext(hw_spec)
        _ = mlp_nonfused(ctx_nonfused, x.clone())

        nonfused_compute_ns = sum(e.ideal_compute_ns for e in ctx_nonfused.estimates)
        nonfused_memory_ns = sum(e.ideal_memory_ns for e in ctx_nonfused.estimates)
        nonfused_total_ns = ctx_nonfused.total_time_ns()
        nonfused_flops = ctx_nonfused.total_flops()
        nonfused_bytes = ctx_nonfused.total_bytes()

        # Determine bottleneck
        if nonfused_compute_ns > nonfused_memory_ns * 1.5:
            nonfused_bottleneck = "COMPUTE"
        elif nonfused_memory_ns > nonfused_compute_ns * 1.5:
            nonfused_bottleneck = "DRAM"
        else:
            nonfused_bottleneck = "BOTH"

        nonfused_ai = nonfused_flops / nonfused_bytes if nonfused_bytes > 0 else 0

        model_results["nonfused"] = {
            "compute_ms": nonfused_compute_ns / 1e6,
            "memory_ms": nonfused_memory_ns / 1e6,
            "total_ms": nonfused_total_ns / 1e6,
            "bottleneck": nonfused_bottleneck,
            "flops": nonfused_flops,
            "bytes": nonfused_bytes,
            "ai": nonfused_ai,
        }

        baseline_time = nonfused_total_ns

        print(
            f"{config.name:<15} {'Non-fused':<25} {nonfused_compute_ns/1e6:<12.4f} "
            f"{nonfused_memory_ns/1e6:<12.4f} {nonfused_total_ns/1e6:<12.4f} "
            f"{'1.00x':<10} {nonfused_bottleneck:<12} {nonfused_ai:<10.1f}"
        )

        all_results.append(
            {
                "model": config.name,
                "impl": "Non-fused",
                "compute_ns": nonfused_compute_ns,
                "memory_ns": nonfused_memory_ns,
                "total_ns": nonfused_total_ns,
                "flops": nonfused_flops,
                "bytes": nonfused_bytes,
                "ai": nonfused_ai,
                "bottleneck": nonfused_bottleneck,
            }
        )

        # 2. Fused ROW_MCAST
        mlp_row_mcast = MockLlamaMLPFused(
            embedding_size=config.embedding_size,
            intermediate_dim=config.intermediate_dim,
            impl=SwiGLUFusedImpl.ROW_MCAST,
        )
        ctx_row_mcast = RooflineContext(hw_spec)
        _ = mlp_row_mcast(ctx_row_mcast, x.clone())

        row_mcast_compute_ns = sum(e.ideal_compute_ns for e in ctx_row_mcast.estimates)
        row_mcast_memory_ns = sum(e.ideal_memory_ns for e in ctx_row_mcast.estimates)
        row_mcast_total_ns = ctx_row_mcast.total_time_ns()
        row_mcast_flops = ctx_row_mcast.total_flops()
        row_mcast_bytes = ctx_row_mcast.total_bytes()

        if row_mcast_compute_ns > row_mcast_memory_ns * 1.5:
            row_mcast_bottleneck = "COMPUTE"
        elif row_mcast_memory_ns > row_mcast_compute_ns * 1.5:
            row_mcast_bottleneck = "DRAM"
        else:
            row_mcast_bottleneck = "BOTH"

        row_mcast_ai = row_mcast_flops / row_mcast_bytes if row_mcast_bytes > 0 else 0
        row_mcast_speedup = (
            baseline_time / row_mcast_total_ns if row_mcast_total_ns > 0 else 0
        )

        model_results["row_mcast"] = {
            "compute_ms": row_mcast_compute_ns / 1e6,
            "memory_ms": row_mcast_memory_ns / 1e6,
            "total_ms": row_mcast_total_ns / 1e6,
            "bottleneck": row_mcast_bottleneck,
            "flops": row_mcast_flops,
            "bytes": row_mcast_bytes,
            "ai": row_mcast_ai,
        }

        print(
            f"{'':<15} {'Fused (row_mcast)':<25} {row_mcast_compute_ns/1e6:<12.4f} "
            f"{row_mcast_memory_ns/1e6:<12.4f} {row_mcast_total_ns/1e6:<12.4f} "
            f"{row_mcast_speedup:.2f}x{'':<6} {row_mcast_bottleneck:<12} {row_mcast_ai:<10.1f}"
        )

        all_results.append(
            {
                "model": config.name,
                "impl": "Fused (row_mcast)",
                "compute_ns": row_mcast_compute_ns,
                "memory_ns": row_mcast_memory_ns,
                "total_ns": row_mcast_total_ns,
                "flops": row_mcast_flops,
                "bytes": row_mcast_bytes,
                "ai": row_mcast_ai,
                "bottleneck": row_mcast_bottleneck,
            }
        )

        # 3. Fused MCAST (optimal)
        mlp_mcast = MockLlamaMLPFused(
            embedding_size=config.embedding_size,
            intermediate_dim=config.intermediate_dim,
            impl=SwiGLUFusedImpl.MCAST,
        )
        ctx_mcast = RooflineContext(hw_spec)
        _ = mlp_mcast(ctx_mcast, x.clone())

        mcast_compute_ns = sum(e.ideal_compute_ns for e in ctx_mcast.estimates)
        mcast_memory_ns = sum(e.ideal_memory_ns for e in ctx_mcast.estimates)
        mcast_total_ns = ctx_mcast.total_time_ns()
        mcast_flops = ctx_mcast.total_flops()
        mcast_bytes = ctx_mcast.total_bytes()

        if mcast_compute_ns > mcast_memory_ns * 1.5:
            mcast_bottleneck = "COMPUTE"
        elif mcast_memory_ns > mcast_compute_ns * 1.5:
            mcast_bottleneck = "DRAM"
        else:
            mcast_bottleneck = "BOTH"

        mcast_ai = mcast_flops / mcast_bytes if mcast_bytes > 0 else 0
        mcast_speedup = baseline_time / mcast_total_ns if mcast_total_ns > 0 else 0

        model_results["mcast"] = {
            "compute_ms": mcast_compute_ns / 1e6,
            "memory_ms": mcast_memory_ns / 1e6,
            "total_ms": mcast_total_ns / 1e6,
            "bottleneck": mcast_bottleneck,
            "flops": mcast_flops,
            "bytes": mcast_bytes,
            "ai": mcast_ai,
        }

        print(
            f"{'':<15} {'Fused (mcast)':<25} {mcast_compute_ns/1e6:<12.4f} "
            f"{mcast_memory_ns/1e6:<12.4f} {mcast_total_ns/1e6:<12.4f} "
            f"{mcast_speedup:.2f}x{'':<6} {mcast_bottleneck:<12} {mcast_ai:<10.1f}"
        )

        all_results.append(
            {
                "model": config.name,
                "impl": "Fused (mcast)",
                "compute_ns": mcast_compute_ns,
                "memory_ns": mcast_memory_ns,
                "total_ns": mcast_total_ns,
                "flops": mcast_flops,
                "bytes": mcast_bytes,
                "ai": mcast_ai,
                "bottleneck": mcast_bottleneck,
            }
        )

        print()  # Blank line between models

    print("-" * 130)
    print()

    # Generate roofline plot
    print(f"Generating roofline plot: {output_plot}")
    generate_roofline_plot(hw_spec, all_results, output_plot)
    print(f"Plot saved to: {output_plot}")
    print()
    print("=" * 90)
    print("COMPARISON COMPLETE")
    print("=" * 90)


def generate_roofline_plot(hw_spec, results: List[Dict], output_path: str):
    """Generate a roofline plot with all results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    from roofline.hardware import MathFidelity

    # Calculate roofline parameters
    peak_tflops = hw_spec.tflops_per_chip(MathFidelity.HiFi4)
    dram_bw_tb_s = hw_spec.dram_bw_gb_s / 1000  # Convert to TB/s

    # Critical arithmetic intensity (ridge point)
    critical_ai = peak_tflops / dram_bw_tb_s  # FLOP/byte

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Roofline (log-log scale)
    ai_range = np.logspace(-1, 4, 1000)  # 0.1 to 10000 FLOP/byte

    # Memory-bound region: perf = BW * AI
    # Compute-bound region: perf = peak
    roofline = np.minimum(dram_bw_tb_s * ai_range, peak_tflops)

    ax.loglog(ai_range, roofline, "k-", linewidth=2, label="Roofline")

    # Add ridge point marker
    ax.axvline(x=critical_ai, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Ridge Point\nAI={critical_ai:.1f}",
        xy=(critical_ai, peak_tflops * 0.7),
        fontsize=9,
        ha="center",
    )

    # Plot each result
    markers = {"Non-fused": "o", "Fused (row_mcast)": "s", "Fused (mcast)": "^"}
    colors = {
        "TinyLlama 1.1B": "purple",
        "Llama 3.2 1B": "blue",
        "Llama 3.1 8B": "green",
        "Llama 3.1 405B": "red",
    }

    for result in results:
        ai = result["ai"]
        # Calculate achieved performance
        time_s = result["total_ns"] / 1e9
        achieved_tflops = (result["flops"] / time_s) / 1e12 if time_s > 0 else 0

        marker = markers.get(result["impl"], "x")
        color = colors.get(result["model"], "gray")

        label = f"{result['model']} - {result['impl']}"
        ax.scatter(
            ai,
            achieved_tflops,
            marker=marker,
            c=color,
            s=100,
            label=label,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )

    # Formatting
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOP/s)", fontsize=12)
    ax.set_title(
        f"SwiGLU MLP Roofline Analysis (Forward Pass)\n{hw_spec.name} "
        f"(Peak: {peak_tflops:.1f} TFLOP/s, BW: {hw_spec.dram_bw_gb_s} GB/s)",
        fontsize=14,
    )

    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.1, peak_tflops * 2)

    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SwiGLU MLP Implementation Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m roofline.examples.swiglu_comparison
  python3 -m roofline.examples.swiglu_comparison --hardware p100
  python3 -m roofline.examples.swiglu_comparison --hardware n300 --batch 4 --seq 2048
  python3 -m roofline.examples.swiglu_comparison --output my_roofline.png
""",
    )
    parser.add_argument(
        "--hardware",
        "-hw",
        type=str,
        choices=["n150", "n300", "p100", "p150"],
        default="n150",
        help="Hardware configuration (default: n150)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--seq",
        "-s",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="swiglu_roofline.png",
        help="Output plot filename (default: swiglu_roofline.png)",
    )

    args = parser.parse_args()

    run_comparison(
        hardware=args.hardware,
        batch_size=args.batch,
        seq_len=args.seq,
        output_plot=args.output,
    )


if __name__ == "__main__":
    main()
