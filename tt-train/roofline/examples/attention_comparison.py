#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Scaled Dot-Product Attention implementation comparison for roofline analysis.

This script compares the performance of different SDPA implementations:
1. MockScaledDotProductAttentionOp - Composite/Unfused (materializes full attention matrix)
2. MockScaledDotProductAttentionFusedOp - Fused (Flash Attention style, O(S) memory)

Uses Llama model shapes: TinyLlama, 1B, 8B, and 405B.

Key differences:
- Composite: Materializes full (B, H, S, S) attention weights matrix
- Fused: Never materializes full attention matrix, uses tiling and recomputation

Run from tt-train directory:
    python3 -m roofline.examples.attention_comparison
    python3 -m roofline.examples.attention_comparison --hardware p100
    python3 -m roofline.examples.attention_comparison --hardware n300 --batch 4 --seq 2048
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LlamaAttentionConfig:
    """Configuration for Llama attention layer."""

    name: str
    embedding_size: int
    num_heads: int
    num_groups: int  # For GQA
    max_sequence_length: int
    description: str


# Llama model configurations (Attention dimensions)
LLAMA_CONFIGS: Dict[str, LlamaAttentionConfig] = {
    "tinyllama": LlamaAttentionConfig(
        name="TinyLlama 1.1B",
        embedding_size=2048,
        num_heads=32,
        num_groups=4,  # GQA with 4 groups
        max_sequence_length=2048,  # 2K context window
        description="TinyLlama 1.1B (22 layers, 2K context)",
    ),
    "llama-1b": LlamaAttentionConfig(
        name="Llama 3.2 1B",
        embedding_size=2048,
        num_heads=32,
        num_groups=8,  # GQA with 8 groups
        max_sequence_length=131072,  # 128K context window
        description="Llama 3.2 1B (16 layers, 128K context)",
    ),
    "llama-8b": LlamaAttentionConfig(
        name="Llama 3.1 8B",
        embedding_size=4096,
        num_heads=32,
        num_groups=8,  # GQA with 8 groups
        max_sequence_length=131072,  # 128K context window
        description="Llama 3.1 8B (32 layers, 128K context)",
    ),
    "llama-405b": LlamaAttentionConfig(
        name="Llama 3.1 405B",
        embedding_size=16384,
        num_heads=128,
        num_groups=8,  # GQA with 8 groups
        max_sequence_length=131072,  # 128K context window
        description="Llama 3.1 405B (126 layers, 128K context)",
    ),
}


def run_comparison(
    hardware: str = "n150",
    batch_size: int = 1,
    seq_len: int = None,
    output_plot: str = "attention_roofline.png",
    include_backward: bool = False,
):
    """Run SDPA comparison and generate roofline plot.

    Args:
        hardware: Hardware configuration to use
        batch_size: Batch size
        seq_len: Sequence length (if None, uses each model's max sequence length)
        output_plot: Output filename for roofline plot
        include_backward: Whether to include backward pass
    """
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
    from roofline.mock_tensor import TensorLabel
    from roofline.operations import (
        MockScaledDotProductAttentionOp,
        MockScaledDotProductAttentionFusedOp,
    )

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

    phase_str = "Forward + Backward Pass" if include_backward else "Forward Pass"
    print("=" * 90)
    print(f"SCALED DOT-PRODUCT ATTENTION IMPLEMENTATION COMPARISON ({phase_str})")
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
    if seq_len is None:
        print(f"  seq_len:     (using each model's max sequence length)")
    else:
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
        head_dim = config.embedding_size // config.num_heads

        # Use model's max sequence length if seq_len not specified
        model_seq_len = seq_len if seq_len is not None else config.max_sequence_length

        # Create Q, K, V tensors
        # Q: [B, num_heads, S, head_dim]
        # K, V: [B, num_groups, S, head_dim] for GQA
        q = MockTensor(
            (batch_size, config.num_heads, model_seq_len, head_dim),
            dtype=DataType.BFLOAT16,
            requires_grad=True,
            label=TensorLabel.ACTIVATION,
        )
        k = MockTensor(
            (batch_size, config.num_groups, model_seq_len, head_dim),
            dtype=DataType.BFLOAT16,
            requires_grad=True,
            label=TensorLabel.ACTIVATION,
        )
        v = MockTensor(
            (batch_size, config.num_groups, model_seq_len, head_dim),
            dtype=DataType.BFLOAT16,
            requires_grad=True,
            label=TensorLabel.ACTIVATION,
        )

        # Results for this model
        model_results = {}

        # 1. Composite/Unfused SDPA (materializes full attention matrix)
        ctx_composite = RooflineContext(hw_spec)
        out_composite = MockScaledDotProductAttentionOp.apply(
            ctx_composite, q.clone(), k.clone(), v.clone()
        )

        if include_backward:
            # Simulate backward pass
            grad_output = MockTensor(
                out_composite.shape,
                dtype=DataType.BFLOAT16,
                requires_grad=True,
                label=TensorLabel.GRADIENT,
            )
            _ = out_composite.backward(ctx_composite, grad_output)

        composite_compute_ns = sum(e.ideal_compute_ns for e in ctx_composite.estimates)
        composite_memory_ns = sum(e.ideal_memory_ns for e in ctx_composite.estimates)
        composite_total_ns = ctx_composite.total_time_ns()
        composite_flops = ctx_composite.total_flops()
        composite_bytes = ctx_composite.total_bytes()

        # Get memory statistics
        (
            composite_peak_bytes,
            composite_breakdown,
        ) = ctx_composite.memory_tracker.peak_memory()
        composite_activation_bytes = composite_breakdown.get(TensorLabel.ACTIVATION, 0)
        composite_gradient_bytes = composite_breakdown.get(TensorLabel.GRADIENT, 0)

        # Determine bottleneck
        if composite_compute_ns > composite_memory_ns * 1.5:
            composite_bottleneck = "COMPUTE"
        elif composite_memory_ns > composite_compute_ns * 1.5:
            composite_bottleneck = "DRAM"
        else:
            composite_bottleneck = "BOTH"

        composite_ai = composite_flops / composite_bytes if composite_bytes > 0 else 0

        model_results["composite"] = {
            "compute_ms": composite_compute_ns / 1e6,
            "memory_ms": composite_memory_ns / 1e6,
            "total_ms": composite_total_ns / 1e6,
            "bottleneck": composite_bottleneck,
            "flops": composite_flops,
            "bytes": composite_bytes,
            "ai": composite_ai,
            "peak_memory_bytes": composite_peak_bytes,
            "activation_bytes": composite_activation_bytes,
            "gradient_bytes": composite_gradient_bytes,
        }

        baseline_time = composite_total_ns

        print(
            f"{config.name:<15} {'Composite':<25} {composite_compute_ns/1e6:<12.4f} "
            f"{composite_memory_ns/1e6:<12.4f} {composite_total_ns/1e6:<12.4f} "
            f"{'1.00x':<10} {composite_bottleneck:<12} {composite_ai:<10.1f}"
        )

        all_results.append(
            {
                "model": config.name,
                "impl": "Composite",
                "compute_ns": composite_compute_ns,
                "memory_ns": composite_memory_ns,
                "total_ns": composite_total_ns,
                "flops": composite_flops,
                "bytes": composite_bytes,
                "ai": composite_ai,
                "bottleneck": composite_bottleneck,
                "peak_memory_bytes": composite_peak_bytes,
                "activation_bytes": composite_activation_bytes,
                "gradient_bytes": composite_gradient_bytes,
            }
        )

        # 2. Fused SDPA (Flash Attention style - no full attention matrix)
        ctx_fused = RooflineContext(hw_spec)
        out_fused = MockScaledDotProductAttentionFusedOp.apply(
            ctx_fused, q.clone(), k.clone(), v.clone()
        )

        if include_backward:
            # Simulate backward pass
            grad_output = MockTensor(
                out_fused.shape, dtype=DataType.BFLOAT16, requires_grad=True
            )
            _ = out_fused.backward(ctx_fused, grad_output)

        fused_compute_ns = sum(e.ideal_compute_ns for e in ctx_fused.estimates)
        fused_memory_ns = sum(e.ideal_memory_ns for e in ctx_fused.estimates)
        fused_total_ns = ctx_fused.total_time_ns()
        fused_flops = ctx_fused.total_flops()
        fused_bytes = ctx_fused.total_bytes()

        # Get memory statistics
        fused_peak_bytes, fused_breakdown = ctx_fused.memory_tracker.peak_memory()
        fused_activation_bytes = fused_breakdown.get(TensorLabel.ACTIVATION, 0)
        fused_gradient_bytes = fused_breakdown.get(TensorLabel.GRADIENT, 0)

        if fused_compute_ns > fused_memory_ns * 1.5:
            fused_bottleneck = "COMPUTE"
        elif fused_memory_ns > fused_compute_ns * 1.5:
            fused_bottleneck = "DRAM"
        else:
            fused_bottleneck = "BOTH"

        fused_ai = fused_flops / fused_bytes if fused_bytes > 0 else 0
        fused_speedup = baseline_time / fused_total_ns if fused_total_ns > 0 else 0

        model_results["fused"] = {
            "compute_ms": fused_compute_ns / 1e6,
            "memory_ms": fused_memory_ns / 1e6,
            "total_ms": fused_total_ns / 1e6,
            "bottleneck": fused_bottleneck,
            "flops": fused_flops,
            "bytes": fused_bytes,
            "ai": fused_ai,
            "peak_memory_bytes": fused_peak_bytes,
            "activation_bytes": fused_activation_bytes,
            "gradient_bytes": fused_gradient_bytes,
        }

        print(
            f"{'':<15} {'Fused':<25} {fused_compute_ns/1e6:<12.4f} "
            f"{fused_memory_ns/1e6:<12.4f} {fused_total_ns/1e6:<12.4f} "
            f"{fused_speedup:.2f}x{'':<6} {fused_bottleneck:<12} {fused_ai:<10.1f}"
        )

        all_results.append(
            {
                "model": config.name,
                "impl": "Fused",
                "compute_ns": fused_compute_ns,
                "memory_ns": fused_memory_ns,
                "total_ns": fused_total_ns,
                "flops": fused_flops,
                "bytes": fused_bytes,
                "ai": fused_ai,
                "bottleneck": fused_bottleneck,
                "peak_memory_bytes": fused_peak_bytes,
                "activation_bytes": fused_activation_bytes,
                "gradient_bytes": fused_gradient_bytes,
            }
        )

        print()  # Blank line between models

    print("-" * 130)
    print()

    # Print detailed memory comparison
    print("=" * 130)
    print("PEAK MEMORY USAGE COMPARISON")
    print("=" * 130)
    print()
    print(
        f"{'Model':<15} {'Implementation':<25} {'Peak (GB)':<12} {'Activations':<12} {'Gradients':<12} {'Memory Saved':<15}"
    )
    print("-" * 130)

    for config_name, config in LLAMA_CONFIGS.items():
        # Find results for this model
        composite_result = next(
            (
                r
                for r in all_results
                if r["model"] == config.name and r["impl"] == "Composite"
            ),
            None,
        )
        fused_result = next(
            (
                r
                for r in all_results
                if r["model"] == config.name and r["impl"] == "Fused"
            ),
        )

        if composite_result and fused_result:
            comp_peak_gb = composite_result.get("peak_memory_bytes", 0) / 1e9
            comp_act_gb = composite_result.get("activation_bytes", 0) / 1e9
            comp_grad_gb = composite_result.get("gradient_bytes", 0) / 1e9

            fused_peak_gb = fused_result.get("peak_memory_bytes", 0) / 1e9
            fused_act_gb = fused_result.get("activation_bytes", 0) / 1e9
            fused_grad_gb = fused_result.get("gradient_bytes", 0) / 1e9

            memory_saved_gb = comp_peak_gb - fused_peak_gb

            print(
                f"{config.name:<15} {'Composite':<25} {comp_peak_gb:<12.4f} {comp_act_gb:<12.4f} {comp_grad_gb:<12.4f} {'Baseline':<15}"
            )
            print(
                f"{'':<15} {'Fused':<25} {fused_peak_gb:<12.4f} {fused_act_gb:<12.4f} {fused_grad_gb:<12.4f} "
                f"{memory_saved_gb:.4f} GB saved"
            )
            print()

    print("-" * 130)
    print()

    # Generate roofline plot
    print(f"Generating roofline plot: {output_plot}")
    generate_roofline_plot(hw_spec, all_results, output_plot, phase_str)
    print(f"Plot saved to: {output_plot}")
    print()
    print("=" * 90)
    print("COMPARISON COMPLETE")
    print("=" * 90)


def generate_roofline_plot(
    hw_spec, results: List[Dict], output_path: str, phase_str: str
):
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
    markers = {"Composite": "o", "Fused (Flash Attn)": "^"}
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
        f"Scaled Dot-Product Attention Roofline Analysis ({phase_str})\n{hw_spec.name} "
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
        description="Scaled Dot-Product Attention Implementation Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m roofline.examples.attention_comparison
  python3 -m roofline.examples.attention_comparison --hardware p100
  python3 -m roofline.examples.attention_comparison --hardware n300 --batch 4 --seq 2048
  python3 -m roofline.examples.attention_comparison --output my_roofline.png --backward
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
        default=None,
        help="Sequence length (default: None, uses each model's max sequence length)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="attention_roofline.png",
        help="Output plot filename (default: attention_roofline.png)",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Include backward pass in comparison (default: forward only)",
    )

    args = parser.parse_args()

    run_comparison(
        hardware=args.hardware,
        batch_size=args.batch,
        seq_len=args.seq,
        output_plot=args.output,
        include_backward=args.backward,
    )


if __name__ == "__main__":
    main()
