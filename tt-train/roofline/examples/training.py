#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Transformer model roofline analysis example.

This script demonstrates roofline estimation for transformer models (GPT and Llama),
providing performance analysis for forward pass, backward pass,
and full training iteration including optimizer step.

Supported Models:
    GPT Models:  nanogpt-char, nanogpt-bpe, gpt2-small, gpt2-medium, gpt2-large
    Llama Models: nanollama, tinyllama

Run from tt-train directory:
    # GPT models
    python3 -m roofline.examples.nanogpt                             # Default: nanogpt-char
    python3 -m roofline.examples.nanogpt --model nanogpt-char -b 64 -s 256
    python3 -m roofline.examples.nanogpt --model gpt2-small -b 4 -s 1024

    # Llama models
    python3 -m roofline.examples.nanogpt --model tinyllama -b 1 -s 2048
    python3 -m roofline.examples.nanogpt --model nanollama -b 64 -s 256

    # List all available models
    python3 -m roofline.examples.nanogpt --list
"""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class ModelType(Enum):
    """Type of transformer model."""

    GPT = "gpt"
    LLAMA = "llama"


# Preset model configurations
MODEL_PRESETS: Dict[str, dict] = {
    # ==================== GPT Models ====================
    # Char tokenizer models (smaller vocab)
    "nanogpt-char": {
        "model_type": ModelType.GPT,
        "vocab_size": 96,  # ~65 chars + special tokens, rounded up to multiple of 32
        "block_size": 256,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.2,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "NanoGPT with char tokenizer (Shakespeare)",
    },
    # BPE tokenizer models (GPT-2 vocab)
    "nanogpt-bpe": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,  # GPT-2 BPE vocab size
        "block_size": 256,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.2,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "NanoGPT with BPE tokenizer",
    },
    "gpt2-small": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "dropout": 0.2,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "GPT-2 Small (124M params)",
    },
    "gpt2-medium": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "dropout": 0.1,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "GPT-2 Medium (355M params)",
    },
    "gpt2-large": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "dropout": 0.1,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "GPT-2 Large (774M params)",
    },
    # ==================== Llama Models ====================
    "nanollama": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 32000,
        "max_sequence_length": 256,
        "embedding_dim": 384,
        "num_heads": 6,
        "num_groups": 3,
        "num_blocks": 6,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Nano Llama (Shakespeare, ~10M params)",
    },
    "nanollama-char": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 96,
        "max_sequence_length": 256,
        "embedding_dim": 384,
        "num_heads": 6,
        "num_groups": 3,
        "num_blocks": 6,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Nano Llama (char tokenizer, ~10M params)",
    },
    "tinyllama": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 32000,
        "max_sequence_length": 2048,
        "embedding_dim": 2048,
        "num_heads": 32,
        "num_groups": 4,
        "num_blocks": 22,
        "dropout": 0.0,
        "theta": 10000.0,
        "weight_tying": False,
        "description": "TinyLlama 1.1B (1.1B params)",
    },
    "tinyllama-char": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 96,
        "max_sequence_length": 2048,
        "embedding_dim": 2048,
        "num_heads": 32,
        "num_groups": 4,
        "num_blocks": 22,
        "dropout": 0.0,
        "theta": 10000.0,
        "weight_tying": False,
        "description": "TinyLlama 1.1B (char tokenizer, 0.96B params)",
    },
}


def run_model_roofline(
    model_name: str = "nanogpt-char",
    batch_size: int = 64,
    seq_len: int = 256,
    hardware: str = "n150",
    plot_memory: bool = True,
):
    """Run transformer model roofline analysis (supports both GPT and Llama models).

    Args:
        model_name: Name of the model preset to use
        batch_size: Batch size for the analysis
        seq_len: Sequence length for the analysis
        hardware: Hardware configuration to use
        plot_memory: Whether to generate memory usage plot
    """
    from roofline import (
        MockTensor,
        MockNanoGPT,
        MockNanoGPTConfig,
        MockLlama,
        MockLlamaConfig,
        RooflineContext,
        WORMHOLE_N150,
        WORMHOLE_N300,
        BLACKHOLE_P100,
        BLACKHOLE_P150,
        DataType,
        MathFidelity,
        MockAdamW,
        mock_clip_grad_norm,
        MockCrossEntropyLossOp,
        TensorLabel,
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

    print("=" * 70)
    print("TRANSFORMER MODEL ROOFLINE ANALYSIS")
    print("=" * 70)
    print()
    print(f"Hardware: {hw_spec.name}")
    print(f"  Cores:      {hw_spec.tensix_cores_per_chip}")
    print(f"  Clock:      {hw_spec.clock_ghz} GHz")
    print(f"  DRAM BW:    {hw_spec.dram_bw_gb_s} GB/s")
    print(f"  Peak (HiFi4): {hw_spec.tflops_per_chip(MathFidelity.HiFi4):.1f} TFLOPs")
    print()

    # Get preset or use default
    if model_name not in MODEL_PRESETS:
        print(f"Unknown model: {model_name}")
        print(f"Available presets: {', '.join(MODEL_PRESETS.keys())}")
        return

    preset = MODEL_PRESETS[model_name]
    model_type = preset["model_type"]

    # Create roofline context FIRST so that parameter tensors are tracked
    # (Memory tracking is enabled when context is created)
    ctx = RooflineContext(hw_spec)

    # Create model based on type (parameters will now be tracked)
    if model_type == ModelType.GPT:
        config = MockNanoGPTConfig(
            vocab_size=preset["vocab_size"],
            block_size=preset["block_size"],
            n_embd=preset["n_embd"],
            n_layer=preset["n_layer"],
            n_head=preset["n_head"],
            dropout=preset["dropout"],
        )
        model = MockNanoGPT(config)
        max_seq_len = config.block_size

        # Print GPT-specific config
        print(f"Model: {model_name} (GPT)")
        print(f"Description: {preset['description']}")
        print()
        print(f"Model Configuration:")
        print(f"  vocab_size:  {config.vocab_size:,}")
        print(f"  block_size:  {config.block_size}")
        print(f"  n_embd:      {config.n_embd}")
        print(f"  n_layer:     {config.n_layer}")
        print(f"  n_head:      {config.n_head}")
        print(f"  dropout:     {config.dropout}")
        print()

    elif model_type == ModelType.LLAMA:
        config = MockLlamaConfig(
            vocab_size=preset["vocab_size"],
            max_sequence_length=preset["max_sequence_length"],
            embedding_dim=preset["embedding_dim"],
            num_heads=preset["num_heads"],
            num_groups=preset["num_groups"],
            num_blocks=preset["num_blocks"],
            dropout_prob=preset["dropout"],
            theta=preset["theta"],
            weight_tying=preset["weight_tying"],
        )
        model = MockLlama(config)
        max_seq_len = config.max_sequence_length

        # Print Llama-specific config
        print(f"Model: {model_name} (Llama)")
        print(f"Description: {preset['description']}")
        print()
        print(f"Model Configuration:")
        print(f"  vocab_size:         {config.vocab_size:,}")
        print(f"  max_sequence_length: {config.max_sequence_length}")
        print(f"  embedding_dim:      {config.embedding_dim}")
        print(f"  num_blocks:         {config.num_blocks}")
        print(f"  num_heads:          {config.num_heads}")
        print(f"  num_groups:         {config.num_groups}")
        print(f"  dropout:            {config.dropout_prob}")
        print(f"  theta:              {config.theta}")
        print()

    else:
        print(f"Unknown model type: {model_type}")
        return

    # Clamp seq_len to max sequence length
    if seq_len > max_seq_len:
        print(
            f"Warning: seq_len ({seq_len}) > max_seq_len ({max_seq_len}), clamping to max_seq_len"
        )
        seq_len = max_seq_len

    print(f"Batch Configuration:")
    print(f"  batch_size:  {batch_size}")
    print(f"  seq_len:     {seq_len}")
    print()

    # Count parameters
    params = model.parameters()
    total_params = sum(p.logical_volume() for p in params.values())
    param_memory = sum(p.bytes() for p in params.values())

    print(f"Model Statistics:")
    print(f"  Parameters:  {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Param Memory: {param_memory/1e9:.3f} GB (BF16)")
    print()

    print("-" * 70)
    print(f"Running Analysis: B={batch_size}, S={seq_len}")
    print("-" * 70)

    # Helper to print memory snapshot
    def print_memory_snapshot(label: str):
        if ctx.memory_tracker is not None:
            current_bytes, breakdown = ctx.memory_tracker.current_memory()
            print(f"--- {label} ---")
            print(f"  Current Memory: {current_bytes / 1e6:.2f} MB")

    # Snapshot after model creation
    print_memory_snapshot("AFTER_MODEL_CREATION")

    # Create optimizer right after model (like ttml)
    # This allocates optimizer state (m and v tensors for AdamW)
    optimizer = MockAdamW(params, lr=1e-4, weight_decay=0.1)

    # Snapshot after optimizer creation
    print_memory_snapshot("AFTER_OPTIMIZER_CREATION")

    # Create input tensors (not tracked as training tensors)
    # Note: indices shape is [batch, 1, 1, seq_len] for ttml
    indices = MockTensor(
        (batch_size, 1, 1, seq_len),
        dtype=DataType.BFLOAT16,  # Will be cast to int internally
        requires_grad=False,
        label=TensorLabel.ACTIVATION,  # Input data, not a training tensor
        name="indices",
    )

    # Target for loss (same shape as indices)
    targets = MockTensor(
        (batch_size, 1, 1, seq_len),
        dtype=DataType.BFLOAT16,
        requires_grad=False,
        label=TensorLabel.ACTIVATION,  # Target data, not a training tensor
        name="targets",
    )

    # Forward pass
    logits = model(ctx, indices)

    # Compute loss
    loss = MockCrossEntropyLossOp.apply(ctx, logits, targets)

    # Snapshot after forward pass
    print_memory_snapshot("AFTER_FORWARD_PASS")

    forward_time_ms = ctx.total_time_ms()
    forward_flops = ctx.total_flops()

    # Backward pass (retain_graph=False to deallocate activations/gradients early)
    loss.backward(ctx, retain_graph=False)

    # Snapshot after backward pass
    print_memory_snapshot("AFTER_BACKWARD_PASS")

    backward_time_ms = ctx.total_time_ms() - forward_time_ms
    backward_flops = ctx.total_flops() - forward_flops
    after_backward_time_ms = ctx.total_time_ms()
    after_backward_flops = ctx.total_flops()

    # Optimizer step (optimizer already created above)
    optimizer.step(ctx)

    # Snapshot after optimizer step
    print_memory_snapshot("AFTER_OPTIMIZER_STEP")

    optimizer_time_ms = ctx.total_time_ms() - after_backward_time_ms
    optimizer_flops = ctx.total_flops() - after_backward_flops

    # Gradient clipping (optional)
    after_optimizer_time_ms = ctx.total_time_ms()
    after_optimizer_flops = ctx.total_flops()
    mock_clip_grad_norm(ctx, params, max_norm=1.0)

    # Snapshot after iteration complete
    print_memory_snapshot("ITERATION_COMPLETE")
    ctx.print_peak_memory()

    grad_clip_time_ms = ctx.total_time_ms() - after_optimizer_time_ms
    grad_clip_flops = ctx.total_flops() - after_optimizer_flops

    # Final metrics
    iteration_time_ms = ctx.total_time_ms()
    iteration_flops = ctx.total_flops()
    tokens_per_iteration = batch_size * seq_len
    tokens_per_second = tokens_per_iteration / (iteration_time_ms / 1000)

    print()
    print("Timing Breakdown:")
    print(f"  Forward:     {forward_time_ms:.4f} ms ({forward_flops/1e12:.4f} TFLOPs)")
    print(
        f"  Backward:    {backward_time_ms:.4f} ms ({backward_flops/1e12:.4f} TFLOPs)"
    )
    print(
        f"  Optimizer:   {optimizer_time_ms:.4f} ms ({optimizer_flops/1e12:.4f} TFLOPs)"
    )
    print(
        f"  Grad Clip:   {grad_clip_time_ms:.4f} ms ({grad_clip_flops/1e12:.4f} TFLOPs)"
    )
    print(
        f"  Total:       {iteration_time_ms:.4f} ms ({iteration_flops/1e12:.4f} TFLOPs)"
    )
    print()
    print("Throughput:")
    print(f"  Tokens/iter: {tokens_per_iteration:,}")
    print(f"  Tokens/sec:  {tokens_per_second:,.0f}")
    print(f"  TFLOPs:      {ctx.achieved_tflops():.2f}")
    print()

    # Bottleneck analysis
    breakdown = ctx.bottleneck_breakdown()
    print("Bottleneck Analysis:")
    for btype, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {btype.value}: {count} ops")

    # Peak memory analysis from tracking
    print()
    print("-" * 70)
    print("Memory Tracking Analysis")
    print("-" * 70)
    ctx.print_peak_memory()

    # Generate memory usage plots
    if plot_memory:
        # Stacked area plot (overview)
        plot_filename = f"memory_usage_{model_name}_b{batch_size}_s{seq_len}.png"
        ctx.plot_memory_usage(
            filename=plot_filename,
            title=f"Memory Usage: {model_name} (B={batch_size}, S={seq_len})",
            stacked=True,
        )
        # Detailed per-category plots (shows individual fluctuations)
        detail_filename = f"memory_detail_{model_name}_b{batch_size}_s{seq_len}.png"
        ctx.plot_memory_usage(
            filename=detail_filename,
            title=f"Memory Detail: {model_name} (B={batch_size}, S={seq_len})",
            stacked=False,
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Disable memory tracking when done
    ctx.disable_memory_tracking()


def list_models():
    """List available model presets."""
    print("Available model presets:")
    print("-" * 70)

    # Group by model type
    gpt_models = {
        k: v for k, v in MODEL_PRESETS.items() if v["model_type"] == ModelType.GPT
    }
    llama_models = {
        k: v for k, v in MODEL_PRESETS.items() if v["model_type"] == ModelType.LLAMA
    }

    if gpt_models:
        print("GPT Models:")
        for name, preset in gpt_models.items():
            print(f"  {name:<20} {preset['description']}")
        print()

    if llama_models:
        print("Llama Models:")
        for name, preset in llama_models.items():
            print(f"  {name:<20} {preset['description']}")
        print()

    print("-" * 70)


def run_single_block_analysis():
    """Detailed analysis of a single transformer block."""
    from roofline import (
        MockTensor,
        MockGPTBlock,
        RooflineContext,
        WORMHOLE_N150,
        BLACKHOLE_P100,
        DataType,
        TensorLabel,
    )

    print()
    print("=" * 70)
    print("SINGLE TRANSFORMER BLOCK ANALYSIS")
    print("=" * 70)
    print()

    # Block configuration
    embedding_dim = 768
    num_heads = 12
    batch_size = 4
    seq_len = 1024

    print(f"Configuration:")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  num_heads:     {num_heads}")
    print(f"  batch_size:    {batch_size}")
    print(f"  seq_len:       {seq_len}")
    print()

    # Create block
    block = MockGPTBlock(embedding_dim, num_heads, dropout=0.1, bias=True)

    # Create context and input
    ctx = RooflineContext(WORMHOLE_N150)
    x = MockTensor(
        (batch_size, 1, seq_len, embedding_dim),
        dtype=DataType.BFLOAT16,
        requires_grad=True,
        label=TensorLabel.ACTIVATION,
    )

    # Forward pass
    y = block(ctx, x)

    # Backward pass
    y.backward(ctx)

    # Print detailed results
    print(ctx.summary(block))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformer Model Roofline Analysis (GPT and Llama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPT models
  python3 -m roofline.examples.nanogpt                            # Default: nanogpt-char, B=64, S=256, n150
  python3 -m roofline.examples.nanogpt -b 16 -s 512               # Custom batch/seq
  python3 -m roofline.examples.nanogpt --model gpt2-small -b 4 -s 1024

  # Llama models
  python3 -m roofline.examples.nanogpt --model nanollama -b 64 -s 256
  python3 -m roofline.examples.nanogpt --model tinyllama -b 1 -s 2048

  # Hardware configurations
  python3 -m roofline.examples.nanogpt --hardware n300            # Wormhole n300
  python3 -m roofline.examples.nanogpt --hardware p100            # Blackhole P100
  python3 -m roofline.examples.nanogpt --hardware p150            # Blackhole P150

  # Utilities
  python3 -m roofline.examples.nanogpt --list                     # List available presets
  python3 -m roofline.examples.nanogpt --detailed                 # Single-block analysis
""",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model preset name (see --list for available presets)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--seq",
        "-s",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
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
        "--list",
        "-l",
        action="store_true",
        help="List available model presets",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed single-block analysis (GPT only)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable memory usage plot generation",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # Determine model name
    if args.model:
        model_name = args.model
    else:
        model_name = "nanogpt-char"  # Default

    run_model_roofline(
        model_name,
        batch_size=args.batch,
        seq_len=args.seq,
        hardware=args.hardware,
        plot_memory=not args.no_plot,
    )

    if args.detailed:
        run_single_block_analysis()


if __name__ == "__main__":
    main()
