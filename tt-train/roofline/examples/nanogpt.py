#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT roofline analysis example.

This script demonstrates roofline estimation for the NanoGPT model,
providing performance analysis for forward pass, backward pass,
and full training iteration including optimizer step.

Run from tt-train directory:
    python3 -m roofline.examples.nanogpt
    python3 -m roofline.examples.nanogpt --tokenizer char -b 64 -s 256
    python3 -m roofline.examples.nanogpt --model gpt2-small -b 4 -s 1024
"""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class TokenizerType(Enum):
    """Tokenizer type for NanoGPT."""

    CHAR = "char"
    BPE = "bpe"


# Preset model configurations
MODEL_PRESETS: Dict[str, dict] = {
    # Char tokenizer models (smaller vocab)
    "nanogpt-char": {
        "vocab_size": 96,  # ~65 chars + special tokens, rounded up to multiple of 32
        "block_size": 256,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.2,
        "bias": True,
        "tokenizer": TokenizerType.CHAR,
        "description": "NanoGPT with char tokenizer (Shakespeare)",
    },
    # BPE tokenizer models (GPT-2 vocab)
    "nanogpt-bpe": {
        "vocab_size": 50257,  # GPT-2 BPE vocab size
        "block_size": 256,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.2,
        "bias": True,
        "tokenizer": TokenizerType.BPE,
        "description": "NanoGPT with BPE tokenizer",
    },
    "gpt2-small": {
        "vocab_size": 50257,  # GPT-2 vocab padded to 64
        "block_size": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "dropout": 0.2,
        "bias": True,
        "tokenizer": TokenizerType.BPE,
        "description": "GPT-2 Small (124M params)",
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "dropout": 0.1,
        "bias": True,
        "tokenizer": TokenizerType.BPE,
        "description": "GPT-2 Medium (355M params)",
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "dropout": 0.1,
        "bias": True,
        "tokenizer": TokenizerType.BPE,
        "description": "GPT-2 Large (774M params)",
    },
}


def run_nanogpt_roofline(
    model_name: str = "nanogpt-char", batch_size: int = 64, seq_len: int = 256
):
    """Run NanoGPT roofline analysis."""
    from roofline import (
        MockTensor,
        MockNanoGPT,
        MockNanoGPTConfig,
        RooflineContext,
        WORMHOLE_N150,
        DataType,
        MockAdamW,
        mock_clip_grad_norm,
        MockCrossEntropyLossOp,
    )

    print("=" * 70)
    print("NANOGPT ROOFLINE ANALYSIS")
    print("=" * 70)
    print()

    # Get preset or use default
    if model_name not in MODEL_PRESETS:
        print(f"Unknown model: {model_name}")
        print(f"Available presets: {', '.join(MODEL_PRESETS.keys())}")
        return

    preset = MODEL_PRESETS[model_name]

    # Create config from preset
    config = MockNanoGPTConfig(
        vocab_size=preset["vocab_size"],
        block_size=preset["block_size"],
        n_embd=preset["n_embd"],
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        dropout=preset["dropout"],
        bias=preset["bias"],
    )

    tokenizer_type = preset["tokenizer"]

    # Clamp seq_len to block_size
    if seq_len > config.block_size:
        print(
            f"Warning: seq_len ({seq_len}) > block_size ({config.block_size}), clamping to block_size"
        )
        seq_len = config.block_size

    print(f"Model: {model_name}")
    print(f"Description: {preset['description']}")
    print(f"Tokenizer: {tokenizer_type.value}")
    print()
    print(f"Model Configuration:")
    print(f"  vocab_size:  {config.vocab_size:,}")
    print(f"  block_size:  {config.block_size}")
    print(f"  n_embd:      {config.n_embd}")
    print(f"  n_layer:     {config.n_layer}")
    print(f"  n_head:      {config.n_head}")
    print(f"  dropout:     {config.dropout}")
    print()
    print(f"Batch Configuration:")
    print(f"  batch_size:  {batch_size}")
    print(f"  seq_len:     {seq_len}")
    print()

    # Create model
    model = MockNanoGPT(config)

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

    # Create roofline context
    ctx = RooflineContext(WORMHOLE_N150)

    # Create input tensors
    # Note: indices shape is [batch, 1, 1, seq_len] for ttml
    indices = MockTensor(
        (batch_size, 1, 1, seq_len),
        dtype=DataType.BFLOAT16,  # Will be cast to int internally
        requires_grad=False,
    )

    # Target for loss (same shape as indices)
    targets = MockTensor(
        (batch_size, 1, 1, seq_len),
        dtype=DataType.BFLOAT16,
        requires_grad=False,
    )

    # Forward pass
    logits = model(ctx, indices)

    # Compute loss
    loss = MockCrossEntropyLossOp.apply(ctx, logits, targets)

    forward_time_ms = ctx.total_time_ms()
    forward_flops = ctx.total_flops()

    # Backward pass
    loss.backward(ctx)

    total_time_ms = ctx.total_time_ms()
    backward_time_ms = total_time_ms - forward_time_ms
    backward_flops = ctx.total_flops() - forward_flops

    # Optimizer step
    optimizer = MockAdamW(params, lr=1e-4, weight_decay=0.1)
    optimizer.step(ctx)

    optimizer_time_ms = ctx.total_time_ms() - total_time_ms

    # Gradient clipping (optional)
    mock_clip_grad_norm(ctx, params, max_norm=1.0)
    grad_clip_time_ms = ctx.total_time_ms() - total_time_ms - optimizer_time_ms

    # Final metrics
    iteration_time_ms = ctx.total_time_ms()
    tokens_per_iteration = batch_size * seq_len
    tokens_per_second = tokens_per_iteration / (iteration_time_ms / 1000)

    # Memory estimates
    activation_memory = ctx.estimate_activation_memory()
    gradient_memory = ctx.estimate_gradient_memory(model)
    optimizer_memory = optimizer.estimate_memory()
    total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory

    print()
    print("Timing Breakdown:")
    print(f"  Forward:     {forward_time_ms:.4f} ms ({forward_flops/1e12:.4f} TFLOPs)")
    print(
        f"  Backward:    {backward_time_ms:.4f} ms ({backward_flops/1e12:.4f} TFLOPs)"
    )
    print(f"  Optimizer:   {optimizer_time_ms:.4f} ms")
    print(f"  Grad Clip:   {grad_clip_time_ms:.4f} ms")
    print(f"  Total:       {iteration_time_ms:.4f} ms")
    print()
    print("Throughput:")
    print(f"  Tokens/iter: {tokens_per_iteration:,}")
    print(f"  Tokens/sec:  {tokens_per_second:,.0f}")
    print(f"  TFLOPs:      {ctx.achieved_tflops():.2f}")
    print()
    print("Memory Estimate:")
    print(f"  Parameters:   {param_memory/1e9:.3f} GB")
    print(f"  Activations:  {activation_memory/1e9:.3f} GB")
    print(f"  Gradients:    {gradient_memory/1e9:.3f} GB")
    print(f"  Optimizer:    {optimizer_memory/1e9:.3f} GB")
    print(f"  Total:        {total_memory/1e9:.3f} GB")
    print()

    # Bottleneck analysis
    breakdown = ctx.bottleneck_breakdown()
    print("Bottleneck Analysis:")
    for btype, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {btype.value}: {count} ops")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def list_models():
    """List available model presets."""
    print("Available model presets:")
    print("-" * 70)
    for name, preset in MODEL_PRESETS.items():
        tokenizer = preset["tokenizer"].value
        params_approx = ""
        if "124M" in preset["description"]:
            params_approx = " (~124M params)"
        elif "355M" in preset["description"]:
            params_approx = " (~355M params)"
        elif "774M" in preset["description"]:
            params_approx = " (~774M params)"
        print(f"  {name:<20} [{tokenizer}] {preset['description']}{params_approx}")
    print("-" * 70)


def run_single_block_analysis():
    """Detailed analysis of a single transformer block."""
    from roofline import (
        MockTensor,
        MockGPTBlock,
        RooflineContext,
        WORMHOLE_N150,
        DataType,
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
        description="NanoGPT Roofline Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m roofline.examples.nanogpt                       # Default: nanogpt-char, B=64, S=256
  python3 -m roofline.examples.nanogpt -b 16 -s 512          # Custom batch/seq
  python3 -m roofline.examples.nanogpt --model gpt2-small -b 4 -s 1024
  python3 -m roofline.examples.nanogpt --tokenizer bpe       # Same as nanogpt-bpe
  python3 -m roofline.examples.nanogpt --list                # List available presets
  python3 -m roofline.examples.nanogpt --detailed            # Include single-block analysis
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
        "--tokenizer",
        "-t",
        type=str,
        choices=["char", "bpe"],
        default=None,
        help="Tokenizer type (shortcut for nanogpt-char or nanogpt-bpe)",
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
        "--list",
        "-l",
        action="store_true",
        help="List available model presets",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed single-block analysis",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # Determine model name
    if args.model:
        model_name = args.model
    elif args.tokenizer:
        model_name = f"nanogpt-{args.tokenizer}"
    else:
        model_name = "nanogpt-char"  # Default

    run_nanogpt_roofline(model_name, batch_size=args.batch, seq_len=args.seq)

    if args.detailed:
        run_single_block_analysis()


if __name__ == "__main__":
    main()
