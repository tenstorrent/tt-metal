# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_expert_weights fused micro-op.

Tests the transformation of routing weights from [B*S, K] to [K, 1, B*S, H]
format with broadcast across the hidden dimension.

Includes:
- Functional correctness tests (golden vs fused op)
- Comparison tests (fused vs unfused TTNN ops)
- Device performance tests using Tracy profiler
"""

import json
import math
import os
from collections import defaultdict

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

# Import the fused op
from models.demos.gpt_oss.micro_ops.prepare_expert_weights.op import (
    PrepareGptOssExpertsTensorSingleCore,
    PrepareGptOssExpertsTensorPipelined,
)

# Performance test configuration
DEVICE_PERF_ENV_VAR = "PREPARE_EXPERT_WEIGHTS_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.15


def prepare_expert_weights_unfused(
    topk_expert_weights: ttnn.Tensor,
    num_experts_per_tok: int,
    hidden_size: int,
) -> ttnn.Tensor:
    """Original unfused implementation for comparison.

    This is the exact sequence of ops that the fused kernel replaces.
    """
    topk_expert_weights = ttnn.reshape(topk_expert_weights, (-1, 1, 1, num_experts_per_tok))
    topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    topk_weights_rm = ttnn.repeat(topk_weights_rm, ttnn.Shape((1, 1, hidden_size, 1)))
    topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 0, 2))
    topk_weights_reshaped = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(topk_weights_rm)
    return topk_weights_reshaped


def create_sharded_input_tensor(
    torch_tensor: torch.Tensor,
    device,
    tile_shape: tuple = (1, 32),
) -> ttnn.Tensor:
    """Create a sharded input tensor on a single core."""
    shape = torch_tensor.shape

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    tile = ttnn.Tile(tile_shape)
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )


def create_sharded_output_tensor(
    torch_tensor: torch.Tensor,
    device,
    tile_shape: tuple = (32, 32),
) -> ttnn.Tensor:
    """Create a sharded output tensor on a single core."""
    shape = torch_tensor.shape
    # Flatten to 2D for sharding
    flat_shape = (shape[0] * shape[1] * shape[2], shape[3])

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        flat_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    tile = ttnn.Tile(tile_shape)
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )


def _measure_perf_us(
    device,
    op_fn,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    """Measure operation performance in microseconds."""
    ttnn.synchronize_device(device)

    # Warmup
    for _ in range(warmup_iters):
        result = op_fn()
        ttnn.synchronize_device(device)

    # Measure
    profiler.clear()
    profiler.start("op_perf")
    for _ in range(measure_iters):
        result = op_fn()
        ttnn.synchronize_device(device)
    profiler.end("op_perf", PERF_CNT=measure_iters)

    return profiler.get("op_perf") * 1e6


# =============================================================================
# Golden Function Tests (No Device Required)
# =============================================================================


class TestGoldenFunction:
    """Test the PyTorch golden reference implementation."""

    @pytest.mark.parametrize(
        "batch_seq,num_experts_per_tok,hidden_size",
        [
            (1, 8, 256),
            (4, 8, 512),
            (8, 4, 1024),
            (16, 8, 2048),
            (32, 8, 7168),
        ],
    )
    def test_golden_output_shape(self, batch_seq, num_experts_per_tok, hidden_size):
        """Test that golden function produces correct output shape."""
        torch.manual_seed(42)
        input_weights = torch.randn(batch_seq, num_experts_per_tok, dtype=torch.bfloat16)

        output = PrepareGptOssExpertsTensorSingleCore.golden(
            input_weights, num_experts_per_tok, hidden_size
        )

        expected_shape = (num_experts_per_tok, 1, batch_seq, hidden_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        logger.info(f"✓ Golden shape test passed: {input_weights.shape} -> {output.shape}")

    @pytest.mark.parametrize("hidden_size", [256, 512, 1024, 2048, 7168])
    def test_broadcast_correctness(self, hidden_size):
        """Verify that each weight is correctly broadcast across hidden dimension."""
        batch_seq = 4
        num_experts_per_tok = 8

        torch.manual_seed(123)
        input_weights = torch.randn(batch_seq, num_experts_per_tok, dtype=torch.bfloat16)

        output = PrepareGptOssExpertsTensorSingleCore.golden(
            input_weights, num_experts_per_tok, hidden_size
        )

        # Check each weight is correctly broadcast
        for k in range(num_experts_per_tok):
            for bs in range(batch_seq):
                expected_value = input_weights[bs, k]
                actual_row = output[k, 0, bs, :]
                assert torch.allclose(actual_row, expected_value.expand(hidden_size), atol=1e-4), \
                    f"Broadcast mismatch at k={k}, bs={bs}"

        logger.info(f"✓ Broadcast correctness verified for H={hidden_size}")


# =============================================================================
# Functional Correctness Tests (Device Required)
# =============================================================================


class TestFusedOpCorrectness:
    """Test fused op correctness against golden reference."""

    @pytest.mark.parametrize(
        "batch_seq,num_experts_per_tok,hidden_size",
        [
            (1, 8, 256),
            (4, 8, 512),
            (8, 4, 1024),
        ],
    )
    def test_fused_op_matches_golden(self, device, batch_seq, num_experts_per_tok, hidden_size):
        """Test that fused kernel output matches golden reference."""
        torch.manual_seed(42)
        input_weights = torch.randn(batch_seq, num_experts_per_tok, dtype=torch.bfloat16)

        # Compute expected output using golden
        torch_expected = PrepareGptOssExpertsTensorSingleCore.golden(
            input_weights, num_experts_per_tok, hidden_size
        )

        logger.info(f"Testing fused op: [{batch_seq}, {num_experts_per_tok}] -> "
                   f"[{num_experts_per_tok}, 1, {batch_seq}, {hidden_size}]")

        # Pad input to fit tile dimensions
        input_tile_shape = (1, 32)
        padded_k = math.ceil(num_experts_per_tok / input_tile_shape[1]) * input_tile_shape[1]
        padded_bs = math.ceil(batch_seq / input_tile_shape[0]) * input_tile_shape[0]

        input_padded = torch.zeros(padded_bs, padded_k, dtype=torch.bfloat16)
        input_padded[:batch_seq, :num_experts_per_tok] = input_weights

        # Create sharded input tensor
        ttnn_input = create_sharded_input_tensor(input_padded, device, input_tile_shape)

        # Prepare output shape
        output_tile_shape = (32, 32)
        padded_h = math.ceil(hidden_size / output_tile_shape[1]) * output_tile_shape[1]
        output_shape_4d = (num_experts_per_tok, 1, batch_seq, padded_h)
        output_zeros = torch.zeros(output_shape_4d, dtype=torch.bfloat16)

        ttnn_output = create_sharded_output_tensor(output_zeros, device, output_tile_shape)

        # Run fused op
        result = PrepareGptOssExpertsTensorSingleCore.op(
            ttnn_input,
            ttnn_output,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
        )

        # Convert back to torch
        output_torch = ttnn.to_torch(result)
        output_torch = output_torch[:num_experts_per_tok, :1, :batch_seq, :hidden_size]

        # Verify
        assert output_torch.shape == torch_expected.shape, \
            f"Shape mismatch: expected {torch_expected.shape}, got {output_torch.shape}"

        passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.99)
        logger.info(pcc_message)
        assert passing, f"PCC test failed: {pcc_message}"
        logger.info("✓ Fused op matches golden!")


# =============================================================================
# Comparison Tests: Fused vs Unfused (Device Required)
# =============================================================================


class TestFusedVsUnfused:
    """Compare fused op output with original unfused TTNN implementation."""

    @pytest.mark.parametrize(
        "batch,seq,num_experts_per_tok,hidden_size",
        [
            (1, 1, 8, 256),
            (1, 4, 8, 512),
            (2, 4, 4, 1024),
            (4, 8, 8, 2048),
        ],
    )
    def test_fused_matches_unfused_output(self, device, batch, seq, num_experts_per_tok, hidden_size):
        """Verify fused op produces identical results to unfused TTNN ops."""
        torch.manual_seed(42)
        input_shape = (batch, 1, seq, num_experts_per_tok)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

        logger.info(f"Comparing fused vs unfused for input shape {input_shape}, H={hidden_size}")

        # Run unfused implementation
        ttnn_input_unfused = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        unfused_output = prepare_expert_weights_unfused(
            ttnn_input_unfused, num_experts_per_tok, hidden_size
        )
        unfused_torch = ttnn.to_torch(unfused_output)
        ttnn.deallocate(unfused_output)

        # Run fused golden reference (PyTorch)
        batch_seq = batch * seq
        torch_input_2d = torch_input.reshape(batch_seq, num_experts_per_tok)
        fused_expected = PrepareGptOssExpertsTensorSingleCore.golden(
            torch_input_2d, num_experts_per_tok, hidden_size
        )

        # Compare unfused TTNN output with fused golden
        passing, pcc_message = comp_pcc(unfused_torch, fused_expected, 0.999)
        logger.info(f"Unfused TTNN vs Fused Golden PCC: {pcc_message}")
        assert passing, f"Fused golden does not match unfused TTNN: {pcc_message}"
        logger.info("✓ Fused implementation matches unfused TTNN output!")


# =============================================================================
# End-to-End Performance Tests (Device Required)
# =============================================================================


class TestE2EPerformance:
    """End-to-end performance comparison tests."""

    @pytest.mark.parametrize(
        "batch_seq,num_experts_per_tok,hidden_size",
        [
            (32, 8, 7168),  # Realistic gpt-oss configuration
        ],
    )
    def test_fused_vs_unfused_perf(self, device, batch_seq, num_experts_per_tok, hidden_size):
        """Compare e2e performance of fused vs unfused implementations."""
        torch.manual_seed(42)

        # Original 4D input shape for unfused
        input_shape_4d = (batch_seq, 1, 1, num_experts_per_tok)
        torch_input_4d = torch.randn(input_shape_4d, dtype=torch.bfloat16)

        # 2D input for fused
        torch_input_2d = torch_input_4d.reshape(batch_seq, num_experts_per_tok)

        logger.info(f"Performance test: B*S={batch_seq}, K={num_experts_per_tok}, H={hidden_size}")

        # Prepare unfused op
        ttnn_input_unfused = ttnn.from_torch(
            torch_input_4d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        def unfused_op():
            return prepare_expert_weights_unfused(
                ttnn_input_unfused, num_experts_per_tok, hidden_size
            )

        # Measure unfused performance
        unfused_perf_us = _measure_perf_us(
            device, unfused_op, PERF_WARMUP_ITERS, PERF_MEASURE_ITERS
        )
        logger.info(f"Unfused avg: {unfused_perf_us:.3f} us over {PERF_MEASURE_ITERS} iters")

        # Prepare fused op inputs
        input_tile_shape = (1, 32)
        padded_k = math.ceil(num_experts_per_tok / input_tile_shape[1]) * input_tile_shape[1]
        padded_bs = math.ceil(batch_seq / input_tile_shape[0]) * input_tile_shape[0]

        input_padded = torch.zeros(padded_bs, padded_k, dtype=torch.bfloat16)
        input_padded[:batch_seq, :num_experts_per_tok] = torch_input_2d

        ttnn_input_fused = create_sharded_input_tensor(input_padded, device, input_tile_shape)

        output_tile_shape = (32, 32)
        padded_h = math.ceil(hidden_size / output_tile_shape[1]) * output_tile_shape[1]
        output_shape_4d = (num_experts_per_tok, 1, batch_seq, padded_h)
        output_zeros = torch.zeros(output_shape_4d, dtype=torch.bfloat16)

        ttnn_output_fused = create_sharded_output_tensor(output_zeros, device, output_tile_shape)

        def fused_op():
            return PrepareGptOssExpertsTensorSingleCore.op(
                ttnn_input_fused,
                ttnn_output_fused,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
            )

        # Measure fused performance
        fused_perf_us = _measure_perf_us(
            device, fused_op, PERF_WARMUP_ITERS, PERF_MEASURE_ITERS
        )
        logger.info(f"Fused avg: {fused_perf_us:.3f} us over {PERF_MEASURE_ITERS} iters")

        # Calculate speedup
        speedup = unfused_perf_us / fused_perf_us if fused_perf_us > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x (unfused/fused)")

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Performance Summary: B*S={batch_seq}, K={num_experts_per_tok}, H={hidden_size}")
        logger.info(f"  Unfused: {unfused_perf_us:.3f} us")
        logger.info(f"  Fused:   {fused_perf_us:.3f} us")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"{'='*60}\n")


# =============================================================================
# Device Performance Tests (Tracy Profiler)
# =============================================================================


class TestDevicePerformance:
    """Device-level performance tests using Tracy profiler."""

    @pytest.mark.parametrize(
        "batch_seq,num_experts_per_tok,hidden_size",
        [
            (32, 8, 7168),
        ],
    )
    @pytest.mark.parametrize("use_signposts", [True])
    def test_device_perf_unfused(
        self, device, batch_seq, num_experts_per_tok, hidden_size, use_signposts
    ):
        """Measure device kernel duration for unfused implementation.

        Run with: pytest ... --capture=no
        Enable Tracy: TT_METAL_DEVICE_PROFILER=1 pytest ...
        """
        from tracy import signpost

        torch.manual_seed(42)
        input_shape = (batch_seq, 1, 1, num_experts_per_tok)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        def op_fn():
            return prepare_expert_weights_unfused(
                ttnn_input, num_experts_per_tok, hidden_size
            )

        # Warmup
        for _ in range(PERF_WARMUP_ITERS):
            result = op_fn()
            ttnn.synchronize_device(device)
            ttnn.deallocate(result)

        ttnn.synchronize_device(device)

        # Measure with signposts for Tracy
        if use_signposts:
            signpost("start")

        for _ in range(DEVICE_PERF_ITERS):
            result = op_fn()
            ttnn.synchronize_device(device)
            ttnn.deallocate(result)

        if use_signposts:
            signpost("stop")

        logger.info(f"Device perf test (unfused) completed: {DEVICE_PERF_ITERS} iterations")
        logger.info("Use Tracy to analyze: tt-perf-report -d generated/profiler/...")

    @pytest.mark.parametrize(
        "batch_seq,num_experts_per_tok,hidden_size",
        [
            (32, 8, 7168),
        ],
    )
    @pytest.mark.parametrize("use_signposts", [True])
    def test_device_perf_fused(
        self, device, batch_seq, num_experts_per_tok, hidden_size, use_signposts
    ):
        """Measure device kernel duration for fused implementation.

        Run with: pytest ... --capture=no
        Enable Tracy: TT_METAL_DEVICE_PROFILER=1 pytest ...
        """
        from tracy import signpost

        torch.manual_seed(42)
        torch_input = torch.randn(batch_seq, num_experts_per_tok, dtype=torch.bfloat16)

        # Prepare inputs
        input_tile_shape = (1, 32)
        padded_k = math.ceil(num_experts_per_tok / input_tile_shape[1]) * input_tile_shape[1]
        padded_bs = math.ceil(batch_seq / input_tile_shape[0]) * input_tile_shape[0]

        input_padded = torch.zeros(padded_bs, padded_k, dtype=torch.bfloat16)
        input_padded[:batch_seq, :num_experts_per_tok] = torch_input

        ttnn_input = create_sharded_input_tensor(input_padded, device, input_tile_shape)

        output_tile_shape = (32, 32)
        padded_h = math.ceil(hidden_size / output_tile_shape[1]) * output_tile_shape[1]
        output_shape_4d = (num_experts_per_tok, 1, batch_seq, padded_h)
        output_zeros = torch.zeros(output_shape_4d, dtype=torch.bfloat16)

        ttnn_output = create_sharded_output_tensor(output_zeros, device, output_tile_shape)

        def op_fn():
            return PrepareGptOssExpertsTensorSingleCore.op(
                ttnn_input,
                ttnn_output,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
            )

        # Warmup
        for _ in range(PERF_WARMUP_ITERS):
            result = op_fn()
            ttnn.synchronize_device(device)

        ttnn.synchronize_device(device)

        # Measure with signposts for Tracy
        if use_signposts:
            signpost("start")

        for _ in range(DEVICE_PERF_ITERS):
            result = op_fn()
            ttnn.synchronize_device(device)

        if use_signposts:
            signpost("stop")

        logger.info(f"Device perf test (fused) completed: {DEVICE_PERF_ITERS} iterations")
        logger.info("Use Tracy to analyze: tt-perf-report -d generated/profiler/...")


# =============================================================================
# Benchmark Data Collection
# =============================================================================


@pytest.mark.parametrize(
    "batch_seq,num_experts_per_tok,hidden_size,expected_fused_us,expected_unfused_us",
    [
        # Configuration, expected fused (us), expected unfused (us)
        # TODO: Fill in measured baselines
        (32, 8, 7168, 0, 0),
    ],
)
def test_perf_regression(
    device,
    batch_seq,
    num_experts_per_tok,
    hidden_size,
    expected_fused_us,
    expected_unfused_us,
):
    """Performance regression test with expected baselines."""
    torch.manual_seed(42)

    # Measure unfused
    input_shape_4d = (batch_seq, 1, 1, num_experts_per_tok)
    torch_input_4d = torch.randn(input_shape_4d, dtype=torch.bfloat16)

    ttnn_input_unfused = ttnn.from_torch(
        torch_input_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    def unfused_op():
        return prepare_expert_weights_unfused(
            ttnn_input_unfused, num_experts_per_tok, hidden_size
        )

    unfused_perf_us = _measure_perf_us(
        device, unfused_op, PERF_WARMUP_ITERS, PERF_MEASURE_ITERS
    )

    # Measure fused
    torch_input_2d = torch_input_4d.reshape(batch_seq, num_experts_per_tok)
    input_tile_shape = (1, 32)
    padded_k = math.ceil(num_experts_per_tok / input_tile_shape[1]) * input_tile_shape[1]
    padded_bs = math.ceil(batch_seq / input_tile_shape[0]) * input_tile_shape[0]

    input_padded = torch.zeros(padded_bs, padded_k, dtype=torch.bfloat16)
    input_padded[:batch_seq, :num_experts_per_tok] = torch_input_2d

    ttnn_input_fused = create_sharded_input_tensor(input_padded, device, input_tile_shape)

    output_tile_shape = (32, 32)
    padded_h = math.ceil(hidden_size / output_tile_shape[1]) * output_tile_shape[1]
    output_shape_4d = (num_experts_per_tok, 1, batch_seq, padded_h)
    output_zeros = torch.zeros(output_shape_4d, dtype=torch.bfloat16)

    ttnn_output_fused = create_sharded_output_tensor(output_zeros, device, output_tile_shape)

    def fused_op():
        return PrepareGptOssExpertsTensorSingleCore.op(
            ttnn_input_fused,
            ttnn_output_fused,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
        )

    fused_perf_us = _measure_perf_us(
        device, fused_op, PERF_WARMUP_ITERS, PERF_MEASURE_ITERS
    )

    # Log results
    speedup = unfused_perf_us / fused_perf_us if fused_perf_us > 0 else float('inf')

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"prepare_expert_weights_B{batch_seq}_K{num_experts_per_tok}_H{hidden_size}"

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    perf_profiler.end(step_name)
    perf_profiler.end("run")

    benchmark_data.add_measurement(perf_profiler, 0, step_name, "unfused_us", unfused_perf_us)
    benchmark_data.add_measurement(perf_profiler, 0, step_name, "fused_us", fused_perf_us)
    benchmark_data.add_measurement(perf_profiler, 0, step_name, "speedup", speedup)

    benchmark_data.save_partial_run_json(
        perf_profiler,
        run_type="prepare_expert_weights_perf",
        ml_model_name="gpt-oss",
        batch_size=batch_seq,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Perf Regression Test: B*S={batch_seq}, K={num_experts_per_tok}, H={hidden_size}")
    logger.info(f"  Unfused: {unfused_perf_us:.3f} us")
    logger.info(f"  Fused:   {fused_perf_us:.3f} us")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"{'='*60}\n")

    # Check regression if baselines are set
    if expected_fused_us > 0:
        margin = DEVICE_PERF_MARGIN
        assert fused_perf_us <= expected_fused_us * (1 + margin), \
            f"Fused perf regression: {fused_perf_us:.3f}us > {expected_fused_us:.3f}us (+{margin:.0%})"

    if expected_unfused_us > 0:
        margin = DEVICE_PERF_MARGIN
        assert unfused_perf_us <= expected_unfused_us * (1 + margin), \
            f"Unfused perf regression: {unfused_perf_us:.3f}us > {expected_unfused_us:.3f}us (+{margin:.0%})"
