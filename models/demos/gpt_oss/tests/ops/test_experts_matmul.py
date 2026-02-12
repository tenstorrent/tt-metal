# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for experts sparse matmul operations.

This module provides focused unit tests for the sparse matmul operations used in
the expert MLP layer. Tests can run on a single device since they don't require
inter-device communication.

Test Types:
1. Stress Test: Runs the matmul operation many times to verify stability
2. Parameter Sweep: Sweeps program config parameters to profile and find optimal settings
"""

import os
from dataclasses import asdict, dataclass
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tt.experts_throughput.config import ThroughputExpertConfig, ThroughputProgramConfig
from models.demos.gpt_oss.tt.experts_throughput.weights import ThroughputExpertWeights

# ==============================================================================
# Test Configuration
# ==============================================================================
STRESS_TEST_ITERS = 100  # Number of iterations for stress test
STRESS_TEST_WARMUP = 10  # Warmup iterations before stress test

PARAM_SWEEP_WARMUP = 5  # Warmup iterations for each parameter config
PARAM_SWEEP_ITERS = 20  # Measurement iterations for each parameter config

# Output directory for parameter sweep results
PARAM_SWEEP_OUTPUT_DIR = "param_sweep_results"


@dataclass
class MatmulResult:
    """Performance result for a single matmul configuration."""

    # Configuration parameters
    cores: tuple[int, int]
    in0_block_w: int
    out_subblock_h: int
    out_subblock_w: int
    per_core_M: int

    # Problem dimensions
    batch_size: int
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_experts_per_device: int

    # Validity
    passed: bool
    pcc: Optional[float] = None
    error_msg: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert tuple to list for JSON
        result["cores"] = list(result["cores"])
        return result


# ==============================================================================
# Reference Implementation
# ==============================================================================
def _apply_swiglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """PyTorch reference implementation for SwiGLU activation."""
    gate_clamped = torch.clamp(gate, max=limit)
    up_clamped = torch.clamp(up, min=-limit, max=limit)
    gate_alpha = gate_clamped * alpha
    gate_sigmoid = torch.sigmoid(gate_alpha)
    glu = gate_clamped * gate_sigmoid
    up_plus_one = up_clamped + 1.0
    result = up_plus_one * glu
    return result


def experts_matmul_reference(
    input_tensor: torch.Tensor,
    w1: torch.Tensor,
    config: ThroughputExpertConfig,
) -> torch.Tensor:
    """PyTorch reference implementation for expert matmul forward pass.

    Args:
        input_tensor: Input [1, 1, B*S, H]
        w1: Gate weights [E, H, I]
        config: Expert configuration

    Returns:
        Output tensor [E, B*S, H]
    """
    # Reshape input: [1, 1, B*S, H] -> [B*S, H]
    total_tokens = input_tensor.shape[2]
    x = input_tensor.reshape(total_tokens, config.hidden_size)

    # Expand for all experts: [B*S, H] -> [E, B*S, H]
    num_experts = w1.shape[0]
    x_expanded = x.unsqueeze(0).expand(num_experts, -1, -1)

    # Gate projection: [E, B*S, H] @ [E, H, I] -> [E, B*S, I]
    output = torch.bmm(x_expanded, w1)

    return output


# ==============================================================================
# TTNN Implementation
# ==============================================================================
def experts_matmul_ttnn(
    input_tensor: ttnn.Tensor,
    weights: ThroughputExpertWeights,
    program_config: ThroughputProgramConfig,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """TTNN implementation of expert matmul operations.

    This mirrors the expert_mlp_forward function but is self-contained for unit testing.

    Args:
        input_tensor: Input tensor [1, 1, B*S, H] in TILE layout
        sparsity: Sparsity tensor [B*S/block, 1, 1, E]
        weights: Expert weights
        config: Expert configuration
        program_config: Program configuration
        memory_config: Memory configuration

    Returns:
        Output tensor [E, 1, B*S, H] in ROW_MAJOR layout
    """
    # Gate projection (w1)
    output = ttnn.matmul(
        input_tensor,
        weights.w1,
        memory_config=memory_config,
        program_config=program_config,
    )

    return output


# ==============================================================================
# Helper Functions
# ==============================================================================
def create_test_weights(
    device,
    config: ThroughputExpertConfig,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> tuple[ThroughputExpertWeights, dict]:
    """Create random weights for testing.

    Returns:
        Tuple of (TTNN weights, reference weights dict)
    """
    num_experts = config.num_experts_per_device
    hidden = config.hidden_size
    intermediate = config.intermediate_size

    # Create random reference weights
    ref_weights = {
        "w1": torch.randn(num_experts, hidden, intermediate, dtype=torch.float32),
        "w1_bias": torch.randn(num_experts, 1, intermediate, dtype=torch.float32),
        "w3": torch.randn(num_experts, hidden, intermediate, dtype=torch.float32),
        "w3_bias": torch.randn(num_experts, 1, intermediate, dtype=torch.float32),
        "w2": torch.randn(num_experts, intermediate, hidden, dtype=torch.float32),
        "w2_bias": torch.randn(num_experts, 1, hidden, dtype=torch.float32),
    }

    # Convert to TTNN tensors
    w1_tt = ttnn.from_torch(
        ref_weights["w1"].unsqueeze(0).bfloat16(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )
    w2_tt = ttnn.from_torch(
        ref_weights["w2"].unsqueeze(0).bfloat16(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )
    w3_tt = ttnn.from_torch(
        ref_weights["w3"].unsqueeze(0).bfloat16(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    # Reshape biases for sparse matmul format: [1, 1, 1, E, 1, dim]
    w1_bias_tt = ttnn.from_torch(
        ref_weights["w1_bias"].reshape(1, 1, 1, num_experts, 1, intermediate).bfloat16(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )
    w2_bias_tt = ttnn.from_torch(
        ref_weights["w2_bias"].reshape(1, num_experts, 1, hidden).bfloat16(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )
    w3_bias_tt = ttnn.from_torch(
        ref_weights["w3_bias"].reshape(1, 1, 1, num_experts, 1, intermediate).bfloat16(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    weights = ThroughputExpertWeights(
        w1=w1_tt,
        w2=w2_tt,
        w3=w3_tt,
        w1_bias=w1_bias_tt,
        w2_bias=w2_bias_tt,
        w3_bias=w3_bias_tt,
    )

    return weights, ref_weights


def create_test_inputs(
    device,
    batch_size: int,
    seq_len: int,
    config: ThroughputExpertConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor, torch.Tensor, torch.Tensor]:
    """Create test input tensors.

    Returns:
        Tuple of (TTNN input, TTNN sparsity, torch input, torch sparsity)
    """
    total_tokens = batch_size * seq_len

    # Create random input
    input_torch = torch.randn(1, 1, total_tokens, config.hidden_size, dtype=torch.bfloat16)

    # Convert to TTNN
    input_tt = ttnn.from_torch(
        input_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    input_tt = ttnn.repeat(input_tt, ttnn.Shape((1, config.num_experts_per_device, 1, 1)))

    return input_tt, input_torch


def validate_output(
    tt_output: torch.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float = 0.998,
) -> tuple[bool, float]:
    """Validate TTNN output against reference.

    Returns:
        Tuple of (passed, pcc)
    """
    passing, pcc = comp_pcc(ref_output.float(), tt_output.float(), expected_pcc)
    return passing, pcc


# ==============================================================================
# Stress Test
# ==============================================================================
@pytest.mark.parametrize("batch_size", [32], ids=["batch32"])
@pytest.mark.parametrize("seq_len", [1], ids=["seq1"])
@pytest.mark.parametrize("num_iterations", [100], ids=["iter100"])
def test_experts_matmul_stress(
    batch_size: int,
    seq_len: int,
    num_iterations: int,
    device,
):
    """Stress test: Run experts matmul many times to verify stability.

    This test runs the matmul operation many times in a row to ensure:
    1. Memory is properly managed (no leaks or corruption)
    2. Results remain consistent
    3. Performance is stable

    Args:
        batch_size: Batch size per device
        seq_len: Sequence length
        num_iterations: Number of times to run the operation
        device: TTNN device fixture
    """
    logger.info(f"Starting stress test: {num_iterations} iterations, batch={batch_size}, seq_len={seq_len}")

    # Get model config
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"), trust_remote_code=True)
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts
    num_experts_per_tok = config.num_experts_per_tok

    # Create expert config for single device
    expert_config = ThroughputExpertConfig(
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=1,
    )

    # Create weights and inputs
    weights, ref_weights = create_test_weights(device, expert_config)
    input_tt, input_torch = create_test_inputs(device, batch_size, seq_len, expert_config)

    # Create program config
    program_config = ThroughputProgramConfig()
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Compute reference once
    ref_output = experts_matmul_reference(
        input_tensor=input_torch.float(),
        w1=ref_weights["w1"],
        config=expert_config,
    )

    # Warmup
    logger.info(f"Warmup: {STRESS_TEST_WARMUP} iterations")
    for i in range(STRESS_TEST_WARMUP):
        sparsity_fresh = ttnn.from_torch(
            sparsity_torch,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        output = experts_matmul_ttnn(input_tt, sparsity_fresh, weights, expert_config, program_config, memory_config)
        ttnn.synchronize_device(device)
        ttnn.deallocate(output)

    # Stress test iterations
    logger.info(f"Running {num_iterations} stress test iterations")
    pccs = []
    times = []

    for i in range(num_iterations):
        # Create fresh sparsity tensor each iteration (required)
        sparsity_fresh = ttnn.from_torch(
            sparsity_torch,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        output = experts_matmul_ttnn(input_tt, sparsity_fresh, weights, expert_config, program_config, memory_config)

        # Validate every 10th iteration
        if i % 10 == 0 or i == num_iterations - 1:
            output_torch = ttnn.to_torch(output)
            output_torch = output_torch.reshape(expert_config.num_experts_per_device, batch_size * seq_len, hidden_size)
            passed, pcc = validate_output(output_torch, ref_output)
            pccs.append(pcc)
            if not passed:
                logger.error(f"Iteration {i}: Validation failed with PCC={pcc}")
                assert False, f"Stress test failed at iteration {i} with PCC={pcc}"

        ttnn.deallocate(output)

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{num_iterations} iterations completed")

    # Report results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    avg_pcc = sum(pccs) / len(pccs)
    min_pcc = min(pccs)

    logger.info("=" * 80)
    logger.info("STRESS TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Iterations: {num_iterations}")
    logger.info(f"Batch size: {batch_size}, Seq len: {seq_len}")
    logger.info(f"Performance (us): avg={avg_time:.2f}, min={min_time:.2f}, max={max_time:.2f}, std={std_time:.2f}")
    logger.info(f"PCC: avg={avg_pcc:.6f}, min={min_pcc:.6f}")
    logger.info(f"Status: PASSED")
    logger.info("=" * 80)

    # Cleanup
    ttnn.deallocate(input_tt)


# ==============================================================================
# Parameter Sweep Test
# ==============================================================================
def generate_program_configs():
    """Generate program configurations to sweep.

    Yields candidate configurations for gate_up and down matmuls.
    """
    # Core grid options - must fit within device grid (typically 8x8 or similar)
    core_grids = [
        (5, 9),  # Default
        (8, 8),
        (8, 7),
        (7, 8),
        (4, 8),
        (5, 8),
    ]

    # in0_block_w options - affects K dimension blocking
    # For hidden_size=2880: 2880/32 = 90 tiles, so factors: 1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90
    in0_block_w_options = [1, 2, 5, 10, 15, 30]
    # in0_block_w_options = [10, 15, 30]

    # Subblock options
    subblock_options = [(1, 1), (1, 2), (2, 1)]
    # subblock_options = [(1, 1)]

    # per_core_M options
    per_core_m_options = [1, 2]
    # per_core_m_options = [1]

    # per_core_N options
    # per_core_n_options = [1, 2]
    per_core_n_options = [2]

    # Generate all combinations (this could be large!)
    # Start with a subset for initial testing
    for cores in core_grids[:3]:  # Test first 3 core grids
        for in0_block_w in in0_block_w_options[:4]:  # Test first 4 block widths
            for out_subblock_h, out_subblock_w in subblock_options:
                for per_core_M in per_core_m_options:
                    for per_core_N in per_core_n_options:
                        # yield ThroughputProgramConfig(
                        #     gate_up_cores=gate_up_cores,
                        #     down_cores=down_cores,
                        #     in0_block_w=in0_block_w,
                        #     out_subblock_h=out_subblock_h,
                        #     out_subblock_w=out_subblock_w,
                        #     per_core_M=per_core_M,
                        # )
                        yield ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=ttnn.CoreCoord(cores[0], cores[1]),
                            in0_block_w=in0_block_w,
                            out_subblock_h=out_subblock_h,
                            out_subblock_w=out_subblock_w,
                            per_core_M=per_core_M,
                            per_core_N=per_core_N,
                            fuse_batch=False,
                            fused_activation=None,
                            mcast_in0=True,
                        )


@pytest.mark.parametrize("batch_size", [32], ids=["batch32"])
@pytest.mark.parametrize("seq_len", [1], ids=["seq1"])
def test_experts_matmul_param_sweep(
    batch_size: int,
    seq_len: int,
    mesh_device,
):
    """Parameter sweep: Test different program configs to find optimal performance.

    This test sweeps through different program configuration parameters:
    - Core grid sizes (gate_up_cores, down_cores)
    - Block sizes (in0_block_w)
    - Subblock sizes (out_subblock_h, out_subblock_w)
    - Per-core parameters (per_core_M)

    Results are saved to JSON for analysis.

    Args:
        batch_size: Batch size per device
        seq_len: Sequence length
        device: TTNN device fixture
    """
    logger.info("=" * 80)
    logger.info("PARAMETER SWEEP TEST")
    logger.info("=" * 80)
    logger.info(f"Batch size: {batch_size}, Seq len: {seq_len}")

    # Get model config
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"), trust_remote_code=True)
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts
    num_experts_per_device = num_experts // 32
    num_experts_per_tok = config.num_experts_per_tok

    # Create expert config
    expert_config = ThroughputExpertConfig(
        intermediate_size=intermediate_size,
        num_experts=num_experts_per_device,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=1,
    )

    # Create weights and inputs (reuse across all configs)
    weights, ref_weights = create_test_weights(mesh_device, expert_config)
    input_tt, input_torch = create_test_inputs(mesh_device, batch_size, seq_len, expert_config)
    # Compute reference once
    ref_output = experts_matmul_reference(
        input_tensor=input_torch.float(),
        w1=ref_weights["w1"],
        config=expert_config,
    )

    memory_config = ttnn.DRAM_MEMORY_CONFIG
    results = []

    # Generate and test program configs
    configs = list(generate_program_configs())
    logger.info(f"Testing {len(configs)} program configurations")

    for config_idx, program_config in enumerate(configs):
        logger.info(f"\nTesting config {config_idx + 1}/{len(configs)}")
        logger.info(f"  cores: {program_config.compute_with_storage_grid_size}")
        logger.info(f"  in0_block_w: {program_config.in0_block_w}")
        logger.info(f"  out_subblock: ({program_config.out_subblock_h}, {program_config.out_subblock_w})")
        logger.info(f"  per_core_M: {program_config.per_core_M}")

        try:
            # Run config
            output = experts_matmul_ttnn(input_tt, weights, program_config, memory_config)

            # Validate
            output_torch = ttnn.to_torch(
                output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=mesh_device.shape)
            )[:1, : expert_config.num_experts_per_device, :, :]
            passed, pcc = validate_output(output_torch, ref_output)
            assert passed, f"Matmul failed with PCC={pcc}"
            results.append(
                MatmulResult(
                    cores=program_config.compute_with_storage_grid_size,
                    in0_block_w=program_config.in0_block_w,
                    out_subblock_h=program_config.out_subblock_h,
                    out_subblock_w=program_config.out_subblock_w,
                    per_core_M=program_config.per_core_M,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts_per_device=expert_config.num_experts_per_device,
                    passed=passed,
                    pcc=pcc,
                )
            )
            logger.info(f"Matmul passed with PCC={pcc}")

        except Exception as e:
            logger.error(f"  Config failed with error: {e}")
            results.append(
                MatmulResult(
                    cores=program_config.compute_with_storage_grid_size,
                    in0_block_w=program_config.in0_block_w,
                    out_subblock_h=program_config.out_subblock_h,
                    out_subblock_w=program_config.out_subblock_w,
                    per_core_M=program_config.per_core_M,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts_per_device=expert_config.num_experts_per_device,
                    passed=False,
                    error_msg=str(e),
                )
            )

    # Summarize results
    passed_results = [r for r in results if r.passed]
    logger.info("=" * 80)
    logger.info("PARAMETER SWEEP SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configs tested: {len(results)}")
    logger.info(f"Passed: {len(passed_results)}")
    logger.info(f"Failed: {len(results) - len(passed_results)}")

    # Cleanup
    ttnn.deallocate(input_tt)

    # At least one config should work
    assert len(passed_results) > 0, "No program configurations passed validation"
