# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test to prove double activation bug when both program_config.fused_activation
and activation parameter are provided to ttnn.matmul/linear.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_double_activation_bug(device):
    """
    This test proves that when BOTH:
    1. program_config.fused_activation is set (e.g., SILU)
    2. activation parameter is also set (e.g., "silu")
    3. core_grid is NOT provided

    The activation is applied TWICE, which is a bug.
    """
    torch.manual_seed(0)

    # Use sizes that work with the program config
    batch_size = 1
    m_size = 32
    k_size = 64
    n_size = 128

    # Create input tensors with smaller values so matmul output is in range where
    # SILU shows clear difference between single and double application
    # SILU(x) = x * sigmoid(x), biggest diff is around x in [-2, 2]
    torch_input_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16) * 0.1
    torch_input_b = torch.randn((k_size, n_size), dtype=torch.bfloat16) * 0.1

    # Compute expected outputs
    torch_matmul = torch_input_a @ torch_input_b
    torch_single_silu = torch.nn.functional.silu(torch_matmul)
    torch_double_silu = torch.nn.functional.silu(torch_single_silu)  # SILU applied twice

    logger.info(f"torch_matmul sample: {torch_matmul[0, 0, :8]}")
    logger.info(f"torch_single_silu sample: {torch_single_silu[0, 0, :8]}")
    logger.info(f"torch_double_silu sample: {torch_double_silu[0, 0, :8]}")

    # Show the difference between single and double
    diff = torch.abs(torch_single_silu - torch_double_silu)
    logger.info(f"Absolute diff (single vs double): max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Create ttnn tensors (NON-SHARDED to bypass the sharded check)
    input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device)

    logger.info(f"input_a is_sharded: {input_a.is_sharded()}")
    logger.info(f"input_b is_sharded: {input_b.is_sharded()}")

    # Create program config with fused_activation=SILU
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=1,
        out_block_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),  # <-- FIRST SILU
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=0,
        untilize_out=False,
    )

    # Call matmul with BOTH program_config.fused_activation AND activation parameter
    # This should trigger the double activation bug
    output = ttnn.matmul(
        input_a,
        input_b,
        program_config=program_config,
        activation="silu",  # <-- SECOND SILU (bug: this gets applied as post-process)
        # NOTE: no core_grid provided, so user_core_coord is nullopt
    )

    output_torch = ttnn.to_torch(output)
    logger.info(f"ttnn output sample: {output_torch[0, 0, :8]}")

    # Calculate absolute differences (more reliable than PCC for this test)
    diff_vs_single = torch.abs(output_torch - torch_single_silu)
    diff_vs_double = torch.abs(output_torch - torch_double_silu)
    mean_diff_single = diff_vs_single.mean().item()
    mean_diff_double = diff_vs_double.mean().item()

    logger.info(f"Diff vs SINGLE SILU: max={diff_vs_single.max():.6f}, mean={mean_diff_single:.6f}")
    logger.info(f"Diff vs DOUBLE SILU: max={diff_vs_double.max():.6f}, mean={mean_diff_double:.6f}")

    # If output is closer to double SILU than single SILU, the bug exists!
    if mean_diff_double < mean_diff_single:
        logger.error("=" * 60)
        logger.error("BUG CONFIRMED: Output matches DOUBLE SILU better than SINGLE SILU!")
        logger.error(f"  Mean diff vs SINGLE SILU: {mean_diff_single:.6f}")
        logger.error(f"  Mean diff vs DOUBLE SILU: {mean_diff_double:.6f}")
        logger.error(f"  Ratio: {mean_diff_single / mean_diff_double:.1f}x closer to double SILU")
        logger.error("=" * 60)
        # Fail the test to prove the bug
        assert False, (
            f"DOUBLE ACTIVATION BUG! Output is {mean_diff_single / mean_diff_double:.1f}x closer to "
            f"double SILU (diff={mean_diff_double:.6f}) than single SILU (diff={mean_diff_single:.6f})"
        )
    else:
        logger.info("No bug detected - output matches single SILU as expected")
        assert_with_pcc(torch_single_silu, output_torch, 0.99)


def test_no_double_activation_with_core_grid(device):
    """
    This test shows that when core_grid IS provided, there's no double activation
    because user_core_coord.has_value() is true, skipping the post-process activation.
    """
    torch.manual_seed(0)

    batch_size = 1
    m_size = 32
    k_size = 64
    n_size = 128

    torch_input_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)

    torch_matmul = torch_input_a @ torch_input_b
    torch_single_silu = torch.nn.functional.silu(torch_matmul)

    input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device)

    # With core_grid provided, activation goes into auto-generated program config
    # and is NOT double-applied
    output = ttnn.matmul(
        input_a,
        input_b,
        activation="silu",
        core_grid=device.core_grid,  # <-- This prevents double activation
    )

    output_torch = ttnn.to_torch(output)

    # This should pass - single SILU as expected
    assert_with_pcc(torch_single_silu, output_torch, 0.99)
    logger.info("PASS: With core_grid, activation is applied only once")


def test_program_config_fused_activation_only(device):
    """
    Control test: only program_config.fused_activation, no activation parameter.
    This should work correctly with single SILU.
    """
    torch.manual_seed(0)

    batch_size = 1
    m_size = 32
    k_size = 64
    n_size = 128

    torch_input_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)

    torch_matmul = torch_input_a @ torch_input_b
    torch_single_silu = torch.nn.functional.silu(torch_matmul)

    input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=1,
        out_block_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=0,
        untilize_out=False,
    )

    # Only program_config.fused_activation, NO activation parameter
    output = ttnn.matmul(
        input_a,
        input_b,
        program_config=program_config,
        # NO activation parameter
    )

    output_torch = ttnn.to_torch(output)

    # This should pass - single SILU as expected
    assert_with_pcc(torch_single_silu, output_torch, 0.99)
    logger.info("PASS: With only program_config.fused_activation, activation is applied once")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
