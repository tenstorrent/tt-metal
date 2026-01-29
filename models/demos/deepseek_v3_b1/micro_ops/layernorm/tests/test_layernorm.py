# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LayerNorm single-core generic op."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.layernorm.op import (
    CB_BETA_RM,
    CB_BETA_TILED,
    CB_GAMMA_RM,
    CB_GAMMA_TILED,
    CB_INPUT_RM,
    CB_INPUT_TILED,
    CB_INTERM,
    CB_OUTPUT_RM,
    CB_OUTPUT_TILED,
    CB_SCALARS,
    LayerNormSingleCore,
    _calculate_sizes,
    _create_cb_descriptors,
)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32],
        [1, 64],
        [4, 128],
    ],
)
def test_golden_vs_torch(shape):
    """
    Test that LayerNormSingleCore.golden() matches torch.nn.functional.layer_norm().

    Pass criteria: torch.allclose(golden, torch_ref, rtol=1e-4, atol=1e-4) for all shapes.
    """
    torch.manual_seed(42)

    # Create input tensor with random values
    input_tensor = torch.randn(shape, dtype=torch.float32)

    # Create gamma (scale) and beta (shift) parameters with shape matching last dimension
    W = shape[-1]
    gamma_tensor = torch.randn(W, dtype=torch.float32)
    beta_tensor = torch.randn(W, dtype=torch.float32)

    epsilon = 1e-6

    # Compute using our golden implementation
    golden_output = LayerNormSingleCore.golden(input_tensor, gamma_tensor, beta_tensor, epsilon)

    # Compute using torch.nn.functional.layer_norm (reference)
    torch_output = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=[W],
        weight=gamma_tensor,
        bias=beta_tensor,
        eps=epsilon,
    )

    # Verify outputs match
    assert torch.allclose(golden_output, torch_output, rtol=1e-4, atol=1e-4), (
        f"Golden output does not match torch.nn.functional.layer_norm for shape {shape}\n"
        f"Max absolute difference: {(golden_output - torch_output).abs().max().item()}"
    )


# =============================================================================
# Step 1.3.4: CB Configuration Tests
# =============================================================================


def test_cb_indices():
    """
    Test that CB indices are correctly defined and unique.
    """
    # Verify input/working CBs are in range 0-15
    input_cbs = [
        CB_INPUT_RM,
        CB_INPUT_TILED,
        CB_GAMMA_RM,
        CB_GAMMA_TILED,
        CB_BETA_RM,
        CB_BETA_TILED,
        CB_SCALARS,
        CB_INTERM,
    ]
    for cb in input_cbs:
        assert 0 <= cb <= 15, f"Input CB {cb} should be in range 0-15"

    # Verify output CBs are >= 16
    output_cbs = [CB_OUTPUT_TILED, CB_OUTPUT_RM]
    for cb in output_cbs:
        assert cb >= 16, f"Output CB {cb} should be >= 16"

    # Verify all CBs are unique
    all_cbs = input_cbs + output_cbs
    assert len(all_cbs) == len(set(all_cbs)), "CB indices must be unique"


@pytest.mark.parametrize(
    "shape,expected_tiles_per_row",
    [
        ([1, 32], 1),  # Single tile width
        ([1, 64], 2),  # Two tiles
        ([4, 128], 4),  # Four tiles
        ([2, 2, 256], 8),  # 3D input, 8 tiles
    ],
)
def test_calculate_sizes(shape, expected_tiles_per_row):
    """
    Test that _calculate_sizes correctly computes buffer dimensions.
    """
    dtype = ttnn.bfloat16
    sizes = _calculate_sizes(shape, dtype)

    # Verify W (final dimension)
    assert sizes["W"] == shape[-1], f"W should be {shape[-1]}"

    # Verify num_rows (product of all dims except last)
    expected_num_rows = 1
    for dim in shape[:-1]:
        expected_num_rows *= dim
    assert sizes["num_rows"] == expected_num_rows, f"num_rows should be {expected_num_rows}"

    # Verify tiles_per_row
    assert sizes["tiles_per_row"] == expected_tiles_per_row, f"tiles_per_row should be {expected_tiles_per_row}"

    # Verify stick_size is properly aligned to 32 bytes
    assert sizes["stick_size"] % 32 == 0, "stick_size must be aligned to 32 bytes"
    assert sizes["stick_size"] >= shape[-1] * sizes["element_size"], "stick_size must hold at least W elements"

    # Verify tile_size (32x32 * element_size)
    assert sizes["tile_size"] == 32 * 32 * sizes["element_size"], "tile_size should be 32*32*element_size"

    # Verify element_size for bfloat16
    assert sizes["element_size"] == 2, "bfloat16 element_size should be 2"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32],
        [1, 64],
        [4, 128],
    ],
)
def test_cb_configuration(shape):
    """
    Test that CB descriptors are created correctly without errors.

    Verifies:
    - All 10 CBs are created
    - Sizes are positive
    - Sizes are properly aligned
    """
    dtype = ttnn.bfloat16
    sizes = _calculate_sizes(shape, dtype)

    # Create a single core grid
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Create CB descriptors
    cb_descriptors = _create_cb_descriptors(core_grid, sizes, dtype)

    # Verify we have exactly 10 CB descriptors
    assert len(cb_descriptors) == 10, f"Expected 10 CB descriptors, got {len(cb_descriptors)}"

    # Verify each CB descriptor has positive, properly sized buffers
    stick_size = sizes["stick_size"]
    tile_size = sizes["tile_size"]
    tiles_per_row = sizes["tiles_per_row"]

    expected_sizes = {
        CB_INPUT_RM: 2 * stick_size,  # Double buffer
        CB_INPUT_TILED: tiles_per_row * tile_size,
        CB_GAMMA_RM: stick_size,  # Single buffer
        CB_GAMMA_TILED: tiles_per_row * tile_size,
        CB_BETA_RM: stick_size,  # Single buffer
        CB_BETA_TILED: tiles_per_row * tile_size,
        CB_SCALARS: tile_size,  # 1 tile
        CB_INTERM: tiles_per_row * tile_size,
        CB_OUTPUT_TILED: tiles_per_row * tile_size,
        CB_OUTPUT_RM: 2 * stick_size,  # Double buffer
    }

    for cb_desc in cb_descriptors:
        # Check that total_size is positive
        assert cb_desc.total_size > 0, f"CB total_size must be positive"

        # Check that format_descriptors has at least one entry
        assert len(cb_desc.format_descriptors) >= 1, "CB must have at least one format descriptor"

        # Get buffer_index from format descriptor
        buffer_index = cb_desc.format_descriptors[0].buffer_index

        # Verify total_size matches expected
        if buffer_index in expected_sizes:
            assert (
                cb_desc.total_size == expected_sizes[buffer_index]
            ), f"CB {buffer_index} total_size should be {expected_sizes[buffer_index]}, got {cb_desc.total_size}"
