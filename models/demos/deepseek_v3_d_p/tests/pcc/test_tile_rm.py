# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC (correctness) tests for tilize/untilize operations.

Tests 4K x 7K bfloat16 matrix conversions between TILE and ROW_MAJOR layouts.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "M, N, pcc_threshold",
    [
        pytest.param(4096, 7168, 0.9999, id="4Kx7K"),  # 128x224 tiles
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chip")], indirect=["mesh_device"])
def test_tilize(mesh_device, M, N, pcc_threshold) -> None:
    """Test tilize: ROW_MAJOR bfloat16 -> TILE bfloat16, verify round-trip correctness."""
    # Create random bfloat16 tensor
    torch_tensor = torch.rand(1, 1, M, N).bfloat16()

    # Create ttnn tensor in ROW_MAJOR layout
    tt_input = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )

    # Tilize: ROW_MAJOR -> TILE
    tt_tiled = ttnn.tilize(tt_input, use_multicore=True)

    # Verify layout changed
    assert tt_tiled.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {tt_tiled.layout}"

    # Convert back to ROW_MAJOR to verify data integrity
    tt_untiled = ttnn.untilize(tt_tiled, use_multicore=True)
    result = ttnn.to_torch(tt_untiled)

    # Cleanup
    tt_input.deallocate(True)
    tt_tiled.deallocate(True)
    tt_untiled.deallocate(True)

    # Verify round-trip correctness
    passed, pcc_value = comp_pcc(torch_tensor, result, pcc=pcc_threshold)
    assert passed, f"Tilize round-trip failed: PCC {pcc_value:.6f} < {pcc_threshold}"


@pytest.mark.parametrize(
    "M, N, pcc_threshold",
    [
        pytest.param(4096, 7168, 0.9999, id="4Kx7K"),  # 128x224 tiles
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chip")], indirect=["mesh_device"])
def test_untilize(mesh_device, M, N, pcc_threshold) -> None:
    """Test untilize: TILE bfloat16 -> ROW_MAJOR bfloat16, verify correctness."""
    # Create random bfloat16 tensor
    torch_tensor = torch.rand(1, 1, M, N).bfloat16()

    # Create ttnn tensor directly in TILE layout
    tt_tiled = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    # Verify initial layout
    assert tt_tiled.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {tt_tiled.layout}"

    # Untilize: TILE -> ROW_MAJOR
    tt_rm = ttnn.untilize(tt_tiled, use_multicore=True)

    # Verify layout changed
    assert tt_rm.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected ROW_MAJOR_LAYOUT, got {tt_rm.layout}"

    result = ttnn.to_torch(tt_rm)

    # Cleanup
    tt_tiled.deallocate(True)
    tt_rm.deallocate(True)

    # Verify correctness
    passed, pcc_value = comp_pcc(torch_tensor, result, pcc=pcc_threshold)
    assert passed, f"Untilize failed: PCC {pcc_value:.6f} < {pcc_threshold}"
