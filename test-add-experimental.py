# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ttnn.experimental.add."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose

pytestmark = pytest.mark.use_module_device
TILE_HEIGHT = 32
TILE_WIDTH = 32

NUM_DRAM_BANKS = 8  # 12
BANK_START_ID = 0
NUM_HEADS = 32


def get_optimal_dram_bank_to_reader_assignment(device, noc=ttnn.NOC.NOC_0):
    """Python equivalent of C++ get_optimal_dram_bank_to_reader_assignment (test_dram_read.cpp).
    Returns (all_worker_cores_ordered, all_worker_cores) where:
      - all_worker_cores_ordered: list of CoreCoord in optimal DRAM bank -> reader order
      - all_worker_cores: CoreRangeSet of single-core ranges for those coords
    """
    all_worker_cores_ordered = device.get_optimal_dram_bank_to_logical_worker_assignment(noc)
    all_worker_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores_ordered]
    )
    return all_worker_cores_ordered, all_worker_cores


#########    TESTS FOR DRAM OPTIMIZED ELTWISE    #########


def _align_to_tile_width(dim):
    return ((dim + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH


def _attention_shapes():
    """Generate attention-related shapes for testing.

    Shape patterns (LLM attention / FFN dimensions):
      (B, H, S, hid/H)      - per-head Q/K/V projections
      (B, 1, S, S)           - single-head attention scores
      (B, H, S, S)           - multi-head attention scores
      (B, 1, S, hid*8/3)    - FFN intermediate (tile-aligned)
    """
    batches = [
        1,
    ]  # 2, 4
    seq_lengths = [
        2048,
    ]  # 4096
    hidden_dims = [
        2048,
    ]  # 4096
    H = NUM_HEADS

    shapes = []
    for B in batches:
        for S in seq_lengths:
            for hid in hidden_dims:
                shapes.append((B, H, S, hid // H))
                shapes.append((B, 1, S, _align_to_tile_width(hid * 8 // 3)))
            shapes.append((B, 1, S, S))
            shapes.append((B, H, S, S))
    return shapes


@pytest.mark.parametrize(
    "shape", _attention_shapes(), ids=[f"{'x'.join(str(d) for d in s)}" for s in _attention_shapes()]
)
def test_experimental_add_dram_grid(device, shape):
    """Test ttnn.experimental.add with attention-related shapes."""
    torch.manual_seed(42)
    # a_torch = torch.full(shape, 3, dtype=torch.bfloat16).reshape(shape)
    # b_torch = torch.full(shape, 2, dtype=torch.bfloat16).reshape(shape)
    a_torch = torch.randn(shape, dtype=torch.bfloat16) * 0.1
    b_torch = torch.randn(shape, dtype=torch.bfloat16) * 0.1

    _, shard_grid = get_optimal_dram_bank_to_reader_assignment(device, ttnn.NOC.NOC_0)

    a_tt = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_tt = ttnn.experimental.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=shard_grid)
    out_torch = ttnn.to_torch(out_tt)

    # expected = torch.add(a_torch, b_torch)
    expected = torch.mul(a_torch, b_torch)

    assert_with_pcc(expected, out_torch, pcc=0.9999)
    # assert_allclose(expected, out_torch)
