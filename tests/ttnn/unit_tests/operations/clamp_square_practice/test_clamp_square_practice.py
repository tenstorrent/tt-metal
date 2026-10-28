# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for clamp_square_practice (Phase 0) — the immutable spec.

y = clamp(x, min_val, max_val) ** 2, reference torch.clamp(x, min_val, max_val) ** 2.

Phase 0 contract: bfloat16, TILE_LAYOUT, rank >= 2, H/W multiples of 32, interleaved DRAM.
Output: same shape / dtype / layout, interleaved DRAM. One device dispatch, multi-core.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.clamp_square_practice import clamp_square_practice

PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

SHAPES = [
    pytest.param((32, 32), id="rank2_single_tile"),
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 64, 128), id="multi_tile"),
    pytest.param((1, 1, 96, 64), id="non_square_tall"),
    pytest.param((1, 1, 32, 1024), id="one_tile_row_wide"),
    pytest.param((4, 64, 96), id="rank3"),
    pytest.param((2, 3, 64, 96), id="multi_batch"),
    pytest.param((3, 5, 96, 224), id="ragged_per_core_tail"),  # 315 tiles: uneven per-core split
    pytest.param((2, 4, 256, 512), id="many_blocks_per_core"),  # 1024 tiles
]

BOUNDS = [
    pytest.param(-1.0, 1.0, id="symmetric"),
    pytest.param(0.0, 2.5, id="asymmetric"),
    pytest.param(-3.0, 0.5, id="neg_heavy"),
    pytest.param(-0.3, 0.7, id="non_bf16_exact_bounds"),
]


def _reference(torch_input_bf16, min_val, max_val):
    return torch.clamp(torch_input_bf16.to(torch.float32), min_val, max_val) ** 2


def _make_input(shape, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input


def _check(tt_output, expected, shape):
    assert list(tt_output.shape) == list(shape)
    assert tt_output.dtype == ttnn.bfloat16
    assert tt_output.layout == ttnn.TILE_LAYOUT
    mem = tt_output.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert mem.buffer_type == ttnn.BufferType.DRAM

    actual = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(expected, actual, PCC[ttnn.bfloat16])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", BOUNDS)
def test_clamp_square_practice(device, shape, min_val, max_val):
    torch_input, tt_input = _make_input(shape, device)
    expected = _reference(torch_input, min_val, max_val)
    tt_output = clamp_square_practice(tt_input, min_val, max_val)
    _check(tt_output, expected, shape)


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (2, 3, 64, 96)])
def test_clamp_square_practice_keyword_args(device, shape):
    torch_input, tt_input = _make_input(shape, device)
    expected = _reference(torch_input, -3.0, 0.5)
    tt_output = clamp_square_practice(tt_input, min_val=-3.0, max_val=0.5)
    _check(tt_output, expected, shape)


def test_clamp_square_practice_wide_bounds_is_square(device):
    """Bounds wider than the data: the op reduces to x**2."""
    shape = (1, 1, 128, 256)
    torch_input, tt_input = _make_input(shape, device)
    expected = torch_input.to(torch.float32) ** 2
    tt_output = clamp_square_practice(tt_input, -100.0, 100.0)
    _check(tt_output, expected, shape)


def test_clamp_square_practice_equal_bounds(device):
    """min_val == max_val is legal: every output element is min_val ** 2."""
    shape = (1, 1, 64, 64)
    _, tt_input = _make_input(shape, device)
    tt_output = clamp_square_practice(tt_input, 0.5, 0.5)
    actual = ttnn.to_torch(tt_output).to(torch.float32)
    assert list(actual.shape) == list(shape)
    assert torch.allclose(actual, torch.full(shape, 0.25), atol=1e-3)


# ---------------------------------------------------------------------------
# Validation — must raise ValueError or RuntimeError on the Python side.
# ---------------------------------------------------------------------------

VALIDATION_ERRORS = (ValueError, RuntimeError)


def test_rejects_wrong_dtype(device, expect_error):
    _, tt_input = _make_input((1, 1, 32, 32), device, dtype=ttnn.float32)
    with expect_error(VALIDATION_ERRORS, ""):
        clamp_square_practice(tt_input, -1.0, 1.0)


def test_rejects_row_major_layout(device, expect_error):
    _, tt_input = _make_input((1, 1, 32, 32), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(VALIDATION_ERRORS, ""):
        clamp_square_practice(tt_input, -1.0, 1.0)


def test_rejects_rank_below_two(device, expect_error):
    try:
        _, tt_input = _make_input((64,), device)
    except Exception as e:  # tensor creation itself may refuse a rank-1 tiled tensor
        pytest.skip(f"cannot construct a rank-1 TILE tensor on this build: {e}")
    with expect_error(VALIDATION_ERRORS, ""):
        clamp_square_practice(tt_input, -1.0, 1.0)


@pytest.mark.parametrize("shape", [(1, 1, 30, 64), (1, 1, 64, 40)], ids=["h_not_32", "w_not_32"])
def test_rejects_non_tile_aligned(device, shape, expect_error):
    _, tt_input = _make_input(shape, device)
    with expect_error(VALIDATION_ERRORS, ""):
        clamp_square_practice(tt_input, -1.0, 1.0)


def test_rejects_l1_interleaved(device, expect_error):
    _, tt_input = _make_input((1, 1, 32, 32), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error(VALIDATION_ERRORS, ""):
        clamp_square_practice(tt_input, -1.0, 1.0)


def test_rejects_sharded(device, expect_error):
    _, tt_input = _make_input((1, 1, 64, 64), device)
    sharded_config = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_sharded = ttnn.to_memory_config(tt_input, sharded_config)
    with expect_error(VALIDATION_ERRORS, ""):
        clamp_square_practice(tt_sharded, -1.0, 1.0)


def test_rejects_min_greater_than_max(device, expect_error):
    _, tt_input = _make_input((1, 1, 32, 32), device)
    with expect_error(VALIDATION_ERRORS, ""):
        clamp_square_practice(tt_input, 1.0, -1.0)
