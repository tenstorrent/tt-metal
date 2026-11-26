# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_slice (Phase 0: bfloat16, TILE, DRAM interleaved, host begins, unit steps).

Immutable spec for the implementer. Cases are production calls of the golden case list
(eval/golden_tests/g6_slice/scenarios.py, ids in the comments) plus the cases the spec's
rules require (BB-22): the two-input-tile row shift, a begin inside a tile of the last dim
(Shape handling: "a begin may lie anywhere inside a tile of the last two dims"), repeated
calls with different begins (BB-07 cache key), the refusal of a step > 1 and the host-side
ValueError checks. Grading is bit-exact (helpers.TOLERANCES[bfloat16] == "exact").
"""

import pytest
import torch

import ttnn
from ttnn.operations.g6_slice import g6_slice


def pytorch_g6_slice(x, starts, ends, steps=None):
    steps = steps or [1] * x.dim()
    return x[tuple(slice(b, e, s) for b, e, s in zip(starts, ends, steps))]


def _bits(t):
    return t.contiguous().view(torch.int16)


def _make_input(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _to_device(torch_x, device):
    return ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _check(out, x_dev, expected):
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config() == x_dev.memory_config()
    assert tuple(out.shape) == tuple(expected.shape)
    actual = ttnn.to_torch(out)
    assert tuple(actual.shape) == tuple(expected.shape)
    assert torch.equal(_bits(actual), _bits(expected)), "output differs bitwise from x[slices]"


# (shape, starts, ends, steps) -- steps None means the call passes no steps.
CASES = [
    # --- regime R1: begins on tile boundaries -----------------------------------------
    pytest.param((1, 256), [0, 0], [1, 196], [1, 1], id="c182_rank2_w_partial"),
    pytest.param((32, 2560), [0, 0], [1, 2560], [1, 1], id="c192_rank2_one_row"),
    pytest.param((16, 32, 12), [9, 0, 0], [16, 32, 12], [1, 1, 1], id="c1155_dim0_w12"),
    pytest.param((2, 128, 64), [1, 0, 0], [2, 128, 64], [1, 1, 1], id="c1405_dim0"),
    pytest.param((1, 256, 128), [0, 0, 0], [1, 196, 128], [1, 1, 1], id="c181_h_partial_end"),
    pytest.param((1, 2304, 3072), [0, 0, 0], [1, 2048, 3072], [1, 1, 1], id="c179_rows_multi_block"),
    pytest.param((8, 1, 4160), [0, 0, 4128], [8, 1, 4140], [1, 1, 1], id="c1732_12_cols"),
    pytest.param((1, 64, 4160), [0, 0, 4128], [1, 64, 4140], [1, 1, 1], id="c1454_12_cols_tall"),
    pytest.param((8, 1, 4160), [0, 0, 2560], [8, 1, 4096], [1, 1, 1], id="c1734_last_dim"),
    pytest.param((1, 64, 3584), [0, 0, 2048], [1, 64, 3584], [1, 1, 1], id="c1442_last_dim"),
    pytest.param((2, 1, 128, 256), [1, 0, 0, 0], [2, 1, 128, 256], [1, 1, 1, 1], id="c1394_rank4_dim0"),
    pytest.param((2, 1, 128, 256), [1, 0, 0, 64], [2, 1, 128, 256], [1, 1, 1, 1], id="c1400_rank4_dim0_and_w"),
    pytest.param((1, 8, 1, 256), [0, 0, 0, 64], [1, 8, 1, 256], [1, 1, 1, 1], id="c1714_rank4_h1"),
    pytest.param((1, 1, 64, 256), [0, 0, 0, 64], [1, 1, 64, 256], [1, 1, 1, 1], id="c1443_rank4_w"),
    pytest.param((1, 1, 256, 256), [0, 0, 192, 0], [1, 1, 193, 256], [1, 1, 1, 1], id="c232_row_on_tile_edge"),
    pytest.param((1, 32, 64, 2048), [0, 0, 0, 1024], [1, 32, 64, 2048], None, id="no_steps_rank4"),
    # --- regime R2: dim r-2 begin inside a tile (row shift) ---------------------------
    pytest.param((1, 31, 5120), [0, 30, 0], [1, 31, 5120], [1, 1, 1], id="c1459_row30_of_31"),
    pytest.param((1, 1, 256, 256), [0, 0, 196, 0], [1, 1, 197, 256], [1, 1, 1, 1], id="c191_row196"),
    pytest.param((16, 4096, 1280), [0, 4095, 0], [16, 4096, 1280], [1, 1, 1], id="c587_row4095_offset31"),
    # rule case: rows 17..81 -> 3 output tile rows, each from two input tiles
    pytest.param((1, 1, 128, 64), [0, 0, 17, 0], [1, 1, 82, 64], None, id="rule_two_input_tiles"),
    # --- regime R2: begins inside a tile of both last dims (row + column shift) -------
    pytest.param((1, 1, 96, 128), [0, 0, 3, 5], [1, 1, 70, 120], None, id="rule_row_and_col_shift"),
]


@pytest.mark.parametrize("shape, starts, ends, steps", CASES)
def test_g6_slice(device, shape, starts, ends, steps):
    torch_x = _make_input(shape)
    x_dev = _to_device(torch_x, device)
    expected = pytorch_g6_slice(torch_x, starts, ends, steps)
    out = g6_slice(x_dev, starts, ends) if steps is None else g6_slice(x_dev, starts, ends, steps)
    _check(out, x_dev, expected)


def test_g6_slice_negative_and_clamped_bounds(device):
    """PyTorch semantics: negative begins/ends count from the end, an end beyond N is clamped."""
    shape = (2, 96, 160)
    torch_x = _make_input(shape)
    x_dev = _to_device(torch_x, device)
    starts, ends = [-1, -64, 32], [5, 96, 10_000]
    expected = torch_x[-1:5, -64:96, 32:10_000]
    _check(g6_slice(x_dev, starts, ends, [1, 1, 1]), x_dev, expected)


def test_g6_slice_same_shape_different_begins(device):
    """Two calls on one x that differ only in begins (both halves, then a row shift) must each
    return their own slice: the begins are not baked into a cached program (BB-07)."""
    shape = (2, 128, 512)
    torch_x = _make_input(shape)
    x_dev = _to_device(torch_x, device)
    for starts, ends in (
        ([0, 0, 256], [2, 128, 512]),
        ([0, 0, 0], [2, 128, 256]),
        ([0, 0, 256], [2, 128, 512]),
        ([0, 37, 0], [2, 38, 512]),
        ([0, 63, 0], [2, 64, 512]),
    ):
        expected = pytorch_g6_slice(torch_x, starts, ends)
        _check(g6_slice(x_dev, starts, ends, [1, 1, 1]), x_dev, expected)


def test_g6_slice_repeat_bit_identical(device):
    """BB-06: the same call twice gives bit-identical outputs."""
    shape = (1, 64, 4160)
    torch_x = _make_input(shape)
    x_dev = _to_device(torch_x, device)
    first = ttnn.to_torch(g6_slice(x_dev, [0, 0, 2560], [1, 64, 4096], [1, 1, 1]))
    second = ttnn.to_torch(g6_slice(x_dev, [0, 0, 2560], [1, 64, 4096], [1, 1, 1]))
    assert torch.equal(_bits(first), _bits(second))


def test_g6_slice_step_above_one_is_refused(device, expect_error):
    """step_mode "strided" is not supported: a support refusal (NotImplementedError)."""
    x_dev = _to_device(_make_input((1, 1, 64, 64)), device)
    with expect_error(NotImplementedError, ""):
        g6_slice(x_dev, [0, 0, 0, 0], [1, 1, 64, 64], [1, 1, 2, 1])


@pytest.mark.parametrize(
    "starts, ends, steps, kwargs",
    [
        pytest.param([0, 0], [1, 64, 64], None, {}, id="starts_wrong_length"),
        pytest.param([0, 0, 0], [1, 64], None, {}, id="ends_wrong_length"),
        pytest.param([0, 0, 0], [1, 64, 64], [1, 1], {}, id="steps_wrong_length"),
        pytest.param([0, 0, 0], [1, 64, 64], [1, 0, 1], {}, id="step_zero"),
        pytest.param([0, 10, 0], [1, 10, 64], None, {}, id="empty_dim"),
        pytest.param([0, 0, 0], [1, 64, 64], None, {"slice_dim": 1, "num_devices": 2}, id="slice_dim_with_lists"),
        pytest.param([0, 0, 0], [1, 64, 64], None, {"slice_dim": 1}, id="slice_dim_without_num_devices"),
        pytest.param([0, 0, 0], [1, 64, 64], None, {"num_devices": 2}, id="num_devices_without_slice_dim"),
    ],
)
def test_g6_slice_invalid_arguments(device, expect_error, starts, ends, steps, kwargs):
    x_dev = _to_device(_make_input((1, 64, 64)), device)
    with expect_error((ValueError, RuntimeError), ""):
        if steps is None:
            g6_slice(x_dev, starts, ends, **kwargs)
        else:
            g6_slice(x_dev, starts, ends, steps, **kwargs)
