# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_slice, Phase 0: bfloat16, TILE, DRAM interleaved, host-list begins, unit steps.

Cases named cNNN are production calls of eval/golden_tests/g6_slice/scenarios.py (case id NNN).
Cases named rule_* are required by the contract and have no production call:
  - rule_two_tiles: rows 17..81 of (1, 1, 128, 64), output tiles drawn from two input tiles;
  - rule_col16 / rule_row17_col5: last-dim begins inside a tile ("a begin may lie anywhere inside a
    tile of the last two dims"), one per stencil shape of the element_shift regime;
  - c1484_negative_clamped: production call 1484 written with a negative begin and an end past N.
Graded bit-exact: helpers.TOLERANCES[ttnn.bfloat16] == "exact".
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_slice import g6_slice, g6_slice_plan


def torch_g6_slice(x, starts, ends, steps=None):
    steps = steps or [1] * x.dim()
    return x[tuple(slice(b, e, s) for b, e, s in zip(starts, ends, steps))]


def _to_device(x, device):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _call(x_dev, starts, ends, steps):
    if steps is None:
        return g6_slice(x_dev, starts, ends)
    return g6_slice(x_dev, starts, ends, steps)


def _assert_same_placement(out, x_dev, expected_shape):
    assert out.dtype == x_dev.dtype
    assert out.layout == x_dev.layout
    assert out.memory_config() == x_dev.memory_config()
    assert tuple(out.shape) == tuple(expected_shape)


def _assert_bit_exact(actual, expected):
    assert tuple(actual.shape) == tuple(expected.shape)
    assert actual.dtype == torch.bfloat16
    a = actual.contiguous().view(torch.int16)
    e = expected.contiguous().view(torch.int16)
    mismatches = int((a != e).sum())
    assert mismatches == 0, f"{mismatches} of {e.numel()} elements differ bitwise"


# (id, shape, starts, ends, steps, regime, full_grid)
CASES = [
    ("c1478_single_tile", (1, 31, 4160), [0, 0, 4128], [1, 31, 4140], [1, 1, 1], "page_copy", False),
    ("c182_rank2", (1, 256), [0, 0], [1, 196], [1, 1], "page_copy", False),
    ("c1484_row32", (1, 33, 5120), [0, 32, 0], [1, 33, 5120], [1, 1, 1], "page_copy", False),
    ("c1484_negative_clamped", (1, 33, 5120), [0, -1, 0], [1, 1000, 5120], [1, 1, 1], "page_copy", False),
    ("c621_lead_and_last", (16, 1, 4096, 256), [15, 0, 0, 64], [16, 1, 4096, 256], [1, 1, 1, 1], "page_copy", False),
    ("c588_multibatch", (16, 6, 4096, 256), [15, 0, 0, 0], [16, 6, 4096, 256], [1, 1, 1, 1], "page_copy", True),
    ("c701_multiblock", (16, 4096, 3584), [0, 0, 0], [16, 4096, 2048], [1, 1, 1], "page_copy", True),
    ("c587_row4095", (16, 4096, 1280), [0, 4095, 0], [16, 4096, 1280], [1, 1, 1], "row_shift", False),
    ("c1183_row7_wide", (1, 1, 16, 62080), [0, 0, 7, 0], [1, 1, 8, 62080], [1, 1, 1, 1], "row_shift", False),
    ("rule_two_tiles", (1, 1, 128, 64), [0, 0, 17, 0], [1, 1, 82, 64], None, "row_shift", False),
    ("rule_col16", (1, 1, 128, 64), [0, 0, 0, 16], [1, 1, 128, 48], [1, 1, 1, 1], "element_shift", False),
    ("rule_row17_col5", (1, 1, 128, 64), [0, 0, 17, 5], [1, 1, 82, 61], [1, 1, 1, 1], "element_shift", False),
]


@pytest.mark.parametrize(
    "shape, starts, ends, steps, regime, full_grid",
    [pytest.param(*case[1:], id=case[0]) for case in CASES],
)
def test_g6_slice(device, shape, starts, ends, steps, regime, full_grid):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    x_dev = _to_device(x, device)
    expected = torch_g6_slice(x, starts, ends, steps)

    plan = g6_slice_plan(x_dev, starts, ends, steps)
    assert plan["regime"] == regime
    if full_grid:
        grid = device.compute_with_storage_grid_size()
        assert plan["num_cores"] == grid.x * grid.y

    out = _call(x_dev, starts, ends, steps)
    _assert_same_placement(out, x_dev, expected.shape)
    _assert_bit_exact(ttnn.to_torch(out), expected)


@pytest.mark.parametrize(
    "shape, starts, ends",
    [
        pytest.param((1, 1, 128, 64), [0, 0, 17, 0], [1, 1, 82, 64], id="row_shift"),
        pytest.param((1, 1, 128, 64), [0, 0, 17, 5], [1, 1, 82, 61], id="element_shift"),
        pytest.param((16, 4096, 1280), [0, 4095, 0], [16, 4096, 1280], id="c587_row4095"),
    ],
)
def test_g6_slice_special_values(device, shape, starts, ends):
    """NaN at the first and +inf at the last selected element are copied like any other value."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    x[tuple(starts)] = float("nan")
    x[tuple(e - 1 for e in ends)] = float("inf")
    x_dev = _to_device(x, device)

    out = g6_slice(x_dev, starts, ends)
    _assert_bit_exact(ttnn.to_torch(out), torch_g6_slice(x, starts, ends))


def test_g6_slice_begin_is_runtime(device):
    """Production calls 621 and 623 differ only in the dim-0 begin: the second must reuse the program."""
    torch.manual_seed(42)
    shape = (16, 1, 4096, 256)
    x = torch.randn(shape, dtype=torch.bfloat16)
    x_dev = _to_device(x, device)
    steps = [1, 1, 1, 1]
    first = ([15, 0, 0, 64], [16, 1, 4096, 256])
    second = ([14, 0, 0, 64], [15, 1, 4096, 256])

    out_first = g6_slice(x_dev, *first, steps)
    entries = device.num_program_cache_entries()
    out_second = g6_slice(x_dev, *second, steps)
    assert device.num_program_cache_entries() == entries

    _assert_bit_exact(ttnn.to_torch(out_first), torch_g6_slice(x, *first, steps))
    _assert_bit_exact(ttnn.to_torch(out_second), torch_g6_slice(x, *second, steps))


def test_g6_slice_validation(device, expect_error):
    torch.manual_seed(42)
    x_dev = _to_device(torch.randn((1, 256), dtype=torch.bfloat16), device)

    def begins_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _assert_validation_error(fn):
        with expect_error((ValueError, RuntimeError), "g6_slice") as info:
            fn()
        assert not isinstance(
            info.value, NotImplementedError
        ), f"a malformed call was refused as unsupported: {info.value!r}"

    _assert_validation_error(lambda: g6_slice(x_dev, [0], [1], [1]))
    _assert_validation_error(lambda: g6_slice(x_dev, [0, 0], [1, 196, 1]))
    _assert_validation_error(lambda: g6_slice(x_dev, [0, 0], [1, 196], [1, 0]))
    _assert_validation_error(lambda: g6_slice(x_dev, [0, 100], [1, 100]))
    _assert_validation_error(lambda: g6_slice(x_dev, [0, 0], [1, 196], slice_dim=1))
    _assert_validation_error(lambda: g6_slice(x_dev, [0, 0], [1, 196], slice_dim=1, num_devices=2))

    starts_t = begins_tensor([0, 0])
    _assert_validation_error(lambda: g6_slice(x_dev, starts_t, starts_t, slice_dim=1))
    _assert_validation_error(lambda: g6_slice(x_dev, starts_t, starts_t, [1, 1], slice_dim=1, num_devices=2))
    _assert_validation_error(lambda: g6_slice(x_dev, starts_t, starts_t, slice_dim=2, num_devices=2))
    _assert_validation_error(lambda: g6_slice(x_dev, starts_t, starts_t, slice_dim=1, num_devices=3))
    wrong_shape = begins_tensor([0, 0, 0])
    _assert_validation_error(lambda: g6_slice(x_dev, wrong_shape, wrong_shape, slice_dim=1, num_devices=2))

    with expect_error(NotImplementedError, "g6_slice"):
        g6_slice(x_dev, [0, 0], [1, 196], [1, 2])
