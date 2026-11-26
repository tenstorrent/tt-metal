# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_slice (Phase 0: bfloat16, TILE, DRAM interleaved, host-list begins, unit steps).

Shapes are production cases of eval/golden_tests/g6_slice/scenarios.py (id in the test id) plus the
cases the contract requires (BB-22). Grading is bit-exact (helpers.TOLERANCES["bfloat16"] == "exact").
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_slice import g6_slice


def torch_g6_slice(x, starts, ends, steps=None):
    """PyTorch basic indexing x[starts[0]:ends[0]:steps[0], ...]."""
    steps = [1] * x.dim() if steps is None else steps
    return x[tuple(slice(b, e, s) for b, e, s in zip(starts, ends, steps))]


def _bits(t):
    return t.contiguous().view(torch.int16)


def _to_device(device, torch_x):
    return ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _random_input(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _make_input(device, shape):
    torch_x = _random_input(shape)
    return torch_x, _to_device(device, torch_x)


def _check(out, x, expected):
    assert out.dtype == x.dtype
    assert out.layout == x.layout
    assert out.memory_config() == x.memory_config()
    assert tuple(out.shape) == tuple(expected.shape)
    actual = ttnn.to_torch(out)
    assert tuple(actual.shape) == tuple(expected.shape)
    diff = _bits(actual) != _bits(expected)
    assert not bool(diff.any()), f"{int(diff.sum())} of {diff.numel()} elements differ bitwise"
    return actual


# (label, shape, starts, ends, steps). Regime per op_design.md: R1 page copy, R2 row band, R3 column shift.
CASES = [
    # R1 — tile-aligned begins, M[-2] >= 32
    ("p819_cols_nonsquare", (1, 3844, 3584), (0, 0, 2048), (1, 3844, 3584), (1, 1, 1)),
    ("p654_lead_dim", (16, 4096, 64), (15, 0, 0), (16, 4096, 64), (1, 1, 1)),
    ("p621_rank4_dims_m4_m1", (16, 1, 4096, 256), (15, 0, 0, 64), (16, 1, 4096, 256), (1, 1, 1, 1)),
    ("p1443_multi_tile", (1, 1, 64, 256), (0, 0, 0, 64), (1, 1, 64, 256), (1, 1, 1, 1)),
    ("p705_w12_lead_dim", (16, 4096, 12), (9, 0, 0), (16, 4096, 12), (1, 1, 1)),
    ("p179_rows_aligned", (1, 2304, 3072), (0, 0, 0), (1, 2048, 3072), (1, 1, 1)),
    ("p181_rows_196", (1, 256, 128), (0, 0, 0), (1, 196, 128), (1, 1, 1)),
    # R2 — begin of dim r-2 inside a tile, or M[-2] < 32
    ("p1478_single_tile", (1, 31, 4160), (0, 0, 4128), (1, 31, 4140), (1, 1, 1)),
    ("p587_row_offset31", (16, 4096, 1280), (0, 4095, 0), (16, 4096, 1280), (1, 1, 1)),
    ("p1183_row_offset7", (1, 1, 16, 62080), (0, 0, 7, 0), (1, 1, 8, 62080), (1, 1, 1, 1)),
    ("p213_row_offset25", (1, 1, 256, 256), (0, 0, 217, 0), (1, 1, 218, 256), (1, 1, 1, 1)),
    ("p815_last_row_partial_tile", (1, 3844, 5120), (0, 3843, 0), (1, 3844, 5120), (1, 1, 1)),
    ("p1171_one_row_input", (16, 1, 4160), (0, 0, 2560), (16, 1, 4096), (1, 1, 1)),
    ("p1434_rows_0_5", (1, 1, 8, 62080), (0, 0, 0, 0), (1, 1, 5, 62080), (1, 1, 1, 1)),
    ("p192_rank2_row0", (32, 2560), (0, 0), (1, 2560), (1, 1)),
    ("p182_rank2_w196", (1, 256), (0, 0), (1, 196), (1, 1)),
    # Contract: one output tile draws its rows from two input tiles (rows 17..81).
    ("rule_two_tiles", (1, 1, 128, 64), (0, 0, 17, 0), (1, 1, 82, 64), (1, 1, 1, 1)),
    # Contract: a begin inside a tile of both last dims (R3).
    ("rule_begin_in_tile_both_dims", (1, 1, 64, 128), (0, 0, 17, 5), (1, 1, 50, 69), (1, 1, 1, 1)),
]


@pytest.mark.parametrize("label,shape,starts,ends,steps", CASES, ids=[c[0] for c in CASES])
def test_g6_slice(device, label, shape, starts, ends, steps):
    torch_x, x = _make_input(device, shape)
    out = g6_slice(x, list(starts), list(ends), list(steps))
    _check(out, x, torch_g6_slice(torch_x, starts, ends, steps))


def test_g6_slice_no_steps(device):
    torch_x, x = _make_input(device, (1, 1, 64, 256))
    out = g6_slice(x, [0, 0, 0, 64], [1, 1, 64, 256])
    _check(out, x, torch_g6_slice(torch_x, (0, 0, 0, 64), (1, 1, 64, 256)))


def test_g6_slice_negative_and_clamped_bounds(device):
    """A negative begin counts from the end of its dim; an end beyond N_d is clamped (PyTorch)."""
    torch_x, x = _make_input(device, (1, 1, 256, 256))
    starts, ends = [0, 0, -39, -192], [1, 1, -38, 100000]
    out = g6_slice(x, starts, ends, [1, 1, 1, 1])
    _check(out, x, torch_g6_slice(torch_x, starts, ends))


def test_g6_slice_nan_inf_copied(device):
    torch_x = _random_input((1, 1, 256, 256))
    torch_x[0, 0, 217, 0] = float("nan")
    torch_x[0, 0, 217, 255] = float("inf")
    x = _to_device(device, torch_x)
    starts, ends = (0, 0, 217, 0), (1, 1, 218, 256)
    out = g6_slice(x, list(starts), list(ends), [1, 1, 1, 1])
    _check(out, x, torch_g6_slice(torch_x, starts, ends))


def test_g6_slice_two_begins_one_input(device):
    """Two calls on one x that differ only in begins (cases 621, 622): both bit-exact."""
    torch_x, x = _make_input(device, (16, 1, 4096, 256))
    for starts, ends in (((15, 0, 0, 64), (16, 1, 4096, 256)), ((15, 0, 0, 0), (16, 1, 4096, 64))):
        out = g6_slice(x, list(starts), list(ends), [1, 1, 1, 1])
        _check(out, x, torch_g6_slice(torch_x, starts, ends))


def test_g6_slice_repeat_is_cached_and_identical(device):
    """BB-06 / BB-07: a repeated call returns identical values and builds no new program."""
    torch_x, x = _make_input(device, (1, 1, 256, 256))
    starts, ends = (0, 0, 217, 0), (1, 1, 218, 256)
    expected = torch_g6_slice(torch_x, starts, ends)
    first = _check(g6_slice(x, list(starts), list(ends), [1, 1, 1, 1]), x, expected)
    entries = device.num_program_cache_entries()
    second = _check(g6_slice(x, list(starts), list(ends), [1, 1, 1, 1]), x, expected)
    assert device.num_program_cache_entries() == entries
    assert torch.equal(_bits(first), _bits(second))


def _begins_tensor(device, values, dtype=ttnn.uint32):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_g6_slice_validation(device, expect_error):
    _, x = _make_input(device, (1, 1, 64, 256))
    bad_calls = [
        lambda: g6_slice(x, [0, 0, 0], [1, 1, 64], [1, 1, 1]),  # length != rank
        lambda: g6_slice(x, [0, 0, 0, 0], [1, 1, 64, 256], [1, 1, 0, 1]),  # step < 1
        lambda: g6_slice(x, [0, 0, 10, 0], [1, 1, 10, 256], [1, 1, 1, 1]),  # empty dim
        lambda: g6_slice(x, [0, 0, 0, 0], [1, 1, 64, 256], slice_dim=2, num_devices=2),  # list begins
    ]
    starts_t = _begins_tensor(device, [0, 0, 0, 0])
    bad_calls += [
        lambda: g6_slice(x, starts_t, starts_t, num_devices=2),  # num_devices without slice_dim
        lambda: g6_slice(x, starts_t, starts_t, slice_dim=2),  # slice_dim without num_devices
        lambda: g6_slice(x, starts_t, starts_t, [1, 1, 1, 1], slice_dim=2, num_devices=2),  # steps with tensors
        lambda: g6_slice(x, starts_t, starts_t, slice_dim=4, num_devices=2),  # slice_dim out of range
        lambda: g6_slice(x, starts_t, starts_t, slice_dim=2, num_devices=3),  # 64 % 3 != 0
    ]
    int_t = _begins_tensor(device, [0, 0, 0, 0], dtype=ttnn.int32)
    bad_calls.append(lambda: g6_slice(x, int_t, int_t, slice_dim=2, num_devices=2))  # not uint32
    short_t = _begins_tensor(device, [0, 0, 0])
    bad_calls.append(lambda: g6_slice(x, short_t, short_t, slice_dim=2, num_devices=2))  # shape != (r,)
    for call in bad_calls:
        with expect_error((ValueError, RuntimeError), ""):
            call()


def test_g6_slice_strided_refused(device, expect_error):
    """step_mode "strided" is outside TARGET: validate() refuses it (UnsupportedAxisValue)."""
    _, x = _make_input(device, (1, 1, 64, 64))
    with expect_error(NotImplementedError, ""):
        g6_slice(x, [0, 0, 0, 0], [1, 1, 64, 64], [1, 1, 2, 1])
