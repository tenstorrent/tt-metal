# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_slice, Phase 0: bfloat16, TILE, DRAM interleaved, host-list begins, unit steps.

The op copies bits, so every check is bit-exact (helpers.TOLERANCES[bfloat16] == "exact").
Shapes are production cases of eval/golden_tests/g6_slice/scenarios.py (ids in the test ids) plus the
rule cases the spec requires: an output tile drawing rows from two input tiles, a last-dim begin inside
a tile, PyTorch begin/end normalization, NaN/inf passthrough, determinism, and one program per begin.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_slice import describe_plan, g6_slice

g6_module = importlib.import_module("ttnn.operations.g6_slice.g6_slice")


def _make_input(shape, device, special_values=None):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    for index, value in special_values or []:
        torch_x[index] = value
    x = ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_x, x


def _reference(torch_x, starts, ends, steps):
    steps = steps or (1,) * len(starts)
    return torch_x[tuple(slice(b, e, s) for b, e, s in zip(starts, ends, steps))]


def _call(op, x, starts, ends, steps):
    if steps is None:
        return op(x, list(starts), list(ends))
    return op(x, list(starts), list(ends), list(steps))


def _assert_bit_exact(out, x, expected):
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config() == x.memory_config()
    assert tuple(out.shape) == tuple(expected.shape)
    actual = ttnn.to_torch(out)
    assert actual.dtype == torch.bfloat16
    diff = actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    if bool(diff.any()):
        first = tuple(int(i) for i in diff.nonzero()[0])
        pytest.fail(f"{int(diff.sum())} of {diff.numel()} elements differ bitwise, first at {first}")
    return actual


# (shape, starts, ends, steps, regime). Regime per op_design.md: col_realign if b[-1] % 32,
# else page_copy if b[-2] % 32 == 0 and M[-2] % 32 == 0, else row_runs.
CASES = {
    # production cases
    "qwen1478_single_tile": ((1, 31, 4160), (0, 0, 4128), (1, 31, 4140), (1, 1, 1), "row_runs"),
    "c1443_rank4_last_dim": ((1, 1, 64, 256), (0, 0, 0, 64), (1, 1, 64, 256), (1, 1, 1, 1), "page_copy"),
    "c186_multi_channel": ((1, 4, 256, 128), (0, 0, 0, 64), (1, 4, 256, 128), (1, 1, 1, 1), "page_copy"),
    "c1811_lead_and_last_dim": ((8, 6, 128, 256), (0, 0, 0, 0), (1, 6, 128, 64), (1, 1, 1, 1), "page_copy"),
    "c1810_rank3_batch": ((8, 128, 64), (0, 0, 0), (1, 128, 64), (1, 1, 1), "page_copy"),
    "c179_non_square_rows": ((1, 2304, 3072), (0, 0, 0), (1, 2048, 3072), (1, 1, 1), "page_copy"),
    "c705_narrow_input": ((16, 4096, 12), (9, 0, 0), (16, 4096, 12), (1, 1, 1), "page_copy"),
    "c1966_end_inside_tile": ((1, 128, 4160), (0, 0, 4096), (1, 128, 4108), (1, 1, 1), "page_copy"),
    "pattern_no_steps": ((1, 32, 64, 2048), (0, 0, 0, 1024), (1, 32, 64, 2048), None, "page_copy"),
    "c587_row_4095": ((16, 4096, 1280), (0, 4095, 0), (16, 4096, 1280), (1, 1, 1), "row_runs"),
    "c1946_row_127": ((1, 128, 5120), (0, 127, 0), (1, 128, 5120), (1, 1, 1), "row_runs"),
    "c815_last_row_partial_input": ((1, 3844, 5120), (0, 3843, 0), (1, 3844, 5120), (1, 1, 1), "row_runs"),
    "c213_row_217": ((1, 1, 256, 256), (0, 0, 217, 0), (1, 1, 218, 256), (1, 1, 1, 1), "row_runs"),
    "c1175_row_15_wide": ((1, 1, 16, 62080), (0, 0, 15, 0), (1, 1, 16, 62080), (1, 1, 1, 1), "row_runs"),
    "c1390_row_1_odd_offset": ((1, 1, 2, 62080), (0, 0, 1, 0), (1, 1, 2, 62080), (1, 1, 1, 1), "row_runs"),
    "c1434_rows_0_to_5": ((1, 1, 8, 62080), (0, 0, 0, 0), (1, 1, 5, 62080), (1, 1, 1, 1), "row_runs"),
    "c1171_one_row_images": ((16, 1, 4160), (0, 0, 2560), (16, 1, 4096), (1, 1, 1), "row_runs"),
    "c1886_partial_last_tile_row": ((1, 6, 100, 256), (0, 0, 0, 0), (1, 6, 100, 64), (1, 1, 1, 1), "row_runs"),
    "c182_rank2_last_dim": ((1, 256), (0, 0), (1, 196), (1, 1), "row_runs"),
    "c192_rank2_first_row": ((32, 2560), (0, 0), (1, 2560), (1, 1), "row_runs"),
    # rule cases
    "rule_two_tiles": ((1, 1, 128, 64), (0, 0, 17, 0), (1, 1, 82, 64), None, "row_runs"),
    "rule_negative_begin_clamped_end": ((1, 1, 64, 256), (0, 0, -33, -128), (1, 1, 10**6, 256), None, "row_runs"),
    "rule_w_begin_in_tile": ((1, 1, 64, 128), (0, 0, 0, 5), (1, 1, 64, 77), None, "col_realign"),
    "rule_w_and_h_begin_in_tile": ((1, 1, 96, 128), (0, 0, 17, 20), (1, 1, 90, 120), None, "col_realign"),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_g6_slice(device, case):
    shape, starts, ends, steps, regime = CASES[case]
    torch_x, x = _make_input(shape, device)

    assert _call(describe_plan, x, starts, ends, steps)["regime"] == regime

    out = _call(g6_slice, x, starts, ends, steps)
    _assert_bit_exact(out, x, _reference(torch_x, starts, ends, steps))


def test_phase0_spreads_over_full_grid(device):
    shape, starts, ends, steps, _ = CASES["c179_non_square_rows"]
    _, x = _make_input(shape, device)
    grid = device.compute_with_storage_grid_size()
    plan = _call(describe_plan, x, starts, ends, steps)
    assert plan["num_cores"] == min(6144, grid.x * grid.y)


@pytest.mark.parametrize("case", ["c179_non_square_rows", "c1175_row_15_wide"])
def test_several_blocks_per_core(device, monkeypatch, case):
    """A small BLOCK_PAGES_CAP forces several blocks per core: CB ring wrap and ragged last blocks."""
    monkeypatch.setattr(g6_module, "BLOCK_PAGES_CAP", 3)
    shape, starts, ends, steps, _ = CASES[case]
    torch_x, x = _make_input(shape, device)

    plan = _call(describe_plan, x, starts, ends, steps)
    assert plan["block_pages"] == 3

    out = _call(g6_slice, x, starts, ends, steps)
    _assert_bit_exact(out, x, _reference(torch_x, starts, ends, steps))


def test_repeat_is_bit_identical(device):
    shape, starts, ends, steps, _ = CASES["c1175_row_15_wide"]
    torch_x, x = _make_input(shape, device)
    expected = _reference(torch_x, starts, ends, steps)
    first = _assert_bit_exact(_call(g6_slice, x, starts, ends, steps), x, expected)
    second = _assert_bit_exact(_call(g6_slice, x, starts, ends, steps), x, expected)
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))


def test_one_program_for_two_begins(device):
    """Begins are runtime arguments: a second row of the same shape class builds no new program."""
    torch_x, x = _make_input((1, 1, 256, 256), device)
    first = ((0, 0, 217, 0), (1, 1, 218, 256), (1, 1, 1, 1))
    second = ((0, 0, 218, 0), (1, 1, 219, 256), (1, 1, 1, 1))

    out = _call(g6_slice, x, *first)
    _assert_bit_exact(out, x, _reference(torch_x, *first))
    entries = device.num_program_cache_entries()

    out = _call(g6_slice, x, *second)
    assert device.num_program_cache_entries() == entries
    _assert_bit_exact(out, x, _reference(torch_x, *second))


def test_nan_and_inf_are_copied(device):
    shape, starts, ends, steps, _ = CASES["c213_row_217"]
    torch_x, x = _make_input(
        shape, device, special_values=[((0, 0, 217, 0), float("nan")), ((0, 0, 217, 255), float("inf"))]
    )
    out = _call(g6_slice, x, starts, ends, steps)
    _assert_bit_exact(out, x, _reference(torch_x, starts, ends, steps))


def test_step_two_is_refused(device, expect_error):
    _, x = _make_input((1, 1, 64, 64), device)
    with expect_error(NotImplementedError, ""):
        g6_slice(x, [0, 0, 0, 0], [1, 1, 64, 64], [1, 1, 2, 1])


def _begins_tensor(values, device, dtype=ttnn.uint32):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


LIST_FORM_ERRORS = {
    "starts_length": (([0, 0, 0], [1, 1, 64, 64]), {}),
    "ends_length": (([0, 0, 0, 0], [1, 1, 64]), {}),
    "steps_length": (([0, 0, 0, 0], [1, 1, 64, 64], [1, 1, 1]), {}),
    "step_zero": (([0, 0, 0, 0], [1, 1, 64, 64], [1, 1, 0, 1]), {}),
    "step_negative": (([0, 0, 0, 0], [1, 1, 64, 64], [1, 1, -1, 1]), {}),
    "empty_dim": (([0, 0, 10, 0], [1, 1, 10, 64]), {}),
    "slice_dim_with_list_begins": (([0, 0, 0, 0], [1, 1, 64, 64]), {"slice_dim": 2}),
    "slice_dim_and_num_devices_with_list_begins": (([0, 0, 0, 0], [1, 1, 64, 64]), {"slice_dim": 2, "num_devices": 2}),
}


@pytest.mark.parametrize("case", list(LIST_FORM_ERRORS), ids=list(LIST_FORM_ERRORS))
def test_list_form_validation(device, expect_error, case):
    args, kwargs = LIST_FORM_ERRORS[case]
    _, x = _make_input((1, 1, 64, 64), device)
    with expect_error((ValueError, RuntimeError), "") as excinfo:
        g6_slice(x, *args, **kwargs)
    assert not isinstance(excinfo.value, NotImplementedError), "a malformed call must not read as a support refusal"


TENSOR_FORM_ERRORS = {
    "steps_with_tensor_begins": {"steps": [1, 1, 1], "slice_dim": 1, "num_devices": 7},
    "slice_dim_too_large": {"slice_dim": 3, "num_devices": 7},
    "slice_dim_too_small": {"slice_dim": -4, "num_devices": 7},
    "num_devices_missing": {"slice_dim": 1},
    "slice_dim_missing": {"num_devices": 7},
    "num_devices_does_not_divide": {"slice_dim": 1, "num_devices": 5},
    "begins_not_uint32": {"slice_dim": 1, "num_devices": 7, "begins_dtype": ttnn.int32},
    "begins_wrong_shape": {"slice_dim": 1, "num_devices": 7, "begins_values": [0, 0, 0, 0]},
}


@pytest.mark.parametrize("case", list(TENSOR_FORM_ERRORS), ids=list(TENSOR_FORM_ERRORS))
def test_tensor_form_validation(device, expect_error, case):
    spec = dict(TENSOR_FORM_ERRORS[case])
    begins = _begins_tensor(spec.pop("begins_values", [0, 0, 0]), device, spec.pop("begins_dtype", ttnn.uint32))
    steps = spec.pop("steps", None)
    _, x = _make_input((1, 224, 5120), device)
    with expect_error((ValueError, RuntimeError), "") as excinfo:
        if steps is None:
            g6_slice(x, begins, begins, **spec)
        else:
            g6_slice(x, begins, begins, steps, **spec)
    assert not isinstance(excinfo.value, NotImplementedError), "a malformed call must not read as a support refusal"
