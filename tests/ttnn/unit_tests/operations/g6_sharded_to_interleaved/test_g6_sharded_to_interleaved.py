# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_sharded_to_interleaved.

A height- or width-sharded L1 TILE tensor is copied, bit for bit, into a newly allocated
interleaved tensor in the buffer type `memory_config` names. The cases are the two production
cases and the cases the spec's rules require, nothing else (BB-22): another shard spec per case
(column-major, shorter last shard, other core ranges), the destination taken from memory_config,
a dirty destination, an intact input, a repeat, special values, a cache order and the validation
refusals. The gate is exact for both dtypes (helpers.TOLERANCES holds ("exact", None)).

A cell outside the op's current SUPPORTED must be refused with a support refusal; once SUPPORTED
grows to hold it, the same parametrization runs the full exact check.
"""

import importlib
import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved

TILE = 32

_OP_MODULE = importlib.import_module("ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved")

# Production cases (eval/golden_tests/g6_sharded_to_interleaved/feature_spec.py INPUTS).
P2 = {
    "shape": (1, 5120),
    "dtype": ttnn.bfloat16,
    "in_scheme": "width_sharded",
    "shard_shape": (32, 160),
    "grid": (((0, 0), (7, 3)),),
    "orientation": "row_major",
    "buffer": "dram",
}
P1 = {
    "shape": (1, 128, 256, 32),
    "dtype": ttnn.bfloat8_b,
    "in_scheme": "height_sharded",
    "shard_shape": (320, 32),
    "grid": (((0, 0), (10, 8)), ((0, 9), (3, 9))),
    "orientation": "row_major",
    "buffer": "l1",
}
# Another shard spec per case: column-major walk, a shorter last shard, other core ranges.
P2_OTHER_SPEC = {**P2, "shard_shape": (32, 96), "grid": (((0, 0), (8, 5)),), "orientation": "column_major"}
P1_OTHER_SPEC = {
    **P1,
    "shard_shape": (352, 32),
    "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
    "orientation": "column_major",
}
# Destination from memory_config: the bfloat16 width-sharded case written to L1.
P2_TO_L1 = {**P2, "buffer": "l1"}

CASES = {
    "p2_production": P2,
    "p2_other_spec": P2_OTHER_SPEC,
    "p2_to_l1": P2_TO_L1,
    "p1_production": P1,
    "p1_other_spec": P1_OTHER_SPEC,
}

_IN_SCHEME = {
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
}
_ORIENTATION = {
    "row_major": ttnn.ShardOrientation.ROW_MAJOR,
    "column_major": ttnn.ShardOrientation.COL_MAJOR,
}
_OUTPUT_MEMORY_CONFIG = {
    "dram": ttnn.DRAM_MEMORY_CONFIG,
    "l1": ttnn.L1_MEMORY_CONFIG,
}
_BUFFER_TYPE = {
    "dram": ttnn.BufferType.DRAM,
    "l1": ttnn.BufferType.L1,
}
_BIT_VIEW = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
}

# One per tile of the first tile row, so a page-map error moves them visibly.
_SPECIAL_VALUES = (
    (0, 0 * TILE + 0, -0.0),
    (0, 1 * TILE + 1, float("inf")),
    (0, 2 * TILE + 2, float("-inf")),
    (0, 3 * TILE + 3, float("nan")),
    (0, 4 * TILE + 4, 1.0e-39),  # bfloat16 subnormal
)


def pytorch_reference(input_tensor):
    return torch.clone(input_tensor)


def _supported(scenario):
    supported = _OP_MODULE.SUPPORTED
    axes = {
        "dtype": scenario["dtype"],
        "layout": ttnn.TILE_LAYOUT,
        "in_scheme": scenario["in_scheme"],
        "buffer": scenario["buffer"],
    }
    return all(value in supported.get(axis, ()) for axis, value in axes.items())


def _make_torch_input(shape, *, special_values=False):
    """randn of the logical shape, each 32x32 tile of the 2-D view scaled by 2 ** (t % 16 - 8).

    Neighbouring tiles differ by powers of two, so a tile on the wrong page (or a bfloat8_b tile
    carrying another tile's shared exponents) is a wrong value.
    """
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols, dtype=torch.float32)
    tile_cols = -(-cols // TILE)
    tile_index = (torch.arange(rows) // TILE)[:, None] * tile_cols + (torch.arange(cols) // TILE)[None, :]
    x = x * torch.pow(2.0, (tile_index % 16 - 8).to(torch.float32))
    if special_values:
        for row, col, value in _SPECIAL_VALUES:
            x[row, col] = value
    return x.reshape(shape).to(torch.bfloat16)


def _input_memory_config(device, scenario):
    size = device.compute_with_storage_grid_size()
    for (_, _), (x1, y1) in scenario["grid"]:
        if x1 > size.x - 1 or y1 > size.y - 1:
            pytest.skip(f"shard grid end ({x1},{y1}) exceeds the worker grid ({size.x},{size.y})")
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in scenario["grid"]]
    )
    spec = ttnn.ShardSpec(grid, tuple(scenario["shard_shape"]), _ORIENTATION[scenario["orientation"]])
    return ttnn.MemoryConfig(_IN_SCHEME[scenario["in_scheme"]], ttnn.BufferType.L1, spec)


def _build_input(device, scenario, *, special_values=False):
    """The sharded device input and the host values the output must equal bit for bit."""
    torch_input = _make_torch_input(scenario["shape"], special_values=special_values)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=scenario["dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_input_memory_config(device, scenario),
    )
    # bfloat16 is stored as given; bfloat8_b as the host packed it, so the reference is its readback.
    source = torch_input if scenario["dtype"] == ttnn.bfloat16 else ttnn.to_torch(tt_input)
    return tt_input, pytorch_reference(source)


def _placement(tensor):
    mc = tensor.memory_config()
    spec = mc.shard_spec
    shard = None
    if spec is not None:
        ranges = tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges())
        shard = (tuple(spec.shape), spec.orientation, ranges)
    return (mc.memory_layout, mc.buffer_type, shard)


def _check_exact(actual, expected, what):
    assert tuple(actual.shape) == tuple(
        expected.shape
    ), f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert (
        actual.dtype == expected.dtype and actual.dtype in _BIT_VIEW
    ), f"{what}: dtype {actual.dtype} != {expected.dtype}"
    view = _BIT_VIEW[actual.dtype]
    differ = actual.contiguous().view(view) != expected.contiguous().view(view)
    count = int(differ.sum())
    if count:
        cols = actual.shape[-1]
        where = torch.nonzero(differ.reshape(-1, cols))[:4].tolist()
        tiles = sorted({(row // TILE, col // TILE) for row, col in where})
        pytest.fail(f"{what}: {count} of {differ.numel()} logical elements differ; first in tiles (row, col) {tiles}")


def _check_output_contract(output, scenario):
    assert tuple(output.shape) == tuple(scenario["shape"])
    assert output.dtype == scenario["dtype"]
    assert output.layout == ttnn.TILE_LAYOUT
    mc = output.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert mc.buffer_type == _BUFFER_TYPE[scenario["buffer"]]


def _check_input_intact(tt_input, source, placement, address):
    assert tt_input.is_allocated(), "the op freed input_tensor"
    assert _placement(tt_input) == placement, "input_tensor's placement changed"
    assert tt_input.buffer_address() == address, "input_tensor's buffer moved"
    _check_exact(ttnn.to_torch(tt_input), source, "input")


def _free(*tensors):
    for tensor in tensors:
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


def _run_and_check(device, expect_error, scenario, *, special_values=False):
    """One graded call: the output exact, in a new buffer, the input intact."""
    tt_input, expected = _build_input(device, scenario, special_values=special_values)
    output_mc = _OUTPUT_MEMORY_CONFIG[scenario["buffer"]]
    tt_output = None
    try:
        if not _supported(scenario):
            with expect_error(SupportRefusal, ""):
                tt_output = g6_sharded_to_interleaved(tt_input, output_mc)
            pytest.xfail("cell outside the op's SUPPORTED: refused as required")
        placement = _placement(tt_input)
        address = tt_input.buffer_address()
        tt_output = g6_sharded_to_interleaved(tt_input, output_mc)
        _check_output_contract(tt_output, scenario)
        _check_exact(ttnn.to_torch(tt_output), expected, "output")
        if scenario["buffer"] == "l1":
            assert tt_output.buffer_address() != address, "the output shares input_tensor's L1 buffer"
        _check_input_intact(tt_input, expected, placement, address)
    finally:
        _free(tt_output, tt_input)


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_g6_sharded_to_interleaved(device, expect_error, case):
    torch.manual_seed(42)
    _run_and_check(device, expect_error, CASES[case])


def test_special_values(device, expect_error):
    """-0.0, +-inf, NaN and a subnormal move as stored bit patterns (bfloat16)."""
    torch.manual_seed(42)
    _run_and_check(device, expect_error, P2, special_values=True)


def test_dirty_destination(device):
    """The output lands on a freed range pre-filled with 0x7F7F: every page must be written by the call."""
    torch.manual_seed(42)
    scenario = P2
    if not _supported(scenario):
        pytest.xfail("cell outside the op's SUPPORTED")
    tt_input, expected = _build_input(device, scenario)
    output_mc = _OUTPUT_MEMORY_CONFIG[scenario["buffer"]]
    tt_output = None
    try:
        loud = torch.full(scenario["shape"], 0x7F7F, dtype=torch.int16).view(torch.bfloat16)
        dirty = ttnn.from_torch(
            loud, dtype=scenario["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=output_mc
        )
        freed_address = dirty.buffer_address()
        ttnn.deallocate(dirty)

        tt_output = g6_sharded_to_interleaved(tt_input, output_mc)
        assert tt_output.buffer_address() == freed_address, (
            f"output at {tt_output.buffer_address():#x}, freed dirty range at {freed_address:#x}: "
            "the case cannot confirm that every page was overwritten"
        )
        _check_output_contract(tt_output, scenario)
        _check_exact(ttnn.to_torch(tt_output), expected, "output")
    finally:
        _free(tt_output, tt_input)


def test_repeat_bit_identical(device):
    """Two calls on one input: bit-identical outputs, each in a buffer of its own (BB-06)."""
    torch.manual_seed(42)
    scenario = P2
    if not _supported(scenario):
        pytest.xfail("cell outside the op's SUPPORTED")
    tt_input, expected = _build_input(device, scenario)
    output_mc = _OUTPUT_MEMORY_CONFIG[scenario["buffer"]]
    first = second = None
    try:
        placement = _placement(tt_input)
        address = tt_input.buffer_address()
        first = g6_sharded_to_interleaved(tt_input, output_mc)
        second = g6_sharded_to_interleaved(tt_input, output_mc)
        assert second.buffer_address() != first.buffer_address(), "a later call returned the first call's buffer"
        first_torch = ttnn.to_torch(first)
        _check_exact(first_torch, expected, "first")
        _check_exact(ttnn.to_torch(second), first_torch, "repeat")
        _check_input_intact(tt_input, expected, placement, address)
    finally:
        _free(first, second, tt_input)


def test_cache_order(device, expect_error):
    """Production spec, other spec, production spec, then the production spec walked column-major:
    a program cached without the shard spec or orientation replays the wrong page map (BB-07)."""
    torch.manual_seed(42)
    sequence = (
        P2,
        P2_OTHER_SPEC,
        P2,
        {**P2, "orientation": "column_major"},
    )
    for scenario in sequence:
        _run_and_check(device, expect_error, scenario)


def _validation_input(device, kind):
    """(input_tensor, memory_config) that the op must refuse with ValueError or RuntimeError."""
    torch.manual_seed(42)
    p2_mc = _input_memory_config(device, P2)
    if kind == "host_tensor":
        x = ttnn.from_torch(_make_torch_input(P2["shape"]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return x, ttnn.DRAM_MEMORY_CONFIG
    if kind == "interleaved_input":
        x = ttnn.from_torch(
            _make_torch_input(P2["shape"]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return x, ttnn.DRAM_MEMORY_CONFIG
    if kind == "row_major_input":
        x = ttnn.from_torch(
            _make_torch_input((32, 5120)),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=p2_mc,
        )
        return x, ttnn.DRAM_MEMORY_CONFIG
    if kind == "block_sharded_input":
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
        spec = ttnn.ShardSpec(grid, (32, 128), ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)
        x = ttnn.from_torch(
            _make_torch_input((64, 256)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        return x, ttnn.DRAM_MEMORY_CONFIG
    if kind == "rank_3_input":
        x = ttnn.from_torch(
            _make_torch_input((1, 32, 5120)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=p2_mc,
        )
        return x, ttnn.DRAM_MEMORY_CONFIG
    if kind == "sharded_memory_config":
        x = ttnn.from_torch(
            _make_torch_input(P2["shape"]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=p2_mc,
        )
        return x, p2_mc
    raise ValueError(f"unknown validation kind {kind}")


@pytest.mark.parametrize(
    "kind",
    [
        "host_tensor",
        "interleaved_input",
        "row_major_input",
        "block_sharded_input",
        "rank_3_input",
        "sharded_memory_config",
    ],
)
def test_validation_refusals(device, expect_error, kind):
    x, memory_config = _validation_input(device, kind)
    try:
        with expect_error((ValueError, RuntimeError), ""):
            out = g6_sharded_to_interleaved(x, memory_config)
            _free(out)
    finally:
        if ttnn.is_tensor_storage_on_device(x):
            _free(x)
