# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_sharded_to_interleaved.

The op copies a height- or width-sharded L1 TILE tensor into a new interleaved
tensor in DRAM or L1 (as memory_config names), bit for bit: torch.clone on the
logical values. The cases are the two production calls, one other shard spec
per call, and the cases the contract's rules require (BB-22). A cell outside
the op's current SUPPORTED xfails on a support refusal; a refusal of a cell
inside SUPPORTED fails.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_sharded_to_interleaved import SUPPORTED, g6_sharded_to_interleaved

TILE = 32

# FLUX.2-dev: 160 tiles, 32 width shards of 1 x 5 tiles, written to DRAM.
P2_FLUX = {
    "shape": (1, 5120),
    "dtype": ttnn.bfloat16,
    "in_scheme": "width_sharded",
    "shard_shape": (32, 160),
    "grid": (((0, 0), (7, 3)),),
    "orientation": "row_major",
    "buffer": "dram",
}
# Panoptic-DeepLab: 1024 tiles, 103 height shards of 10 tile rows on two core
# ranges (the last shard holds 4), written to L1.
P1_PANOPTIC = {
    "shape": (1, 128, 256, 32),
    "dtype": ttnn.bfloat8_b,
    "in_scheme": "height_sharded",
    "shard_shape": (320, 32),
    "grid": (((0, 0), (10, 8)), ((0, 9), (3, 9))),
    "orientation": "row_major",
    "buffer": "l1",
}
# Another spec per call: other shard shape and core ranges, a shorter last
# shard, walked column-major.
P2_OTHER_SPEC = {**P2_FLUX, "shard_shape": (32, 96), "grid": (((0, 0), (8, 5)),), "orientation": "column_major"}
P1_OTHER_SPEC = {
    **P1_PANOPTIC,
    "shard_shape": (352, 32),
    "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
    "orientation": "column_major",
}
# Destination from memory_config: the bfloat16 width-sharded call written to L1.
P2_FLUX_TO_L1 = {**P2_FLUX, "buffer": "l1"}

_IN_SCHEME = {
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
}
_ORIENTATION = {
    "row_major": ttnn.ShardOrientation.ROW_MAJOR,
    "column_major": ttnn.ShardOrientation.COL_MAJOR,
}
_OUTPUT_MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
_BUFFER_TYPE = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}
_BIT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def pytorch_g6_sharded_to_interleaved(input_tensor):
    return torch.clone(input_tensor)


def _scenario_id(scenario):
    shape = "x".join(str(d) for d in scenario["shape"])
    shard = "x".join(str(d) for d in scenario["shard_shape"])
    cores = "+".join(f"{x0}.{y0}-{x1}.{y1}" for (x0, y0), (x1, y1) in scenario["grid"])
    return f"{shape}-shard{shard}-{cores}-{scenario['orientation']}-to_{scenario['buffer']}"


def _cell(scenario):
    return {
        "dtype": scenario["dtype"],
        "layout": ttnn.TILE_LAYOUT,
        "in_scheme": scenario["in_scheme"],
        "buffer": scenario["buffer"],
    }


def _in_supported(scenario):
    return all(value in SUPPORTED.get(axis, ()) for axis, value in _cell(scenario).items())


def _core_ranges(grid):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in grid]
    )


def _skip_if_grid_too_small(device, grid):
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in grid:
        if x1 >= size.x or y1 >= size.y:
            pytest.skip(f"shard grid end ({x1},{y1}) exceeds the device worker grid ({size.x},{size.y})")


def _sharded_memory_config(device, scenario, memory_layout=None):
    _skip_if_grid_too_small(device, scenario["grid"])
    spec = ttnn.ShardSpec(
        _core_ranges(scenario["grid"]), tuple(scenario["shard_shape"]), _ORIENTATION[scenario["orientation"]]
    )
    return ttnn.MemoryConfig(memory_layout or _IN_SCHEME[scenario["in_scheme"]], ttnn.BufferType.L1, spec)


def _torch_input(shape, *, special_values=False):
    """randn scaled per 32x32 tile by 2 ** (t % 16 - 8), t the row-major tile index of the 2-D view.

    Neighbouring tiles differ, so a wrong page map cannot pass on alike tiles,
    and a bfloat8_b tile carrying another tile's exponents changes value.
    """
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols, dtype=torch.float32)
    tile_index = (torch.arange(rows) // TILE)[:, None] * math.ceil(cols / TILE) + (torch.arange(cols) // TILE)[None, :]
    x = x * torch.pow(2.0, (tile_index % 16 - 8).to(torch.float32))
    if special_values:
        # One per tile of the first tile row; 1e-39 is a bfloat16 subnormal.
        for t, value in enumerate((-0.0, float("inf"), float("-inf"), float("nan"), 1.0e-39)):
            x[0, t * TILE + t] = value
    return x.reshape(shape).to(torch.bfloat16)


def _make_input(device, scenario, created, *, special_values=False):
    """Sharded device input and its reference values (as stored on the device)."""
    memory_config = _sharded_memory_config(device, scenario)
    torch_input = _torch_input(scenario["shape"], special_values=special_values)
    tt_input = ttnn.from_torch(
        torch_input, dtype=scenario["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    created.append(tt_input)
    # bfloat16 is stored as given; bfloat8_b as the host packed it, so its reference is the readback.
    expected = torch_input if scenario["dtype"] == ttnn.bfloat16 else ttnn.to_torch(tt_input)
    return tt_input, expected


def _free(tensors):
    for tensor in tensors:
        if tensor.is_allocated():
            ttnn.deallocate(tensor)


def _placement(tensor):
    mc = tensor.memory_config()
    spec = mc.shard_spec
    shard = None
    if spec is not None:
        ranges = tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges())
        shard = (tuple(spec.shape), spec.orientation, ranges)
    return (mc.memory_layout, mc.buffer_type, shard)


def _assert_bit_equal(actual, expected, what):
    """Bit equality of the logical values; NaN, -0.0 and subnormals compare by pattern."""
    assert tuple(actual.shape) == tuple(
        expected.shape
    ), f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert (
        actual.dtype == expected.dtype and actual.dtype in _BIT_VIEW
    ), f"{what}: readback dtype {actual.dtype} vs {expected.dtype}"
    view = _BIT_VIEW[actual.dtype]
    differ = actual.contiguous().view(view) != expected.contiguous().view(view)
    count = int(differ.sum())
    if count:
        where = torch.nonzero(differ.reshape(-1, actual.shape[-1]))[:4].tolist()
        tiles = sorted({(row // TILE, col // TILE) for row, col in where})
        pytest.fail(f"{what}: {count} of {differ.numel()} logical elements differ; first in tiles (row, col) {tiles}")


def _assert_output_contract(output, scenario):
    assert tuple(output.shape) == tuple(scenario["shape"]), f"output shape {tuple(output.shape)}"
    assert output.dtype == scenario["dtype"], f"output dtype {output.dtype}"
    assert output.layout == ttnn.TILE_LAYOUT, f"output layout {output.layout}"
    mc = output.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"output memory layout {mc.memory_layout}"
    assert mc.buffer_type == _BUFFER_TYPE[scenario["buffer"]], f"output buffer type {mc.buffer_type}"


def _call(tt_input, scenario):
    try:
        return g6_sharded_to_interleaved(tt_input, _OUTPUT_MEMORY_CONFIG[scenario["buffer"]])
    except SupportRefusal as refusal:
        if _in_supported(scenario):
            raise
        pytest.xfail(f"cell outside SUPPORTED {_cell(scenario)}: {refusal}")


def _checked_call(tt_input, expected, scenario, created):
    """One call, graded: exact output, output contract, input allocated, in place and unchanged."""
    placement = _placement(tt_input)
    input_address = tt_input.buffer_address()
    output = _call(tt_input, scenario)
    created.append(output)
    _assert_output_contract(output, scenario)
    output_torch = ttnn.to_torch(output)
    _assert_bit_equal(output_torch, pytorch_g6_sharded_to_interleaved(expected), "output")
    assert tt_input.is_allocated(), "the op deallocated input_tensor"
    assert _placement(tt_input) == placement, f"input placement changed: {placement} -> {_placement(tt_input)}"
    assert tt_input.buffer_address() == input_address, "input_tensor's buffer moved during the call"
    if scenario["buffer"] == "l1":
        assert output.buffer_address() != input_address, "the output shares input_tensor's L1 buffer"
    _assert_bit_equal(ttnn.to_torch(tt_input), expected, "input")
    return output, output_torch


@pytest.mark.parametrize(
    "scenario", [P2_FLUX, P1_PANOPTIC, P2_OTHER_SPEC, P1_OTHER_SPEC, P2_FLUX_TO_L1], ids=_scenario_id
)
def test_g6_sharded_to_interleaved(device, scenario):
    torch.manual_seed(42)
    created = []
    try:
        tt_input, expected = _make_input(device, scenario, created)
        _checked_call(tt_input, expected, scenario, created)
    finally:
        _free(created)


def test_dirty_destination(device):
    """The output lands on a freed range pre-filled with 0x7F7F: every page must be written by the call."""
    scenario = P2_FLUX
    torch.manual_seed(42)
    created = []
    try:
        tt_input, expected = _make_input(device, scenario, created)
        loud = torch.full(scenario["shape"], 0x7F7F, dtype=torch.int16).view(torch.bfloat16)
        dirty = ttnn.from_torch(
            loud,
            dtype=scenario["dtype"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_OUTPUT_MEMORY_CONFIG[scenario["buffer"]],
        )
        freed_address = dirty.buffer_address()
        ttnn.deallocate(dirty)
        output, _ = _checked_call(tt_input, expected, scenario, created)
        assert output.buffer_address() == freed_address, (
            f"output at {output.buffer_address():#x}, freed dirty range at {freed_address:#x}: "
            "the case cannot confirm that every page was overwritten"
        )
    finally:
        _free(created)


def test_two_live_inputs_alternate(device):
    """Two inputs of one spec, both alive, called a, b, a: a reused program must take each call's own addresses."""
    scenario = P2_FLUX
    torch.manual_seed(42)
    created = []
    try:
        input_a, expected_a = _make_input(device, scenario, created)
        input_b, expected_b = _make_input(device, scenario, created)
        for tt_input, expected in ((input_a, expected_a), (input_b, expected_b), (input_a, expected_a)):
            _checked_call(tt_input, expected, scenario, created)
    finally:
        _free(created)


def test_cache_order(device):
    """Production spec, other spec, production spec, then the production spec walked column-major."""
    torch.manual_seed(42)
    for scenario in (P2_FLUX, P2_OTHER_SPEC, P2_FLUX, {**P2_FLUX, "orientation": "column_major"}):
        created = []
        try:
            tt_input, expected = _make_input(device, scenario, created)
            _checked_call(tt_input, expected, scenario, created)
        finally:
            _free(created)


def test_repeat_bit_identical(device):
    """Two calls on one input give bit-identical outputs in two buffers."""
    scenario = P2_FLUX
    torch.manual_seed(42)
    created = []
    try:
        tt_input, expected = _make_input(device, scenario, created)
        first, first_torch = _checked_call(tt_input, expected, scenario, created)
        second, second_torch = _checked_call(tt_input, expected, scenario, created)
        assert second.buffer_address() != first.buffer_address(), "a later call returned the first call's buffer"
        _assert_bit_equal(second_torch, first_torch, "repeat")
    finally:
        _free(created)


def test_program_cache_reuse(device):
    """A repeat call with the same shape, dtype and placements builds no new program."""
    scenario = P2_FLUX
    torch.manual_seed(42)
    created = []
    try:
        warm_input, warm_expected = _make_input(device, scenario, created)
        _checked_call(warm_input, warm_expected, scenario, created)
        entries = device.num_program_cache_entries()
        tt_input, expected = _make_input(device, scenario, created)
        _checked_call(tt_input, expected, scenario, created)
        assert device.num_program_cache_entries() == entries, "a repeat call added a program cache entry"
    finally:
        _free(created)


def test_special_values(device):
    """-0.0, +-inf, NaN and a subnormal move as stored bit patterns (bfloat16)."""
    scenario = P2_FLUX
    torch.manual_seed(42)
    created = []
    try:
        tt_input, expected = _make_input(device, scenario, created, special_values=True)
        _checked_call(tt_input, expected, scenario, created)
    finally:
        _free(created)


# --- Validation: structural errors raise ValueError or RuntimeError, never a support refusal ---


def _build_or_skip(build):
    try:
        return build()
    except (RuntimeError, ValueError) as error:
        pytest.skip(f"invalid input could not be constructed on this device: {error}")


def _assert_rejected(expect_error, tt_input, memory_config):
    # The message is the implementer's wording; only the exception type is the contract.
    with expect_error((ValueError, RuntimeError), ""):
        g6_sharded_to_interleaved(tt_input, memory_config)


def test_rejects_host_tensor(device, expect_error):
    torch.manual_seed(42)
    host_tensor = ttnn.from_torch(_torch_input((1, 5120)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    _assert_rejected(expect_error, host_tensor, ttnn.DRAM_MEMORY_CONFIG)


def test_rejects_interleaved_input(device, expect_error):
    torch.manual_seed(42)
    tt_input = ttnn.from_torch(
        _torch_input((1, 5120)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    try:
        _assert_rejected(expect_error, tt_input, ttnn.DRAM_MEMORY_CONFIG)
    finally:
        _free([tt_input])


def test_rejects_block_sharded_input(device, expect_error):
    torch.manual_seed(42)
    scenario = {**P2_FLUX, "shard_shape": (32, 640), "grid": (((0, 0), (7, 0)),)}
    memory_config = _sharded_memory_config(device, scenario, ttnn.TensorMemoryLayout.BLOCK_SHARDED)
    tt_input = _build_or_skip(
        lambda: ttnn.from_torch(
            _torch_input(scenario["shape"]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
    )
    try:
        _assert_rejected(expect_error, tt_input, ttnn.DRAM_MEMORY_CONFIG)
    finally:
        _free([tt_input])


def test_rejects_row_major_input(device, expect_error):
    torch.manual_seed(42)
    scenario = {**P2_FLUX, "shard_shape": (1, 160)}
    memory_config = _sharded_memory_config(device, scenario)
    tt_input = _build_or_skip(
        lambda: ttnn.from_torch(
            _torch_input(scenario["shape"]),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
    )
    try:
        _assert_rejected(expect_error, tt_input, ttnn.DRAM_MEMORY_CONFIG)
    finally:
        _free([tt_input])


def test_rejects_rank_3_input(device, expect_error):
    torch.manual_seed(42)
    scenario = {**P2_FLUX, "shape": (1, 1, 5120)}
    created = []
    try:
        tt_input, _ = _make_input(device, scenario, created)
        _assert_rejected(expect_error, tt_input, ttnn.DRAM_MEMORY_CONFIG)
    finally:
        _free(created)


def test_rejects_sharded_output_config(device, expect_error):
    torch.manual_seed(42)
    created = []
    try:
        tt_input, _ = _make_input(device, P2_FLUX, created)
        _assert_rejected(expect_error, tt_input, tt_input.memory_config())
    finally:
        _free(created)
