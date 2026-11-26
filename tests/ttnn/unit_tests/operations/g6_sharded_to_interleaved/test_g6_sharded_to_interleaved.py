# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_sharded_to_interleaved (immutable spec for the implementer).

The op copies a height- or width-sharded L1 TILE tensor into a new interleaved
TILE tensor (DRAM or L1, as memory_config names): torch.clone on the logical
values, bit for bit. The cases are the two production cases, the other shard
spec of each, the bfloat16 case written to L1, and the checks the spec's rules
require (dirty destination, repeat, input intact, cache order, special values,
validation); nothing else (BB-22).

A case outside the op's SUPPORTED/EXCLUSIONS contract must be refused with a
support refusal and is then reported xfail, so this file needs no edit as
SUPPORTED grows toward TARGET.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_sharded_to_interleaved import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    g6_sharded_to_interleaved,
)

TILE = 32

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

# Production cases (feature_spec.py INPUTS) and the other shard spec of each.
P2_FLUX = {
    "shape": (1, 5120),
    "dtype": ttnn.bfloat16,
    "layout": ttnn.TILE_LAYOUT,
    "in_scheme": "width_sharded",
    "shard_shape": (32, 160),
    "grid": (((0, 0), (7, 3)),),
    "orientation": "row_major",
    "buffer": "dram",
}
P1_PANOPTIC = {
    "shape": (1, 128, 256, 32),
    "dtype": ttnn.bfloat8_b,
    "layout": ttnn.TILE_LAYOUT,
    "in_scheme": "height_sharded",
    "shard_shape": (320, 32),
    "grid": (((0, 0), (10, 8)), ((0, 9), (3, 9))),
    "orientation": "row_major",
    "buffer": "l1",
}
# 54 shards on (0,0)-(8,5) walked column-major, 53 of 3 tiles and the last of 1.
P2_OTHER_SPEC = {**P2_FLUX, "shard_shape": (32, 96), "grid": (((0, 0), (8, 5)),), "orientation": "column_major"}
# 94 shards on two ranges walked column-major, 93 of 11 tile rows and the last of 1.
P1_OTHER_SPEC = {
    **P1_PANOPTIC,
    "shard_shape": (352, 32),
    "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
    "orientation": "column_major",
}
# The destination comes from memory_config, not from the dtype or the strategy.
P2_TO_L1 = {**P2_FLUX, "buffer": "l1"}


def _scenario_id(scenario):
    shape = "x".join(str(d) for d in scenario["shape"])
    shard = "x".join(str(d) for d in scenario["shard_shape"])
    cores = "+".join(f"{x0}.{y0}-{x1}.{y1}" for (x0, y0), (x1, y1) in scenario["grid"])
    return f"{shape}-shard{shard}-{cores}-{scenario['orientation']}-to_{scenario['buffer']}"


def _is_supported(scenario):
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)
    if any(axes[axis] not in allowed for axis, allowed in SUPPORTED.items() if axis in axes):
        return False
    return not any(all(axes.get(k) == v for k, v in exc.items()) for exc in EXCLUSIONS)


def _skip_if_grid_too_small(device, grid):
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in grid:
        if x1 > size.x - 1 or y1 > size.y - 1:
            pytest.skip(f"shard grid end ({x1},{y1}) exceeds the device worker grid ({size.x},{size.y})")


def _input_memory_config(scenario):
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in scenario["grid"]]
    )
    spec = ttnn.ShardSpec(grid, tuple(scenario["shard_shape"]), _ORIENTATION[scenario["orientation"]])
    return ttnn.MemoryConfig(_IN_SCHEME[scenario["in_scheme"]], ttnn.BufferType.L1, spec)


def _make_torch_input(shape, *, salt=0, special_values=False):
    """randn with every 32x32 tile scaled by its own power of two.

    Alike tiles would hide a wrong page map: the scale makes each tile differ
    from its neighbours, also after bfloat8_b packing. `salt` shifts the scales
    so that no two calls of this file expect the same bytes: a freed output of
    an earlier call reused at the same address must not pass by chance.
    """
    torch.manual_seed(42)
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols, dtype=torch.float32)
    tile_cols = -(-cols // TILE)
    tile_index = (torch.arange(rows) // TILE)[:, None] * tile_cols + (torch.arange(cols) // TILE)[None, :]
    x = x * torch.pow(2.0, ((tile_index + salt) % 16 - 8).to(torch.float32))
    if special_values:
        for col, value in ((0, -0.0), (33, float("inf")), (66, float("-inf")), (99, float("nan")), (132, 1.0e-39)):
            x[0, col] = value
    return x.reshape(shape).to(torch.bfloat16)


def pytorch_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.clone(input_tensor)


_BIT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def _assert_bit_equal(actual, expected, what):
    """Exact gate (helpers.TOLERANCES: ("exact", None)); NaN, -0.0 and subnormals compare by pattern."""
    assert tuple(actual.shape) == tuple(
        expected.shape
    ), f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert (
        actual.dtype == expected.dtype and actual.dtype in _BIT_VIEW
    ), f"{what}: readback dtype {actual.dtype}, expected {expected.dtype}"
    view = _BIT_VIEW[actual.dtype]
    differ = actual.contiguous().view(view) != expected.contiguous().view(view)
    count = int(differ.sum())
    if count:
        where = torch.nonzero(differ.reshape(-1, actual.shape[-1]))[:4].tolist()
        tiles = sorted({(r // TILE, c // TILE) for r, c in where})
        pytest.fail(f"{what}: {count} of {differ.numel()} logical elements differ; first in tiles (row, col) {tiles}")


def _assert_output_contract(output, scenario):
    assert tuple(output.shape) == tuple(scenario["shape"])
    assert output.dtype == scenario["dtype"]
    assert output.layout == ttnn.TILE_LAYOUT
    mc = output.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert mc.buffer_type == _BUFFER_TYPE[scenario["buffer"]]


def _placement(tensor):
    mc = tensor.memory_config()
    spec = mc.shard_spec
    ranges = tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges())
    return (mc.memory_layout, mc.buffer_type, tuple(spec.shape), spec.orientation, ranges)


class _Case:
    """A sharded device input built from a scenario, its expected values and placement."""

    def __init__(self, device, scenario, *, salt, special_values=False):
        _skip_if_grid_too_small(device, scenario["grid"])
        self.scenario = scenario
        self.output_mc = _OUTPUT_MEMORY_CONFIG[scenario["buffer"]]
        torch_input = _make_torch_input(scenario["shape"], salt=salt, special_values=special_values)
        self.tt_input = ttnn.from_torch(
            torch_input,
            dtype=scenario["dtype"],
            layout=scenario["layout"],
            device=device,
            memory_config=_input_memory_config(scenario),
        )
        # bfloat16 is stored as given; bfloat8_b as the host packed it.
        self.source = torch_input if scenario["dtype"] == ttnn.bfloat16 else ttnn.to_torch(self.tt_input)
        self.expected = pytorch_reference(self.source)
        self.placement = _placement(self.tt_input)
        self.input_address = self.tt_input.buffer_address()

    def check_output(self, output, what="output"):
        _assert_output_contract(output, self.scenario)
        _assert_bit_equal(ttnn.to_torch(output), self.expected, what)

    def check_input_intact(self):
        assert self.tt_input.is_allocated(), "the op deallocated input_tensor"
        assert _placement(self.tt_input) == self.placement, "input_tensor placement changed"
        assert self.tt_input.buffer_address() == self.input_address, "input_tensor buffer moved"
        _assert_bit_equal(ttnn.to_torch(self.tt_input), self.source, "input")


def _refuse_if_unsupported(case, expect_error):
    if _is_supported(case.scenario):
        return
    with expect_error(SupportRefusal, ".*"):
        g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    ttnn.deallocate(case.tt_input)
    pytest.xfail(f"outside SUPPORTED: {_scenario_id(case.scenario)}")


@pytest.mark.parametrize(
    "scenario,salt",
    [
        pytest.param(P2_FLUX, 0, id=_scenario_id(P2_FLUX)),
        pytest.param(P1_PANOPTIC, 1, id=_scenario_id(P1_PANOPTIC)),
        pytest.param(P2_OTHER_SPEC, 2, id=_scenario_id(P2_OTHER_SPEC)),
        pytest.param(P1_OTHER_SPEC, 3, id=_scenario_id(P1_OTHER_SPEC)),
        pytest.param(P2_TO_L1, 4, id=_scenario_id(P2_TO_L1)),
    ],
)
def test_g6_sharded_to_interleaved(device, scenario, salt, expect_error):
    case = _Case(device, scenario, salt=salt)
    _refuse_if_unsupported(case, expect_error)
    output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    case.check_output(output)
    if scenario["buffer"] == "l1":
        assert output.buffer_address() != case.input_address, "the output shares input_tensor's L1 buffer"
    case.check_input_intact()
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)


def test_g6_sharded_to_interleaved_dirty_destination(device, expect_error):
    """Every page is written by the call: the output lands on a freed range pre-filled with 0x7F7F."""
    case = _Case(device, P2_FLUX, salt=5)
    _refuse_if_unsupported(case, expect_error)
    loud = torch.full(P2_FLUX["shape"], 0x7F7F, dtype=torch.int16).view(torch.bfloat16)
    dirty = ttnn.from_torch(
        loud, dtype=P2_FLUX["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=case.output_mc
    )
    freed_address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    assert output.buffer_address() == freed_address, (
        f"output at {output.buffer_address():#x}, freed dirty range at {freed_address:#x}: "
        "cannot confirm that every page was overwritten"
    )
    case.check_output(output)
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)


def test_g6_sharded_to_interleaved_repeat(device, expect_error):
    """Two calls on one input: bit-identical outputs in two new buffers, input unchanged."""
    case = _Case(device, P2_FLUX, salt=6)
    _refuse_if_unsupported(case, expect_error)
    first = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    second = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    assert second.buffer_address() != first.buffer_address(), "a later call returned the first call's buffer"
    case.check_output(first, "first call")
    case.check_output(second, "second call")
    case.check_input_intact()
    for tensor in (first, second, case.tt_input):
        ttnn.deallocate(tensor)


def test_g6_sharded_to_interleaved_cache_order(device, expect_error):
    """Program-cache key: specs that differ in shard shape, then in orientation alone, each map right."""
    sequence = (
        P2_FLUX,
        P2_OTHER_SPEC,
        P2_FLUX,
        {**P2_FLUX, "orientation": "column_major"},
        P2_FLUX,
    )
    for step, scenario in enumerate(sequence):
        case = _Case(device, scenario, salt=7 + step)
        _refuse_if_unsupported(case, expect_error)
        output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
        case.check_output(output, f"call {step} ({_scenario_id(scenario)})")
        ttnn.deallocate(output)
        ttnn.deallocate(case.tt_input)


def test_g6_sharded_to_interleaved_special_values(device, expect_error):
    """-0.0, +-inf, NaN and a subnormal move as stored bit patterns."""
    case = _Case(device, P2_FLUX, salt=12, special_values=True)
    _refuse_if_unsupported(case, expect_error)
    output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    case.check_output(output)
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)


def _assert_structural_refusal(expect_error, callable_):
    with expect_error((ValueError, RuntimeError), ".*") as info:
        callable_()
    assert not isinstance(
        info.value, SupportRefusal
    ), f"a structural input error must raise ValueError/RuntimeError, not a support refusal: {info.value!r}"


def test_g6_sharded_to_interleaved_rejects_unsharded_input(device, expect_error):
    tt_input = ttnn.from_torch(
        _make_torch_input(P2_FLUX["shape"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _assert_structural_refusal(expect_error, lambda: g6_sharded_to_interleaved(tt_input, ttnn.DRAM_MEMORY_CONFIG))
    ttnn.deallocate(tt_input)


def test_g6_sharded_to_interleaved_rejects_sharded_memory_config(device, expect_error):
    case = _Case(device, P2_FLUX, salt=13)
    sharded_mc = _input_memory_config(P2_FLUX)
    _assert_structural_refusal(expect_error, lambda: g6_sharded_to_interleaved(case.tt_input, sharded_mc))
    ttnn.deallocate(case.tt_input)
