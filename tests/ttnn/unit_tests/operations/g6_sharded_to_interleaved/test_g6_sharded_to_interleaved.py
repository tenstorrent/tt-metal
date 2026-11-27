# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_sharded_to_interleaved (immutable spec for the implementer).

The op moves a height- or width-sharded L1 TILE tensor to a new interleaved
tensor (DRAM or L1, as memory_config names), bit for bit: torch.clone on the
logical values. The gate is exact (helpers.TOLERANCES holds ("exact", None)
for both dtypes): bfloat16 compared as stored bit patterns, bfloat8_b compared
after the host unpacks input and output.

Shapes are limited to the production cases and the cases the spec's rules
require (BB-22): the two production specs, one other shard spec per case
(column-major, ragged last shard), the destination taken from memory_config,
a dirty destination, repeat / input-intact, cache order, special values and
the validation refusals. A cell outside the op's SUPPORTED must refuse with a
typed support refusal and is reported as xfail; once a refinement adds the
cell to SUPPORTED the same test runs it and gates it exactly.
"""

import importlib
import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved
from ttnn.operations._op_contract import SupportRefusal

TILE = 32

_OP_MODULE = importlib.import_module("ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved")

# Phase 0 baseline the op must keep claiming (spec: Phase 0 SUPPORTED).
_PHASE0 = {
    "dtype": ttnn.bfloat16,
    "layout": ttnn.TILE_LAYOUT,
    "in_scheme": "width_sharded",
    "buffer": "dram",
}

# --- Scenarios: production specs and the specs the rules require ------------------

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
# 54 shards on (0,0)-(8,5): 53 of 3 tiles and the last of 1, walked column-major.
P2_OTHER_SPEC = {**P2_FLUX, "shard_shape": (32, 96), "grid": (((0, 0), (8, 5)),), "orientation": "column_major"}
# Same tiles, destination named by memory_config: L1 interleaved.
P2_TO_L1 = {**P2_FLUX, "buffer": "l1"}
# Production spec, orientation alone differs.
P2_COLUMN_MAJOR = {**P2_FLUX, "orientation": "column_major"}

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
# 94 shards on a 9 x 10 and a 1 x 4 range: 93 of 11 tile rows and the last of 1, column-major.
P1_OTHER_SPEC = {
    **P1_PANOPTIC,
    "shard_shape": (352, 32),
    "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
    "orientation": "column_major",
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


def _scenario_id(scn):
    shape = "x".join(str(d) for d in scn["shape"])
    shard = "x".join(str(d) for d in scn["shard_shape"])
    cores = "+".join(f"{x0}.{y0}-{x1}.{y1}" for (x0, y0), (x1, y1) in scn["grid"])
    return f"{shape}-{scn['in_scheme']}-shard{shard}-{cores}-{scn['orientation']}-to_{scn['buffer']}"


# --- Helpers ------------------------------------------------------------------------


def pytorch_reference(input_tensor):
    """torch.clone: the op converts placement and changes no value."""
    return torch.clone(input_tensor)


def _make_torch_input(shape, *, special_values=False):
    """randn, each 32x32 tile of the 2-D view scaled by 2 ** (t % 16 - 8).

    Alike tiles would hide a wrong page map; the per-tile scale makes every
    neighbouring tile differ, also after bfloat8_b packing.
    """
    torch.manual_seed(42)
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols, dtype=torch.float32)
    tile_cols = -(-cols // TILE)
    tile_index = (torch.arange(rows) // TILE)[:, None] * tile_cols + (torch.arange(cols) // TILE)[None, :]
    x = x * torch.pow(2.0, (tile_index % 16 - 8).to(torch.float32))
    if special_values:
        for row, col, value in (
            (0, 0 * TILE + 0, -0.0),
            (0, 1 * TILE + 1, float("inf")),
            (0, 2 * TILE + 2, float("-inf")),
            (0, 3 * TILE + 3, float("nan")),
            (0, 4 * TILE + 4, 1.0e-39),
        ):
            x[row, col] = value
    return x.reshape(shape).to(torch.bfloat16)


def _skip_if_grid_too_small(device, grid):
    size = device.compute_with_storage_grid_size()
    for (_, _), (x1, y1) in grid:
        if x1 > size.x - 1 or y1 > size.y - 1:
            pytest.skip(f"shard grid end ({x1},{y1}) exceeds the device worker grid ({size.x},{size.y})")


def _input_memory_config(device, scn):
    _skip_if_grid_too_small(device, scn["grid"])
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in scn["grid"]]
    )
    spec = ttnn.ShardSpec(grid, tuple(scn["shard_shape"]), _ORIENTATION[scn["orientation"]])
    return ttnn.MemoryConfig(_IN_SCHEME[scn["in_scheme"]], ttnn.BufferType.L1, spec)


def _make_input(device, scn, *, special_values=False):
    torch_input = _make_torch_input(scn["shape"], special_values=special_values)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=scn["dtype"],
        layout=scn["layout"],
        device=device,
        memory_config=_input_memory_config(device, scn),
    )
    # bfloat16 is stored as given; bfloat8_b as the host packed it.
    source = torch_input if scn["dtype"] == ttnn.bfloat16 else ttnn.to_torch(tt_input)
    return tt_input, source


def _is_supported(scn):
    axes = {"dtype": scn["dtype"], "layout": scn["layout"], "in_scheme": scn["in_scheme"], "buffer": scn["buffer"]}
    if any(axes[axis] not in allowed for axis, allowed in _OP_MODULE.SUPPORTED.items() if axis in axes):
        return False
    return not any(all(axes.get(k) == v for k, v in exc.items()) for exc in _OP_MODULE.EXCLUSIONS)


def _expect_refusal_then_xfail(device, expect_error, scn):
    tt_input, _ = _make_input(device, scn)
    try:
        with expect_error(SupportRefusal, ""):
            g6_sharded_to_interleaved(tt_input, _OUTPUT_MEMORY_CONFIG[scn["buffer"]])
    finally:
        ttnn.deallocate(tt_input)
    pytest.xfail(f"outside SUPPORTED (refinement candidate): {_scenario_id(scn)}")


# Bit patterns, so NaN, -0.0 and subnormals compare exactly.
_BIT_VIEW = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
}


def _assert_bit_equal(actual, expected, what):
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
        cols = actual.shape[-1]
        where = torch.nonzero(differ.reshape(-1, cols))[:4].tolist()
        tiles = sorted({(r // TILE, c // TILE) for r, c in where})
        pytest.fail(f"{what}: {count} of {differ.numel()} logical elements differ; first in tiles (row, col) {tiles}")


def _assert_output_contract(output, scn):
    assert tuple(output.shape) == tuple(scn["shape"]), f"shape {tuple(output.shape)} != {scn['shape']}"
    assert output.dtype == scn["dtype"], f"dtype {output.dtype} != {scn['dtype']}"
    assert output.layout == ttnn.TILE_LAYOUT, f"layout {output.layout}"
    mc = output.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"memory_layout {mc.memory_layout}"
    assert mc.buffer_type == _BUFFER_TYPE[scn["buffer"]], f"buffer_type {mc.buffer_type}"


def _placement(tensor):
    mc = tensor.memory_config()
    spec = mc.shard_spec
    ranges = tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges())
    return (mc.memory_layout, mc.buffer_type, tuple(spec.shape), spec.orientation, ranges)


def _run_and_check(device, scn, *, special_values=False):
    """One graded call: exact output, output contract, input intact and not reused."""
    tt_input, source = _make_input(device, scn, special_values=special_values)
    outputs = []
    try:
        placement_before = _placement(tt_input)
        input_address = tt_input.buffer_address()
        expected = pytorch_reference(source)

        tt_output = g6_sharded_to_interleaved(tt_input, _OUTPUT_MEMORY_CONFIG[scn["buffer"]])
        outputs.append(tt_output)
        _assert_output_contract(tt_output, scn)
        _assert_bit_equal(ttnn.to_torch(tt_output), expected, "output")

        assert tt_input.is_allocated(), "the op deallocated input_tensor"
        assert _placement(tt_input) == placement_before, "input placement changed"
        assert tt_input.buffer_address() == input_address, "input buffer moved"
        if scn["buffer"] == "l1":
            assert tt_output.buffer_address() != input_address, "output shares input_tensor's L1 buffer"
        _assert_bit_equal(ttnn.to_torch(tt_input), source, "input")
    finally:
        for tensor in outputs + [tt_input]:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)


# --- Tests ---------------------------------------------------------------------------


def test_phase0_support_floor():
    """The Phase 0 cell stays claimed, and no tagger beyond the four TARGET axes exists."""
    for axis, value in _PHASE0.items():
        assert value in _OP_MODULE.SUPPORTED.get(axis, []), f"SUPPORTED[{axis!r}] must hold {value!r}"
    assert set(_OP_MODULE.INPUT_TAGGERS) == {"dtype", "layout", "in_scheme", "buffer"}
    for exc in _OP_MODULE.EXCLUSIONS:
        assert not all(_PHASE0.get(k) == v for k, v in exc.items()), f"EXCLUSIONS refuses the Phase 0 cell: {exc}"


@pytest.mark.parametrize(
    "scn",
    [P2_FLUX, P2_OTHER_SPEC, P2_TO_L1, P1_PANOPTIC, P1_OTHER_SPEC],
    ids=_scenario_id,
)
def test_g6_sharded_to_interleaved(device, expect_error, scn):
    """Production specs, the other shard spec of each case, destination from memory_config."""
    if not _is_supported(scn):
        _expect_refusal_then_xfail(device, expect_error, scn)
    _run_and_check(device, scn)


def test_dirty_destination(device, expect_error):
    """Every page is written by the call: the output lands on a freed range pre-filled with 0x7F7F."""
    scn = P2_FLUX
    if not _is_supported(scn):
        _expect_refusal_then_xfail(device, expect_error, scn)
    output_mc = _OUTPUT_MEMORY_CONFIG[scn["buffer"]]
    tt_input, source = _make_input(device, scn)
    tt_output = None
    try:
        loud = torch.full(scn["shape"], 0x7F7F, dtype=torch.int16).view(torch.bfloat16)
        dirty = ttnn.from_torch(
            loud, dtype=scn["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=output_mc
        )
        freed_address = dirty.buffer_address()
        ttnn.deallocate(dirty)

        tt_output = g6_sharded_to_interleaved(tt_input, output_mc)
        assert tt_output.buffer_address() == freed_address, (
            f"output at {tt_output.buffer_address():#x}, freed dirty range at {freed_address:#x}: "
            "the case cannot confirm that every page was overwritten"
        )
        _assert_output_contract(tt_output, scn)
        _assert_bit_equal(ttnn.to_torch(tt_output), pytorch_reference(source), "output")
    finally:
        for tensor in [tt_output, tt_input]:
            if tensor is not None and tensor.is_allocated():
                ttnn.deallocate(tensor)


def test_repeat_new_storage(device, expect_error):
    """Two calls on one input: bit-identical outputs in two new buffers; the input stays intact."""
    scn = P2_FLUX
    if not _is_supported(scn):
        _expect_refusal_then_xfail(device, expect_error, scn)
    output_mc = _OUTPUT_MEMORY_CONFIG[scn["buffer"]]
    tt_input, source = _make_input(device, scn)
    outputs = []
    try:
        input_address = tt_input.buffer_address()
        placement_before = _placement(tt_input)
        for _ in range(2):
            outputs.append(g6_sharded_to_interleaved(tt_input, output_mc))
        assert (
            outputs[0].buffer_address() != outputs[1].buffer_address()
        ), "a later call returned the first call's buffer"
        first = ttnn.to_torch(outputs[0])
        _assert_bit_equal(first, pytorch_reference(source), "output")
        _assert_bit_equal(ttnn.to_torch(outputs[1]), first, "repeat")
        assert tt_input.is_allocated() and tt_input.buffer_address() == input_address, "input buffer freed or moved"
        assert _placement(tt_input) == placement_before, "input placement changed"
        _assert_bit_equal(ttnn.to_torch(tt_input), source, "input")
    finally:
        for tensor in outputs + [tt_input]:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)


def test_cache_order(device, expect_error):
    """Program-cache key completeness (BB-07): specs alternate, the last differs in orientation alone."""
    for scn in (P2_FLUX, P2_OTHER_SPEC, P2_FLUX, P2_COLUMN_MAJOR):
        if not _is_supported(scn):
            _expect_refusal_then_xfail(device, expect_error, scn)
        _run_and_check(device, scn)


def test_special_values(device, expect_error):
    """-0.0, +-inf, NaN and a subnormal move as stored bit patterns (bfloat16)."""
    scn = P2_FLUX
    if not _is_supported(scn):
        _expect_refusal_then_xfail(device, expect_error, scn)
    _run_and_check(device, scn, special_values=True)


# --- Validation: structural refusals are ValueError / RuntimeError, never a support refusal ---


def _build_or_skip(build):
    try:
        return build()
    except Exception as exc:  # noqa: BLE001 - an input the runtime cannot build tests nothing about the op
        pytest.skip(f"cannot build the invalid input on this device: {exc!r}")


def _invalid_interleaved_input(device):
    return ttnn.from_torch(
        _make_torch_input(P2_FLUX["shape"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _invalid_host_input(device):
    return ttnn.from_torch(_make_torch_input(P2_FLUX["shape"]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _invalid_block_sharded_input(device):
    _skip_if_grid_too_small(device, (((0, 0), (7, 0)),))
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 640), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        _make_torch_input(P2_FLUX["shape"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )


def _invalid_row_major_input(device):
    _skip_if_grid_too_small(device, P2_FLUX["grid"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (1, 160), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        _make_torch_input(P2_FLUX["shape"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )


def _invalid_dram_sharded_input(device):
    dram = device.dram_grid_size()
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, 0))])
    shard_width = -(-P2_FLUX["shape"][-1] // (dram.x * TILE)) * TILE
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (32, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        _make_torch_input(P2_FLUX["shape"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )


def _invalid_rank3_input(device):
    torch_input = _make_torch_input(P2_FLUX["shape"]).reshape(1, 1, P2_FLUX["shape"][-1])
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_input_memory_config(device, P2_FLUX),
    )


def _valid_input(device):
    tt_input, _ = _make_input(device, P2_FLUX)
    return tt_input


@pytest.mark.parametrize(
    "build_input, memory_config",
    [
        pytest.param(_invalid_host_input, ttnn.DRAM_MEMORY_CONFIG, id="input_not_on_device"),
        pytest.param(_invalid_interleaved_input, ttnn.DRAM_MEMORY_CONFIG, id="input_not_sharded"),
        pytest.param(_invalid_block_sharded_input, ttnn.DRAM_MEMORY_CONFIG, id="input_block_sharded"),
        pytest.param(_invalid_row_major_input, ttnn.DRAM_MEMORY_CONFIG, id="input_row_major"),
        pytest.param(_invalid_dram_sharded_input, ttnn.DRAM_MEMORY_CONFIG, id="input_not_in_l1"),
        pytest.param(_invalid_rank3_input, ttnn.DRAM_MEMORY_CONFIG, id="input_rank_3"),
        pytest.param(_valid_input, "input_memory_config", id="memory_config_not_interleaved"),
    ],
)
def test_validation_refusals(device, expect_error, build_input, memory_config):
    tt_input = _build_or_skip(lambda: build_input(device))
    try:
        if memory_config == "input_memory_config":
            memory_config = tt_input.memory_config()
        with expect_error((ValueError, RuntimeError), "") as excinfo:
            g6_sharded_to_interleaved(tt_input, memory_config)
        assert not isinstance(
            excinfo.value, SupportRefusal
        ), f"a structural refusal must not read as a missing feature: {excinfo.value!r}"
    finally:
        if tt_input.storage_type() == ttnn.StorageType.DEVICE and tt_input.is_allocated():
            ttnn.deallocate(tt_input)
