# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshard.

g6_reshard(x, memory_config) returns a new BLOCK_SHARDED L1 tensor in the placement memory_config
names whose every tile is x's tile at the same tile index, moved as stored. The production shape
is the only shape (BB-22); the cases vary placements instead: P is the production pair, I0 / I3 /
I5 vary the input placement (identity, orientation, grid off the origin), O1 the output placement
(off the origin, COL_MAJOR, cut-short last block column). Graded bit-exactly: a PCC stays near 1.0
when a few blocks land on the wrong core.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_reshard import g6_reshard

SHAPE = (1, 1, 2048, 1024)
DTYPE = ttnn.bfloat8_b
LAYOUT = ttnn.TILE_LAYOUT
ROW_MAJOR = ttnn.ShardOrientation.ROW_MAJOR
COL_MAJOR = ttnn.ShardOrientation.COL_MAJOR

# name: (input grid, input shard, input orientation, output grid, output shard, output orientation)
PLACEMENTS = {
    "P": (((0, 0), (10, 9)), (224, 96), ROW_MAJOR, ((0, 0), (7, 7)), (256, 128), ROW_MAJOR),
    "I0": (((0, 0), (7, 7)), (256, 128), ROW_MAJOR, ((0, 0), (7, 7)), (256, 128), ROW_MAJOR),
    "I3": (((0, 0), (7, 7)), (256, 128), COL_MAJOR, ((0, 0), (7, 7)), (256, 128), ROW_MAJOR),
    "I5": (((2, 1), (9, 8)), (256, 128), ROW_MAJOR, ((0, 0), (7, 7)), (256, 128), ROW_MAJOR),
    "O1": (((0, 0), (10, 9)), (224, 96), ROW_MAJOR, ((1, 0), (8, 6)), (256, 160), COL_MAJOR),
}


def pytorch_g6_reshard(x):
    """Reference: the values of x, bit for bit. The placement decides where they live, not what they are."""
    return x.clone(memory_format=torch.contiguous_format)


def _core_range_set(*grids):
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in grids}
    )


def _sharded(grid, shard, orientation, *, layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer=ttnn.BufferType.L1):
    spec = ttnn.ShardSpec(_core_range_set(grid), list(shard), orientation)
    return ttnn.MemoryConfig(layout, buffer, spec)


def _require_grid(device, *grids):
    # A rectangle beyond the worker grid fails instead of skipping: the production case must not
    # pass vacuously on a smaller part.
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in grids:
        assert x1 < size.x and y1 < size.y, (
            f"core rectangle ending at ({x1}, {y1}) needs a worker grid of at least "
            f"{x1 + 1}x{y1 + 1}; this device has {size.x}x{size.y}"
        )


def _random_input(shape=SHAPE, seed=42):
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _place_input(device, name):
    """x in the input placement of `name`, x read back (the reference values), and out's config."""
    in_grid, in_shard, in_orientation, out_grid, out_shard, out_orientation = PLACEMENTS[name]
    _require_grid(device, in_grid, out_grid)
    x = ttnn.from_torch(
        _random_input(),
        dtype=DTYPE,
        layout=LAYOUT,
        device=device,
        memory_config=_sharded(in_grid, in_shard, in_orientation),
    )
    return x, ttnn.to_torch(x), _sharded(out_grid, out_shard, out_orientation)


def _assert_bit_exact(actual, expected, what):
    # Integer views keep -0.0 apart from +0.0 and NaN payloads apart.
    assert actual.dtype == expected.dtype, f"{what}: readback dtype {actual.dtype}, expected {expected.dtype}"
    assert list(actual.shape) == list(expected.shape), f"{what}: shape {list(actual.shape)}"
    differ = actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    if bool(differ.any()):
        width = expected.shape[-1]
        tiles = differ.reshape(-1, 32, width // 32, 32).any(dim=3).any(dim=1).nonzero().tolist()
        raise AssertionError(
            f"{what}: {int(differ.sum())} of {differ.numel()} elements differ; "
            f"{len(tiles)} tiles, first (tile row, tile col): {[tuple(t) for t in tiles[:8]]}"
        )


def _rectangles(core_range_set):
    return sorted(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in core_range_set.ranges())


def _assert_placement(out, out_cfg):
    assert list(out.shape) == list(SHAPE)
    assert out.dtype == DTYPE
    assert out.layout == LAYOUT
    got = out.memory_config()
    assert got.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert got.buffer_type == ttnn.BufferType.L1
    spec, want = got.shard_spec, out_cfg.shard_spec
    assert spec is not None
    assert _rectangles(spec.grid) == _rectangles(want.grid)
    assert list(spec.shape) == list(want.shape)
    assert spec.orientation == want.orientation


def _run_and_check(device, name):
    x, x_host, out_cfg = _place_input(device, name)
    expected = pytorch_g6_reshard(x_host)
    out = g6_reshard(x, out_cfg)
    _assert_placement(out, out_cfg)
    out_host = ttnn.to_torch(out)
    _assert_bit_exact(out_host, x_host, f"{name}: out against x read back")
    _assert_bit_exact(out_host, expected, f"{name}: out against the reference")
    return x, x_host, out


@pytest.mark.parametrize("name", list(PLACEMENTS))
def test_g6_reshard(device, name):
    x, x_host, out = _run_and_check(device, name)
    assert out is not x
    _assert_bit_exact(ttnn.to_torch(x), x_host, f"{name}: input_tensor after the call")


def test_g6_reshard_new_tensor_on_identity_placement(device):
    """Same placement on both sides: out is still a new buffer and x is unchanged."""
    x, x_host, out = _run_and_check(device, "I0")
    assert out.buffer_address() != x.buffer_address(), "out shares input_tensor's L1 buffer"
    _assert_bit_exact(ttnn.to_torch(x), x_host, "I0: input_tensor after the call")


def test_g6_reshard_program_cache_placements(device):
    """I0 and I3 differ only in the input orientation, P and O1 only in memory_config: a cache key
    missing either replays a wrong program."""
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    for name in ("I0", "I3", "P", "O1", "P", "I5"):
        _run_and_check(device, name)


def test_g6_reshard_repeat_determinism(device):
    """Three calls on one x, every output kept alive and bit-identical to x."""
    x, x_host, out_cfg = _place_input(device, "P")
    outs = [g6_reshard(x, out_cfg) for _ in range(3)]
    for i, out in enumerate(outs):
        _assert_placement(out, out_cfg)
        _assert_bit_exact(ttnn.to_torch(out), x_host, f"P: call {i}")
    _assert_bit_exact(ttnn.to_torch(x), x_host, "P: input_tensor after three calls")


# --- Validation: ValueError or RuntimeError, never a support refusal --------------------------
# The message is left free (empty pattern): the spec names the error types, not their text.

_P_IN = PLACEMENTS["P"][:3]
_P_OUT = PLACEMENTS["P"][3:]


def _input_on_host(device):
    return ttnn.from_torch(_random_input(), dtype=DTYPE, layout=LAYOUT)


def _input_interleaved_l1(device):
    return ttnn.from_torch(
        _random_input(), dtype=DTYPE, layout=LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def _input_interleaved_dram(device):
    return ttnn.from_torch(
        _random_input(), dtype=DTYPE, layout=LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _input_rank_2(device):
    return ttnn.from_torch(
        _random_input(shape=SHAPE[-2:]), dtype=DTYPE, layout=LAYOUT, device=device, memory_config=_sharded(*_P_IN)
    )


BAD_INPUTS = {
    "on_host": _input_on_host,
    "interleaved_l1": _input_interleaved_l1,
    "interleaved_dram": _input_interleaved_dram,
    "rank_2": _input_rank_2,
}


@pytest.mark.parametrize("case", list(BAD_INPUTS))
def test_g6_reshard_rejects_input(device, expect_error, case):
    _require_grid(device, _P_IN[0], _P_OUT[0])
    x = BAD_INPUTS[case](device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_reshard(x, _sharded(*_P_OUT))


def _outside_worker_grid(device):
    size = device.compute_with_storage_grid_size()
    return _sharded(((size.x - 4, 0), (size.x + 3, 7)), (256, 128), ROW_MAJOR)


BAD_MEMORY_CONFIGS = {
    "interleaved": lambda device: ttnn.L1_MEMORY_CONFIG,
    "height_sharded": lambda device: _sharded(
        ((0, 0), (7, 7)), (32, 1024), ROW_MAJOR, layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    ),
    "dram": lambda device: _sharded(((0, 0), (7, 7)), (256, 128), ROW_MAJOR, buffer=ttnn.BufferType.DRAM),
    "no_shard_spec": lambda device: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1),
    "shard_not_tile_multiple": lambda device: _sharded(((0, 0), (7, 8)), (250, 128), ROW_MAJOR),
    "too_few_core_rows": lambda device: _sharded(((0, 0), (7, 6)), (256, 128), ROW_MAJOR),
    "too_few_core_columns_col_major": lambda device: _sharded(((0, 0), (6, 7)), (256, 128), COL_MAJOR),
    "not_one_rectangle": lambda device: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core_range_set(((0, 0), (3, 7)), ((5, 0), (8, 7))), [256, 128], ROW_MAJOR),
    ),
    "outside_worker_grid": _outside_worker_grid,
}


@pytest.mark.parametrize("case", list(BAD_MEMORY_CONFIGS))
def test_g6_reshard_rejects_memory_config(device, expect_error, case):
    x, _, _ = _place_input(device, "P")
    with expect_error((ValueError, RuntimeError), ""):
        g6_reshard(x, BAD_MEMORY_CONFIGS[case](device))
