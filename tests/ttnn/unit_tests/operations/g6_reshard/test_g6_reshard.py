# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshard (immutable spec for the implementer).

The production shape is the only shape (BB-22). What varies is the pair of
placements: the production pair and the cases the spec's placement rule
requires (identity, transposed orientation, input grid off the origin,
another output placement with a COL_MAJOR grid off the origin and a cut-short
last block column). Grading is bit equality: the op moves stored tiles and
computes nothing, so helpers.TOLERANCES for this op is ("exact", None).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_reshard import g6_reshard


SHAPE = (1, 1, 2048, 1024)
SEED = 42

_ORIENTATION = {
    "row_major": ttnn.ShardOrientation.ROW_MAJOR,
    "col_major": ttnn.ShardOrientation.COL_MAJOR,
}

# name: (in grid, in shard, in orientation, out grid, out shard, out orientation)
PLACEMENTS = {
    # Production pair: input last block row 32 high, last block column 64 wide.
    "P": (((0, 0), (10, 9)), (224, 96), "row_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    # Same placement on both sides: out must still be a new tensor.
    "I0": (((0, 0), (7, 7)), (256, 128), "row_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    # Differs from I0 only in the input orientation.
    "I3": (((0, 0), (7, 7)), (256, 128), "col_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    # Input grid off core (0, 0): no tile stays on its core.
    "I5": (((2, 1), (9, 8)), (256, 128), "row_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    # Input as P; out COL_MAJOR off the origin, last block column 64 wide.
    "O1": (((0, 0), (10, 9)), (224, 96), "row_major", ((1, 0), (8, 6)), (256, 160), "col_major"),
}


def pytorch_reference(input_tensor):
    """out holds input_tensor's values bit for bit; the placement never enters the values."""
    return input_tensor.clone(memory_format=torch.contiguous_format)


def _core_range_set(grid):
    (x0, y0), (x1, y1) = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _block_sharded(grid, shard, orientation, buffer_type=ttnn.BufferType.L1):
    spec = ttnn.ShardSpec(_core_range_set(grid), [int(shard[0]), int(shard[1])], _ORIENTATION[orientation])
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type, spec)


def _require_grid(device, *grids):
    # Fail rather than skip: the production pair must not pass vacuously on a smaller part.
    size = device.compute_with_storage_grid_size()
    for (_, _), (x1, y1) in grids:
        assert x1 < size.x and y1 < size.y, (
            f"core rectangle ending at ({x1}, {y1}) needs a worker grid of at least "
            f"{x1 + 1}x{y1 + 1}; this device has {size.x}x{size.y}"
        )


def _place_input(device, case, *, shape=SHAPE):
    in_grid, in_shard, in_orientation, out_grid, out_shard, out_orientation = PLACEMENTS[case]
    _require_grid(device, in_grid, out_grid)
    torch.manual_seed(SEED)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_block_sharded(in_grid, in_shard, in_orientation),
    )
    # x read back is the reference input: bfloat8_b quantization happened on the way in.
    return x, ttnn.to_torch(x), _block_sharded(out_grid, out_shard, out_orientation)


# Integer views compare stored bits, so -0.0 and +0.0 (or two NaN payloads) never pass for each other.
_BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def _assert_bits_equal(actual, expected, what):
    assert actual.dtype == expected.dtype, f"{what}: readback dtype {actual.dtype}, expected {expected.dtype}"
    assert list(actual.shape) == list(
        expected.shape
    ), f"{what}: shape {list(actual.shape)}, expected {list(expected.shape)}"
    bits = _BITS[expected.dtype]
    differ = actual.contiguous().view(bits) != expected.contiguous().view(bits)
    if bool(differ.any()):
        width = expected.shape[-1]
        flat = differ.reshape(-1, width)
        tiles = flat.reshape(flat.shape[0] // 32, 32, width // 32, 32).any(dim=3).any(dim=1)
        bad = [tuple(t) for t in tiles.nonzero().tolist()[:8]]
        pytest.fail(
            f"{what}: {int(differ.sum())} of {differ.numel()} elements differ; "
            f"{int(tiles.sum())} tiles, first (tile row, tile col): {bad}"
        )


def _rectangles(core_range_set):
    return sorted(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in core_range_set.ranges())


def _cores(core_range_set):
    return {
        (x, y)
        for (x0, y0), (x1, y1) in _rectangles(core_range_set)
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
    }


def _assert_placement(out, out_cfg, x):
    assert list(out.shape) == list(x.shape)
    assert out.dtype == x.dtype
    assert out.layout == x.layout
    got = out.memory_config()
    assert got.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert got.buffer_type == ttnn.BufferType.L1
    assert got.shard_spec is not None
    want = out_cfg.shard_spec
    assert _rectangles(got.shard_spec.grid) == _rectangles(want.grid)
    assert list(got.shard_spec.shape) == list(want.shape)
    assert got.shard_spec.orientation == want.orientation


def _assert_not_aliased(out, x):
    assert out is not x, "out is input_tensor itself"
    try:
        out_address, x_address = out.buffer_address(), x.buffer_address()
    except RuntimeError:
        return
    shared = _cores(out.memory_config().shard_spec.grid) & _cores(x.memory_config().shard_spec.grid)
    assert not (shared and out_address == x_address), "out shares input_tensor's L1 buffer"


def _call_and_check(x, x_host, out_cfg):
    expected = pytorch_reference(x_host)
    out = g6_reshard(x, out_cfg)
    _assert_placement(out, out_cfg, x)
    _assert_not_aliased(out, x)
    _assert_bits_equal(ttnn.to_torch(out), expected, "out against the reference")
    _assert_bits_equal(ttnn.to_torch(x), x_host, "input_tensor after the call")
    return out


@pytest.mark.parametrize("case", list(PLACEMENTS))
def test_g6_reshard(device, case):
    x, x_host, out_cfg = _place_input(device, case)
    _call_and_check(x, x_host, out_cfg)


def test_g6_reshard_placements_one_process(device):
    """Placements that differ only in input orientation (I0, I3) or only in out's
    placement (P, O1) run back to back with the program cache on: a cache key
    missing any placement field replays the wrong program."""
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    for case in ("I0", "I3", "P", "O1", "P", "I5"):
        x, x_host, out_cfg = _place_input(device, case)
        _call_and_check(x, x_host, out_cfg)


def test_g6_reshard_repeat_determinism(device):
    """Three calls on one input, every output kept alive and bit-identical."""
    x, x_host, out_cfg = _place_input(device, "P")
    outs = [_call_and_check(x, x_host, out_cfg) for _ in range(3)]
    first = ttnn.to_torch(outs[0])
    for out in outs[1:]:
        _assert_bits_equal(ttnn.to_torch(out), first, "repeat call against the first call")


# --- Validation: ValueError or RuntimeError before any dispatch -------------------------


def _config_or_skip(build):
    # A config ttnn itself refuses to construct never reaches the op, so there is nothing to test.
    try:
        return build()
    except (RuntimeError, ValueError, TypeError) as exc:
        pytest.skip(f"ttnn refuses to construct this memory_config: {exc}")


def _expect_refusal(expect_error, x, memory_config):
    # Every validation error is raised by the op's host code, named after the op.
    with expect_error((ValueError, RuntimeError), "g6_reshard"):
        g6_reshard(x, memory_config)


def test_g6_reshard_rejects_interleaved_input(device, expect_error):
    torch.manual_seed(SEED)
    x = ttnn.from_torch(
        torch.randn(SHAPE, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    _, _, _, out_grid, out_shard, out_orientation = PLACEMENTS["I0"]
    _expect_refusal(expect_error, x, _block_sharded(out_grid, out_shard, out_orientation))


def test_g6_reshard_rejects_rank_3(device, expect_error):
    x, _, out_cfg = _place_input(device, "I0", shape=SHAPE[1:])
    _expect_refusal(expect_error, x, out_cfg)


def test_g6_reshard_rejects_interleaved_memory_config(device, expect_error):
    x, _, _ = _place_input(device, "I0")
    _expect_refusal(expect_error, x, ttnn.L1_MEMORY_CONFIG)


def test_g6_reshard_rejects_dram_memory_config(device, expect_error):
    x, _, _ = _place_input(device, "I0")
    cfg = _config_or_skip(lambda: _block_sharded(((0, 0), (7, 7)), (256, 128), "row_major", ttnn.BufferType.DRAM))
    _expect_refusal(expect_error, x, cfg)


def test_g6_reshard_rejects_missing_shard_spec(device, expect_error):
    x, _, _ = _place_input(device, "I0")
    cfg = _config_or_skip(lambda: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1))
    _expect_refusal(expect_error, x, cfg)


def test_g6_reshard_rejects_shard_not_multiple_of_32(device, expect_error):
    x, _, _ = _place_input(device, "I0")
    cfg = _config_or_skip(lambda: _block_sharded(((0, 0), (7, 9)), (240, 128), "row_major"))
    _expect_refusal(expect_error, x, cfg)


def test_g6_reshard_rejects_grid_too_small(device, expect_error):
    # (256, 128) shards make 8 block columns; the grid has 7 core columns.
    x, _, _ = _place_input(device, "I0")
    cfg = _config_or_skip(lambda: _block_sharded(((0, 0), (6, 7)), (256, 128), "row_major"))
    _expect_refusal(expect_error, x, cfg)


def test_g6_reshard_rejects_grid_not_one_rectangle(device, expect_error):
    x, _, _ = _place_input(device, "I0")

    def build():
        grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(8, 7)),
            }
        )
        spec = ttnn.ShardSpec(grid, [256, 128], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

    _expect_refusal(expect_error, x, _config_or_skip(build))


def test_g6_reshard_rejects_grid_outside_worker_grid(device, expect_error):
    x, _, _ = _place_input(device, "I0")
    size = device.compute_with_storage_grid_size()
    cfg = _config_or_skip(lambda: _block_sharded(((0, 0), (size.x, 7)), (256, 128), "row_major"))
    _expect_refusal(expect_error, x, cfg)
