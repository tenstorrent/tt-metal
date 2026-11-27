# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshard: bit-exact block-sharded L1 -> block-sharded L1 move.

Only the production shape (1, 1, 2048, 1024) is used (BB-22). The cases differ
by placement: P is the production pair, and I0, I3, I5, O1 are the placements
rule 2 requires (identity, column-major input, input grid off the origin,
another output placement).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_reshard import g6_reshard

SHAPE = (1, 1, 2048, 1024)
DTYPE = ttnn.bfloat8_b
LAYOUT = ttnn.TILE_LAYOUT

_ORIENTATION = {
    "row_major": ttnn.ShardOrientation.ROW_MAJOR,
    "col_major": ttnn.ShardOrientation.COL_MAJOR,
}

# name -> (in_grid, in_shard, in_orientation, out_grid, out_shard, out_orientation)
SCENARIOS = {
    "P": (((0, 0), (10, 9)), (224, 96), "row_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    "I0": (((0, 0), (7, 7)), (256, 128), "row_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    "I3": (((0, 0), (7, 7)), (256, 128), "col_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    "I5": (((2, 1), (9, 8)), (256, 128), "row_major", ((0, 0), (7, 7)), (256, 128), "row_major"),
    "O1": (((0, 0), (10, 9)), (224, 96), "row_major", ((1, 0), (8, 6)), (256, 160), "col_major"),
}

# Bits, not values: -0.0 must not pass for +0.0.
_BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def pytorch_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """The op moves values and computes none: a copy of the stored values."""
    return input_tensor.clone()


def _block_sharded_l1(device, grid, shard, orientation):
    (x0, y0), (x1, y1) = grid
    size = device.compute_with_storage_grid_size()
    # Fail rather than skip: the production case must not pass vacuously on a smaller grid.
    assert (
        x1 < size.x and y1 < size.y
    ), f"core rectangle {grid} needs a {x1 + 1}x{y1 + 1} worker grid, got {size.x}x{size.y}"
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
    spec = ttnn.ShardSpec(cores, [shard[0], shard[1]], _ORIENTATION[orientation])
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def _configs(device, name):
    in_grid, in_shard, in_orient, out_grid, out_shard, out_orient = SCENARIOS[name]
    return (
        _block_sharded_l1(device, in_grid, in_shard, in_orient),
        _block_sharded_l1(device, out_grid, out_shard, out_orient),
    )


def _make_input(device, in_cfg):
    torch_input = torch.randn(SHAPE, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(torch_input, dtype=DTYPE, layout=LAYOUT, device=device, memory_config=in_cfg)
    # bfloat8_b quantizes on upload: the reference is what the device holds.
    return x, ttnn.to_torch(x)


def _assert_bits_equal(actual, expected, what):
    assert actual.dtype == expected.dtype, f"{what}: readback dtype {actual.dtype}, expected {expected.dtype}"
    assert list(actual.shape) == list(
        expected.shape
    ), f"{what}: shape {list(actual.shape)}, expected {list(expected.shape)}"
    bits = _BITS[expected.dtype]
    differ = actual.contiguous().view(bits) != expected.contiguous().view(bits)
    if bool(differ.any()):
        h, w = expected.shape[-2] * expected.shape[0] * expected.shape[1], expected.shape[-1]
        tiles = differ.reshape(h // 32, 32, w // 32, 32).any(dim=3).any(dim=1)
        bad = [tuple(t) for t in tiles.nonzero().tolist()[:8]]
        pytest.fail(
            f"{what}: {int(differ.sum())} elements differ in {int(tiles.sum())} tiles, first (tile row, tile col): {bad}"
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
    assert got.shard_spec is not None
    assert _rectangles(got.shard_spec.grid) == _rectangles(out_cfg.shard_spec.grid)
    assert list(got.shard_spec.shape) == list(out_cfg.shard_spec.shape)
    assert got.shard_spec.orientation == out_cfg.shard_spec.orientation


def _cores(core_range_set):
    return {
        (x, y)
        for (x0, y0), (x1, y1) in _rectangles(core_range_set)
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
    }


def _assert_new_tensor(out, x):
    assert out is not x, "out is input_tensor itself"
    shared = _cores(out.memory_config().shard_spec.grid) & _cores(x.memory_config().shard_spec.grid)
    if shared:
        assert out.buffer_address() != x.buffer_address(), "out shares input_tensor's L1 buffer"


def _check_call(out, x_host, out_cfg):
    _assert_placement(out, out_cfg)
    _assert_bits_equal(ttnn.to_torch(out), pytorch_reference(x_host), "out against the reference")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_g6_reshard(device, name):
    torch.manual_seed(42)
    in_cfg, out_cfg = _configs(device, name)
    x, x_host = _make_input(device, in_cfg)

    out = g6_reshard(x, out_cfg)

    _check_call(out, x_host, out_cfg)
    _assert_new_tensor(out, x)
    _assert_bits_equal(ttnn.to_torch(x), x_host, "input_tensor after the call")


def test_g6_reshard_placements_one_process(device):
    """I0/I3 differ only in orientation and P/O1 only in out's placement: a
    cache key that misses either replays a wrong program (BB-07)."""
    torch.manual_seed(42)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    # Every tensor stays alive so a later call cannot land on a buffer that already holds the right tiles.
    alive = []
    for name in ["I0", "I3", "P", "O1", "P", "I5"]:
        in_cfg, out_cfg = _configs(device, name)
        x, x_host = _make_input(device, in_cfg)
        out = g6_reshard(x, out_cfg)
        alive.append((x, out))
        _check_call(out, x_host, out_cfg)


def test_g6_reshard_repeat(device):
    """P three times on one input, every output kept alive: each call must
    write its own fresh buffer completely and identically (BB-06, BB-03)."""
    torch.manual_seed(42)
    in_cfg, out_cfg = _configs(device, "P")
    x, x_host = _make_input(device, in_cfg)
    outs = []
    for _ in range(3):
        out = g6_reshard(x, out_cfg)
        outs.append(out)
        _check_call(out, x_host, out_cfg)
    _assert_bits_equal(ttnn.to_torch(x), x_host, "input_tensor after the calls")
