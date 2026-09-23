# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4: padding (auto / explicit, all fill signs, non-aligned H / W, rank 0 / 1).

Contract (feature_spec.py, op_requirements.md):
    to_torch(out) == x                                   (the logical shape never grows)
    out.cpu().to_torch_with_padded_shape() == F.pad(x, value=fill)
The output is allocated at the padded shape and returned as a zero-copy view at the input's
logical shape; the stick reader walks the padded tile grid and fills everything the input
does not cover (W tail, H tail, whole pad sticks / tile-rows / images).
Every case is bit-exact at bf16. Device is module-scoped by this directory's conftest.py.
"""
import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize

L1 = ttnn.BufferType.L1
ROW = ttnn.ShardOrientation.ROW_MAJOR
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _auto_padded(shape, tile_h):
    shape = [1] * max(0, 2 - len(shape)) + list(shape)
    return shape[:-2] + [-(-shape[-2] // tile_h) * tile_h, -(-shape[-1] // 32) * 32]


def _expected_padded(x, padded_shape, fill):
    x = x.reshape((1,) * (len(padded_shape) - x.dim()) + tuple(x.shape))
    pads = []
    for d, p in reversed(list(zip(x.shape, padded_shape))):
        pads += [0, p - d]
    return torch.nn.functional.pad(x, pads, value=fill)


def _run(device, shape, *, fill, padded_shape=None, in_mc=ttnn.DRAM_MEMORY_CONFIG, out_mc=None, tile_h=32):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    kwargs = {"pad_value": fill}
    if padded_shape is not None:
        kwargs["output_padded_shape"] = padded_shape
    if tile_h != 32:
        kwargs["tile"] = ttnn.Tile([tile_h, 32])
    out = tilize(t, memory_config=out_mc, **kwargs)
    assert out.layout == ttnn.TILE_LAYOUT
    y = ttnn.to_torch(out)
    assert list(y.shape) == list(shape)
    assert torch.equal(y, x), f"logical view differs: max diff {(y.float() - x.float()).abs().max()}"
    target = list(padded_shape) if padded_shape is not None else _auto_padded(shape, tile_h)
    assert list(out.padded_shape) == target
    rb = out.cpu().to_torch_with_padded_shape()
    exp = _expected_padded(x, target, float(fill))
    assert list(rb.shape) == list(exp.shape)
    bad = (rb != exp).sum().item()
    assert bad == 0, f"{bad} of {exp.numel()} padded elements differ"
    return out


@pytest.mark.parametrize(
    "shape, fill",
    [
        pytest.param([1, 1, 32, 50], 0, id="w_tail_zero"),
        pytest.param([1, 1, 50, 64], 3, id="h_tail_positive"),
        pytest.param([1, 1, 50, 50], -7, id="hw_tails_negative"),
        pytest.param([1, 1, 30, 32], 0, id="subtile_both"),
        pytest.param([50, 50], 0, id="rank2"),
        pytest.param([3, 50, 64], 2, id="rank3_fold_across_images"),
        pytest.param([1, 2, 1, 50, 50], -1, id="rank5"),
        pytest.param([64], 0, id="rank1"),
        pytest.param([50], 1.5, id="rank1_w_tail"),
        pytest.param([], 0, id="rank0"),
        pytest.param([], -2, id="rank0_negative"),
        pytest.param([1, 1, 1, 60], 4, id="single_stick"),
        pytest.param([2, 3, 70, 100], -0.5, id="multi_image_multi_tile_rows"),
        pytest.param([4, 1, 1000, 72], 9, id="multicore_tall"),
        pytest.param([1, 1, 64, 64], 5, id="aligned_nothing_to_fill"),
    ],
)
def test_pad_auto(device, shape, fill):
    _run(device, shape, fill=fill)


@pytest.mark.parametrize(
    "shape, padded, fill",
    [
        pytest.param([1, 1, 32, 50], [1, 1, 32, 128], 0, id="beyond_round_w"),
        pytest.param([1, 1, 50, 50], [1, 1, 128, 128], 5, id="beyond_round_hw"),
        pytest.param([1, 1, 32, 64], [1, 1, 64, 128], 0, id="whole_pad_tiles"),
        pytest.param([1, 1, 30, 32], [1, 1, 32, 32], -4, id="exact_round_negative"),
        pytest.param([1, 1, 32, 64], [3, 1, 32, 64], 7, id="whole_pad_images"),
        pytest.param([2, 3, 40, 40], [3, 3, 64, 96], -3, id="outer_lead_dim_and_tails"),
        pytest.param([1, 3, 40, 40], [2, 5, 64, 96], -3, id="unit_outer_lead_dims_and_tails"),
        pytest.param([40, 40], [2, 64, 64], 1, id="rank_grows"),
        pytest.param([1, 1, 32, 32], [1, 1, 32, 32], 3, id="explicit_no_growth"),
    ],
)
def test_pad_explicit(device, shape, padded, fill):
    _run(device, shape, fill=fill, padded_shape=padded)


def test_pad_l1_to_l1(device):
    _run(device, [1, 1, 32, 50], fill=0, in_mc=ttnn.L1_MEMORY_CONFIG, out_mc=ttnn.L1_MEMORY_CONFIG)


def test_pad_to_height_sharded(device):
    out_mc = ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(_crs(0, 0, 1, 0), (32, 64), ROW))
    _run(device, [1, 1, 50, 64], fill=0, out_mc=out_mc)


def test_pad_width_sharded_input(device):
    # A WIDTH-sharded Layout::ROW_MAJOR input: pages are shard-width sticks, so a padded segment's
    # data bytes split at page boundaries; the output streams to DRAM.
    in_mc = ttnn.MemoryConfig(WIDTH, L1, ttnn.ShardSpec(_crs(0, 0, 1, 0), (50, 48), ROW))
    _run(device, [1, 1, 50, 96], fill=-1, in_mc=in_mc, out_mc=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("tile_h", [16, 8, 1])
def test_pad_tiny_tile(device, tile_h):
    # Alignment is measured against the output tile height.
    _run(device, [2, 1, 21, 40], fill=6, tile_h=tile_h)


def test_pad_program_cache_across_fill_values(device):
    """The fill is an RT arg: a second fill value reuses the program and still fills exactly."""
    device.enable_program_cache()
    _run(device, [1, 1, 50, 50], fill=-7)
    n = device.num_program_cache_entries()
    _run(device, [1, 1, 50, 50], fill=11)
    assert device.num_program_cache_entries() == n


def test_pad_refuses_inner_leading_dim_growth(device, expect_error):
    """[2, 3, ...] -> [3, 4, ...]: the padded readback could be F.pad, but TTNN's logical view of
    that buffer is not the input (logical image k reads padded image k), so the op refuses."""
    t = ttnn.from_torch(torch.zeros([2, 3, 40, 40]).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(NotImplementedError, "grows a leading dim"):
        tilize(t, output_padded_shape=[3, 4, 64, 96], pad_value=0)
