# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-call-site test for both ttnn.reshape calls the tt-forge ResNet-50 graph issues, run on Quasar
via ttnn.experimental.quasar.reshape.

reshape is value-preserving: it changes the logical shape and must round-trip every element, so the
golden is torch's own reshape and the bar is PCC ~1.0.

  image_flatten  [1,224,224,3] -> [1,1,50176,3]   L1 INTERLEAVED TILE -> L1 INTERLEAVED TILE
        collapses the NHWC-permuted input image into the flat [1, 1, N*H*W, C] form the stem conv
        takes. Note both sides are TILE here: Forge tilizes the image BEFORE permuting and
        reshaping it, so this is a tiled reshape whose last dim is 3 -- a 3-element-wide tile row,
        the awkward case for a tile-aware reshape.

  pool_squeeze   [1,1,1,2048]  -> [1,2048]        L1 BLOCK_SHARDED TILE, 16 cores (8x2),
                                                  shard [32, 256], on BOTH sides
        drops the leading unit dims of the global-avg-pool output before the fc matmul, in place,
        with the sharded layout preserved -- a rank change on a block-sharded tensor, which a naive
        reshape implementation gets wrong.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_reshape_forge.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.999  # pure metadata change -- values must round-trip

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
CR16 = (((0, 0), (7, 1)),)  # 8x2 = 16 cores

# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
# fmt: off
L1_IL       = ("INTERLEAVED",   "L1", None, None,      "TILE")
BS16_32x256 = ("BLOCK_SHARDED", "L1", CR16, (32, 256), "TILE")

# (idx, tag,             input_shape,       target_shape,       input_mem,   output_mem)
RESHAPE_CASES = [
    (0, "image_flatten", (1, 224, 224, 3), (1, 1, 50176, 3), L1_IL,       L1_IL),
    (1, "pool_squeeze",  (1, 1, 1, 2048),  (1, 2048),        BS16_32x256, BS16_32x256),
]
# fmt: on


def _mem(spec):
    """Frozen Forge memory-config tuple -> a real ttnn.MemoryConfig."""
    memory_layout, buffer_type, core_ranges, shard_shape, _page_layout = spec
    layout = getattr(ttnn.TensorMemoryLayout, memory_layout)
    buffer = getattr(ttnn.BufferType, buffer_type)
    if core_ranges is None:
        return ttnn.MemoryConfig(layout, buffer, None)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*lo), ttnn.CoreCoord(*hi)) for lo, hi in core_ranges])
    return ttnn.MemoryConfig(layout, buffer, ttnn.ShardSpec(ranges, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR))


def _page(spec):
    return ttnn.TILE_LAYOUT if spec[4] == "TILE" else ttnn.ROW_MAJOR_LAYOUT


def _to_device(x, spec, device):
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=_page(spec))
    try:
        return tt.to(device, _mem(spec))
    except Exception as e:
        raise AssertionError(
            "could not place a %s tensor in the Forge memory config (%s/%s/%s, shard %s over %s): %s"
            % (tuple(x.shape), spec[1], spec[0], spec[4], spec[3], spec[2], e)
        ) from e


def _require_grid(device, *specs):
    """Skip unless the device compute grid can hold every Forge core range in play."""
    grid = device.compute_with_storage_grid_size()
    for spec in specs:
        if spec[2] is None:
            continue
        need_x = max(hi[0] for _lo, hi in spec[2]) + 1
        need_y = max(hi[1] for _lo, hi in spec[2]) + 1
        if need_x > grid.x or need_y > grid.y:
            pytest.skip(
                "Forge pins a %dx%d core grid but this device grid is %dx%d. These configs need a "
                "full Quasar part; ../ops/ covers the same kernels with device-derived sharding."
                % (need_x, need_y, grid.x, grid.y)
            )


def _id(case):
    idx, tag, in_shape, target, _in_mem, _out_mem = case
    return "%d_%s_%s_to_%s" % (
        idx,
        tag,
        "x".join(str(v) for v in in_shape),
        "x".join(str(v) for v in target),
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("case", RESHAPE_CASES, ids=[_id(c) for c in RESHAPE_CASES])
def test_forge_reshape(mesh_device, case):
    device = mesh_device
    torch.manual_seed(0)

    idx, tag, in_shape, target, in_mem, out_mem = case
    _require_grid(device, in_mem, out_mem)

    n_in, n_out = 1, 1
    for d in in_shape:
        n_in *= d
    for d in target:
        n_out *= d
    assert n_in == n_out, "%s changes the element count: %d -> %d" % (tag, n_in, n_out)

    x_torch = torch.rand(in_shape, dtype=torch.bfloat16)
    golden = x_torch.reshape(target)

    x = _to_device(x_torch, in_mem, device)
    out = ttnn.experimental.quasar.reshape(x, list(target), memory_config=_mem(out_mem))
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == target, "%s output %s, Forge IR says %s" % (tag, tuple(out.shape), target)
    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == target, "%s host output %s, expected %s" % (tag, tuple(got.shape), target)
    assert_with_pcc(golden, got, pcc=PCC)
