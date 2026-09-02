# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-call-site test for every ttnn.to_memory_config the tt-forge ResNet-50 graph issues, run on
Quasar via ttnn.experimental.quasar.to_memory_config.

4 cases. to_memory_config is value-preserving -- it re-lays the same elements across a different
memory layout without touching shape or values -- so the golden is the input and the bar is
PCC ~1.0. Each case pins BOTH the source and destination configs Forge chose:

  layer2->layer3 reshard A   [1,1,784,256]  L1 HEIGHT_SHARDED 25 cores, shard [ 32, 256]
                          -> L1 BLOCK_SHARDED 56 cores (8x7),          shard [128,  32]
        re-lays layer3.0.conv1's output into the block-sharded layer3 layout, feeding
        layer3.0.conv2.

  layer2->layer3 reshard B   [1,1,784,512]  L1 HEIGHT_SHARDED 25 cores, shard [ 32, 512]
                          -> L1 BLOCK_SHARDED 56 cores (8x7),          shard [128,  64]
        the same reshard for the skip branch, feeding layer3.0.downsample.

  pre-fc gather              [1,2048]       L1 BLOCK_SHARDED 16 cores (8x2), shard [32, 256]
                          -> L1 INTERLEAVED
        gathers the pooled feature vector before the fc matmul -- Forge feeds its 1D-mcast linear
        an INTERLEAVED activation, not a width-sharded one.

  final gather               [1,1000]       L1 WIDTH_SHARDED 32 cores (8x4), shard [32, 32]
                          -> DRAM INTERLEAVED
        the fc output back to DRAM; the last op in the graph.

The two sharded->sharded cases are the interesting ones: a full reshard across a DIFFERENT core
count AND a different shard geometry (25 -> 56 cores, height -> block), which is the data-movement
pattern most likely to be missing or wrong on Quasar. They are also what makes layer3.0.conv2 and
layer3.0.downsample block-sharded while layer3.0.conv1 is still height-sharded.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_to_memory_config_forge.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.999  # pure data movement -- values must round-trip

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
CR56 = (((0, 0), (7, 6)),)  # 8x7     = 56 cores
CR32 = (((0, 0), (7, 3)),)  # 8x4     = 32 cores
CR25 = (((0, 0), (7, 2)), ((0, 3), (0, 3)))  # 8x3 + 1 = 25 cores
CR16 = (((0, 0), (7, 1)),)  # 8x2     = 16 cores

# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
# fmt: off
L1_IL       = ("INTERLEAVED",    "L1",   None, None,       "TILE")
DRAM_IL     = ("INTERLEAVED",    "DRAM", None, None,       "TILE")
HS25_32x256 = ("HEIGHT_SHARDED", "L1",   CR25, ( 32, 256), "TILE")
HS25_32x512 = ("HEIGHT_SHARDED", "L1",   CR25, ( 32, 512), "TILE")
BS56_128x32 = ("BLOCK_SHARDED",  "L1",   CR56, (128,  32), "TILE")
BS56_128x64 = ("BLOCK_SHARDED",  "L1",   CR56, (128,  64), "TILE")
BS16_32x256 = ("BLOCK_SHARDED",  "L1",   CR16, ( 32, 256), "TILE")
WS32_32x32  = ("WIDTH_SHARDED",  "L1",   CR32, ( 32,  32), "TILE")

# (idx, tag,                shape,              src_mem,      dst_mem)
TO_MEMORY_CONFIG_CASES = [
    (0, "L2toL3_reshard_a", (1, 1, 784, 256), HS25_32x256, BS56_128x32),
    (1, "L2toL3_reshard_b", (1, 1, 784, 512), HS25_32x512, BS56_128x64),
    (2, "pre_fc_gather",    (1, 2048),        BS16_32x256, L1_IL),
    (3, "final_gather",     (1, 1000),        WS32_32x32,  DRAM_IL),
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


def _mem_tag(spec):
    kind = {
        "INTERLEAVED": "il",
        "HEIGHT_SHARDED": "hs",
        "WIDTH_SHARDED": "ws",
        "BLOCK_SHARDED": "bs",
    }[spec[0]]
    if spec[2] is None:
        return "%s%s" % (spec[1], kind)
    cores = sum((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) for lo, hi in spec[2])
    return "%s%s%d" % (spec[1], kind, cores)


def _id(case):
    idx, tag, shape, src, dst = case
    return "%d_%s_%s_to_%s" % (idx, tag, _mem_tag(src), _mem_tag(dst))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("case", TO_MEMORY_CONFIG_CASES, ids=[_id(c) for c in TO_MEMORY_CONFIG_CASES])
def test_forge_to_memory_config(mesh_device, case):
    device = mesh_device
    torch.manual_seed(0)

    idx, tag, shape, src_spec, dst_spec = case
    _require_grid(device, src_spec, dst_spec)

    x_torch = torch.rand(shape, dtype=torch.bfloat16)
    src = _to_device(x_torch, src_spec, device)

    dst_cfg = _mem(dst_spec)
    out = ttnn.experimental.quasar.to_memory_config(src, dst_cfg)
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == shape, "to_memory_config %s changed shape: %s -> %s" % (tag, shape, tuple(out.shape))
    got_mem = out.memory_config()
    assert got_mem.memory_layout == dst_cfg.memory_layout, "%s landed in %s but Forge asked for %s" % (
        tag,
        got_mem.memory_layout,
        dst_spec[0],
    )
    if dst_spec[3] is not None:
        # MemoryConfig.shard_spec is optional -- a missing one means the reshard landed unsharded,
        # which is a real mismatch, not an attribute error.
        assert got_mem.shard_spec is not None, "%s output has no shard spec, but Forge asked for shard %s over %s" % (
            tag,
            list(dst_spec[3]),
            dst_spec[2],
        )
        assert list(got_mem.shard_spec.shape) == list(dst_spec[3]), "%s shard shape %s but Forge asked for %s" % (
            tag,
            list(got_mem.shard_spec.shape),
            list(dst_spec[3]),
        )

    assert_with_pcc(x_torch, ttnn.to_torch(out).to(torch.bfloat16), pcc=PCC)
