# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the Quasar (Metal-2) `ttnn.experimental.quasar.reallocate`.

WHERE IT COMES FROM
-------------------
`reallocate` is a thin wrapper over the Quasar `move` op (reallocate.cpp -> quasar::move). resnet50/quasar
uses it to defragment L1 after a tensor is freed, so the following convs get contiguous free space. The
on-resnet-path call-sites (ttnn_functional_resnet50.py) all pass a SHARDED tensor with no memory_config:

    ds_out = ttnn.experimental.quasar.reallocate(ds_out)   # downsample-out defrag (TILE, HEIGHT_SHARDED)
    x      = ttnn.experimental.quasar.reallocate(x_rm)      # conv2 pre-projection defrag (ROW_MAJOR, HEIGHT_SHARDED)

A sharded input dispatches to `move_sharded` -> MULTI_CORE_SHARDED -> the Quasar Metal-2
`MoveShardedProgramFactory` (the ported path). This test only exercises that sharded path. The interleaved
path (`move_impl` -> legacy `MoveProgramFactory`, ProgramDescriptor-based, NOT Quasar-ported) is never
reached on the resnet path, so the test deliberately stays sharded.

WHAT IT VALIDATES
-----------------
reallocate/move is a pure data-relocation no-op: it deallocates the input and re-creates it (usually at a
new L1 address) with the SAME shape/layout/dtype/memory-config and identical values. So `ttnn.to_torch` of
the output must round-trip the original torch tensor (PCC ~1.0), and shape/layout/memory-config must be
preserved. NOTE: `move` deallocates its input, so the golden torch tensor is captured before the call.

RUN (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_reallocate.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _fit_num_cores(num_tiles, grid):
    """Largest core count <= device grid that evenly divides num_tiles (keeps shards tile-aligned)."""
    cap = grid.x * grid.y
    n = min(cap, num_tiles)
    while n > 1 and num_tiles % n != 0:
        n -= 1
    return n


# (height_tiles, width, layout, id): height-sharded tensors representative of the resnet reallocate sites.
CASES = [
    (32, 256, ttnn.TILE_LAYOUT, "downsample_out_tile_h1024_c256"),  # ds_out defrag (line ~251)
    (32, 256, ttnn.ROW_MAJOR_LAYOUT, "conv2_defrag_rm_h1024_c256"),  # x_rm defrag (line ~335)
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("height_tiles, width, layout, tid", CASES, ids=[c[-1] for c in CASES])
def test_quasar_reallocate_sharded(mesh_device, height_tiles, width, layout, tid):
    torch.manual_seed(0)
    device = mesh_device

    # Build a HEIGHT_SHARDED L1 input that shards evenly across the device grid.
    grid = device.compute_with_storage_grid_size()
    num_cores = _fit_num_cores(height_tiles, grid)
    shard_height = (height_tiles // num_cores) * 32
    input_shape = (1, 1, height_tiles * 32, width)

    x = torch.rand(input_shape, dtype=torch.bfloat16)

    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=sharded_mem_config,
    )

    in_layout = tt_in.layout
    in_mem_config = tt_in.memory_config()
    in_addr = tt_in.buffer().address()

    # On-path call signature: reallocate(tensor) with no memory_config (output keeps the input's config).
    # NOTE: move deallocates tt_in, so tt_in must not be used after this call.
    out = ttnn.experimental.quasar.reallocate(tt_in)

    # Shape / layout / memory-config preserved (pure relocation).
    assert tuple(out.shape) == tuple(input_shape)
    assert out.layout == in_layout
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert out.memory_config() == in_mem_config

    # The relocation typically lands at a new L1 address; only informational (no space -> same address is legal).
    out_addr = out.buffer().address()
    if out_addr == in_addr:
        print(f"[reallocate] output reused input address {in_addr} (no free space to move)")

    # Primary check: values are preserved exactly.
    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    assert_with_pcc(x, got, pcc=0.9999)
