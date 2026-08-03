# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op repro for the Quasar (Metal-2) sharded_to_interleaved HANG.

WHERE IT COMES FROM
-------------------
The Quasar conv2d DRAM-slicing path (used by the resnet stem, M=112*112=12544 overflows the 1 MB DFB
ring) first converts the ROW_MAJOR HEIGHT_SHARDED L1 activation into interleaved DRAM so `padded_slice`
can read spatial slices. That conversion is `ttnn::prim::qsr::sharded_to_interleaved`, reached from
`ttnn.experimental.quasar.to_memory_config(sharded_rm_tensor, interleaved_dram_config)`. It uses:
    reader_unary_sharded.cpp                                    (producer: marks the resident shard ready)
    writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp   (consumer: async_write each stick
                                                                         to interleaved DRAM, then barrier)

THE BUG (see project_quasar_matmul_async_full_barrier_47797 / #47797)
--------------------------------------------------------------------
The writer hangs in `noc.async_write` (the per-stick loop, BEFORE its `async_write_barrier`/`pop_front`).
The Quasar NIU dispatches ~1 nonposted write (`NIU_MST_NONPOSTED_WR_REQ_SENT` freezes at 1) and then
stalls — the write-ack never returns — so command buffer 0 never frees and the next `async_write` spins
forever at `noc_command_ready()` (noc.h:545). This is the SAME NIU nonposted-write stall that wedges the
split-conv matmul's in1 sender in `async_full_barrier`. It is NOT tile-counter reuse (a fully serialized
dispatch still hangs) and it is NOT a DFB `finish()` issue.

This test exercises that conversion IN ISOLATION so the runtime/NOC team has a minimal repro without the
whole split-conv pipeline. On a healthy NIU it round-trips (PCC ~1.0); on the current emulator it HANGS in
the writer. It runs directly (no skip) so it IS the repro — the 600 s timeout bounds the wedge so it can't
hang a suite forever.

RUN (emulator, slow dispatch + forced JIT, all-core dprint to watch the writer stall in async_write):
  TT_METAL_DPRINT_CORES=all TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_sharded_to_interleaved.py

  # smallest/fastest single case:
  TT_METAL_DPRINT_CORES=all TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_sharded_to_interleaved.py -k small
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _fit_num_cores(num_sticks, grid):
    """Largest core count <= device grid that evenly divides num_sticks (exact, unpadded height shards)."""
    cap = grid.x * grid.y
    n = min(cap, num_sticks)
    while n > 1 and num_sticks % n != 0:
        n -= 1
    return n


# (nhw, c, id): ROW_MAJOR height-sharded [1,1,nhw,c] L1 -> interleaved DRAM.
#   stem_c32 mirrors the resnet stem conv2d activation scale (~6624 sticks/core on a 2-core split);
#   small_c32 is a tiny control that still issues enough nonposted writes to hit the NIU stall.
CASES = [
    (12544, 32, "stem_scale_nhw12544_c32"),  # M = 112*112, the stem's DRAM-slicing activation width
    (256, 32, "small_nhw256_c32"),  # minimal repro (NIU stalls after ~1 write regardless of size)
]


@pytest.mark.timeout(600)  # bounds the NIU-stall wedge (#47797) so this repro can't hang forever
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("nhw, c, tid", CASES, ids=[cse[-1] for cse in CASES])
def test_quasar_sharded_to_interleaved_rm(mesh_device, nhw, c, tid):
    """ROW_MAJOR HEIGHT_SHARDED L1 -> interleaved DRAM (the conv DRAM-slicing s2i). Round-trip must PCC ~1.0."""
    torch.manual_seed(0)
    device = mesh_device

    grid = device.compute_with_storage_grid_size()
    num_cores = _fit_num_cores(nhw, grid)
    shard_h = nhw // num_cores

    x = torch.rand((1, 1, nhw, c), dtype=torch.bfloat16)

    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, c),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # ROW_MAJOR + HEIGHT_SHARDED L1 source -> routes to writer_unary_stick_layout on the s2i.
    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    # sharded -> interleaved DRAM: dispatches to ttnn::prim::qsr::sharded_to_interleaved. HANGS today.
    out = ttnn.experimental.quasar.to_memory_config(tt_in, ttnn.DRAM_MEMORY_CONFIG)

    assert_with_pcc(x, ttnn.to_torch(out), 0.9999)
