# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated per-op repro for the Quasar (Metal-2) sharded_to_interleaved DFB CREDIT-CAPACITY assert.

WHERE IT COMES FROM
-------------------
The Quasar conv2d DRAM-slicing path (conv2d_DRAM, used for large layer convs on the 2-core emulator)
first converts the ROW_MAJOR HEIGHT_SHARDED L1 activation to interleaved DRAM so `padded_slice` can read
spatial slices. That conversion is `ttnn::prim::qsr::sharded_to_interleaved`, reached from
`ttnn.experimental.quasar.to_memory_config(sharded_rm_tensor, interleaved_dram_config)`. Its reader
(reader_unary_sharded.cpp) is a fake-push producer over a DFB borrowed onto the resident shard: for a
ROW_MAJOR height shard it uses ONE per-stick credit per output stick, so it pushes `shard_h` credits/core
in a single `cb_in0.push_back(shard_h)`.

THE BUG (root-caused on-device via DPRINT, 2026-08-05) -- 8-BIT DFB CAPACITY TRUNCATION
--------------------------------------------------------------------------------------
`push_back(shard_h)` trips `ASSERT(overlay::llk_intf_get_capacity(...) >= num_entries)` at
tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:190 (the watcher reports "line 190" but a stale kernel
name; with watcher OFF the trapped reader RISC shows up as a "Not done phys cores" hang instead). The
reader DPRINT prints the device credit capacity vs what it pushes:
    shard_h=1568 -> `[S2IR] pre-push push=1568 cap=32`     (1568 & 0xFF == 32)
    shard_h=6272 -> `[S2IR] pre-push push=6272 cap=128`    (6272 & 0xFF == 128)
So `cap == num_entries & 0xFF` -- the tile-counter credit capacity is TRUNCATED TO 8 BITS (mod 256).

ROOT CAUSE (shared DFB infra, NOT a quasar-op bug, NOT a HW limit):
  - dataflow_buffer_config.h:197  `uint8_t capacity;`  in dfb_hart_init_entry_t (the per-hart init entry)
  - dataflow_buffer_init.h:104/133 `h.capacity = static_cast<uint8_t>(w0 >> 24);`  (device unpack, 8 bits)
  - dataflow_buffer_init.h:933      `buf_capacity = eh.capacity;`                    (programs the TC)
`num_entries` is uint16_t (config.h:214) and the overlay register is 16-bit, but the init-entry `capacity`
is serialized through an 8-bit field, so any DFB with capacity > 255 is silently truncated. FIX = widen the
init-entry capacity field and its w0[31:24] bit-packing to >= 16 bits (DFB/LLK team).

BOUNDARY: shard_h <= 255 round-trips (cap == shard_h); shard_h >= 256 asserts (cap = shard_h & 0xFF < shard_h;
e.g. 256 -> cap=0). A ROW_MAJOR height shard uses one per-stick credit per stick, so shard_h IS the credit
count -- the resnet conv2d_DRAM activations (1568, 6272 sticks/core) are far over 255.

HOW TO RUN
----------
`shard_h >= 256` trips a HW assert (watcher on) / hang (watcher off); run the over-limit cases ONE AT A TIME
with `-k`. `sh255` is the max-capacity control that should round-trip (PCC ~1.0). Reader DPRINT shows push=/cap=:

  TT_METAL_DPRINT_CORES=all TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_sharded_to_interleaved.py -k sh255

  # the real resnet layer2 conv2 conv2d_DRAM input (shard_h=1568 -> cap=32) / stem-scale (6272 -> cap=128):
  ... -k sh1568   /   ... -k sh6272

NOTE: once push_back succeeds (shard_h <= 255), the writer may separately hit the NIU nonposted-write stall
tracked as #47797 -- a DISTINCT downstream issue reachable only after this capacity truncation is fixed.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (shard_h, num_cores, id).  num_units_per_shard for a ROW_MAJOR height shard == shard_h (independent of C),
# which is exactly the count the reader push_back()es. The init-entry capacity field is 8-bit, so
# cap == shard_h & 0xFF. Sweep across the 255/256 boundary:
#   sh255 : push=255  -- max-capacity control, round-trips (cap == 255)
#   sh256 : push=256  -- first over the 8-bit field (cap = 256 & 0xFF = 0) -> asserts
#   sh1568: push=1568 -- real resnet layer2 conv2 conv2d_DRAM activation (cap = 1568 & 0xFF = 32) -> asserts
#   sh6272: push=6272 -- real resnet stem-scale activation (cap = 6272 & 0xFF = 128) -> asserts
CASES = [
    (255, 1, "sh255_control"),
    (256, 1, "sh256_over_8bit"),
    (1568, 2, "sh1568_resnet_layer2_conv2"),
    (6272, 2, "sh6272_stem_scale"),
]


@pytest.mark.timeout(300)  # bounds any wedge so this repro can't hang a suite forever
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("shard_h, num_cores, tid", CASES, ids=[cse[-1] for cse in CASES])
def test_quasar_s2i_credit_capacity(mesh_device, shard_h, num_cores, tid):
    """ROW_MAJOR HEIGHT_SHARDED L1 [1,1,shard_h*num_cores,C] -> interleaved DRAM (the conv2d_DRAM s2i).

    The reader fake-pushes shard_h per-stick credits into a borrowed DFB; the device credit capacity is 32,
    so shard_h over the limit asserts at push_back (dataflow_buffer.inl:190). shard_h within the limit must
    round-trip (PCC ~1.0)."""
    torch.manual_seed(0)
    device = mesh_device
    c = 128  # matches the resnet conv2 activation channel count; num_units_per_shard is C-independent here

    grid = device.compute_with_storage_grid_size()
    if num_cores > grid.x * grid.y:
        pytest.skip(f"case needs {num_cores} cores; device grid has {grid.x * grid.y}")

    nhw = shard_h * num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, c),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x = torch.rand((1, 1, nhw, c), dtype=torch.bfloat16)
    # ROW_MAJOR + HEIGHT_SHARDED L1 source -> the s2i per-stick fake-push path (shard_h credits/core).
    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    # sharded L1 -> interleaved DRAM: dispatches to ttnn::prim::qsr::sharded_to_interleaved.
    # Reader DPRINT prints "[S2IR] pre-push push={shard_h} cap={device_credit_capacity}" before the assert.
    out = ttnn.experimental.quasar.to_memory_config(tt_in, ttnn.DRAM_MEMORY_CONFIG)

    assert_with_pcc(x, ttnn.to_torch(out), 0.9999)
