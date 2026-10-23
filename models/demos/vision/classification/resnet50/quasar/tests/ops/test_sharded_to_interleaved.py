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

THE UNDERLYING INFRA BUG (root-caused on-device via DPRINT) -- 8-BIT DFB CAPACITY TRUNCATION
-------------------------------------------------------------------------------------------
Before the workaround below, `push_back(shard_h)` tripped `ASSERT(overlay::llk_intf_get_capacity(...) >=
num_entries)` at tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:190 (watcher reports "line 190" w/ a
stale kernel name; with watcher OFF the trapped reader RISC shows up as a "Not done phys cores" hang). The
reader DPRINT printed the device credit capacity vs what it pushed:
    shard_h=1568 -> `[S2IR] pre-push push=1568 cap=32`     (1568 & 0xFF == 32)
    shard_h=6272 -> `[S2IR] pre-push push=6272 cap=128`    (6272 & 0xFF == 128)
i.e. `cap == num_entries & 0xFF` -- the tile-counter credit capacity is TRUNCATED TO 8 BITS (mod 256):
  - dataflow_buffer_config.h:197   `uint8_t capacity;`  in dfb_hart_init_entry_t (per-hart init entry)
  - dataflow_buffer_init.h:104/133 `h.capacity = static_cast<uint8_t>(w0 >> 24);`  (device unpack, 8 bits)
  - dataflow_buffer_init.h:933      `buf_capacity = eh.capacity;`                    (programs the TC)
`num_entries` is uint16_t (config.h:214) and the overlay register is 16-bit, but the init-entry `capacity`
is serialized through an 8-bit field, so any DFB with capacity > 255 is silently truncated. The proper fix
is to widen that field + its w0[31:24] bit-packing to >= 16 bits (DFB/LLK-team infra change, tracked
separately). A "credit" here is one STICK (one C-channel row), not a tile.

THE FIX (infra, abhullar/max-pool-cap commit f5b5ce6a7d0, cherry-picked onto this branch)
-----------------------------------------------------------------------------------------
"[Bug fix]: capacity should be 16bits not 8 for DFB config" widens the DFB init-entry capacity field from
8-bit to 16-bit (matching HW BUFFER_CAPACITY): it moves `capacity` out of w0[31:24] into the former _pad2 at
bytes 26-27 (dfb_hart_init_entry_t::capacity -> uint16_t; device unpack reads w6>>16), and adds a
dfb_narrow_field<> guard that TT_FATALs on truncation instead of silently wrapping. Ceiling is now 65535
entries, so the per-stick s2i credits (1568 / 6272) fit with no op-side change. This is the authoritative fix;
the earlier op-level credit-coarsening workaround was reverted in favour of it.

So with the fix in place ALL cases below must round-trip (PCC ~1.0); this test guards that. Run:

  TT_METAL_DPRINT_CORES=all TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_sharded_to_interleaved.py

The reader DPRINT should now show cap == push (the FULL shard_h, no longer truncated mod 256). sh256 is the
first case that previously asserted; sh1568 / sh6272 are the real resnet layer2-conv2 / stem-scale activations.

NOTE: once push_back succeeds, the writer may separately hit the NIU nonposted-write stall tracked as #47797
-- a DISTINCT downstream issue.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc

# (shard_h, num_cores, id).  num_units_per_shard for a ROW_MAJOR height shard == shard_h (independent of C).
# With the 16-bit-capacity infra fix, ALL of these must round-trip; the shard_h > 255 cases are the ones that
# previously tripped the 8-bit capacity assert / hang (now they get their full shard_h credits):
#   sh255 : <= 255, always worked (control)
#   sh256 : first case that previously asserted (256 & 0xFF = 0)
#   sh1568: real resnet layer2 conv2 conv2d_DRAM activation (previously cap=32)
#   sh6272: real resnet stem-scale activation (previously cap=128)
CASES = [
    (255, 1, "sh255_control"),
    (256, 1, "sh256_was_over_8bit"),
    (1568, 2, "sh1568_resnet_layer2_conv2"),
    (6272, 2, "sh6272_stem_scale"),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("shard_h, num_cores, tid", CASES, ids=[cse[-1] for cse in CASES])
def test_quasar_s2i_credit_capacity(mesh_device, shard_h, num_cores, tid):
    """ROW_MAJOR HEIGHT_SHARDED L1 [1,1,shard_h*num_cores,C] -> interleaved DRAM (the conv2d_DRAM s2i).

    The reader fake-pushes shard_h per-stick credits into a borrowed DFB. With the 16-bit-capacity infra fix
    the credit capacity is no longer truncated mod 256, so every case must round-trip (PCC ~1.0)."""
    # The stem-scale shard (6272 sticks) needs ~1.6 MB/bank; WH's L1 bank is ~1.37 MB -> OOM (bank_manager).
    # It fits the 2-core Quasar emulator's larger banks. Smaller-shard cases still run on WH. Run on Quasar.
    if is_wormhole_b0() and tid == "sh6272_stem_scale":
        pytest.skip(
            "stem-scale shard (6272 sticks) OOMs WH's ~1.37 MB L1 bank; fits the 2-core Quasar emulator. Run on Quasar."
        )
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
