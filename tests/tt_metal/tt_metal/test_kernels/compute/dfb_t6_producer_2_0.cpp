// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side producer for the single-DFB matrix
// sweep (TRISC → DFB → DM case).
//
// This kernel adds Tensix-as-producer coverage. Because TRISC compute kernels
// can't NoC-read from DRAM in this test setup, the host pre-fills the DFB's L1
// ring directly via WriteToDeviceL1 before the program launches. The kernel
// itself only does reserve_back / push_back — it posts the credits that say
// "one tile is available", which the downstream DM consumer waits on.
//
// Flow per test invocation:
//   1. Host pre-fills the DFB L1 ring with the input data.
//   2. This kernel calls reserve_back + push_back num_entries_per_producer
//      times, then dfb.finish().
//   3. The DM consumer drains those credits, reads from L1, and NoC-writes
//      out to DRAM.
//   4. Host reads DRAM to verify.
//
// Bindings (set by host KernelSpec):
//   dfb::out — PRODUCER (host pre-binds the same DFB the DM consumer reads).

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"          // for dummy_pack (TEN-4746 no-write pack ordering)
#include "api/compute/tile_move_copy.h"  // for copy_init (see the init note in kernel_main)
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);

    DataflowBuffer dfb(dfb::out);

    // Both inits are needed before dummy_pack, even though this kernel only packs. llk_pack_dummy
    // aims its PACR_STRIDE at bd_table index 0, which its own comment notes "falls in the unpack
    // partition [0,16), not pack's" -- it is only meant to be safe because PACK_STRIDE_NO_WRITE
    // suppresses the store so the descriptor is never read. Empirically it is read anyway: without
    // copy_init programming that entry the PACR faults with PACKER_0 ILLEGAL_TILE_SIZE.
    compute_kernel_hw_startup(dfb::out, dfb::out);
    copy_init(dfb::out);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        dfb.reserve_back(1);
        // TEN-4746: a real packer op must sit between reserve_back's WAIT_FREE and push_back's
        // PUSH_TILES, or the PUSH can retire before the space is available. This kernel packs
        // nothing (the host pre-fills the ring), so a no-write dummy pack supplies that op.
        // Without it the ring-pressure tests over-post the tile counter, which the hardware
        // reports as a TILE_COUNTERS fault; the non-ring-pressure cases never notice because
        // free space is always available and the WAIT_FREE is satisfied on arrival.
        ckernel::dummy_pack(dfb::out);
        dfb.push_back(1);
    }
    dfb.finish();
}
