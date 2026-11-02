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
// posts the credits that say "one tile is available", which the downstream DM
// consumer waits on. The producing thread also stamps word 0 of every entry it
// pushes with its thread id, so the host can see which thread pushed it.
//
// Flow per test invocation:
//   1. Host pre-fills the DFB L1 ring with the input data.
//   2. This kernel calls reserve_back + stamp + push_back num_entries_per_producer
//      times, then dfb.finish().
//   3. The DM consumer drains those credits, reads from L1, and NoC-writes
//      out to DRAM.
//   4. Host reads DRAM to verify, expecting the stamp in word 0 of every entry.
//
// Bindings (set by host KernelSpec):
//   dfb::out — PRODUCER (host pre-binds the same DFB the DM consumer reads).

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"  // for dummy_pack (TEN-4746 no-write pack ordering)
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr bool producer_on_trisc3 = get_arg(args::producer_on_trisc3) != 0;
    constexpr uint32_t producer_stamp_base = get_arg(args::producer_stamp_base);
    constexpr bool i_produce = ckernel::TRISC_ID == (producer_on_trisc3 ? 3u : 2u);

    DataflowBuffer dfb(dfb::out);

    // dummy_pack's PACR_STRIDE validates a pack-partition bd_table entry; compute_kernel_hw_startup
    // is what runs llk_pack_init and programs that entry. copy_init is not needed: this kernel
    // never unpacks.
    compute_kernel_hw_startup(dfb::out, dfb::out);

    // One reserve/push per share (1 on a plain ring, more when a side is BLOCKED). Nothing is
    // packed: the host pre-fills the ring.
    const uint32_t share = dfb.get_producer_share();
    const uint32_t entry_bytes = dfb.get_entry_size();
    const uint32_t stride_tiles = dfb.get_producer_stride_tiles();
    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id += share) {
        dfb.reserve_back(share);
        // TEN-4746: a real packer op must sit between reserve_back's WAIT_FREE and push_back's
        // PUSH_TILES. The host pre-fills the ring, so a no-write dummy pack supplies that op
        // without modifying the payload.
        ckernel::dummy_pack(dfb::out);
        if constexpr (i_produce) {
            // Make sure the space is free before storing: tensix_sync waits for the pack queued behind reserve_back.
            ckernel::tensix_sync();
            const uint32_t share_base = dfb.get_write_ptr() << 4;
            for (uint32_t j = 0; j < share; ++j) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(share_base + j * stride_tiles * entry_bytes) =
                    producer_stamp_base + ckernel::TRISC_ID;
            }
            asm volatile("fence" ::: "memory");  // the stamps land before push_back posts the credit
        }
        dfb.push_back(share);
    }
    dfb.finish();
}
