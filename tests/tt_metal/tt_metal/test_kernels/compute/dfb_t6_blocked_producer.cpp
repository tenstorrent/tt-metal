// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side BLOCKED producer (TRISC -> DFB -> DM).
// The host pre-fills the ring; this kernel only posts credits, one block per reserve/push.

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"  // for dummy_pack (TEN-4746 no-write pack ordering)
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t block_size = get_arg(args::block_size);

    DataflowBuffer dfb(dfb::out);

    // dummy_pack's PACR_STRIDE validates a pack-partition bd_table entry; compute_kernel_hw_startup
    // is what runs llk_pack_init and programs that entry. copy_init is not needed: this kernel
    // never unpacks.
    compute_kernel_hw_startup(dfb::out, dfb::out);

    const uint32_t num_blocks = num_entries_per_producer / block_size;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        dfb.reserve_back(block_size);
        // TEN-4746: a real packer op must sit between reserve_back's WAIT_FREE and push_back's
        // PUSH_TILES. The host pre-fills the ring, so a no-write dummy pack supplies that op
        // without modifying the payload.
        ckernel::dummy_pack(dfb::out);
        dfb.push_back(block_size);
    }
    dfb.finish();
}
