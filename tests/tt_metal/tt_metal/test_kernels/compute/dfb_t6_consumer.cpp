// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // CTA[1] = blocked_consumer, shared with dfb_consumer.cpp layout
    constexpr uint32_t blocked_consumer = get_compile_time_arg_val(1);

    uint32_t num_entries    = get_arg_val<uint32_t>(0);
    uint32_t consumer_mask  = get_arg_val<uint32_t>(1);
    uint32_t logical_dfb_id = get_arg_val<uint32_t>(2);
    const uint32_t num_consumers = static_cast<uint32_t>(__builtin_popcount(consumer_mask));

    // Compute which logical consumer slot this TRISC occupies within the mask.
    uint32_t trisc_id     = static_cast<uint32_t>(ckernel::csr_read<ckernel::CSR::TRISC_ID>());
    uint32_t consumer_idx = static_cast<uint32_t>(__builtin_popcount(consumer_mask & ((1u << trisc_id) - 1u)));

    experimental::DataflowBuffer dfb(logical_dfb_id);

    // DPRINT << "t6 consumer trisc_id: " << trisc_id << " consumer_idx: " << consumer_idx
    //        << " num_entries: " << num_entries << ENDL();

    for (uint32_t tile_id = 0; tile_id < num_entries; tile_id++) {
        // Blocked: every consumer processes all tiles.
        // Strided: each consumer owns every num_consumers-th tile starting at consumer_idx.
        if constexpr (!blocked_consumer) {
            if (tile_id % num_consumers != consumer_idx) {
                continue;
            }
        }
        DPRINT << "consumer tile id " << tile_id << ENDL();
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
    dfb.finish();
    DPRINT << "CBWD" << ENDL();
}
