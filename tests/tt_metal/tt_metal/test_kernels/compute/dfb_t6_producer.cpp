// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t num_entries   = get_arg_val<uint32_t>(0);
    uint32_t producer_mask = get_arg_val<uint32_t>(1);
    const uint32_t num_producers = static_cast<uint32_t>(__builtin_popcount(producer_mask));

    // Compute which logical producer slot this TRISC occupies within the mask.
    uint32_t trisc_id    = static_cast<uint32_t>(ckernel::csr_read<ckernel::CSR::TRISC_ID>());
    uint32_t producer_idx = static_cast<uint32_t>(__builtin_popcount(producer_mask & ((1u << trisc_id) - 1u)));

    experimental::DataflowBuffer dfb(0);

    // DPRINT << "t6 producer trisc_id: " << trisc_id << " producer_idx: " << producer_idx
    //        << " num_entries: " << num_entries << ENDL();

    for (uint32_t tile_id = 0; tile_id < num_entries; tile_id++) {
        if (tile_id % num_producers != producer_idx) {
            continue;
        }
        DPRINT << "producer tile id " << tile_id << ENDL();
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    DPRINT << "PFW" << ENDL();
    dfb.finish();
    DPRINT << "PFD" << ENDL();
}
