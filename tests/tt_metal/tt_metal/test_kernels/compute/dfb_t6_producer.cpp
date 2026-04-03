// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // CTA layout mirrors dfb_producer.cpp: [src_addr, num_entries_per_producer, implicit_sync, blocked_consumer, ...]
    // src_addr (CTA[0]), implicit_sync (CTA[2]), blocked_consumer (CTA[3]), and TensorAccessorArgs are
    // unused: the Tensix producer doesn't do NOC reads. The host pre-fills DFB L1 and the kernel
    // only posts credits so the DM consumer knows data is available.
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(1);

    // RTA layout mirrors dfb_producer.cpp: [producer_mask, chunk_offset]
    // chunk_offset (RTA[1]) is unused since Tensix doesn't read from DRAM.
    uint32_t producer_mask = get_arg_val<uint32_t>(0);

    // Compute which logical producer slot this TRISC occupies within the mask.
    uint32_t trisc_id     = static_cast<uint32_t>(ckernel::csr_read<ckernel::CSR::TRISC_ID>());
    uint32_t producer_idx = static_cast<uint32_t>(__builtin_popcount(producer_mask & ((1u << trisc_id) - 1u)));

    experimental::DataflowBuffer dfb(0);

    // DPRINT << "t6 producer trisc_id: " << trisc_id << " producer_idx: " << producer_idx
    //        << " num_entries_per_producer: " << num_entries_per_producer << ENDL();

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        DPRINT << "producer tile id " << tile_id << ENDL();
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    DPRINT << "PFW" << ENDL();
    dfb.finish();
    DPRINT << "PFD" << ENDL();
}
