// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"

// Reads one tile from each of four DRAM buffers into CBs c_0..c_3 (one page each, sized by the CB's own
// tile size so the CBs may carry different formats). Runtime args: four (address, bank id) pairs.
void kernel_main() {
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;

    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t src_addr = get_arg_val<uint32_t>(2 * i);
        const uint32_t bank_id = get_arg_val<uint32_t>(2 * i + 1);
        CircularBuffer cb(i);
        cb.reserve_back(1);
        noc.async_read(dram_src, cb, cb.get_tile_size(), {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        cb.push_back(1);
    }
}
