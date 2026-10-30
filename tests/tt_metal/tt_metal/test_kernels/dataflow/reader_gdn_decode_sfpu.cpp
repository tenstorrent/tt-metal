// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GDN decode SFPU gtest reader: reads NUM_IN single-page DRAM buffers into circular buffers c_0 .. c_{NUM_IN-1}, one
// tile at a time; the tile size of each CB (which may be a tiny tile) comes from the CB's own page size. Runtime args
// per input i: {address, bank id, number of tiles}; an input with zero tiles is skipped.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t NUM_IN = get_compile_time_arg_val(0);
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    for (uint32_t i = 0; i < NUM_IN; i++) {
        uint32_t addr = get_arg_val<uint32_t>(3 * i);
        const uint32_t bank = get_arg_val<uint32_t>(3 * i + 1);
        const uint32_t ntiles = get_arg_val<uint32_t>(3 * i + 2);
        CircularBuffer cb(i);
        const uint32_t tile_bytes = cb.get_tile_size();
        for (uint32_t t = 0; t < ntiles; t++) {
            cb.reserve_back(1);
            noc.async_read(dram, cb, tile_bytes, {.bank_id = bank, .addr = addr}, {});
            noc.async_read_barrier();
            cb.push_back(1);
            addr += tile_bytes;
        }
    }
}
