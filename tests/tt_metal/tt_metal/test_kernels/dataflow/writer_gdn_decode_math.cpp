// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GDN decode step 1a gtest writer: drains NUM_OUT circular buffers, one after the other, into single-page DRAM
// buffers one tile at a time (tile size = the CB's page size). The host sizes every output CB for all repetitions, so
// the order does not matter and the pack thread never waits here. Runtime args per output o:
// {cb id, address, bank id, number of tiles}.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t NUM_OUT = get_compile_time_arg_val(0);
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    for (uint32_t o = 0; o < NUM_OUT; o++) {
        const uint32_t cb_id = get_arg_val<uint32_t>(4 * o);
        uint32_t addr = get_arg_val<uint32_t>(4 * o + 1);
        const uint32_t bank = get_arg_val<uint32_t>(4 * o + 2);
        const uint32_t ntiles = get_arg_val<uint32_t>(4 * o + 3);
        CircularBuffer cb(cb_id);
        const uint32_t tile_bytes = cb.get_tile_size();
        for (uint32_t t = 0; t < ntiles; t++) {
            cb.wait_front(1);
            noc.async_write(cb, dram, tile_bytes, {}, {.bank_id = bank, .addr = addr});
            noc.async_write_barrier();
            cb.pop_front(1);
            addr += tile_bytes;
        }
    }
}
