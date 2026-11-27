// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 version of writer_cb_test.cpp for DFBs. Drains each DFB in turn to consecutive DRAM ranges.
// The host binds the DFBs as dfb_0, dfb_1, ... on one node of a fresh program, so their slots are
// consecutive from dfb_0.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dfbs = get_arg(args::num_dfbs);
    constexpr uint32_t num_tiles_per_dfb = get_arg(args::num_tiles_per_dfb);
    constexpr uint32_t ublock_size_tiles = get_arg(args::ublock_size_tiles);
    uint32_t dram_address = get_arg(args::dram_address);

    Noc noc;
    for (uint32_t dfb_index = 0; dfb_index < num_dfbs; dfb_index++) {
        DataflowBufferAnyPattern buffer(static_cast<uint16_t>(dfb::dfb_0 + dfb_index));
        const uint32_t ublock_size_bytes = buffer.get_entry_size() * ublock_size_tiles;
        for (uint32_t tile = 0; tile < num_tiles_per_dfb; tile += ublock_size_tiles) {
            buffer.wait_front(ublock_size_tiles);
            noc.async_write(
                buffer,
                AllocatorBank<AllocatorBankType::DRAM>{},
                ublock_size_bytes,
                {},
                {.bank_id = 0, .addr = dram_address});
            noc.async_write_barrier();
            buffer.pop_front(ublock_size_tiles);
            dram_address += ublock_size_bytes;
        }
    }
}
