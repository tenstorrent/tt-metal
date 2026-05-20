// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr uint32_t dfb_ids[6] = {
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
    };

    const uint32_t src_addrs[6] = {
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(10),
    };
    const uint32_t src_bank_ids[6] = {
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(11),
    };
    const uint32_t num_tiles = get_arg_val<uint32_t>(12);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;

    for (uint32_t i = 0; i < 6; ++i) {
        DataflowBuffer dfb(dfb_ids[i]);
        const uint32_t bytes_per_tile = dfb.get_entry_size();
        uint32_t src_addr = src_addrs[i];
        for (uint32_t t = 0; t < num_tiles; ++t) {
            dfb.reserve_back(1);
            noc.async_read(dram_src, dfb, bytes_per_tile, {.bank_id = src_bank_ids[i], .addr = src_addr}, {});
            noc.async_read_barrier();
            dfb.push_back(1);
            src_addr += bytes_per_tile;
        }
    }
}
