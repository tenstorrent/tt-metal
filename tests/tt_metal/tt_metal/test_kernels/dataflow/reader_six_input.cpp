// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t dfb_ids[6] = {
        dfb::in0,
        dfb::in1,
        dfb::in2,
        dfb::in3,
        dfb::in4,
        dfb::in5,
    };

    const uint32_t src_addrs[6] = {
        get_arg(args::src0_addr),
        get_arg(args::src1_addr),
        get_arg(args::src2_addr),
        get_arg(args::src3_addr),
        get_arg(args::src4_addr),
        get_arg(args::src5_addr),
    };
    const uint32_t src_bank_ids[6] = {
        get_arg(args::src0_bank_id),
        get_arg(args::src1_bank_id),
        get_arg(args::src2_bank_id),
        get_arg(args::src3_bank_id),
        get_arg(args::src4_bank_id),
        get_arg(args::src5_bank_id),
    };
    const uint32_t num_tiles = get_arg(args::num_tiles);

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
