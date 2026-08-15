// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Explicit-sync reader for the BFD datacopy POC. Streams three DRAM inputs into
// three single-producer DFBs (in0/in1/in2) with reserve_back/async_read/barrier/
// push_back (same pattern as reader_binary_2_0.cpp). One tile per input per
// iteration, so the compute's round-robin (in0, in1, in2, ...) consumer stays fed
// and each input DFB sees its tiles in order 0..num_tiles-1.

#include <stdint.h>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    std::uint32_t src0_addr = get_arg(args::src0_addr);
    std::uint32_t src1_addr = get_arg(args::src1_addr);
    std::uint32_t src2_addr = get_arg(args::src2_addr);
    std::uint32_t bank_id = get_arg(args::bank_id);
    std::uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    constexpr std::uint32_t ublock = 1;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb2(dfb::in2);
    const std::uint32_t sz0 = dfb0.get_entry_size() * ublock;
    const std::uint32_t sz1 = dfb1.get_entry_size() * ublock;
    const std::uint32_t sz2 = dfb2.get_entry_size() * ublock;

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        dfb0.reserve_back(ublock);
        noc.async_read(dram_src, dfb0, sz0, {.bank_id = bank_id, .addr = src0_addr}, {});
        noc.async_read_barrier();
        dfb0.push_back(ublock);

        dfb1.reserve_back(ublock);
        noc.async_read(dram_src, dfb1, sz1, {.bank_id = bank_id, .addr = src1_addr}, {});
        noc.async_read_barrier();
        dfb1.push_back(ublock);

        dfb2.reserve_back(ublock);
        noc.async_read(dram_src, dfb2, sz2, {.bank_id = bank_id, .addr = src2_addr}, {});
        noc.async_read_barrier();
        dfb2.push_back(ublock);

        src0_addr += sz0;
        src1_addr += sz1;
        src2_addr += sz2;
    }
}
