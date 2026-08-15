// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Explicit-sync reader for the BFD datacopy POC. Streams three DRAM inputs into
// three single-producer DFBs (in0/in1/in2) with reserve_back/async_read/barrier/
// push_back. Sequential per input (all of in0, then in1, then in2) to match the
// compute kernel, which processes one input's block fully before switching operands.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

namespace {
void read_block(
    Noc& noc,
    AllocatorBank<AllocatorBankType::DRAM>& src,
    DataflowBuffer& dfb,
    std::uint32_t addr,
    std::uint32_t bank_id,
    std::uint32_t n_tiles) {
    const std::uint32_t sz = dfb.get_entry_size();
    for (std::uint32_t t = 0; t < n_tiles; ++t) {
        dfb.reserve_back(1);
        noc.async_read(src, dfb, sz, {.bank_id = bank_id, .addr = addr}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
        addr += sz;
    }
}
}  // namespace

void kernel_main() {
    std::uint32_t src0_addr = get_arg(args::src0_addr);
    std::uint32_t src1_addr = get_arg(args::src1_addr);
    std::uint32_t src2_addr = get_arg(args::src2_addr);
    std::uint32_t bank_id = get_arg(args::bank_id);
    std::uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb2(dfb::in2);

    read_block(noc, dram_src, dfb0, src0_addr, bank_id, num_tiles);
    read_block(noc, dram_src, dfb1, src1_addr, bank_id, num_tiles);
    read_block(noc, dram_src, dfb2, src2_addr, bank_id, num_tiles);
}
