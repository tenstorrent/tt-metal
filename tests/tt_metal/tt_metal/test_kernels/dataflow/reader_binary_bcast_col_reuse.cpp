// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the blocked bcast-col SUB path with SrcB reuse.
//
// Unlike reader_binary_2_0.cpp, the two inputs are NOT read in lockstep: src1 (the bcast-col
// operand) is num_bcast_tiles tiles, one per row of a block, read once and pushed once; while
// src0 streams num_tiles column tiles. The compute kernel holds src1 at the front of its buffer for
// the whole run so every block re-reads the same L1 tiles, this reuse is the point of the custom op.

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    std::uint32_t src0_addr = get_arg(args::src0_addr);
    std::uint32_t src0_bank_id = get_arg(args::src0_bank_id);
    std::uint32_t src1_addr = get_arg(args::src1_addr);
    std::uint32_t src1_bank_id = get_arg(args::src1_bank_id);
    std::uint32_t num_tiles = get_arg(args::num_tiles);
    std::uint32_t num_bcast_tiles = get_arg(args::num_bcast_tiles);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    constexpr std::uint32_t ublock_size_tiles = 1;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    const std::uint32_t ublock_size_bytes_0 = dfb0.get_entry_size() * ublock_size_tiles;
    const std::uint32_t ublock_size_bytes_1 = dfb1.get_entry_size() * ublock_size_tiles;

    // src1: num_bcast_tiles tiles (one per row of a block), read up front and never re-read.
    for (std::uint32_t i = 0; i < num_bcast_tiles; i += ublock_size_tiles) {
        dfb1.reserve_back(ublock_size_tiles);
        noc.async_read(dram_src, dfb1, ublock_size_bytes_1, {.bank_id = src1_bank_id, .addr = src1_addr}, {});
        noc.async_read_barrier();
        dfb1.push_back(ublock_size_tiles);
        src1_addr += ublock_size_bytes_1;
    }

    // src0: num_tiles column tiles, one at a time. The buffer holds one block, so this
    // self-throttles against the compute kernel's per-block pop_front.
    for (std::uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        dfb0.reserve_back(ublock_size_tiles);
        noc.async_read(dram_src, dfb0, ublock_size_bytes_0, {.bank_id = src0_bank_id, .addr = src0_addr}, {});
        noc.async_read_barrier();
        dfb0.push_back(ublock_size_tiles);
        src0_addr += ublock_size_bytes_0;
    }
}
