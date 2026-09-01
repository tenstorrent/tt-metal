// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t src0_addr = get_arg(args::src0_addr);
    uint32_t src0_bank_id = get_arg(args::src0_bank_id);
    uint32_t src1_addr = get_arg(args::src1_addr);
    uint32_t src1_bank_id = get_arg(args::src1_bank_id);
    uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    uint32_t ublock_size_tiles = 1;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    uint32_t ublock_size_bytes_0 = dfb0.get_entry_size() * ublock_size_tiles;
    uint32_t ublock_size_bytes_1 = dfb1.get_entry_size() * ublock_size_tiles;

    // read ublocks from src0/src1 to DFB0/DFB1, then push ublocks to compute (unpacker)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        dfb0.reserve_back(ublock_size_tiles);
#ifndef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
        dfb1.reserve_back(ublock_size_tiles);
#endif
        noc.async_read(dram_src, dfb0, ublock_size_bytes_0, {.bank_id = src0_bank_id, .addr = src0_addr}, {});
#ifndef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
        noc.async_read(dram_src, dfb1, ublock_size_bytes_1, {.bank_id = src1_bank_id, .addr = src1_addr}, {});
#endif
        noc.async_read_barrier();
        dfb0.push_back(ublock_size_tiles);
#ifndef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
        dfb1.push_back(ublock_size_tiles);
#endif
        src0_addr += ublock_size_bytes_0;
#ifndef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
        src1_addr += ublock_size_bytes_1;
#endif
    }

#ifdef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
    dfb1.reserve_back(1);
    noc.async_read(dram_src, dfb1, ublock_size_bytes_1, {.bank_id = src1_bank_id, .addr = src1_addr}, {});
    noc.async_read_barrier();
    dfb1.push_back(1);
#endif

    // This input populates dest with values before binary operation
    // executes, this is used to test eltwise binary with dest re-use
    // and eltwise binary with dest accumulation
#if defined(DST_ACCUM_MODE) || defined(LOAD_BUF2_DATA) || defined(ELTWISE_DEST_REUSE_TYPE)
    uint32_t src2_addr = get_arg(args::src2_addr);
    uint32_t src2_bank_id = get_arg(args::src2_bank_id);

    DataflowBuffer dfb2(dfb::in2);
    uint32_t ublock_size_bytes_2 = dfb2.get_entry_size() * ublock_size_tiles;

#ifdef PRECEDE_DEST_REUSE_WITH_COL_BROADCAST
    constexpr uint32_t num_input2_tiles = 1;
#else
    const uint32_t num_input2_tiles = num_tiles;
#endif
    for (uint32_t i = 0; i < num_input2_tiles; i += ublock_size_tiles) {
        dfb2.reserve_back(ublock_size_tiles);
        noc.async_read(dram_src, dfb2, ublock_size_bytes_2, {.bank_id = src2_bank_id, .addr = src2_addr}, {});
        noc.async_read_barrier();
        dfb2.push_back(ublock_size_tiles);
        src2_addr += ublock_size_bytes_2;
    }
#endif
}
