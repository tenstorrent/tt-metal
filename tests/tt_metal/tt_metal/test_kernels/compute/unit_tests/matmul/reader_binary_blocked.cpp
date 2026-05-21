// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

void kernel_main() {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);

    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_dram_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_blocks = get_arg_val<uint32_t>(4);
    uint32_t in0_block_tile_cnt = get_arg_val<uint32_t>(5);
    uint32_t in1_block_tile_cnt = get_arg_val<uint32_t>(6);
    uint32_t in0_block_size_bytes = get_arg_val<uint32_t>(7);
    uint32_t in1_block_size_bytes = get_arg_val<uint32_t>(8);

#ifdef ARCH_QUASAR
    DataflowBuffer cb_in0(in0_cb);
    DataflowBuffer cb_in1(in1_cb);
#else
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
#endif
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src0;
    AllocatorBank<AllocatorBankType::DRAM> dram_src1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    for (uint32_t i = 0; i < num_blocks; i++) {
        cb_in0.reserve_back(in0_block_tile_cnt);
        cb_in1.reserve_back(in1_block_tile_cnt);

        noc.async_read(dram_src0, cb_in0, in0_block_size_bytes, {.bank_id = src0_dram_bank_id, .addr = src0_addr}, {});
        noc.async_read(dram_src1, cb_in1, in1_block_size_bytes, {.bank_id = src1_dram_bank_id, .addr = src1_addr}, {});

        noc.async_read_barrier();

        cb_in0.push_back(in0_block_tile_cnt);
        cb_in1.push_back(in1_block_tile_cnt);

        src0_addr += in0_block_size_bytes;
        src1_addr += in1_block_size_bytes;
    }
}
