// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t src0_addr = get_arg(args::src0_addr);
    uint32_t src0_num_tiles = get_arg(args::src0_num_tiles);
    uint32_t src1_addr = get_arg(args::src1_addr);
    uint32_t src1_num_tiles = get_arg(args::src1_num_tiles);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = dfb_in0.get_entry_size();
    uint32_t ublock_size_bytes_1 = dfb_in1.get_entry_size();
    uint32_t ublock_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles > src1_num_tiles ? src0_num_tiles : src1_num_tiles;

    // The source buffers are mesh buffers, so their accessor args are passed as compile-time varargs.
    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    const auto s0 = TensorAccessor(src0_args, src0_addr);
    const auto s1 = TensorAccessor(src1_args, src1_addr);

    // read ublocks from src0/src1 to DFB0/DFB1, then push ublocks to compute (unpacker)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        if (i < src0_num_tiles) {
            uint64_t src0_noc_addr = get_noc_addr(i, s0);

            dfb_in0.reserve_back(ublock_size_tiles);
            l1_write_addr_in0 = dfb_in0.get_write_ptr();

            noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);

            noc_async_read_barrier();

            dfb_in0.push_back(ublock_size_tiles);
        }

        if (i < src1_num_tiles) {
            uint64_t src1_noc_addr = get_noc_addr(i, s1);

            dfb_in1.reserve_back(ublock_size_tiles);
            l1_write_addr_in1 = dfb_in1.get_write_ptr();

            noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

            noc_async_read_barrier();

            dfb_in1.push_back(ublock_size_tiles);
        }
    }
}
