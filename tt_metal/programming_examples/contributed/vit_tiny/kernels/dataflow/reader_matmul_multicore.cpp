// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multicore matmul reader: each core reads its assigned rows of A and all of B.
// Runtime args: src0_addr, src1_addr, start_row, num_rows, Kt, Nt

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t Kt = get_arg_val<uint32_t>(4);
    uint32_t Nt = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr);

    for (uint32_t mt = 0; mt < num_rows; mt++) {
        uint32_t row = start_row + mt;
        for (uint32_t nt = 0; nt < Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                {
                    uint32_t a_tile_index = row * Kt + kt;
                    cb_reserve_back(cb_in0, 1);
                    uint32_t l1_addr = get_write_ptr(cb_in0);
                    noc_async_read_tile(a_tile_index, s0, l1_addr);
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, 1);
                }
                {
                    uint32_t b_tile_index = kt * Nt + nt;
                    cb_reserve_back(cb_in1, 1);
                    uint32_t l1_addr = get_write_ptr(cb_in1);
                    noc_async_read_tile(b_tile_index, s1, l1_addr);
                    noc_async_read_barrier();
                    cb_push_back(cb_in1, 1);
                }
            }
        }
    }
}
