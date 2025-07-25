// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // Declare address in which we stored the source matrices. We have set the exact same format between CBs and DRAM
    // buffers in the host code, so we can use the same address for both DRAM and CBs.
    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr, get_tile_size(cb_id_in0));
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr, get_tile_size(cb_id_in1));

    // Loop through the dimensions of the matrices. Read them and push to the circular buffers.
    // Dimension names are called M, N and K. `t` in `mt` means tile.
    for (uint32_t mt = 0; mt < Mt; mt++) {
        uint32_t itileB = 0;
        for (uint32_t nt = 0; nt < Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                {                                          // Read A's tile at (mt, kt)
                    uint32_t a_tile_index = mt * Kt + kt;  // A is MK, so we stride by Kt
                    cb_reserve_back(cb_id_in0, 1);
                    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                    noc_async_read_tile(a_tile_index, s0, l1_write_addr_in0);
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in0, 1);
                }

                {                                          // Read B's tile at (kt, nt)
                    uint32_t b_tile_index = kt * Nt + nt;  // B is KN, so we stride by Nt
                    cb_reserve_back(cb_id_in1, 1);
                    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                    noc_async_read_tile(b_tile_index, s1, l1_write_addr_in1);
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in1, 1);
                }
            }  // Kt loop
        }  // Nt loop
    }  // Mt loop
}
