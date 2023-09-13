// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // const uint32_t src_addr                    = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr                    = get_arg_val<uint32_t>(1);
    // const uint32_t num_unpadded_W              = get_arg_val<uint32_t>(2);
    const uint32_t num_total_W                 = get_arg_val<uint32_t>(3);
    // const uint32_t num_unpadded_Z              = get_arg_val<uint32_t>(4);
    const uint32_t num_total_Z                 = get_arg_val<uint32_t>(5);
    // const uint32_t num_unpadded_Y              = get_arg_val<uint32_t>(6);
    const uint32_t num_total_Y                 = get_arg_val<uint32_t>(7);
    // const uint32_t num_unpadded_X              = get_arg_val<uint32_t>(8);
    const uint32_t num_total_X                 = get_arg_val<uint32_t>(9);
    // const uint32_t unpadded_X_nbytes           = get_arg_val<uint32_t>(10);
    const uint32_t padded_X_nbytes             = get_arg_val<uint32_t>(11);
    // const uint32_t padded_X_diff_nbytes        = get_arg_val<uint32_t>(12);
    // const uint32_t pad_value_const_buffer_addr = get_arg_val<uint32_t>(13);
    // const uint32_t pad_value_const_buffer_nbytes = get_arg_val<uint32_t>(14);   // assumed to be 64 bytes. TODO: generalize?
    // const uint32_t pad_value_packed            = get_arg_val<uint32_t>(15);
    // const uint32_t dst_buffer_l1_addr          = get_arg_val<uint32_t>(16);

    constexpr uint32_t cb_id = tt::CB::c_in0;

    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    #define dst_stick_size_is_pow2 get_compile_time_arg_val(4) == 1

    #if (dst_stick_size_is_pow2)
        constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(5);
        const InterleavedPow2AddrGen<dst_is_dram> s1 = {
            .bank_base_address = dst_addr,
            .log_base_2_of_page_size = dst_log_base_2_of_page_size
        };
    #else
        const InterleavedAddrGen<dst_is_dram> s1 = {
            .bank_base_address = dst_addr,
            .page_size = padded_X_nbytes
        };
    #endif

    uint32_t dst_stick_id = 0;
    for (uint32_t w = 0; w < num_total_W; ++ w) {
        for (uint32_t z = 0; z < num_total_Z; ++ z) {
            for (uint32_t y = 0; y < num_total_Y; ++ y) {
                cb_wait_front(cb_id, 1);
                uint32_t l1_addr = get_read_ptr(cb_id);
                uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, s1);
                noc_async_write(l1_addr, dst_noc_addr, padded_X_nbytes);
                noc_async_write_barrier();
                ++ dst_stick_id;
                cb_pop_front(cb_id, 1);
            }
        }
    }
}
