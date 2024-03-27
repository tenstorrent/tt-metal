// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t stick_size               = get_arg_val<uint32_t>(1);
    const uint32_t block_height             = get_arg_val<uint32_t>(2);
    const uint32_t block_width_bytes        = get_arg_val<uint32_t>(3);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(4);
    const bool aligned                      = static_cast<bool>(get_arg_val<uint32_t>(5));
    const uint32_t aligned_input_width_offset_bytes = get_arg_val<uint32_t>(6);
    const uint32_t aligned_block_width_bytes = get_arg_val<uint32_t>(7);
    const uint32_t aligned_offset           = get_arg_val<uint32_t>(8);
    const uint32_t start_id                 = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);

    constexpr bool src0_is_dram          = get_compile_time_arg_val(2) == 1;
    #define src_stick_size_is_pow2 get_compile_time_arg_val(3) == 1
    #if (src_stick_size_is_pow2)
    constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(4);
    const InterleavedPow2AddrGen<src0_is_dram> s0 = {
        .bank_base_address = src_addr + aligned_input_width_offset_bytes,
        .log_base_2_of_page_size = src_log_base_2_of_page_size
    };
    #else
    const InterleavedAddrGen<src0_is_dram> s0 = {
        .bank_base_address = src_addr + aligned_input_width_offset_bytes,
        .page_size = stick_size
    };
    #endif
    uint32_t stick_id = start_id;
    cb_reserve_back(cb_id_in0, block_height);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    if (aligned) {
        for (uint32_t h = 0; h < block_height; ++h) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
            noc_async_read(src_noc_addr, l1_write_addr, block_width_bytes);
            stick_id++;
            l1_write_addr += padded_block_width_bytes;
        }
    } else {
        cb_reserve_back(cb_id_in1, 1);
        uint32_t scratch_l1_write_addr = get_write_ptr(cb_id_in1);
        uint64_t scratch_l1_noc_read_addr = get_noc_addr(scratch_l1_write_addr + aligned_offset);
        for (uint32_t h = 0; h < block_height; ++h) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
            noc_async_read(src_noc_addr, scratch_l1_write_addr, aligned_block_width_bytes);
            noc_async_read_barrier();
            noc_async_read(scratch_l1_noc_read_addr, l1_write_addr, block_width_bytes);
            stick_id++;
            l1_write_addr += padded_block_width_bytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, block_height);
}
