// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t src_base_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_base_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(2);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t stride_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t packed_dst_core = get_compile_time_arg_val(5);

    constexpr uint32_t packet_words = packet_size_bytes / sizeof(uint32_t);

    const uint32_t dst_noc_x = packed_dst_core >> 16;
    const uint32_t dst_noc_y = packed_dst_core & 0xFFFF;

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        uint32_t src_addr = src_base_l1_addr + iter * stride_bytes;
        uint32_t dst_addr = dst_base_l1_addr + iter * stride_bytes;

        volatile tt_l1_ptr uint32_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
        for (uint32_t word = 0; word < packet_words; ++word) {
            src_ptr[word] = (iter << 16) | word;
        }

        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);
        noc_async_write(src_addr, dst_noc_addr, packet_size_bytes);
        // Explicitly flush non-blocking writes before next update/send iteration.
        noc_async_write_barrier();
    }
}
