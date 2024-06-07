// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <array>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"

void kernel_main() {
    uint32_t arg_idx = 0;

    constexpr uint32_t msg_hdr_size = get_compile_time_arg_val(0);
    constexpr bool enable_page_size_variations = get_compile_time_arg_val(1) == 1;

    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cb_page_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_pages = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t num_sizes = 8;
    std::array<uint32_t, num_sizes> sub_sizes = {};
    for (uint32_t i = 0; i < num_sizes; i++) {
        sub_sizes[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    const uint32_t read_page_size = cb_page_size - msg_hdr_size;
    const InterleavedAddrGen<true> src_addr_gen = {.bank_base_address = input_buffer_addr, .page_size = read_page_size};

    auto cb = tt::CB::c_in0;

    uint32_t sub_index = 0;

    for (uint32_t i = 0; i < num_pages; i++) {
        cb_reserve_back(cb, 1);
        volatile uint32_t *page_header_addr = reinterpret_cast<volatile uint32_t *>(get_write_ptr(cb));
        // NOTE THAT msg_hdr_size is doubled on host side to maintain alignment for the DRAM reads in THIS TEST ONLY
        uint32_t data_out_start = reinterpret_cast<uint32_t>(page_header_addr) + msg_hdr_size;
        uint64_t src_noc_addr = get_noc_addr(i, src_addr_gen);
        uint32_t message_header_size =
            (read_page_size >> 4) + 2;  // one for header one for padding to maintain noc word alignment
        if (enable_page_size_variations) {
            if (message_header_size < sub_sizes[sub_index] || sub_index >= 8) {
                DPRINT << "REMOTE SENDER READER ERROR!\n";
            }
            message_header_size -= sub_sizes[sub_index];
            sub_index = sub_index == num_sizes - 1 ? 0 : sub_index + 1;
        }
        page_header_addr[0] = message_header_size;
        page_header_addr[1] = 0;
        page_header_addr[2] = 0;
        page_header_addr[3] = 0;
        page_header_addr[4] = 0;
        page_header_addr[5] = 0;
        page_header_addr[6] = 0;
        page_header_addr[7] = 0;

        noc_async_read(src_noc_addr, data_out_start, read_page_size);

        // TODO: upgrade to look at the writes acked counter instead
        noc_async_read_barrier();
        cb_push_back(cb, 1);
    }
}
