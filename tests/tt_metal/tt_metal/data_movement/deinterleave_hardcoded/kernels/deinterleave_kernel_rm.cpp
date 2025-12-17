// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "risc_common.h"

// Target 8KB of data before a single barrier for 8x8 grid of readers
template <uint32_t payload_size, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    // magic numbers
    if constexpr (payload_size == 64) {
        return 48;
    } else if constexpr (payload_size == 96) {
        return 32;
    } else if constexpr (payload_size == 128) {
        return 16;
    }
    return 4;
}

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t width = get_compile_time_arg_val(2);
    constexpr uint32_t height = get_compile_time_arg_val(3);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);
    constexpr uint32_t barrier_threshold = get_compile_time_arg_val(7) != 0
                                               ? get_compile_time_arg_val(7)
                                               : get_barrier_read_threshold<stick_size_bytes, 2>();
    constexpr uint32_t test_id = get_compile_time_arg_val(8);

    static_assert(stick_size_bytes <= NOC_MAX_BURST_SIZE, "stick size too big, cannot use one_packet API for reads");
    const uint32_t start_x = get_arg_val<uint32_t>(0) + VIRTUAL_TENSIX_START_X;
    const uint32_t end_x = get_arg_val<uint32_t>(1) + VIRTUAL_TENSIX_START_X;
    const uint32_t start_y = get_arg_val<uint32_t>(2) + VIRTUAL_TENSIX_START_Y;
    const uint32_t end_y = get_arg_val<uint32_t>(3) + VIRTUAL_TENSIX_START_Y;
    const uint32_t src_width_stride = get_arg_val<uint32_t>(4);
    const uint32_t src_height_offset_to_next = get_arg_val<uint32_t>(5);
    const uint32_t src_offset = get_arg_val<uint32_t>(6);
    const uint32_t dst_size_bytes = get_arg_val<uint32_t>(7);
    const uint32_t dst_offset = get_arg_val<uint32_t>(8);
    const uint32_t offset_x = get_arg_val<uint32_t>(9) + VIRTUAL_TENSIX_START_X;
    const uint32_t offset_y = get_arg_val<uint32_t>(10) + VIRTUAL_TENSIX_START_Y;
    const uint32_t num_src_cores = get_arg_val<uint32_t>(11);
    const uint32_t dst_rollover_offset = get_arg_val<uint32_t>(12);
    const uint32_t dst_address_get_write_ptr = get_arg_val<uint32_t>(13);
    const uint32_t src_address_get_read_ptr = get_arg_val<uint32_t>(14);

    const uint32_t num_of_transactions = (num_src_cores / 2) * (height / stride_h) * (width / stride_w);
    const uint32_t transaction_size_bytes = stick_size_bytes;

    uint32_t stick_size = stick_size_bytes / 2;

    uint32_t barrier_count = 0;

    // Go through nodes (start_x, start_y) to (end_x, end_y)
    // Copy your stick (dst_batch) to the dst buffer
    // both DM0/DM1 read from all nodes, assuming that's creating uniform load to the NOC
    // they split reading even/odd lines
    auto dst_address = dst_address_get_write_ptr + dst_offset;

    uint32_t src_noc_x = offset_x;
    uint32_t src_noc_y = offset_y;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t src_core = 0; src_core < num_src_cores; src_core += 2) {
            auto src_noc_address = get_noc_addr(src_noc_x, src_noc_y, src_address_get_read_ptr) + src_offset;
            noc_async_read_one_packet_set_state(src_noc_address, stick_size_bytes);

            // Copy half of data data from src to dst
            for (uint32_t h = 0; h < height; h += stride_h) {
                for (uint32_t w = 0; w < width; w += stride_w) {
                    noc_async_read_one_packet_with_state<true>(src_noc_address, dst_address);
                    src_noc_address += src_width_stride;
                    dst_address += stick_size_bytes;
                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                }
                // skip lines to the next src line
                src_noc_address += src_height_offset_to_next;
            }
            dst_address += dst_size_bytes;  // dst_stride - one src image in dst size

            // iterate over src cores, with wrapping
            // to figure out better place for this!
            src_noc_x += 2;
            if (src_noc_x >= end_x) {
                src_noc_x = start_x + src_noc_x - end_x;
                src_noc_y++;
                if (src_noc_y >= end_y) {  // rollover
                    src_noc_y = start_y;
                    dst_address = dst_address_get_write_ptr + dst_rollover_offset;
                }
            }
        }

        noc_async_read_barrier();
    }
}
