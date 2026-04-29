// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Spoofed prefetch kernel
//  - spoofs out the fast dispatch prefetcher to test the dispatcher
//  - data (dispatch commands) magically appears in a buffer and gets sent to dispatcher

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"

constexpr uint32_t dispatch_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t dispatch_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_cb_sem = get_compile_time_arg_val(3);
constexpr uint32_t cmd_cb_base = get_compile_time_arg_val(4);
constexpr uint32_t cmd_cb_pages = get_compile_time_arg_val(5);
constexpr uint32_t page_batch_size = get_compile_time_arg_val(6);

constexpr uint8_t my_noc_index = NOC_INDEX;
constexpr uint32_t dispatch_noc_xy = uint32_t(NOC_XY_ENCODING(DISPATCH_NOC_X, DISPATCH_NOC_Y));
constexpr uint32_t dispatch_cb_page_size = 1 << dispatch_cb_log_page_size;
constexpr uint32_t dispatch_cb_end = dispatch_cb_base + dispatch_cb_page_size * dispatch_cb_pages;

void kernel_main() {
    int iterations = get_arg_val<int>(0);

    uint32_t cmd_ptr;
    uint32_t dispatch_data_ptr = dispatch_cb_base;
    CBWriter<dispatch_cb_sem, my_noc_index, dispatch_noc_xy, dispatch_cb_sem> writer;
    for (int i = 0; i < iterations; i++) {
        cmd_ptr = cmd_cb_base;
        for (uint32_t j = 0; j < (cmd_cb_pages - 1) / page_batch_size; j++) {
            writer.acquire_pages(page_batch_size);
            for (uint32_t k = 0; k < page_batch_size; k++) {
                noc_async_write(
                    cmd_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), dispatch_cb_page_size);
                cmd_ptr += dispatch_cb_page_size;
                dispatch_data_ptr += dispatch_cb_page_size;
                if (dispatch_data_ptr == dispatch_cb_end) {
                    dispatch_data_ptr = dispatch_cb_base;
                }
            }
            writer.release_pages(page_batch_size);
        }
    }

    // Send finish, last cmd in the chain
    writer.acquire_pages(1);
    noc_async_write(cmd_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), dispatch_cb_page_size);
    writer.release_pages(1);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
