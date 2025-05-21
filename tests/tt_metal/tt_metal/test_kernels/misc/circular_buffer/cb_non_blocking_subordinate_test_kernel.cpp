// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include <cstddef>
#include <cstdint>
#include <array>

/*
 * This kernel enumerates over all the CBs and pushes a page at a time. For every page, it checks to see if the
 * non-blocking call to `cb_pages_reservable_at_back` to get the result. It stores the result for the element
 * corresponding to that iteration index, into the output buffer associated with that CB. The buffer can
 * later be readback by host for comparison and checking.
 */
void kernel_main() {
    constexpr int32_t n_cbs = get_compile_time_arg_val(0);
    constexpr int32_t n_pages = get_compile_time_arg_val(1);

    size_t arg_idx = 0;

    auto master_sem_addr = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    auto subordinate_sem_addr = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));

    std::array<uint32_t, n_cbs> output_buffer_addrs;
    for (size_t i = 0; i < n_cbs; i++) {
        output_buffer_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    auto get_idx = [n_pages](size_t i, size_t j) -> size_t { return i * n_pages + j; };

    for (int32_t i = 0; i < n_cbs; i++) {
        auto* const output_buffer = reinterpret_cast<uint8_t*>(output_buffer_addrs[i]);
        for (int32_t j = 0; j < n_pages; j++) {
            // First level signal indicates the writer has pushed new pages to the CB
            noc_semaphore_wait(master_sem_addr, 1);
            noc_semaphore_set(subordinate_sem_addr, 1);

            for (int32_t k = 0; k < n_pages; k++) {
                auto result = cb_pages_available_at_front(i, k);
                output_buffer[get_idx(j, k)] = static_cast<uint8_t>(result);
            }
            noc_semaphore_wait(master_sem_addr, 2);
            noc_semaphore_set(master_sem_addr, 0);
            if (j > 0) {
                cb_wait_front(i, j);
                cb_pop_front(i, j);
            }
            // Second level signal indicates "alignment pages". We signal back that we are
            // done processing this step
            noc_semaphore_set(subordinate_sem_addr, 2);

            if (j > 0) {
                // snap back to alignment
                cb_wait_front(i, n_pages - j);
                cb_pop_front(i, n_pages - j);
            }
        }
    }
}
