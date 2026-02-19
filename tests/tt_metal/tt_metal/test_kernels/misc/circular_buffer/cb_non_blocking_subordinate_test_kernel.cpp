// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"

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

    experimental::Semaphore master_sem(get_arg_val<uint32_t>(arg_idx++));
    experimental::Semaphore subordinate_sem(get_arg_val<uint32_t>(arg_idx++));

    std::array<uint32_t, n_cbs> output_buffer_addrs;
    for (size_t i = 0; i < n_cbs; i++) {
        output_buffer_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    auto get_idx = [n_pages](size_t i, size_t j) -> size_t { return i * n_pages + j; };

    for (int32_t i = 0; i < n_cbs; i++) {
        experimental::CircularBuffer cb(i);
        auto* const output_buffer = reinterpret_cast<uint8_t*>(output_buffer_addrs[i]);
        for (int32_t j = 0; j < n_pages; j++) {
            // First level signal indicates the writer has pushed new pages to the CB
            master_sem.down(1);
            subordinate_sem.up(1);

            for (int32_t k = 0; k < n_pages; k++) {
                auto result = cb.pages_available_at_front(k);
                output_buffer[get_idx(j, k)] = static_cast<uint8_t>(result);
            }
            master_sem.down(1);
            if (j > 0) {
                cb.wait_front(j);
                cb.pop_front(j);
            }
            // Second level signal indicates "alignment pages". We signal back that we are
            // done processing this step
            subordinate_sem.up(1);

            if (j > 0) {
                // snap back to alignment
                cb.wait_front(n_pages - j);
                cb.pop_front(n_pages - j);
            }
        }
    }
}
