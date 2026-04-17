// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the add_bias_bcast_rows isolated test.
// Pushes partials into c_24 per iter, bias into c_2 either once or per-iter
// depending on `bias_one_time_front`.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t partials_addr = get_arg_val<uint32_t>(0);
    uint32_t partials_bank = get_arg_val<uint32_t>(1);
    uint32_t bias_addr = get_arg_val<uint32_t>(2);
    uint32_t bias_bank = get_arg_val<uint32_t>(3);
    uint32_t num_invocations = get_arg_val<uint32_t>(4);
    uint32_t partials_tiles_per_iter = get_arg_val<uint32_t>(5);
    uint32_t bias_ntiles = get_arg_val<uint32_t>(6);
    uint32_t partials_bytes_per_iter = get_arg_val<uint32_t>(7);
    uint32_t bias_bytes_per_push = get_arg_val<uint32_t>(8);
    uint32_t bias_one_time_front = get_arg_val<uint32_t>(9);

    constexpr uint32_t partials_cb = tt::CBIndex::c_24;
    constexpr uint32_t bias_cb = tt::CBIndex::c_2;

    for (uint32_t iter = 0; iter < num_invocations; iter++) {
        // Bias push
        if (bias_one_time_front == 0 || iter == 0) {
            uint64_t bias_noc = get_noc_addr_from_bank_id<true>(bias_bank, bias_addr);
            cb_reserve_back(bias_cb, bias_ntiles);
            uint32_t l1_b = get_write_ptr(bias_cb);
            noc_async_read(bias_noc, l1_b, bias_bytes_per_push);
            noc_async_read_barrier();
            cb_push_back(bias_cb, bias_ntiles);

            // Advance bias DRAM pointer only in per-iter mode.
            if (bias_one_time_front == 0) {
                bias_addr += bias_bytes_per_push;
            }
        }

        // Partials push
        uint64_t partials_noc = get_noc_addr_from_bank_id<true>(partials_bank, partials_addr);
        cb_reserve_back(partials_cb, partials_tiles_per_iter);
        uint32_t l1_p = get_write_ptr(partials_cb);
        noc_async_read(partials_noc, l1_p, partials_bytes_per_iter);
        noc_async_read_barrier();
        cb_push_back(partials_cb, partials_tiles_per_iter);
        partials_addr += partials_bytes_per_iter;
    }
}
