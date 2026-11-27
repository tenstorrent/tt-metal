// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Ablation stub for g6_clone_writer.cpp: same CT/RT layout, block count and CB handshake, no writes.
// Paired with the production reader it measures the read-only time of a plan.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = (num_pages_this_core + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        cb_wait_front(cb_page_staging, block_pages);
        cb_pop_front(cb_page_staging, block_pages);
    }
}
