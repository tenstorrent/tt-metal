// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simple kernel to populate a CB from a sharded tensor.
// For sharded tensors with globally allocated CBs, the data is already in L1,
// but we need to manually reserve_back and push_back to make it visible to the CB interface.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);

    DPRINT << "Populating CB from sharded tensor\n";
    DPRINT << "cb_index: " << (uint32_t)cb_index << "\n";
    DPRINT << "num_pages: " << (uint32_t)num_pages << "\n";

    // For globally allocated CBs pointing to sharded tensors:
    // The data is already in the buffer, we just need to mark it as available
    cb_reserve_back(cb_index, num_pages);
    cb_push_back(cb_index, num_pages);

    DPRINT << "CB populated with " << (uint32_t)num_pages << " pages\n";
}
