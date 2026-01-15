// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

/**
 * Simple in0 reader kernel for replicated input.
 *
 * CB0 is backed directly by the input tensor (replicated on all compute cores).
 * This kernel just signals that tiles are ready for compute - no data movement needed.
 */
void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    // CB0 is backed by the input tensor - data is already there.
    // Just signal that tiles are ready for each block.
    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_push_back(cb_id_in0, in0_block_num_tiles);
    }
}
