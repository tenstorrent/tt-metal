// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

/**
 * Simplified matmul reader kernel for fully sharded inputs.
 *
 * Both in0 and in1 are backed by L1 shards:
 * - in0: 1x7K input tensor, replicated across all cores
 * - in1: Weight tensor, sharded across cores (each core has its slice)
 *
 * Each core processes a single output tile:
 * - M = 1 tile (tiny tile height)
 * - K = full K dimension
 * - N = 1 tile per core (32 elements)
 *
 * This kernel just signals that the sharded CBs are ready.
 */
void kernel_main() {
    // Compile time args
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(2);  // K dimension in tiles

    // Both in0 and in1 are backed by sharded tensors - just signal they're ready
    cb_reserve_back(in0_cb, num_tiles_k);
    cb_push_back(in0_cb, num_tiles_k);

    cb_reserve_back(in1_cb, num_tiles_k);
    cb_push_back(in1_cb, num_tiles_k);
}
