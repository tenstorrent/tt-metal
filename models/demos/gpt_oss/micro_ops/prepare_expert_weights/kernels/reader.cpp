// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

/**
 * Reader kernel for prepare_expert_weights operation.
 *
 * This kernel simply signals that the input buffer (backed by sharded L1)
 * is ready for the compute kernel to read.
 *
 * Compile-time args:
 *   0: input_cb - Circular buffer index for input weights
 *   1: num_tiles - Number of input tiles
 */
void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);

    // Signal that input buffer is ready (backed by L1 shard)
    // The data is already in L1 from the sharded tensor, we just need
    // to signal availability to the compute kernel via the CB protocol
    cb_reserve_back(input_cb, num_tiles);
    cb_push_back(input_cb, num_tiles);
}
