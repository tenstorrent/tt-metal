// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

/**
 * Writer kernel for prepare_expert_weights operation.
 *
 * This kernel waits for all output tiles to be written by the compute kernel.
 * Since the output buffer is backed by sharded L1, we don't need to explicitly
 * write anywhere - we just wait for completion.
 *
 * Compile-time args:
 *   0: output_cb - Circular buffer index for output
 *   1: num_tiles - Number of output tiles to wait for
 */
void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);

    // Wait for all output tiles to be ready
    // This ensures compute kernel has finished writing to the output CB
    cb_wait_front(output_cb, num_tiles);
}
