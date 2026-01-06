// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
/**
 * Simplified matmul writer kernel for width-sharded output.
 *
 * This kernel waits for the output tile to be ready in the output CB.
 * The output CB is backed by a sharded tensor, so data is written directly to L1.
 */
void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);

    constexpr uint32_t num_output_tiles = 1;
    // Wait for all output tiles to be available in CB
    // Note: output_cb is backed by sharded tensor, data will be written directly to L1
    cb_wait_front(output_cb, 1);
}
