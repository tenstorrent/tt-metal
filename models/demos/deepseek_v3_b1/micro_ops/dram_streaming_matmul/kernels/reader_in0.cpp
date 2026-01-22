// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

/**
 * Simple in0 reader kernel for replicated input.
 *
 * CB0 is backed directly by the input tensor (replicated on all compute cores).
 * This kernel just signals that all K tiles are ready - no data movement needed.
 */
void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(1);

    // CB0 is backed by the input tensor - data is already there.
    // Signal that all K tiles are ready for compute.
    cb_push_back(cb_id_in0, num_tiles_k);
}
