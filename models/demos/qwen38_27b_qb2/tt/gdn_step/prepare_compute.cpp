// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "compute_math.hpp"

// Identical FP32 math to the fused path, executed once per shared Q/K head.
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 1, 11);
    for (uint32_t item = 0; item < count; ++item) {
        cb_wait_front(0, 4);
        cb_wait_front(1, 4);
        normalize(0, 11, true);
        normalize(1, 12, false);
        cb_pop_front(0, 4);
        cb_pop_front(1, 4);
        // The writer owns consumption of normalized CB11/12.
    }
}
