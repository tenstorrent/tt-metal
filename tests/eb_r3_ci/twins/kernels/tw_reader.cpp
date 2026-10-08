// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twins: the input CBs sit on L1 tensors (globally allocated), so feeding them is a push without a
// copy. Runtime args: iterations, number of CBs, max pushes per iteration, then per CB (cb, pages per push, pushes per
// iteration); each iteration pushes round-robin over the CBs, push p of every CB that has one.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t iters = get_arg_val<uint32_t>(0);
    const uint32_t num_cbs = get_arg_val<uint32_t>(1);
    const uint32_t max_pushes = get_arg_val<uint32_t>(2);
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t p = 0; p < max_pushes; ++p) {
            for (uint32_t i = 0; i < num_cbs; ++i) {
                const uint32_t cb = get_arg_val<uint32_t>(3 + 3 * i);
                const uint32_t pages = get_arg_val<uint32_t>(4 + 3 * i);
                const uint32_t pushes = get_arg_val<uint32_t>(5 + 3 * i);
                if (p < pushes) {
                    cb_reserve_back(cb, pages);
                    cb_push_back(cb, pages);
                }
            }
        }
    }
}
