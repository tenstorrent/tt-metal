// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twins: the output CBs sit on L1 tensors (globally allocated), so draining them is a pop without a
// copy; the last iteration's pages stay in the tensors. Runtime args as tw_reader.cpp, with pops in place of pushes.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t iters = get_arg_val<uint32_t>(0);
    const uint32_t num_cbs = get_arg_val<uint32_t>(1);
    const uint32_t max_pops = get_arg_val<uint32_t>(2);
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t p = 0; p < max_pops; ++p) {
            for (uint32_t i = 0; i < num_cbs; ++i) {
                const uint32_t cb = get_arg_val<uint32_t>(3 + 3 * i);
                const uint32_t pages = get_arg_val<uint32_t>(4 + 3 * i);
                const uint32_t pops = get_arg_val<uint32_t>(5 + 3 * i);
                if (p < pops) {
                    cb_wait_front(cb, pages);
                    cb_pop_front(cb, pages);
                }
            }
        }
    }
}
