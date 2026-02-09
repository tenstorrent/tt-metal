// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/tilize.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t num_blocks = get_named_compile_time_arg_val("num_blocks");
    constexpr uint32_t block_size = get_named_compile_time_arg_val("block_size");

    compute_kernel_hw_startup(in_cb, out_cb);
    tilize_init(in_cb, block_size, out_cb);

    for (uint32_t i = 0; i < num_blocks; i++) {
        // DPRINT << "BEFORE WAIT FRONT" << ENDL();
        // cb_wait_front(in_cb, block_size);
        // DPRINT << "AFTER WAIT FRONT" << ENDL();
        // cb_reserve_back(out_cb, block_size);
        // DPRINT << "AFTER RESERVE BACK" << ENDL();
        tilize_block(in_cb, block_size, out_cb);
        // DPRINT << "AFTER TILIZE BLOCK" << ENDL();
        // cb_push_back(out_cb, block_size);
        // DPRINT << "AFTER PUSH BACK" << ENDL();
        cb_pop_front(in_cb, block_size);
    }
#endif
}
