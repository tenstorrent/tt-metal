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
    constexpr uint32_t block_size = get_named_compile_time_arg_val("block_size");

    compute_kernel_hw_startup(in_cb, out_cb);
    tilize_init(in_cb, block_size, out_cb);

    tilize_block(in_cb, block_size, out_cb);
    cb_pop_front(in_cb, block_size);
#endif
}
