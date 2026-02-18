// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Tilize 16x32 kernel - matches fast_tilize_test.cpp scheme.
 *
 * Processes BLOCK_CT_DIM tiles in one row. fast_tilize_init(full_dim) and
 * fast_tilize_block(block) use the same dest bank algorithm as fast_tilize_test:
 * - remaining_tiles > 2*dest_size: process dest_size tiles
 * - remaining_tiles > dest_size: split evenly (even_remainder)
 * - else: handle 3-tile odd case or single sequence (not yet implemented for 16xN)
 */

#include "../../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/tilize.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t block_ct_dim = get_named_compile_time_arg_val("block_ct_dim");

    compute_kernel_hw_startup(in_cb, out_cb);
    fast_tilize_init(in_cb, block_ct_dim, out_cb);
    fast_tilize_block(in_cb, block_ct_dim, out_cb, 0, 0);
    cb_pop_front(in_cb, block_ct_dim);
    fast_tilize_uninit(in_cb, out_cb);
#endif
}
