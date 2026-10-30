// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// transpose_split compute: one tile transpose per item; tail items (>= dx_items) are scaled by tail_scale.

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reg_api.h"
#include "api/compute/transpose.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_out = 1;
constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Tt = get_compile_time_arg_val(1);
constexpr uint32_t rows_t = get_compile_time_arg_val(2);
constexpr uint32_t dx_items = get_compile_time_arg_val(3);
constexpr uint32_t scale_bits = get_compile_time_arg_val(4);
constexpr uint32_t one_bits = 0x3F800000U;

void kernel_main() {
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        transpose_init(cb_in);
        transpose_tile(cb_in, 0, 0);
        if constexpr (scale_bits != one_bits) {
            if (work >= dx_items) {
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scale_bits);
            }
        }
        tile_regs_commit();
        pack_and_push(0, cb_out);
        cb_pop_front(cb_in, 1);
    }
}
