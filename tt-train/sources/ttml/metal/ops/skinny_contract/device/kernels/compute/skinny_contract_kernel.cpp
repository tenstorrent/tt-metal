// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// skinny_contract compute: per item the `cols` output tiles (0, n0 + j) = sum_t small_t[0, t] @ big[t, n0 + j]
// accumulate in DEST regs 0..cols-1 (fp32, HiFi4), then * scale, transposed in DEST when TRANSPOSE_OUT, packed bf16.

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "api/compute/transpose_dest.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t cb_small = 0;
constexpr uint32_t cb_big = 1;
constexpr uint32_t cb_out = 2;
constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Nt = get_compile_time_arg_val(2);
constexpr uint32_t cols = get_compile_time_arg_val(3);
constexpr uint32_t prefetch = get_compile_time_arg_val(5);  // K rows per reader batch
constexpr uint32_t scale_bits = get_compile_time_arg_val(4);
constexpr uint32_t one_bits = 0x3F800000U;

void kernel_main() {
    compute_kernel_hw_startup(cb_small, cb_big, cb_out);
    cb_wait_front(cb_small, Kt);
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t n0 = (work_start + item) * cols;
        const uint32_t ncols = (n0 + cols <= Nt) ? cols : (Nt - n0);
        tile_regs_acquire();
#ifdef BIG_TRANSPOSED
        constexpr uint32_t big_tr = 1;  // the big tiles arrive as (n, t): the unpacker transposes in1 for free
#else
        constexpr uint32_t big_tr = 0;
#endif
        matmul_init(cb_small, cb_big, big_tr);
        for (uint32_t t0 = 0; t0 < Kt; t0 += prefetch) {
            const uint32_t rows = (t0 + prefetch <= Kt) ? prefetch : (Kt - t0);
            cb_wait_front(cb_big, prefetch * cols);
            for (uint32_t t = t0; t < t0 + rows; ++t) {
                for (uint32_t j = 0; j < ncols; ++j) {
                    matmul_tiles(cb_small, cb_big, t, (t - t0) * cols + j, j);  // in1 transpose set by the init
                }
            }
            cb_pop_front(cb_big, prefetch * cols);
        }
        if constexpr (scale_bits != one_bits) {
            binop_with_scalar_tile_init();
            for (uint32_t j = 0; j < ncols; ++j) {
                mul_unary_tile(j, scale_bits);
            }
        }
#ifdef TRANSPOSE_OUT
        transpose_dest_init</*is_32bit*/ true>(cb_small);
        for (uint32_t j = 0; j < ncols; ++j) {
            transpose_dest</*is_32bit*/ true>(j);
        }
#endif
        tile_regs_commit();
        cb_reserve_back(cb_out, cols);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        for (uint32_t j = 0; j < cols; ++j) {  // a shorter last item pads to a full block
            pack_tile(j < ncols ? j : 0, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols);
    }
    cb_pop_front(cb_small, Kt);
}
