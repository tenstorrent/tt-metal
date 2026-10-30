// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "log_softmax_sfpu.h"

constexpr uint32_t count = get_compile_time_arg_val(0), wt = get_compile_time_arg_val(1);
constexpr uint32_t width = get_compile_time_arg_val(2), tw = get_compile_time_arg_val(3);
constexpr uint32_t cx = 0, cd = 1, ccols = 2, cmax = 3, cpivot = 4, cnorm = 5, csum = 6, cgmax = 7, cstateout = 8,
                   cstatein = 9, cout = 16;

inline void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
inline void fill(uint32_t dst, float value) {
    fill_tile_init();
    fill_tile(dst, value);
}
inline void columns(uint32_t dst, uint32_t col) {
    load(ccols, 0, dst);
    binop_with_scalar_tile_init();
    const float offset = static_cast<float>(col * tw);
    uint32_t bits;
    __builtin_memcpy(&bits, &offset, sizeof(bits));
    add_unary_tile(dst, bits);
}
inline void sub_col(uint32_t dst, uint32_t vec) {
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(dst, vec);
}
inline void mul_col(uint32_t dst, uint32_t vec) {
    sfpu_bcast_col_init();
    sfpu_mul_bcast_col(dst, vec);
}
inline void exp_value(uint32_t dst) {
    exp_tile_init<false>();
    exp_tile<false>(dst);
}
inline void pack_one(uint32_t reg, uint32_t cb) {
    cb_reserve_back(cb, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}
template <PoolType type>
inline void reduce_row(uint32_t dst) {
    sfpu_reduce_init<type, DataFormat::Float32>();
    sfpu_reduce<type, DataFormat::Float32, ReduceDim::REDUCE_ROW>(dst);
}
#ifdef BACKWARD
inline void gradient(uint32_t dst) {
    load(cx, 0, dst);
    load(cstatein, 0, 2);
    sub_col(dst, 2);
    load(cstatein, 2, 2);
    sub_col(dst, 2);
    exp_value(dst);
    load(csum, 0, 2);
    mul_col(dst, 2);
    load(cd, 0, 2);
    sub_binary_tile_init();
    sub_binary_tile(2, dst, dst);
}
#endif

void kernel_main() {
    compute_kernel_hw_startup(cx, cx, cout);
    cb_wait_front(ccols, 1);
    for (uint32_t row = 0; row < count; ++row) {
#ifndef BACKWARD
        // Pass 1: lane-wise maxima, followed by an accurate horizontal FP32 reduction.
        tile_regs_acquire();
        fill(0, -std::numeric_limits<float>::infinity());
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(cx, 1);
            load(cx, 0, 1);
            columns(2, col);
            log_softmax_mask<0>(1, width);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0);
            cb_pop_front(cx, 1);
        }
        reduce_row<PoolType::MAX>(0);
        pack_one(0, cmax);
        cb_wait_front(cmax, 1);
        // Pass 2: choose one first maximum; indices are FP32 rather than rounded BF16.
        tile_regs_acquire();
        fill(0, std::numeric_limits<float>::infinity());
        load(cmax, 0, 2);
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(cx, 1);
            load(cx, 0, 1);
            sub_col(1, 2);
            columns(3, col);
            log_softmax_mask<1, 2>(1, width);
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);
            cb_pop_front(cx, 1);
        }
        reduce_row<PoolType::MIN>(0);
        pack_one(0, cpivot);
        cb_wait_front(cpivot, 1);
        // Pass 3: never add the selected e^0=1 to the tail (arXiv:2610.31642, Eq. 5).
        tile_regs_acquire();
        fill(0, 0.F);
        load(cpivot, 0, 3);
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(cx, 1);
            load(cx, 0, 1);
            load(cmax, 0, 2);
            sub_col(1, 2);
            exp_value(1);
            columns(2, col);
            log_softmax_mask<2>(1, width);
            sub_col(2, 3);
            log_softmax_mask<3>(1, width);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            cb_pop_front(cx, 1);
        }
        reduce_row<PoolType::SUM>(0);
        log1p_tile_init<false>();
        log1p_tile<false>(0);
        pack_one(0, cnorm);
        cb_wait_front(cnorm, 1);
        tile_regs_acquire();
        load(cmax, 0, 0);
        load(cpivot, 0, 1);
        load(cnorm, 0, 2);
        cb_reserve_back(cstateout, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cstateout);
        for (uint32_t i = 0; i < 3; ++i) pack_tile(i, cstateout);
        tile_regs_release();
        cb_push_back(cstateout, 3);
        // Pass 4: emit native-precision output without a full-size FP32 intermediate.
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(cx, 1);
            tile_regs_acquire();
            load(cx, 0, 1);
            load(cmax, 0, 2);
            sub_col(1, 2);
            load(cnorm, 0, 2);
            sub_col(1, 2);
            pack_one(1, cout);
            cb_pop_front(cx, 1);
        }
        cb_pop_front(cmax, 1);
        cb_pop_front(cpivot, 1);
        cb_pop_front(cnorm, 1);
#else
        cb_wait_front(cstatein, 3);
        // Pass 1: sum upstream gradients in FP32.
        tile_regs_acquire();
        fill(0, 0.F);
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(cx, 1);
            cb_wait_front(cd, 1);
            load(cd, 0, 1);
            columns(2, col);
            log_softmax_mask<2>(1, width);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            cb_pop_front(cx, 1);
            cb_pop_front(cd, 1);
        }
        reduce_row<PoolType::SUM>(0);
        pack_one(0, csum);
        cb_wait_front(csum, 1);
        // Pass 2: sum only non-pivot input gradients before any output rounding.
        tile_regs_acquire();
        fill(0, 0.F);
        load(cstatein, 1, 3);
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(cx, 1);
            cb_wait_front(cd, 1);
            gradient(1);
            columns(2, col);
            log_softmax_mask<2>(1, width);
            sub_col(2, 3);
            log_softmax_mask<3>(1, width);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            cb_pop_front(cx, 1);
            cb_pop_front(cd, 1);
        }
        reduce_row<PoolType::SUM>(0);
        negative_tile_init();
        negative_tile(0);
        pack_one(0, cgmax);
        cb_wait_front(cgmax, 1);
        // Pass 3: reconstruct the pivot as -sum(other gradients), Eq. 7.
        for (uint32_t col = 0; col < wt; ++col) {
            cb_wait_front(cx, 1);
            cb_wait_front(cd, 1);
            tile_regs_acquire();
            gradient(1);
            columns(2, col);
            log_softmax_mask<2>(1, width);
            load(cstatein, 1, 3);
            sub_col(2, 3);
            load(cgmax, 0, 0);
            fill(3, 0.F);
            sfpu_bcast_col_init();
            sfpu_add_bcast_col(3, 0);
            log_softmax_mask<4>(1, width);
            pack_one(1, cout);
            cb_pop_front(cx, 1);
            cb_pop_front(cd, 1);
        }
        cb_pop_front(csum, 1);
        cb_pop_front(cgmax, 1);
        cb_pop_front(cstatein, 3);
#endif
    }
    cb_pop_front(ccols, 1);
}
