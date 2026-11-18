// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused cross-entropy backward: per row, the gradient s * softmax(x) with the target's entry
// s * expm1((x_t - M) - l), s times the row's upstream gradient with UPSTREAM_GRAD; with the vocabulary sharded across
// devices (VP_BACKWARD), from the combined row state. See row_softmax_program.hpp. The row statistics and the
// conventions of the steps are in row_softmax_compute.hpp.
#include "tt-train/sources/ttml/metal/ops/log_softmax/device/kernels/compute/row_softmax_compute.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Cross-entropy gradient: +0 logits x (overwritten), +1 row maximum m, +2 q = s * exp(-l), both broadcast
// across the row: dx = exp(x - m) * q = s * softmax(x). Tile padding holds -inf and gets zero.
inline void calculate_cross_entropy_grad() {
    constexpr uint32_t off_max = 1 * kDstSlotRows, off_q = 2 * kDstSlotRows;
#pragma GCC unroll 0
    for (int i = 0; i < 8; ++i) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat m = sfpi::dst_reg[off_max];
        const sfpi::vFloat e = log_softmax_exp(x - m);
        const sfpi::vFloat q = sfpi::dst_reg[off_q];
        sfpi::dst_reg[0] = e * q;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
namespace ckernel {
// Call log_softmax_online_init first.
ALWI void cross_entropy_grad(uint32_t dst) {
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_cross_entropy_grad, dst, VectorMode::RC));
}
}  // namespace ckernel

constexpr uint32_t chunks = get_compile_time_arg_val(0);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(1);  // s.
// Circular buffers (see row_softmax_program.cpp), passed to the row-statistics steps as their `Cbs`.
struct Cb {
    static constexpr uint32_t cx = 0, ccols = 2, cm = 3, cs = 4, cic = 5, ci = 6, cmax = 7, cpivot = 10, ctail = 11,
                              cd = 12, cnorm = 13, cout = 16, cpart0 = 17, cpart1 = 18, cpart2 = 19, cpivotgrad = 21,
                              cpartout = 23, ctargetx = 25, cupstream = 29;
    static constexpr uint32_t cfp32 = ccols;  // Any FP32 buffer, for source-A reconfigurations.
};

#ifdef UPSTREAM_GRAD
// Multiplies the column vector in DST tile `dst` by each row's upstream gradient g (cupstream, column 0), using DST
// tile dst + 1. Part of a step: the caller holds DST.
inline void multiply_by_upstream_tile(uint32_t dst) {
    copy_init(Cb::cupstream);
    copy_tile(Cb::cupstream, 0, dst + 1);  // dst[dst + 1] = g, the upstream gradient
    mul_binary_tile_init();
    mul_binary_tile(dst, dst + 1, dst);  // dst[dst] *= g
}
#endif

// q = s * exp(-l) (times g with UPSTREAM_GRAD), so that dx = exp(x - M) * q = s * softmax(x): cnorm -> cd.
inline void cross_entropy_scale_tile() {
    cb_wait_front(Cb::cnorm, 1);
#ifdef UPSTREAM_GRAD
    cb_wait_front(Cb::cupstream, 1);
#endif
    tile_regs_acquire();
    copy_init(Cb::cnorm);
    copy_tile(Cb::cnorm, 0, 0);  // dst0 = l
    negative_tile_init();
    negative_tile(0);  // dst0 = -l
    exp_tile_init<USE_APPROXIMATE_EXPONENTIAL>();
    exp_tile<USE_APPROXIMATE_EXPONENTIAL>(0);  // dst0 = exp(-l) = 1 / (1 + T)
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scaler_bits);  // dst0 = q = s * exp(-l)
#ifdef UPSTREAM_GRAD
    multiply_by_upstream_tile(0);  // dst0 = q * g
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cb::cd>(0);  // cd = q
    tile_regs_release();
}

// The target's gradient dx_t = s * (p_t - 1) = s * expm1(y_t) (times g), y_t = (x_t - M) - l: for a target at the
// selected maximum that is -s * T / (1 + T), the sum of the other gradients (arXiv:2610.31642, Eq. 7), without
// subtracting nearly equal values. ctargetx, cmax, cnorm -> cpivotgrad, which the writer substitutes. Pops ctargetx
// (and cupstream).
inline void target_gradient_tile() {
    cb_wait_front(Cb::ctargetx, 1);
    cb_wait_front(Cb::cmax, 1);
    tile_regs_acquire();
    copy_init(Cb::ctargetx);
    copy_tile(Cb::ctargetx, 0, 0);  // dst0 = x_t
    copy_init(Cb::cmax);
    copy_tile(Cb::cmax, 0, 1);  // dst1 = M
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = x_t - M
    copy_init(Cb::cnorm);
    copy_tile(Cb::cnorm, 0, 1);  // dst1 = l
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = y_t = (x_t - M) - l
    expm1_tile_init();
    expm1_tile(0);  // dst0 = p_t - 1 = expm1(y_t)
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scaler_bits);  // dst0 = s * (p_t - 1)
#ifdef UPSTREAM_GRAD
    multiply_by_upstream_tile(0);  // dst0 = s * (p_t - 1) * g
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cb::cpivotgrad>(0);  // cpivotgrad = dx_t, the target's gradient
    tile_regs_release();
    cb_pop_front(Cb::ctargetx, 1);
#ifdef UPSTREAM_GRAD
    cb_pop_front(Cb::cupstream, 1);
#endif
}

// dx = exp(x - M) * q over this unit's logits, with M (cm) and q (cs) broadcast across the row: cx -> cout. Pops
// each logits tile.
inline void cross_entropy_grad_tiles(uint32_t begin, uint32_t end) {
    cb_wait_front(Cb::cm, 1);
    cb_wait_front(Cb::cs, 1);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(Cb::cx, 1);
        tile_regs_acquire();
        copy_init(Cb::cm);
        copy_tile(Cb::cm, 0, 1);  // dst1 = M
        copy_init(Cb::cs);
        copy_tile(Cb::cs, 0, 2);  // dst2 = q
        reconfig_data_format_srca(Cb::cfp32, Cb::cx);
        copy_init(Cb::cx);
        copy_tile(Cb::cx, 0, 0);  // dst0 = x
        reconfig_data_format_srca(Cb::cx, Cb::cfp32);
        cross_entropy_grad(0);  // dst0 = dx = exp(x - M) * q
        tile_regs_commit();
        tile_regs_wait();
        pack_one_tile<Cb::cout>(0);  // cout = dx
        tile_regs_release();
        cb_pop_front(Cb::cx, 1);
    }
}

// One row of the cross-entropy gradient: dx = s * softmax(x), with the writer substituting the target's dx_t.
// Consumes the row statistics cmax, cnorm (and cpivot, which the vocabulary-parallel backward does not have).
inline void cross_entropy_grad_row(uint32_t begin, uint32_t end) {
    cross_entropy_scale_tile();                // cd = q
    broadcast_column_tile<Cb::cd, Cb::cs>(0);  // cs = q in every column
    cb_pop_front(Cb::cd, 1);
    target_gradient_tile();                      // cpivotgrad = dx_t
    broadcast_column_tile<Cb::cmax, Cb::cm>(0);  // cm = M in every column
    cb_pop_front(Cb::cmax, 1);
#ifndef VP_BACKWARD
    cb_pop_front(Cb::cpivot, 1);
#endif
    cb_pop_front(Cb::cnorm, 1);
    cross_entropy_grad_tiles(begin, end);  // cout = dx
    cb_pop_front(Cb::cm, 1);
    cb_pop_front(Cb::cs, 1);
}

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0), begin = get_arg_val<uint32_t>(1), end = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(Cb::cx, Cb::cx, Cb::cout);
    reconfig_data_format_srca(Cb::cx, Cb::cfp32);  // Steps start with source A set for the FP32 statistics.
    cb_wait_front(Cb::ccols, 1);
    for (uint32_t row = 0; row < count; ++row) {
        // VP_BACKWARD has no statistics pass: the reader hands over the combined row state (M, l) in cmax and cnorm.
#ifndef VP_BACKWARD
        row_statistics<Cb, chunks>(begin, end);  // cmax, cpivot, cnorm = M, K, l
#endif
        cross_entropy_grad_row(begin, end);
    }
    cb_pop_front(Cb::ccols, 1);
}
