// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for FusedNormChainDecode: two chained decode-row RMSNorms on this core's `per` column tiles.
// Stage A (as FusedNormDecode's residual form): x^2 (FPU, fp32 dest) -> cb_sq -> FPU row reduce -> partial tile
// (cb_part); after the exchange (cb_recv) the full row sums -> inv1 = rsqrt(sum / W + eps) (SFPU) -> cb_inv;
// h = (x * inv1 (FPU column broadcast) * g1 (FPU, dest reused) + residual (FPU, dest reused)) * scalar (SFPU),
// packed once to bf16 into cb_h (the stage-B input: the chain's own bf16 rounding of h) and once into cb_out0.
// Stage B: h^2 -> partial (cb_part again) -> exchange (cb_recv2) -> inv2 -> y = h * inv2 * g2 -> cb_out1.
// Every SFPU op of a window runs after its last FPU write.
// compile-time args: [cb_x, cb_g1, cb_g2, cb_res, cb_scaler, cb_sq, cb_part, cb_recv, cb_recv2, cb_inv, cb_h,
//                     cb_out0, cb_out1, per, N, RT, eps_bits, inv_w_bits, scalar_bits]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_G1 = get_compile_time_arg_val(1);
constexpr uint32_t CB_G2 = get_compile_time_arg_val(2);
constexpr uint32_t CB_RES = get_compile_time_arg_val(3);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(4);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(5);
constexpr uint32_t CB_PART = get_compile_time_arg_val(6);
constexpr uint32_t CB_RECV = get_compile_time_arg_val(7);
constexpr uint32_t CB_RECV2 = get_compile_time_arg_val(8);
constexpr uint32_t CB_INV = get_compile_time_arg_val(9);
constexpr uint32_t CB_H = get_compile_time_arg_val(10);
constexpr uint32_t CB_OUT0 = get_compile_time_arg_val(11);
constexpr uint32_t CB_OUT1 = get_compile_time_arg_val(12);
constexpr uint32_t PER = get_compile_time_arg_val(13);
constexpr uint32_t N = get_compile_time_arg_val(14);
constexpr uint32_t RT = get_compile_time_arg_val(15);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(16);
constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(17);
constexpr uint32_t SCALAR_BITS = get_compile_time_arg_val(18);
constexpr uint32_t ONE_BITS = 0x3F800000u;

// x^2 of the `per` tiles of cb_x (fp32) into cb_sq, then their row sums into one fp32 partial tile of cb_part
FORCE_INLINE void partial_row_sums(uint32_t cb_x) {
    CircularBuffer csq(CB_SQ), cp(CB_PART), cs(CB_SCALER);
    reconfig_data_format(cb_x, cb_x);
    mul_init(cb_x, cb_x);
    pack_reconfig_data_format(CB_SQ);
    csq.reserve_back(PER);
    tile_regs_acquire();
    for (uint32_t c = 0; c < PER; ++c) {
        mul_tiles(cb_x, cb_x, c, c, c);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t c = 0; c < PER; ++c) {
        pack_tile(c, CB_SQ, c);
    }
    tile_regs_release();
    csq.push_back(PER);
    csq.wait_front(PER);
    cs.wait_front(1);
    reconfig_data_format(CB_SCALER, CB_SQ);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, CB_PART);
    tile_regs_acquire();
    for (uint32_t c = 0; c < PER; ++c) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, c, 0, 0);
    }
    reduce_uninit(CB_SQ);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_PART);
    cp.reserve_back(1);
    pack_tile(0, CB_PART, 0);
    cp.push_back(1);
    tile_regs_release();
    csq.pop_front(PER);
}

// the exchanged partials of a round (cb_recv: the N cores' partials in the columns of RT tiles) -> inv (cb_inv)
FORCE_INLINE void full_row_inv(uint32_t cb_recv) {
    CircularBuffer cr(cb_recv), ci(CB_INV);
    cr.wait_front(RT);
    ci.reserve_back(1);
    reconfig_data_format(CB_SCALER, cb_recv);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_recv, CB_SCALER, CB_INV);
    tile_regs_acquire();
    for (uint32_t t = 0; t < RT; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_recv, CB_SCALER, t, 0, 0);
    }
    reduce_uninit(cb_recv);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_W_BITS);
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile<RsqrtMode::Default>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_INV);
    pack_tile(0, CB_INV, 0);
    ci.push_back(1);
    tile_regs_release();
    cr.pop_front(RT);
}

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_SCALER, CB_OUT0);
    CircularBuffer cx(CB_X), cg1(CB_G1), cg2(CB_G2), cres(CB_RES), ci(CB_INV), cs(CB_SCALER), ch(CB_H), co0(CB_OUT0),
        co1(CB_OUT1);
    cx.wait_front(PER);
    cg1.wait_front(PER);
    cg2.wait_front(PER);
    cres.wait_front(PER);
    // ---- stage A: h = (x * inv1 * g1 + residual) * scalar
    partial_row_sums(CB_X);
    full_row_inv(CB_RECV);
    ci.wait_front(1);
    tile_regs_acquire();
    reconfig_data_format(CB_X, CB_INV);
    mul_bcast_cols_init(CB_X, CB_INV);
    for (uint32_t c = 0; c < PER; ++c) {
        mul_tiles_bcast_cols(CB_X, CB_INV, c, 0, c);
    }
    reconfig_data_format_srca(CB_G1);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_G1);
    for (uint32_t c = 0; c < PER; ++c) {
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_G1, c, c);
    }
    reconfig_data_format_srca(CB_RES);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_RES);
    for (uint32_t c = 0; c < PER; ++c) {
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_RES, c, c);
    }
    if (SCALAR_BITS != ONE_BITS) {  // SFPU tail after every FPU write of the window
        binop_with_scalar_tile_init();
        for (uint32_t c = 0; c < PER; ++c) {
            mul_unary_tile(c, SCALAR_BITS);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_H);
    ch.reserve_back(PER);
    co0.reserve_back(PER);
    for (uint32_t c = 0; c < PER; ++c) {
        pack_tile(c, CB_H, c);
        pack_tile(c, CB_OUT0, c);
    }
    ch.push_back(PER);
    co0.push_back(PER);
    tile_regs_release();
    ci.pop_front(1);
    cx.pop_front(PER);
    cg1.pop_front(PER);
    cres.pop_front(PER);
    // ---- stage B: y = h * inv2 * g2 on the bf16 h
    ch.wait_front(PER);
    partial_row_sums(CB_H);
    full_row_inv(CB_RECV2);
    ci.wait_front(1);
    tile_regs_acquire();
    reconfig_data_format(CB_H, CB_INV);
    mul_bcast_cols_init(CB_H, CB_INV);
    for (uint32_t c = 0; c < PER; ++c) {
        mul_tiles_bcast_cols(CB_H, CB_INV, c, 0, c);
    }
    reconfig_data_format_srca(CB_G2);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_G2);
    for (uint32_t c = 0; c < PER; ++c) {
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_G2, c, c);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_OUT1);
    co1.reserve_back(PER);
    for (uint32_t c = 0; c < PER; ++c) {
        pack_tile(c, CB_OUT1, c);
    }
    co1.push_back(PER);
    tile_regs_release();
    ci.pop_front(1);
    ch.pop_front(PER);
    cg2.pop_front(PER);
    cs.pop_front(1);
}
