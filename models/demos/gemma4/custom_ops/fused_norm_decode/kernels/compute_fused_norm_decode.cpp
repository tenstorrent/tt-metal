// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for FusedNormDecode. Per input i: x_i^2 of this core's `per` tiles (FPU, fp32 dest, one acquire) ->
// cb_sq; their row sums (FPU row reduce with the unit scaler) -> one fp32 partial tile in cb_part (column 0).
// After the exchange (reader: cb_recv tile i*RT + t holds the N cores' partials of input i in its columns): row
// reduce over the RT tiles -> the full row sums; inv_i = rsqrt(sum / W + eps) (SFPU) -> cb_inv (fp32, column 0).
// Per output o, one acquire over the `per` tiles: dst[c] = x0[c] * inv_0 (FPU column broadcast), *= gamma_o[c]
// (FPU, dest reused as srcB; the gamma tiles carry the row replicated), [NIN == 2: dst[per + c] = x1[c] * inv_1 *
// gamma_1[c]; dst[c] += dst[per + c] (SFPU)], [+= residual[c] (FPU dest reuse)], [*= scalar_o (SFPU)], one bf16
// pack per tile. Every SFPU op of a window runs after its last FPU write (an FPU write to a dest tile an earlier
// SFPU op of the window read corrupts it).
// compile-time args: [cb_x0, cb_x1, cb_g0, cb_g1, cb_g2, cb_res, cb_scaler, cb_sq, cb_part, cb_recv, cb_inv,
//                     cb_out0, cb_out1, cb_out2, per, N, RT, NIN, NOUT, has_res, gmask]
// runtime args (every core): [eps_bits, inv_w_bits, scalar0_bits, scalar1_bits, scalar2_bits]. The model's eps /
// 1/W and the layer's scalars are runtime values on purpose: the learned layer scalar differs per layer, and as a
// compile-time arg it made every layer its own JIT build (3 TRISC compiles per layer on the cold decode and
// prefill paths); one binary per form and split serves all layers.
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

constexpr uint32_t CB_X0 = get_compile_time_arg_val(0);
constexpr uint32_t CB_X1 = get_compile_time_arg_val(1);
constexpr uint32_t CB_G0 = get_compile_time_arg_val(2);
constexpr uint32_t CB_G1 = get_compile_time_arg_val(3);
constexpr uint32_t CB_G2 = get_compile_time_arg_val(4);
constexpr uint32_t CB_RES = get_compile_time_arg_val(5);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(6);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(7);
constexpr uint32_t CB_PART = get_compile_time_arg_val(8);
constexpr uint32_t CB_RECV = get_compile_time_arg_val(9);
constexpr uint32_t CB_INV = get_compile_time_arg_val(10);
constexpr uint32_t CB_OUT0 = get_compile_time_arg_val(11);
constexpr uint32_t CB_OUT1 = get_compile_time_arg_val(12);
constexpr uint32_t CB_OUT2 = get_compile_time_arg_val(13);
constexpr uint32_t PER = get_compile_time_arg_val(14);
constexpr uint32_t N = get_compile_time_arg_val(15);
constexpr uint32_t RT = get_compile_time_arg_val(16);
constexpr uint32_t NIN = get_compile_time_arg_val(17);
constexpr uint32_t NOUT = get_compile_time_arg_val(18);
constexpr bool HAS_RES = get_compile_time_arg_val(19) == 1;
constexpr uint32_t GMASK = get_compile_time_arg_val(20);
constexpr uint32_t ONE_BITS = 0x3F800000u;
constexpr uint32_t CB_X[2] = {CB_X0, CB_X1};
constexpr uint32_t CB_G[3] = {CB_G0, CB_G1, CB_G2};
constexpr uint32_t CB_OUT[3] = {CB_OUT0, CB_OUT1, CB_OUT2};

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

void kernel_main() {
    const uint32_t eps_bits = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t scalar_bits[3] = {get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    compute_kernel_hw_startup(CB_X0, CB_SCALER, CB_OUT0);
    CircularBuffer cx0(CB_X0), cx1(CB_X1), cr(CB_RECV), ci(CB_INV), cs(CB_SCALER);
    cx0.wait_front(PER);
    partial_row_sums(CB_X0);
    if constexpr (NIN == 2) {
        cx1.wait_front(PER);
        partial_row_sums(CB_X1);
    }
    // the exchanged partials -> inv per input
    cr.wait_front(NIN * RT);
    ci.reserve_back(NIN);
    for (uint32_t i = 0; i < NIN; ++i) {
        reconfig_data_format(CB_SCALER, CB_RECV);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_RECV, CB_SCALER, CB_INV);
        tile_regs_acquire();
        for (uint32_t t = 0; t < RT; ++t) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_RECV, CB_SCALER, i * RT + t, 0, 0);
        }
        reduce_uninit(CB_RECV);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_w_bits);
        add_unary_tile(0, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile<RsqrtMode::Default>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_INV);
        pack_tile(0, CB_INV, i);
        tile_regs_release();
    }
    ci.push_back(NIN);
    cr.pop_front(NIN * RT);
    ci.wait_front(NIN);
    constexpr uint32_t n_gamma = NIN == 2 ? 2 : NOUT;
    CircularBuffer cg0(CB_G0), cg1(CB_G1), cg2(CB_G2), cres(CB_RES);
    if constexpr ((GMASK & 1) != 0) {
        cg0.wait_front(PER);
    }
    if constexpr ((GMASK & 2) != 0 && n_gamma > 1) {
        cg1.wait_front(PER);
    }
    if constexpr ((GMASK & 4) != 0 && n_gamma > 2) {
        cg2.wait_front(PER);
    }
    if constexpr (HAS_RES) {
        cres.wait_front(PER);
    }
    for (uint32_t o = 0; o < NOUT; ++o) {
        const uint32_t cb_out = CB_OUT[o];
        CircularBuffer co(cb_out);
        tile_regs_acquire();
        // x0 * inv_0 (column broadcast of the fp32 inv tile) -> dst 0..PER-1
        reconfig_data_format(CB_X0, CB_INV);
        mul_bcast_cols_init(CB_X0, CB_INV);
        for (uint32_t c = 0; c < PER; ++c) {
            mul_tiles_bcast_cols(CB_X0, CB_INV, c, 0, c);
        }
        const uint32_t g_first = (NIN == 2) ? 0 : o;
        if ((GMASK >> g_first) & 1) {
            const uint32_t cb_g = CB_G[g_first];
            reconfig_data_format_srca(cb_g);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_g);
            for (uint32_t c = 0; c < PER; ++c) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_g, c, c);
            }
        }
        if constexpr (NIN == 2) {
            reconfig_data_format(CB_X1, CB_INV);
            mul_bcast_cols_init(CB_X1, CB_INV);
            for (uint32_t c = 0; c < PER; ++c) {
                mul_tiles_bcast_cols(CB_X1, CB_INV, c, 1, PER + c);
            }
            if constexpr ((GMASK & 2) != 0) {
                reconfig_data_format_srca(CB_G1);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_G1);
                for (uint32_t c = 0; c < PER; ++c) {
                    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_G1, c, PER + c);
                }
            }
        }
        if constexpr (HAS_RES) {
            reconfig_data_format_srca(CB_RES);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_RES);
            for (uint32_t c = 0; c < PER; ++c) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_RES, c, c);
            }
        }
        // SFPU tail: the two-input sum and the scalar (after every FPU write of this window)
        if constexpr (NIN == 2) {
            add_binary_tile_init();
            for (uint32_t c = 0; c < PER; ++c) {
                add_binary_tile(c, PER + c, c);
            }
        }
        if (scalar_bits[o] != ONE_BITS) {
            binop_with_scalar_tile_init();
            for (uint32_t c = 0; c < PER; ++c) {
                mul_unary_tile(c, scalar_bits[o]);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        co.reserve_back(PER);
        for (uint32_t c = 0; c < PER; ++c) {
            pack_tile(c, cb_out, c);
        }
        co.push_back(PER);
        tile_regs_release();
    }
    ci.pop_front(NIN);
    cx0.pop_front(PER);
    if constexpr (NIN == 2) {
        cx1.pop_front(PER);
    }
    if constexpr ((GMASK & 1) != 0) {
        cg0.pop_front(PER);
    }
    if constexpr ((GMASK & 2) != 0 && n_gamma > 1) {
        cg1.pop_front(PER);
    }
    if constexpr ((GMASK & 4) != 0 && n_gamma > 2) {
        cg2.pop_front(PER);
    }
    if constexpr (HAS_RES) {
        cres.pop_front(PER);
    }
    cs.pop_front(1);
}
