// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFlash2 rung-3 owner core, compute: the owner phases' arithmetic (df2_mk_common.hpp), fp32 DST, HiFi4:
//   RMS / RMS_AR  P = rowsum(x0^2 + x1^2) -> CB_P; inv = rsqrt(rowsum(G0 + G1) / H + eps) (column 0);
//                 y_j = x_j * inv (column broadcast) * w_j (row 0 broadcast) -> bf16 (kernels/head/head_c.cpp)
//   CONV / FIN    the rung-2 fused conv body per column tile (kernels/spec/df2_conv_compute.cpp: exact selection
//                 matmuls into DST, fp32 SFPU products / sums); FIN adds the residual tile before the pack
//   HEADS         a q / k head: inv = rsqrt(rowsum(sum_j x_j^2) / hd + eps); hn_j = x_j * inv * w_j -> bf16;
//                 RoPE (rotate-half over the HT tiles: out_j = hn_j * cos_j + s_j * hn_{j ^ HT/2} * sin_j, s_j = -1 for
//                 j < HT / 2) -> bf16
//   SWI           m_j = up_j * silu(gate_j) -> bf16
// Every fp32 CB read by copy_tile (W, G, HW, INV) is unpacked straight to DST (UnpackToDestFp32, set by the host).
// Runtime args: 0 k (the owner index: which head / which m tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spec/df2_mk_common.hpp"

using namespace df2mk;

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(CT_NPH);
constexpr uint32_t NQH = get_compile_time_arg_val(CT_NQH);
constexpr uint32_t NKVH = get_compile_time_arg_val(CT_NKVH);
constexpr uint32_t HT = get_compile_time_arg_val(CT_HT);
constexpr uint32_t IT = get_compile_time_arg_val(CT_IT);
constexpr uint32_t NTS = get_compile_time_arg_val(CT_NTS);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(CT_EPS);
constexpr uint32_t INV_H_BITS = get_compile_time_arg_val(CT_INVH);
constexpr uint32_t INV_HD_BITS = get_compile_time_arg_val(CT_INVHD);
constexpr uint32_t NH_ALL = NQH + 2 * NKVH;
static_assert(HT % 2 == 0, "rotate-half pairs whole tiles");

// dst[d] <- tile i of cb (exact fp32 for UnpackToDestFp32 CBs; bf16 CBs unpack exactly into the fp32 DST)
FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, i, d);
}

FORCE_INLINE void add_dst(uint32_t a, uint32_t b) {  // dst[a] += dst[b]
    add_binary_tile_init();
    add_binary_tile(a, b, a);
}

FORCE_INLINE void mul_dst(uint32_t a, uint32_t b) {  // dst[a] *= dst[b]
    mul_binary_tile_init();
    mul_binary_tile(a, b, a);
}

FORCE_INLINE void square_dst(uint32_t d) {
    square_tile_init();
    square_tile(d);
}

FORCE_INLINE void row_sum(uint32_t d) {  // column 0 of dst[d] <- the row sums of dst[d]
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(d);
}

// column 0 of dst[d] (per-row sums of squares) -> rsqrt(sum * inv_n + eps)
FORCE_INLINE void rms_inv(uint32_t d, uint32_t inv_n_bits) {
    row_sum(d);
    binop_with_scalar_tile_init();
    mul_unary_tile(d, inv_n_bits);
    add_unary_tile(d, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(d);
}

FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}

// ------------------------------------------------------------------ RMS
FORCE_INLINE void c_rms() {
    cb_wait_front(CB_X, 2);
    cb_wait_front(CB_W, 2);
    cb_reserve_back(CB_P, 1);
    tile_regs_acquire();
    load(CB_X, 0, 0);
    square_dst(0);
    load(CB_X, 1, 1);
    square_dst(1);
    add_dst(0, 1);
    row_sum(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_P);
    tile_regs_release();
    cb_push_back(CB_P, 1);

    cb_wait_front(CB_G, 2);
    cb_reserve_back(CB_OUT, 2);
    tile_regs_acquire();
    load(CB_G, 0, 2);
    load(CB_G, 1, 3);
    add_dst(2, 3);
    rms_inv(2, INV_H_BITS);
    load(CB_X, 0, 0);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 2);
    load(CB_W, 0, 1);
    sfpu_mul_bcast_row_init();
    sfpu_mul_bcast_row(0, 1);
    load(CB_X, 1, 3);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(3, 2);
    load(CB_W, 1, 1);
    sfpu_mul_bcast_row_init();
    sfpu_mul_bcast_row(3, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_OUT);
    pack_to(3, CB_OUT);
    tile_regs_release();
    cb_push_back(CB_OUT, 2);
    cb_pop_front(CB_X, 2);
    cb_pop_front(CB_W, 2);
    cb_pop_front(CB_G, 2);
}

// ------------------------------------------------------------------ CONV / FIN (the rung-2 compute body)
FORCE_INLINE void zero_dst(uint32_t d) {
    copy_init(CB_ZERO);
    copy_tile(CB_ZERO, 0, d);
}

// dst[d] += CF[tap] @ E[tap] + J[tap] @ B   (the coefficient tile of tap t; the J CB holds k0 at 0, k1 at 1)
FORCE_INLINE void coef_into(uint32_t d, uint32_t tap) {
    matmul_init(CB_CF, CB_E);
    matmul_tiles(CB_CF, CB_E, tap, tap, d);
    matmul_init(CB_J, CB_B);
    matmul_tiles(CB_J, CB_B, tap, 0, d);
}

FORCE_INLINE void c_conv(bool fin) {
    reconfig_data_format<SrcOrder::Reverse>(CB_CF, CB_E);  // every conv CB is bf16 (the matmul operand formats)
    pack_reconfig_data_format(CB_OUT);
    cb_wait_front(CB_J, 2);
    cb_wait_front(CB_SH, 1);
    cb_wait_front(CB_ZERO, 1);
    for (uint32_t i = 0; i < 2; ++i) {
        cb_wait_front(CB_X, 1);
        cb_wait_front(CB_CF, 2);
        cb_wait_front(CB_E, 2);
        cb_wait_front(CB_B, 1);
        cb_reserve_back(CB_OUT, 1);
        tile_regs_acquire();
        zero_dst(0);
        zero_dst(1);
        coef_into(1, 1);
        matmul_init(CB_SH, CB_X);
        matmul_tiles(CB_SH, CB_X, 0, 0, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        zero_dst(1);
        coef_into(1, 0);
        copy_init(CB_X);
        copy_tile(CB_X, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(2, 1, 2);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        if (fin) {
            cb_wait_front(CB_RES, 1);
            copy_init(CB_RES);
            copy_tile(CB_RES, 0, 3);
            add_binary_tile_init();
            add_binary_tile(0, 3, 0);  // out = residual + conv (fp32), packed bf16
            cb_pop_front(CB_RES, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_X, 1);
        cb_pop_front(CB_CF, 2);
        cb_pop_front(CB_E, 2);
        cb_pop_front(CB_B, 1);
    }
    cb_pop_front(CB_J, 2);
    cb_pop_front(CB_SH, 1);
    cb_pop_front(CB_ZERO, 1);
}

// ------------------------------------------------------------------ HEADS (q / k heads of owners k < NQH + NKVH)
FORCE_INLINE void c_heads(uint32_t k, uint32_t flag) {
    if (k >= NQH + NKVH || (k < NQH && ((flag >> 9) & 1u))) {
        return;  // a v head (no arithmetic), an idle owner, or a q head of the ctx program (not needed)
    }
    cb_wait_front(CB_QKV, HT);
    cb_wait_front(CB_HW, HT);
    cb_wait_front(CB_COS, HT);
    cb_wait_front(CB_SIN, HT);
    // inv = rsqrt(rowsum(sum_j x_j^2) / hd + eps)
    cb_reserve_back(CB_INV, 1);
    tile_regs_acquire();
    load(CB_QKV, 0, 0);
    square_dst(0);
    for (uint32_t j = 1; j < HT; ++j) {
        load(CB_QKV, j, 1);
        square_dst(1);
        add_dst(0, 1);
    }
    rms_inv(0, INV_HD_BITS);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_INV);
    tile_regs_release();
    cb_push_back(CB_INV, 1);
    // hn_j = x_j * inv * w_j -> bf16
    cb_wait_front(CB_INV, 1);
    cb_reserve_back(CB_HN, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        tile_regs_acquire();
        load(CB_QKV, j, 0);
        load(CB_INV, 0, 1);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        load(CB_HW, j, 1);
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_HN);
        tile_regs_release();
    }
    cb_push_back(CB_HN, HT);
    cb_pop_front(CB_QKV, HT);
    cb_pop_front(CB_HW, HT);
    cb_pop_front(CB_INV, 1);
    // RoPE: out_j = hn_j * cos_j + s_j * hn_{partner} * sin_j -> CB_OUT (q heads: the BRISC) / CB_KOUT (k heads: the
    // NCRISC's cache write; one consumer RISC per CB)
    const uint32_t cb_o = k < NQH ? CB_OUT : CB_KOUT;
    cb_wait_front(CB_HN, HT);
    cb_reserve_back(cb_o, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        const uint32_t half = HT / 2;
        const uint32_t partner = j < half ? j + half : j - half;
        tile_regs_acquire();
        load(CB_HN, j, 0);
        load(CB_COS, j, 1);
        mul_dst(0, 1);
        load(CB_HN, partner, 2);
        load(CB_SIN, j, 3);
        mul_dst(2, 3);
        if (j < half) {
            negative_tile_init();
            negative_tile(2);
        }
        add_dst(0, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_o);
        tile_regs_release();
    }
    cb_push_back(cb_o, HT);
    cb_pop_front(CB_HN, HT);
    cb_pop_front(CB_COS, HT);
    cb_pop_front(CB_SIN, HT);
}

// ------------------------------------------------------------------ SWI
FORCE_INLINE void c_swi(uint32_t k) {
    if (k * NTS >= IT) {
        return;
    }
    cb_wait_front(CB_GU, 2 * NTS);
    cb_reserve_back(CB_OUT, NTS);
    for (uint32_t j = 0; j < NTS; ++j) {
        tile_regs_acquire();
        load(CB_GU, NTS + j, 0);  // gate
        silu_tile_init();
        silu_tile(0);
        load(CB_GU, j, 1);  // up
        mul_dst(1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(1, CB_OUT);
        tile_regs_release();
    }
    cb_push_back(CB_OUT, NTS);
    cb_pop_front(CB_GU, 2 * NTS);
}

template <uint32_t P>
FORCE_INLINE void run(uint32_t k) {
    if constexpr (P < NPH) {
        constexpr uint32_t TYPE = get_compile_time_arg_val(CT_PH0 + 2 * P);
        constexpr uint32_t FLAG = get_compile_time_arg_val(CT_PH0 + 2 * P + 1);
        if constexpr (TYPE == PH_RMS || TYPE == PH_RMS_AR) {
            c_rms();
        } else if constexpr (TYPE == PH_CONV) {
            c_conv(false);
        } else if constexpr (TYPE == PH_FIN) {
            c_conv(true);
        } else if constexpr (TYPE == PH_HEADS) {
            c_heads(k, FLAG);
        } else {
            c_swi(k);
        }
        mk::cb_rewind_range(0, N_CBS);  // the pack / unpack pointers back to the base for the next phase's chunks
        run<P + 1>(k);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t k = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_CF, CB_E, CB_OUT);
    run<0>(k);
}
