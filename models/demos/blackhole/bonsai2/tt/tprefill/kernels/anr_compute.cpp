// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill (residual add +) RMSNorm + rotation (norm_rot.py), compute. This core: one token row tile (32 tokens),
// 1024-wide blocks b0 .. b1 - 1 of its width Kt tiles (PAIRED: the row's other blocks are on a partner core; the two
// add their sums of squares, exchanged by the readers, before the norm).
// A. HAS_ADD: h = a + b per tile (bf16, also streamed to the writer); else the reader put h = x in c_3 directly.
// B. Per token sum of squares: squares 8 tiles at a time (fp32), R += SQ @ ONES (R[r][c] = token r's sum for every c),
//    accumulated in L1 by the packer.
// C. inv = rsqrt(R / K + eps) (every column of a row holds its token's value).
// D. Per tile t: DEST 0 = h_t * inv, then the SFPU multiplies it by the norm weight with the rotation signs folded in
//    (w * d: the 4 lane vectors of column tile t from the vector tile in DEST 7, rows 4 (t % 8) + v) -> nd (bf16);
//    WRITE_N: DEST 1 = h_t * inv times w alone -> n, the normalised activation, to the writer.
// E. Per 1024-wide block: the factored Hadamard of hrot_compute.cpp on nd with the weight tiles +-H32 / 32 (d is
//    already in nd) -> x' to the writer.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

namespace {
constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 2, cb_h = 3, cb_sq = 4, cb_r = 5, cb_inv = 6, cb_nd = 7, cb_rp = 8;
constexpr uint32_t cb_out = 16, cb_n = 17, cb_hout = 18;
constexpr uint32_t ONES = 0, HP = 1, HN = 2, VEC = 3;  // const tiles; VEC: w * d vector tiles, then w vector tiles
ALWI void fmt(uint32_t a, uint32_t b) { reconfig_data_format(a, b); }
ALWI void fmt_mm(uint32_t a, uint32_t b) { reconfig_data_format<SrcOrder::Reverse>(a, b); }
}  // namespace

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Templates k = 0..3: LReg[16] = LREG(4 + k) * VD + 0; macro k loads a row into LREG0, multiplies it by vector k and
// stores it back (t2_compute.cpp).
inline void vec_macro_init() {
    TTI_SFPMUL(p_sfpu::LREG4, 0, p_sfpu::LCONST_0, 12, 0);
    TTI_SFPMUL(p_sfpu::LREG5, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(p_sfpu::LREG6, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_SFPMUL(p_sfpu::LREG7, 0, p_sfpu::LCONST_0, 15, 0);
    constexpr uint32_t store_bits = 0x40 | (2 << 3) | 3;
#define ANR_MACRO(k)                                                   \
    do {                                                               \
        constexpr uint32_t mad_bits = 0x80 | 0x40 | (4 + k);           \
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8));   \
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8)); \
        TTI_SFPCONFIG(0, 4 + k, 0);                                    \
    } while (0)
    ANR_MACRO(0);
    ANR_MACRO(1);
    ANR_MACRO(2);
    ANR_MACRO(3);
#undef ANR_MACRO
    TTI_SFPCONFIG(0x3F0, 8, 1);
    TTI_SFPNOP;
}
#define ANR_ROW(q) \
    TTI_SFPLOADMACRO(((2 * (((q) >> 3) & 1) + ((q) & 1)) << 2), sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_6, 0)
// DEST tile (base) *= the 4 vectors at rows 4 z .. 4 z + 3 of DEST tile src; the dest base is DEST tile `base`.
inline void vec_scale(uint32_t src_off, uint32_t z) {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    const uint32_t sa = src_off + 8 * z;
    TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 0);
    TT_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 2);
    TT_SFPLOAD(p_sfpu::LREG6, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 4);
    TT_SFPLOAD(p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 6);
    TTI_SFPNOP;
    ANR_ROW(0);
    ANR_ROW(1);
    ANR_ROW(2);
    ANR_ROW(3);
    ANR_ROW(4);
    ANR_ROW(5);
    ANR_ROW(6);
    ANR_ROW(7);
    ANR_ROW(8);
    ANR_ROW(9);
    ANR_ROW(10);
    ANR_ROW(11);
    ANR_ROW(12);
    ANR_ROW(13);
    ANR_ROW(14);
    ANR_ROW(15);
    ANR_ROW(16);
    ANR_ROW(17);
    ANR_ROW(18);
    ANR_ROW(19);
    ANR_ROW(20);
    ANR_ROW(21);
    ANR_ROW(22);
    ANR_ROW(23);
    ANR_ROW(24);
    ANR_ROW(25);
    ANR_ROW(26);
    ANR_ROW(27);
    ANR_ROW(28);
    ANR_ROW(29);
    ANR_ROW(30);
    ANR_ROW(31);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
#undef ANR_ROW
// In place over DEST tiles 0..7: tile c0 = sum_a0 (-1)^popcount(c0 & a0) tile a0 (hrot_compute.cpp).
inline void fwht8() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 1
    for (uint32_t q = 0; q < T; ++q) {
#pragma GCC unroll 2
        for (uint32_t h = 0; h < 8; h += 4) {
            const sfpi::vFloat a = sfpi::dst_reg[(h + 0) * T];
            const sfpi::vFloat b = sfpi::dst_reg[(h + 1) * T];
            const sfpi::vFloat c = sfpi::dst_reg[(h + 2) * T];
            const sfpi::vFloat d = sfpi::dst_reg[(h + 3) * T];
            const sfpi::vFloat s0 = a + b, s1 = a - b, s2 = c + d, s3 = c - d;
            sfpi::dst_reg[(h + 0) * T] = s0 + s2;
            sfpi::dst_reg[(h + 1) * T] = s1 + s3;
            sfpi::dst_reg[(h + 2) * T] = s0 - s2;
            sfpi::dst_reg[(h + 3) * T] = s1 - s3;
        }
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 4; ++i) {
            const sfpi::vFloat a = sfpi::dst_reg[i * T];
            const sfpi::vFloat b = sfpi::dst_reg[(i + 4) * T];
            sfpi::dst_reg[i * T] = a + b;
            sfpi::dst_reg[(i + 4) * T] = a - b;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

// DEST tile dst *= vectors z of the vector tile in DEST tile src
ALWI void scale_tile(uint32_t dst, uint32_t src, uint32_t z) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::vec_scale((src - dst) * 64, z);
    math::clear_dst_reg_addr();
#endif
}

void kernel_main() {
    const uint32_t Kt = get_common_arg_val<uint32_t>(0);
    const uint32_t rows = get_common_arg_val<uint32_t>(1);  // row tiles (one per core)
    const uint32_t inv_k = get_common_arg_val<uint32_t>(2), eps = get_common_arg_val<uint32_t>(3);
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1), b1 = get_arg_val<uint32_t>(2), paired = get_arg_val<uint32_t>(3);
    if (me >= rows) {
        return;
    }
    const uint32_t T0 = 32 * b0, Kown = 32 * (b1 - b0);  // this core's tiles: global T0 .. T0 + Kown - 1
    const uint32_t NV = Kt / 8;  // vector tiles per weight
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_c, cb_h, cb_h);

    // A. h
#ifdef HAS_ADD
    fmt(cb_a, cb_b);
    add_init(cb_a, cb_b);
    pack_reconfig_data_format(cb_h);
    for (uint32_t t = 0; t < Kown; ++t) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        tile_regs_acquire();
        add_tiles(cb_a, cb_b, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_h, 1);
        pack_tile(0, cb_h);
        cb_push_back(cb_h, 1);
        cb_reserve_back(cb_hout, 1);
        pack_tile(0, cb_hout);
        cb_push_back(cb_hout, 1);
        tile_regs_release();
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
#endif
    cb_wait_front(cb_h, Kown);
    cb_wait_front(cb_c, VEC + 2 * NV);

    // B. R = sum_t (h_t * h_t) @ ONES, accumulated in L1
    cb_reserve_back(cb_r, 1);
    for (uint32_t i = 0; i < Kown; i += 8) {
        const uint32_t n = Kown - i < 8 ? Kown - i : 8;
        tile_regs_acquire();
        fmt(cb_h, cb_h);
        mul_init(cb_h, cb_h);
        for (uint32_t k = 0; k < n; ++k) {
            mul_tiles(cb_h, cb_h, i + k, i + k, k);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sq);
        cb_reserve_back(cb_sq, n);
        for (uint32_t k = 0; k < n; ++k) {
            pack_tile<true>(k, cb_sq, k);
        }
        cb_push_back(cb_sq, n);
        tile_regs_release();
        cb_wait_front(cb_sq, n);
        tile_regs_acquire();
        fmt_mm(cb_sq, cb_c);
        matmul_init(cb_sq, cb_c);
        for (uint32_t k = 0; k < n; ++k) {
            matmul_tiles(cb_sq, cb_c, k, ONES, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_r);
        pack_reconfig_l1_acc(i > 0 ? 1 : 0);
        pack_tile<true>(0, cb_r, 0);
        pack_reconfig_l1_acc(0);
        tile_regs_release();
        cb_pop_front(cb_sq, n);
    }
    cb_push_back(cb_r, 1);

    // C. inv = rsqrt(R / K + eps)
    cb_wait_front(cb_r, 1);
    tile_regs_acquire();
    if (paired) {  // R = own + partner's partial sums (the partner's arrive in c_8)
        cb_wait_front(cb_rp, 1);
        fmt(cb_r, cb_rp);
        add_init(cb_r, cb_rp);
        add_tiles(cb_r, cb_rp, 0, 0, 0);
    } else {
        fmt(cb_r, cb_r);
        copy_init(cb_r);
        copy_tile(cb_r, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_k);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_inv);
    cb_reserve_back(cb_inv, 1);
    pack_tile(0, cb_inv);
    cb_push_back(cb_inv, 1);
    tile_regs_release();
    cb_pop_front(cb_r, 1);
    if (paired) {
        cb_pop_front(cb_rp, 1);
    }
    cb_wait_front(cb_inv, 1);

    // D + E per 1024-wide block
    for (uint32_t b = b0; b < b1; ++b) {
        // 4 tiles per DEST pass (DEST 0..3; the 8-tile group's vector tile in DEST 7): nd = h * inv * (w * d) and,
        // WRITE_N, a second pass n = h * inv * w
        for (uint32_t t0 = 32 * b; t0 < 32 * b + 32; t0 += 4) {
#ifdef WRITE_N
            for (uint32_t pass = 0; pass < 2; ++pass) {
#else
            for (uint32_t pass = 0; pass < 1; ++pass) {
#endif
                const uint32_t out_cb = pass == 0 ? cb_nd : cb_n;
                tile_regs_acquire();
                fmt(cb_h, cb_inv);
                mul_init(cb_h, cb_inv);
                for (uint32_t i = 0; i < 4; ++i) {
                    mul_tiles(cb_h, cb_inv, t0 - T0 + i, 0, i);
                }
                fmt(cb_c, cb_c);
                copy_init(cb_c);
                copy_tile(cb_c, VEC + pass * NV + t0 / 8, 7);
                MATH((ckernel::sfpu::vec_macro_init()));
                for (uint32_t i = 0; i < 4; ++i) {
                    scale_tile(i, 7, (t0 + i) % 8);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(out_cb);
                cb_reserve_back(out_cb, 4);
                for (uint32_t i = 0; i < 4; ++i) {
                    pack_tile(i, out_cb);
                }
                cb_push_back(out_cb, 4);
                tile_regs_release();
            }
        }
        // rotation of the block: per c1, DEST tile a0 = sum_a1 H4[c1, a1] nd_(8 a1 + a0) @ H32 / 32, then H8 over DEST
        cb_wait_front(cb_nd, 32);
        for (uint32_t c1 = 0; c1 < 4; ++c1) {
            tile_regs_acquire();
            fmt_mm(cb_nd, cb_c);
            matmul_init(cb_nd, cb_c);
            for (uint32_t a0 = 0; a0 < 8; ++a0) {
                for (uint32_t a1 = 0; a1 < 4; ++a1) {
                    matmul_tiles(cb_nd, cb_c, 8 * a1 + a0, (__builtin_popcount(c1 & a1) & 1) ? HN : HP, a0);
                }
            }
            MATH(({
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                ckernel::sfpu::fwht8();
                math::clear_dst_reg_addr();
            }));
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            for (uint32_t c0 = 0; c0 < 8; ++c0) {
                cb_reserve_back(cb_out, 1);
                pack_tile(c0, cb_out);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_nd, 32);
    }
    cb_pop_front(cb_inv, 1);
    cb_pop_front(cb_h, Kown);
    cb_pop_front(cb_c, VEC + 2 * NV);
}
