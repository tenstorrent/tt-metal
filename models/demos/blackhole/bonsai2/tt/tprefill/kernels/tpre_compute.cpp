// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Exact ternary prefill linear, compute. Y[m, n] = sum_kb s[n, kb] * (X'[m, kb] @ T[kb, n]) for 32-row tiles m, with
// T the ternary weights (2-bit Bfp2_b tiles with a forced exponent: exactly -1, 0, +1) and s the fp16 block scales.
// Per output tile pair (NB = 2 columns), per kb (128 = 4 K tiles): 4 LoFi matmuls (ternary in1 = SrcA needs 1 bit, so
// exact at any fidelity) into DEST P tiles, then the SFPU adds P * s (each column's scale, broadcast down the rows)
// into the fp32 accumulator tiles and zeroes P. The scales come from the records' inline fp16 scale tiles (one per 32
// kbs, kb j at tile row j), copied into DEST S tiles. Two kbs per SFPU pass. DEST (full sync, 8 fp32 tiles): acc 0-1
// (one per column), P 2-5 (kb A: 2-3, kb B: 4-5, one per column), S 6-7. RESIDENT: the whole row of x' tiles stays in
// c_0 for all column pairs; else (one pair) it streams in CH-tile chunks.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"  // silu_tile_init

constexpr uint32_t cb_x = tt::CBIndex::c_0;
constexpr uint32_t cb_w = tt::CBIndex::c_1;
constexpr uint32_t cb_s = tt::CBIndex::c_3;  // the fp16 scale tiles (declared Float32: srcA Tf32 holds fp16)
constexpr uint32_t cb_out = tt::CBIndex::c_16;

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// acc += P * (row J of S, broadcast down the rows); P = 0. SFPU geometry of an fp32 DEST tile (tests/probe4.py): face
// f is SFPU rows 8f..8f+7; SFPU row 8f+k holds face rows 4*(k/2)..+3 (one per 8-lane subvector) at the even (k even)
// or odd columns of the face. Row J of S: faces FB (columns 0-15) and FB+1 (16-31), subvector J % 4 of SFPU rows
// 8*face + R + parity. Loading that row into four registers and transposing their subvectors leaves register J % 4
// holding subvector J % 4 in all four subvectors: one column's scale for all four face rows.
// Two kbs at once (j, j + 1, j even), relative to the DEST base the caller sets to the column's acc tile: acc at 0, PA
// at +2 tiles, PB at +4, S at +6 (all indices compile-time: a runtime DEST index costs RISC-V work per access).
// acc += PA * s(j) + PB * s(j + 1); PA = PB = 0. Scales of kbs j and j + 1 are subvectors j % 4 and j % 4 + 1 of the
// same SFPU rows of S, so one load and one subvector transpose give both broadcasts.
template <int J>
inline void pre_scale2() {
    constexpr uint32_t T = 32;
    constexpr uint32_t FB = J < 16 ? 0 : 2;
    constexpr uint32_t RR = 2 * ((J % 16) / 4);
    constexpr uint32_t Q0 = J % 4;
#pragma GCC unroll 2
    for (uint32_t fc = 0; fc < 2; ++fc) {
#pragma GCC unroll 2
        for (uint32_t par = 0; par < 2; ++par) {
            sfpi::vFloat a = sfpi::dst_reg[6 * T + 8 * (FB + fc) + RR + par];
            sfpi::vFloat b = a, c = a, d = a;
            sfpi::subvec_transp(a, b, c, d);
            const sfpi::vFloat sa = Q0 == 0 ? a : c;
            const sfpi::vFloat sb = Q0 == 0 ? b : d;
#pragma GCC unroll 8
            for (uint32_t q = 0; q < 8; ++q) {
                const uint32_t r = 8 * (fc + 2 * (q / 4)) + 2 * (q % 4) + par;
                sfpi::vFloat v = sfpi::dst_reg[2 * T + r] * sa + sfpi::dst_reg[r];
                sfpi::dst_reg[r] = sfpi::dst_reg[4 * T + r] * sb + v;
#ifndef ZACC
                sfpi::dst_reg[2 * T + r] = 0.0f;
                sfpi::dst_reg[4 * T + r] = 0.0f;
#endif
            }
        }
    }
}
// GLU pair (gate in DEST tile 0, up in 1, the base): tile 1 = up * silu(gate) over the whole tile. Needs
// silu_tile_init.
inline void glu_full() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (uint32_t r = 0; r < T; ++r) {
        sfpi::vFloat g = sfpi::dst_reg[r];
        sfpi::vFloat u = sfpi::dst_reg[T + r];
        sfpi::dst_reg[T + r] = u * g * _sfpu_sigmoid_<true>(g);
    }
}
// OVERLAP: two DEST sets of 4 tiles (set s at tile 4s: acc, PA, PB, S); the SFPU scales one set while the FPU fills
// the other. Offsets relative to the set's base tile.
template <int J>
inline void pre_scale_set() {
    constexpr uint32_t T = 32;
    constexpr uint32_t FB = J < 16 ? 0 : 2;
    constexpr uint32_t RR = 2 * ((J % 16) / 4);
    constexpr uint32_t Q0 = J % 4;
#pragma GCC unroll 2
    for (uint32_t fc = 0; fc < 2; ++fc) {
#pragma GCC unroll 2
        for (uint32_t par = 0; par < 2; ++par) {
            sfpi::vFloat a = sfpi::dst_reg[3 * T + 8 * (FB + fc) + RR + par];
            sfpi::vFloat b = a, c = a, d = a;
            sfpi::subvec_transp(a, b, c, d);
            const sfpi::vFloat sa = Q0 == 0 ? a : c;
            const sfpi::vFloat sb = Q0 == 0 ? b : d;
#pragma GCC unroll 8
            for (uint32_t q = 0; q < 8; ++q) {
                const uint32_t r = 8 * (fc + 2 * (q / 4)) + 2 * (q % 4) + par;
                sfpi::vFloat v = sfpi::dst_reg[T + r] * sa + sfpi::dst_reg[r];
                sfpi::dst_reg[r] = sfpi::dst_reg[2 * T + r] * sb + v;
                sfpi::dst_reg[T + r] = 0.0f;
                sfpi::dst_reg[2 * T + r] = 0.0f;
            }
        }
    }
}
// set 0's acc += set 1's acc (base tile 0)
inline void add_sets() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (uint32_t r = 0; r < T; ++r) {
        const sfpi::vFloat a = sfpi::dst_reg[r];
        const sfpi::vFloat b = sfpi::dst_reg[4 * T + r];
        sfpi::dst_reg[r] = a + b;
    }
}
inline void zero8() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (uint32_t r = 0; r < 8 * T; ++r) {
        sfpi::dst_reg[r] = 0.0f;
    }
}
// Zero DEST tiles t0 .. t0 + 5 (acc and P of a column pair at the start of each output tile pair).
inline void zero6(uint32_t t0, uint32_t, uint32_t, uint32_t) {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (uint32_t r = 0; r < 6 * T; ++r) {
        sfpi::dst_reg[t0 * T + r] = 0.0f;
    }
}
}  // namespace ckernel::sfpu

#endif

// Column b's acc += PA * s(j) + PB * s(j + 1) (kbs j even and j + 1); DEST base = tile b for the SFPU pass.
#define PS2_CASE(J) \
    case J: ckernel::sfpu::pre_scale2<J>(); break;
ALWI void scale_pair(uint32_t j, uint32_t b) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(b);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    // clang-format off
    switch (j) {
        PS2_CASE(0) PS2_CASE(2) PS2_CASE(4) PS2_CASE(6) PS2_CASE(8) PS2_CASE(10) PS2_CASE(12) PS2_CASE(14)
        PS2_CASE(16) PS2_CASE(18) PS2_CASE(20) PS2_CASE(22) PS2_CASE(24) PS2_CASE(26) PS2_CASE(28) PS2_CASE(30)
        default: break;
    }
    // clang-format on
    math::clear_dst_reg_addr();
#endif
}

#define PSS_CASE(J) \
    case J: ckernel::sfpu::pre_scale_set<J>(); break;
// The SFPU pass of kb pair (j, j + 1) on DEST set `set`; its stall waits only for the FPU work issued before it.
ALWI void scale_set(uint32_t j, uint32_t set) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(4 * set);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    // clang-format off
    switch (j) {
        PSS_CASE(0) PSS_CASE(2) PSS_CASE(4) PSS_CASE(6) PSS_CASE(8) PSS_CASE(10) PSS_CASE(12) PSS_CASE(14)
        PSS_CASE(16) PSS_CASE(18) PSS_CASE(20) PSS_CASE(22) PSS_CASE(24) PSS_CASE(26) PSS_CASE(28) PSS_CASE(30)
        default: break;
    }
    // clang-format on
    math::clear_dst_reg_addr();
#endif
}
ALWI void sfpu_at0(void (*fn)()) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    fn();
    math::clear_dst_reg_addr();
#endif
}

// ZACC: the P tiles (DEST 2-5) cleared by the FPU's ZEROACC (one 16-row face per instruction, as the unary datacopy
// clears a tile) instead of SFPU stores, once the previous SFPU pass has read them.
ALWI void zero_p_tiles() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    for (uint32_t t = 2; t < 6; ++t) {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(t);
        for (uint32_t f = 0; f < 4; ++f) {
            TT_ZEROACC(p_zeroacc::CLR_16, 1, 0, ADDR_MOD_7, ((t & 3) << 2) + f);
        }
    }
#endif
}

ALWI void ternary_srca() {
    // the decode records' ternary tiles: Bfp2_b codes with no exponent section; the unpacker forces exponent 127
    UNPACK(({
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Bfp2_b, (uint32_t)DataFormat::Float16_b, 256 >> 4, FACE_R_DIM, 4);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
        cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(127);
    }));
}

ALWI void scale_tile_srca() {
    // A 32x32 IEEE fp16 tile (2048 B) in a CB declared Float32 (srcA Tf32, which holds fp16 exactly).
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            (uint32_t)DataFormat::Float16, (uint32_t)DataFormat::Tf32, 2048 >> 4, FACE_R_DIM, 4);
    }));
}

ALWI void matmul_mode(uint32_t ct) {
    reconfig_data_format_srca(cb_w);
    matmul_block_init(cb_x, cb_w, 0, ct, 1, 1);
    ternary_srca();
    addcmul_tile_init();
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t CH = get_compile_time_arg_val(2);
    constexpr bool RESIDENT = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t NB = get_compile_time_arg_val(4);  // columns per batch (2; 1 with OVERLAP)
    constexpr uint32_t G = (KB + 31) / 32;
    const uint32_t Mt = get_common_arg_val<uint32_t>(0);
    const uint32_t n_count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_w, cb_x, cb_out);  // SrcA = in1 (weights), SrcB = in0 (x'), as matmul_init
    matmul_mode(NB);
#ifdef OVERLAP
    static_assert(NB == 1, "OVERLAP takes one column per batch");
#endif
    if (n_count == 0) {  // keep the multicast flowing: every core takes every chunk
        for (uint32_t m = 0; m < Mt; ++m) {
            for (uint32_t c = 0; c < Kt / CH; ++c) {
                cb_wait_front(cb_x, CH);
                cb_pop_front(cb_x, CH);
            }
        }
        return;
    }
    cb_wait_front(cb_w, n_count * Kt);
    cb_wait_front(cb_s, n_count * G);
    for (uint32_t m = 0; m < Mt; ++m) {
        if constexpr (RESIDENT) {
            cb_wait_front(cb_x, Kt);
        }
#ifdef OVERLAP
        for (uint32_t n0 = 0; n0 < n_count; ++n0) {
            tile_regs_acquire();
            MATH((sfpu_at0(ckernel::sfpu::zero8)));
            uint32_t set = 0, prev_j = 0;
            bool pending = false;
            for (uint32_t g = 0; g < G; ++g) {
                MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));  // the old S tiles are free
                copy_init(cb_s);
                scale_tile_srca();
                copy_tile(cb_s, n0 * G + g, 3);
                copy_tile(cb_s, n0 * G + g, 7);
                matmul_mode(1);
                const uint32_t nj = (KB - g * 32 < 32) ? (KB - g * 32) : 32;
                for (uint32_t j = 0; j < nj; j += 2) {
                    if (pending) {
                        scale_set(prev_j, 1 - set);  // the previous pair, while the FPU computes this one
                    }
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint32_t kb = g * 32 + j + h;
                        uint32_t x0 = kb * 4;
                        if constexpr (!RESIDENT) {
                            if (x0 % CH == 0) {
                                cb_wait_front(cb_x, CH);
                            }
                            x0 %= CH;
                        }
                        const uint32_t w0 = n0 * Kt + kb * 4;
                        for (uint32_t i = 0; i < 4; ++i) {
                            matmul_block(cb_x, cb_w, x0 + i, w0 + i, 4 * set + 1 + h, 0, 1, 1, 1);
                        }
                        if constexpr (!RESIDENT) {
                            if ((kb * 4 + 4) % CH == 0) {
                                cb_pop_front(cb_x, CH);
                            }
                        }
                    }
                    prev_j = j;
                    pending = true;
                    set = 1 - set;
                }
                scale_set(prev_j, 1 - set);  // this group's last pair, before its S is replaced
                pending = false;
            }
            MATH((sfpu_at0(ckernel::sfpu::add_sets)));
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
#else
        for (uint32_t n0 = 0; n0 < n_count; n0 += NB) {
            const uint32_t nb = n_count - n0 < NB ? n_count - n0 : NB;
            tile_regs_acquire();
            MATH((SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, zero6, 0, 0, 0, 0, VectorMode::None)));
            for (uint32_t g = 0; g < G; ++g) {
                copy_init(cb_s);
                scale_tile_srca();
                for (uint32_t b = 0; b < nb; ++b) {
                    copy_tile(cb_s, (n0 + b) * G + g, 6 + b);
                }
                matmul_mode(nb);
                const uint32_t nj = (KB - g * 32 < 32) ? (KB - g * 32) : 32;
                for (uint32_t j = 0; j < nj; j += 2) {  // nj is even for every Bonsai K (a multiple of 256)
#ifdef ZACC
                    MATH((zero_p_tiles()));
#endif
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint32_t kb = g * 32 + j + h;
                        uint32_t x0 = kb * 4;
                        if constexpr (!RESIDENT) {
                            if (x0 % CH == 0) {
                                cb_wait_front(cb_x, CH);
                            }
                            x0 %= CH;
                        }
                        // this batch's tiles start at n0 * Kt, laid out [k][column] (kernels/tpre_reader.cpp)
                        const uint32_t w0 = n0 * Kt + kb * 4 * nb;
                        for (uint32_t i = 0; i < 4; ++i) {
#ifndef SKIP_MM
                            matmul_block(cb_x, cb_w, x0 + i, w0 + i * nb, 2 + 2 * h, 0, nb, 1, 1);
#endif
                        }
                        if constexpr (!RESIDENT) {
                            if ((kb * 4 + 4) % CH == 0) {
                                cb_pop_front(cb_x, CH);
                            }
                        }
                    }
                    for (uint32_t b = 0; b < nb; ++b) {
#ifndef SKIP_SFPU
                        scale_pair(j, b);
#endif
                    }
                }
            }
#ifdef GLU_PAIRS
            // columns (gate i, up i): one output tile up * silu(gate)
            silu_tile_init();
            MATH(({
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                ckernel::sfpu::glu_full();
                math::clear_dst_reg_addr();
            }));
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            cb_reserve_back(cb_out, 1);
            pack_tile(1, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
#else
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            for (uint32_t b = 0; b < nb; ++b) {  // one tile per reserve: a pair could straddle the ring's end
                cb_reserve_back(cb_out, 1);
                pack_tile(b, cb_out);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
#endif
        }
#endif
        if constexpr (RESIDENT) {
            cb_pop_front(cb_x, Kt);
        }
    }
}
