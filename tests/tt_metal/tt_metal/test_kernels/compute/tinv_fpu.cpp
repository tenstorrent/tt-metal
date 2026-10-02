// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/triangle_solve.h"
#include "api/dataflow/circular_buffer.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "internal/tt-1xx/risc_common.h"
#endif
#include "../../../../../ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp"

// WY inverse micro-benchmark: T_inv = (I - negN)^-1 of one 32x32 fp32 tile, four implementations plus the
// MOVD2A/MOVD2B probes behind them. Every variant produces one output tile per repetition; the MATH thread
// writes wall-clock timestamps per repetition to L1 (runtime arg 0).
//
//   c_0  : negN tiles (NUM_IN), Float32
//   c_1  : identity tile, Float32
//   c_2  : the three WY quadrant masks (Qtl, Qbr, Q10), Float32
//   c_3..c_8 : fp32 single-tile scratch for invert_block
//   c_16 : output tiles, Float32
//
// Compile-time args:
//   0 VARIANT  0 invert_block (LLK rounds), 1 SFPU triangle_solve_tile, 2 fused FPU Horner, 3 fused FPU square,
//              4 probe MOVD2B (X -> SrcB, x I), 5 probe MOVD2A (X -> SrcA, I x), 6 probe unpack X -> SrcB (X @ I),
//              7 probe unpack X -> SrcA (I @ X), 8 probe MOVD2B/MOVD2A after an unpacker dummy-valid,
//              9 probe MOVD2B/MOVD2A after a MATH-side SETDVALID, 10 probe MOVD2B/MOVD2A over banks a real
//              unpack published, 11 probe MOVD2B of one face over the unpacker's SrcB, 12 probe MOVD2B of X then
//              MOVD2A of one face of I over the unpacker's SrcA, 13 probe the chain's own face product (addr mods,
//              fidelity phases, dest immediates) on X faces x I16, 14/15 the fused Horner / HornerR dataflow through
//              LLK matmul rounds (anchor; inputs carry the quadrants, three output tiles), 16 fused FPU HornerR,
//              17 fused FPU HornerR with the chain issued by the PACK thread after tile_regs_wait (feasibility)
//   1 NUM_IN   negN tiles consumed
//   2 REPS     repetitions per tile
//   3 STALL    1: STALLWAIT(MATH) between the fused chain's stages
//   4 NSRC     0: negN unpacked into SrcA/SrcB, 1: negN copied to DST and moved
//   5 SPLIT    1: variants 2/3 run an inline copy of the wrapper with four timestamps per repetition
//   6 FMT      Src format mask around the moves: 1 pin SrcA tf32, 2 pin SrcB tf32, 4 keep the zero flag; 0 implied;
//              8 (probes only) ALU Fp32_enabled cleared while the moves run
// Runtime args: 0 L1 address of the timestamp buffer (uint32: count, then 4 words per repetition).

namespace {
constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
constexpr uint32_t NUM_IN = get_compile_time_arg_val(1);
constexpr uint32_t REPS = get_compile_time_arg_val(2);
constexpr bool STALL = get_compile_time_arg_val(3) != 0;
constexpr uint32_t NSRC = get_compile_time_arg_val(4);
constexpr bool SPLIT = get_compile_time_arg_val(5) != 0;
constexpr uint32_t FMT = get_compile_time_arg_val(6);
constexpr bool LLK_ROUNDS = VARIANT == 14 || VARIANT == 15;
constexpr uint32_t TILES_PER_IN = LLK_ROUNDS ? 4 : 1;  // negN, N00, N11, N10

constexpr auto cb_n = tt::CBIndex::c_0;
constexpr auto cb_eye = tt::CBIndex::c_1;
constexpr auto cb_mask = tt::CBIndex::c_2;
constexpr auto cb_tmpN = tt::CBIndex::c_3;
constexpr auto cb_tmpT = tt::CBIndex::c_4;
constexpr auto cb_A = tt::CBIndex::c_5;
constexpr auto cb_B = tt::CBIndex::c_6;
constexpr auto cb_C = tt::CBIndex::c_7;
constexpr auto cb_D = tt::CBIndex::c_8;
constexpr auto cb_out = tt::CBIndex::c_16;

using gdn_tinv_fpu::Form;
using gdn_tinv_fpu::NSrc;
constexpr NSrc kNSrc = NSRC == 0 ? NSrc::Unpack : NSrc::Dst;

#ifdef TRISC_PACK
volatile tt_l1_ptr uint32_t* g_pstats = nullptr;  // PACK timestamps, 4 KB above the MATH records
uint32_t g_prec = 0;
inline void precord(uint32_t t0, uint32_t t1, uint32_t t2) {
    g_pstats[1 + 4 * g_prec + 0] = t0;
    g_pstats[1 + 4 * g_prec + 1] = t1;
    g_pstats[1 + 4 * g_prec + 2] = t2;
    g_pstats[1 + 4 * g_prec + 3] = t2;
    g_prec++;
    g_pstats[0] = g_prec;
}
#endif
#ifdef TRISC_MATH
volatile tt_l1_ptr uint32_t* g_stats = nullptr;
uint32_t g_rec = 0;
inline uint32_t now() { return get_timestamp_32b(); }
inline void record(uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    g_stats[1 + 4 * g_rec + 0] = t0;
    g_stats[1 + 4 * g_rec + 1] = t1;
    g_stats[1 + 4 * g_rec + 2] = t2;
    g_stats[1 + 4 * g_rec + 3] = t3;
    g_rec++;
    g_stats[0] = g_rec;
}
// Zero addr mod for the probe moves; the matmul init owns ADDR_MOD_0..5.
inline void probe_math_prologue() {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
    ckernel::addr_mod_t{}.set(ADDR_MOD_7);
    gdn_tinv_fpu::detail::src_format_enter<FMT>();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
}
inline void probe_math_epilogue() { gdn_tinv_fpu::detail::src_format_leave<FMT>(); }
// DST tile T (64 rows) -> SrcB rows 0..63.
template <uint32_t T>
inline void tile_d2b() {
    gdn_tinv_fpu::detail::d2b<0, T * 64 + 0>();
    gdn_tinv_fpu::detail::d2b<16, T * 64 + 16>();
    gdn_tinv_fpu::detail::d2b<32, T * 64 + 32>();
    gdn_tinv_fpu::detail::d2b<48, T * 64 + 48>();
}
template <uint32_t T>
inline void tile_d2a() {
    gdn_tinv_fpu::detail::d2a<0, T * 64 + 0>();
    gdn_tinv_fpu::detail::d2a<16, T * 64 + 16>();
    gdn_tinv_fpu::detail::d2a<32, T * 64 + 32>();
    gdn_tinv_fpu::detail::d2a<48, T * 64 + 48>();
}
#endif

// SFPU forward substitution, as sfpu_tinv in chunk_gdn_math.hpp.
inline void sfpu_variant() {
    cb_reserve_back(cb_out, 1);
    copy_init(cb_eye);
    constexpr uint32_t DST_RHS = 0, DST_X = 1;
    tile_regs_acquire();
    copy_tile(cb_eye, 0, DST_RHS);
    triangle_solve_tile_init();
    triangle_solve_tile<DataFormat::Float32, true /*L_NEGATED*/>(cb_n, 0, DST_RHS, DST_X);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(DST_X, cb_out, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

// The gdn_tinv_fpu::tinv wrapper with MATH timestamps around its phases.
template <Form F>
inline void fpu_split_variant() {
    cb_reserve_back(cb_out, 1);
    copy_init(cb_eye);
#ifdef TRISC_MATH
    const uint32_t t0 = now();
#endif
    tile_regs_acquire();
    copy_tile(cb_eye, 0, gdn_tinv_fpu::kTout);
    if constexpr (kNSrc == NSrc::Unpack) {
        matmul_init(cb_n, cb_n);
        UNPACK((llk_unpack_AB_matmul(cb_n, cb_n, 0, 0)));
    } else {
        copy_tile(cb_n, 0, gdn_tinv_fpu::kTn);
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
    }
#ifdef TRISC_MATH
    const uint32_t t1 = now();
    gdn_tinv_fpu::detail::chain<F, kNSrc, STALL, FMT>();
    const uint32_t t2 = now();
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(gdn_tinv_fpu::kTout, cb_out, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
#ifdef TRISC_MATH
    record(t0, t1, t2, now());
#endif
}

// The fused Horner (S_IN_B = false) / HornerR dataflow through LLK matmul rounds: per 16-block, DST = I then
// += Nq @ S (or S @ Nq) and pack, 15 times; then off = (Bi11 @ N10) @ Bi00. Inputs c_0: negN, N00, N11, N10.
// Outputs (c_16): Bi00 as a full tile, Bi11 as a full tile, off.
template <bool S_IN_B>
inline void llk_rounds_variant() {
    constexpr uint32_t T_N00 = 1, T_N11 = 2, T_N10 = 3;
    // DST = I when with_eye, DST += in0 @ in1 (in0 -> SrcB, in1 -> SrcA), packed to out and also to c_16 when also_out.
    auto round = [](uint32_t in0, uint32_t t0, uint32_t in1, uint32_t t1, uint32_t out, bool with_eye, bool also_out) {
        cb_reserve_back(out, 1);
        if (also_out) {
            cb_reserve_back(cb_out, 1);
        }
        tile_regs_acquire();
        if (with_eye) {
            copy_init(cb_eye);
            copy_tile(cb_eye, 0, 0);
        }
        matmul_init(in0, in1);
        matmul_tiles(in0, in1, t0, t1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, 0);
        if (also_out) {
            pack_tile(0, cb_out, 0);
        }
        tile_regs_release();
        cb_push_back(out, 1);
        if (also_out) {
            cb_push_back(cb_out, 1);
        }
    };
    // block: S_1 = I (cb_eye), S_m alternates cb_A / cb_B, S_16 lands in keep and in c_16.
    auto block = [&](uint32_t nq_tile, uint32_t keep) {
        uint32_t prev = cb_eye;
        for (uint32_t m = 1; m < 16; m++) {
            const bool last = m == 15;
            const uint32_t next = last ? keep : ((m & 1) ? cb_A : cb_B);
            if constexpr (S_IN_B) {
                round(prev, 0, cb_n, nq_tile, next, true, last);
            } else {
                round(cb_n, nq_tile, prev, 0, next, true, last);
            }
            cb_wait_front(next, 1);
            if (prev != cb_eye) {
                cb_pop_front(prev, 1);
            }
            prev = next;
        }
    };
    block(T_N00, cb_C);
    block(T_N11, cb_D);
    round(cb_D, 0, cb_n, T_N10, cb_A, false, false);  // tmp = Bi11 @ N10
    cb_wait_front(cb_A, 1);
    round(cb_A, 0, cb_C, 0, cb_out, false, false);  // off = tmp @ Bi00
    cb_pop_front(cb_A, 1);
    cb_pop_front(cb_C, 1);
    cb_pop_front(cb_D, 1);
}

// gdn_tinv_fpu::tinv with the chain issued by the PACK thread (NSrc::Dst): MATH copies I and negN and commits,
// PACK runs the chain in the committed half, packs and releases.
inline void fpu_pack_issued_variant() {
    cb_reserve_back(cb_out, 1);
    reconfig_data_format_srca(cb_eye);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_eye);
#ifdef TRISC_MATH
    if constexpr (SPLIT) {
        riscv_wait(2000);  // keep MATH's Src use clear of PACK's chain on the previous half
    }
#endif
    tile_regs_acquire();
    copy_tile(cb_eye, 0, gdn_tinv_fpu::kTout);
    copy_tile(cb_n, 0, gdn_tinv_fpu::kTn);
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    tile_regs_commit();
    tile_regs_wait();
#ifdef TRISC_PACK
    const uint32_t p0 = get_timestamp_32b();
    if constexpr (VARIANT == 17) {
        gdn_tinv_fpu::detail::chain<Form::HornerR, NSrc::Dst, STALL, FMT>();
    } else {
        // minimal: I (T0 face 0) -> SrcA/SrcB, I x I into T0 face 1 (zero), so the output shows where PACK's
        // moves read and its MVMULs write
        namespace gf = gdn_tinv_fpu;
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        gf::detail::set_common_mods();
        gf::detail::src_format_enter<FMT>();
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        gf::detail::d2b<0, gf::row(gf::kTout, 0)>();
        gf::detail::d2a<0, gf::row(gf::kTout, 0)>();
        gf::detail::face_mm<gf::row(gf::kTout, 1)>();
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        gf::detail::src_format_leave<FMT>();
    }
    const uint32_t p1 = get_timestamp_32b();
#endif
    pack_tile(gdn_tinv_fpu::kTout, cb_out, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
#ifdef TRISC_PACK
    precord(p0, p1, get_timestamp_32b());
#endif
}

// X (c_0 front tile) through the FPU identity product, the X operand supplied by the path under test.
inline void probe_variant() {
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    if constexpr (VARIANT == 6 || VARIANT == 7) {
        if constexpr (VARIANT == 6) {
            matmul_init(cb_n, cb_eye);
            UNPACK((llk_unpack_AB_matmul(cb_n, cb_eye, 0, 0)));  // X -> SrcB via the unpacker
        } else {
            matmul_init(cb_eye, cb_n);
            UNPACK((llk_unpack_AB_matmul(cb_eye, cb_n, 0, 0)));  // X -> SrcA via the unpacker
        }
#ifdef TRISC_MATH
        gdn_tinv_fpu::detail::src_format_enter<FMT>();
        llk_math_matmul<MATH_FIDELITY>(1);
        probe_math_epilogue();
#endif
    } else {
        copy_init(cb_n);
        copy_tile(cb_n, 0, 0);  // X exact in DST 0 (unpack-to-dest)
        if constexpr (VARIANT >= 8) {
            copy_init(cb_eye);
            copy_tile(cb_eye, 0, 2);  // I in DST 2
        }
        matmul_init(cb_eye, cb_eye);
        if constexpr (VARIANT < 8 || VARIANT >= 10) {
            UNPACK((llk_unpack_AB_matmul(cb_eye, cb_eye, 0, 0)));  // SrcA = SrcB = I, both banks valid
        } else if constexpr (VARIANT == 8) {
            UNPACK((llk_unpack_set_srcb_dummy_valid()));
        }
#ifdef TRISC_MATH
        if constexpr (VARIANT == 9) {
            TTI_SETDVALID(0b11);
        }
        probe_math_prologue();
        if constexpr (FMT & 8) {  // probe: ALU_ACC_CTRL_Fp32_enabled off while the moves read DST
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
        }
        if constexpr (VARIANT == 4) {
            tile_d2b<0>();
        } else if constexpr (VARIANT == 5) {
            tile_d2a<0>();
        } else if constexpr (VARIANT == 11) {
            gdn_tinv_fpu::detail::d2b<0, 0>();  // face 0 of X only
        } else if constexpr (VARIANT == 12) {
            tile_d2b<0>();
            gdn_tinv_fpu::detail::d2a<0, 2 * 64>();  // face 0 of I only, over the unpacker's I
        } else if constexpr (VARIANT == 13) {
            // the fused chain's own face product: X face f (SrcB) x I16 (SrcA) -> DST tile 1 face f
            gdn_tinv_fpu::detail::set_common_mods();
            gdn_tinv_fpu::detail::d2a<0, 2 * 64>();
            gdn_tinv_fpu::detail::d2b<0, 0>();
            gdn_tinv_fpu::detail::face_mm<64>();
            gdn_tinv_fpu::detail::d2b<0, 16>();
            gdn_tinv_fpu::detail::face_mm<64 + 16>();
            gdn_tinv_fpu::detail::d2b<0, 32>();
            gdn_tinv_fpu::detail::face_mm<64 + 32>();
            gdn_tinv_fpu::detail::d2b<0, 48>();
            gdn_tinv_fpu::detail::face_mm<64 + 48>();
        } else {
            tile_d2b<0>();
            tile_d2a<2>();
        }
        if constexpr (FMT & 8) {
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
        }
        if constexpr (VARIANT == 13) {
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        } else {
            llk_math_matmul<MATH_FIDELITY>(1);
        }
        probe_math_epilogue();
#endif
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, cb_out, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
}  // namespace

void kernel_main() {
#ifdef TRISC_MATH
    g_stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    g_stats[0] = 0;
#endif
#ifdef TRISC_PACK
    g_pstats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0) + 4096);
    g_pstats[0] = 0;
#endif
    compute_kernel_hw_startup(cb_n, cb_eye, cb_out);
    cb_wait_front(cb_eye, 1);
    cb_wait_front(cb_mask, 3);

    for (uint32_t i = 0; i < NUM_IN; i++) {
        cb_wait_front(cb_n, TILES_PER_IN);
        for (uint32_t r = 0; r < REPS; r++) {
#ifdef TRISC_MATH
            const uint32_t t0 = now();
#endif
            if constexpr (VARIANT == 0) {
                invert_block(cb_n, 0, cb_out, cb_tmpN, cb_tmpT, cb_eye, cb_mask, cb_A, cb_B, cb_C, cb_D);
            } else if constexpr (VARIANT == 1) {
                sfpu_variant();
            } else if constexpr (VARIANT == 2) {
                if constexpr (SPLIT) {
                    fpu_split_variant<Form::Horner>();
                } else {
                    gdn_tinv_fpu::tinv<Form::Horner, kNSrc, STALL, FMT>(cb_n, cb_eye, cb_out);
                }
            } else if constexpr (VARIANT == 3) {
                if constexpr (SPLIT) {
                    fpu_split_variant<Form::Square>();
                } else {
                    gdn_tinv_fpu::tinv<Form::Square, kNSrc, STALL, FMT>(cb_n, cb_eye, cb_out);
                }
            } else if constexpr (VARIANT == 14) {
                llk_rounds_variant<false>();
            } else if constexpr (VARIANT == 15) {
                llk_rounds_variant<true>();
            } else if constexpr (VARIANT == 16) {
                if constexpr (SPLIT) {
                    fpu_split_variant<Form::HornerR>();
                } else {
                    gdn_tinv_fpu::tinv<Form::HornerR, kNSrc, STALL, FMT>(cb_n, cb_eye, cb_out);
                }
            } else if constexpr (VARIANT == 17 || VARIANT == 18) {
                fpu_pack_issued_variant();
            } else {
                probe_variant();
            }
#ifdef TRISC_MATH
            if constexpr (!(SPLIT && (VARIANT == 2 || VARIANT == 3 || VARIANT == 16))) {
                record(t0, 0, 0, now());
            }
#endif
        }
        cb_pop_front(cb_n, TILES_PER_IN);
    }
}
