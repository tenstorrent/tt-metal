// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// 2D exact ternary prefill linear (linear2.py), compute. This core: token row tiles r < m (m <= 7) of its row group x
// its output tile columns, in blocks of up to NBLK. Per block, group of J = KPC kbs and column c, in one DEST pass: for
// each kb j of the group, P[r] += x'[r, kb] @ T[kb, c] for all r (4 matmul_blocks, rt = m, ternary in1 with a forced
// exponent: exact products), then the SFPU multiplies every P[r] by the column's ratio r_j = s_j / s_(j+1) (s_j for the
// group's last kb; scales.ratio_chain), so the pass ends with sum_j s_j P_j. The ratio vectors come from the column's
// group ratio tile of scales.pack_ratio_group in DEST tile 7 (fp16 hi + lo, summed in fp32 into LREG4..7), then one
// SFPLOADMACRO per SFPU row loads, multiplies and stores the row. The packer adds the pass into the fp32 accumulator
// tile (r, c) in L1 (L1 accumulation; the first group overwrites). After the block every accumulator goes out as bf16
// (GLU_PAIRS: up * silu(gate) per (gate, up) column pair).
// FPU / SFPU overlap (m >= 2): the rows go in two halves, A = rows [0, h) and B = [h, m). The pack thread's SFPU scales
// half A of kb j while the math thread's FPU computes half B of kb j, and half B of kb j during half A of kb j + 1.
// Semaphore FPU_SFPU (math -> pack): a half's matmuls are done; S2F (pack -> math): a half is scaled. With m = 1 the
// math thread scales after the matmuls.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"  // silu_tile_init

constexpr uint32_t cb_x = tt::CBIndex::c_0;
constexpr uint32_t cb_w = tt::CBIndex::c_1;
constexpr uint32_t cb_s = tt::CBIndex::c_3;  // broadcast-scale tiles (fp16 in a CB declared Float32)
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_acc = tt::CBIndex::c_24;  // fp32 accumulators
constexpr uint32_t cb_res = tt::CBIndex::c_4;   // RESIDUAL: tiles added to the outputs

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// P[r] (DEST tiles r < m) *= the column's ratios, one instruction per SFPU row (SFPU row q of a tile is face q / 8,
// column parity q % 2; scales.py). Templates k = 0..3: LReg[16] = LREG(4 + k) * VD + 0 (the 4 ratio vectors in
// LREG4..7); macro k: load the row into VD = LREG0, MAD with template k (VB = VD, result in LReg[16]), store LReg[16]
// back to the row 2 instructions later (the pattern of tt-llk's _mul_int_).
#define T2_ROW(q) \
    TTI_SFPLOADMACRO(((2 * (((q) >> 3) & 1) + ((q) & 1)) << 2), sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_6, 0)
// (a tile's 32 SFPU rows also go into replay buffer entries 0..31: one REPLAY per tile, as the TRISC cannot push 32
// instructions as fast)
inline void t2_macro_init() {
    TTI_SFPMUL(p_sfpu::LREG4, 0, p_sfpu::LCONST_0, 12, 0);
    TTI_SFPMUL(p_sfpu::LREG5, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(p_sfpu::LREG6, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_SFPMUL(p_sfpu::LREG7, 0, p_sfpu::LCONST_0, 15, 0);
    constexpr uint32_t store_bits = 0x40 | (2 << 3) | 3;
#define T2_MACRO(k)                                                    \
    do {                                                               \
        constexpr uint32_t mad_bits = 0x80 | 0x40 | (4 + k);           \
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8));   \
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8)); \
        TTI_SFPCONFIG(0, 4 + k, 0);                                    \
    } while (0)
    T2_MACRO(0);
    T2_MACRO(1);
    T2_MACRO(2);
    T2_MACRO(3);
#undef T2_MACRO
    TTI_SFPCONFIG(0x3F0, 8, 1);  // UsesLoadMod0ForStore for macros 0..3, delay kinds as _init_mul_int_
    TTI_SFPNOP;
#ifndef T2_NOREPLAY
    load_replay_buf(0, 32, [] {
        T2_ROW(0);
        T2_ROW(1);
        T2_ROW(2);
        T2_ROW(3);
        T2_ROW(4);
        T2_ROW(5);
        T2_ROW(6);
        T2_ROW(7);
        T2_ROW(8);
        T2_ROW(9);
        T2_ROW(10);
        T2_ROW(11);
        T2_ROW(12);
        T2_ROW(13);
        T2_ROW(14);
        T2_ROW(15);
        T2_ROW(16);
        T2_ROW(17);
        T2_ROW(18);
        T2_ROW(19);
        T2_ROW(20);
        T2_ROW(21);
        T2_ROW(22);
        T2_ROW(23);
        T2_ROW(24);
        T2_ROW(25);
        T2_ROW(26);
        T2_ROW(27);
        T2_ROW(28);
        T2_ROW(29);
        T2_ROW(30);
        T2_ROW(31);
    });
#endif
}
inline void scale_macro(uint32_t z, uint32_t m, uint32_t r0 = 0) {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    // column z of the unit: hi vectors v = 0..3 at SFPU rows 8 z + v of DEST tile 7, lo vectors at 8 z + 4 + v
    const uint32_t sa = (7 - r0) * 64 + 16 * z;  // relative to the dest base (DEST tile r0)
    TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 0);
    TT_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 2);
    TT_SFPLOAD(p_sfpu::LREG6, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 4);
    TT_SFPLOAD(p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 6);
    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 8);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 10);
    TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 12);
    TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, sa + 14);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG4, 0);  // hi * 1 + lo
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG5, 0);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG7, 0);
    TTI_SFPNOP;  // the macros' MAD reads LREG4..7 without the automatic MAD stall
    TTI_SFPNOP;
    for (uint32_t r = 0; r < m; ++r) {  // ADDR_MOD_6 steps 2 per row: 32 rows reach the next tile
#ifndef T2_NOREPLAY
        lltt::replay(0, 32);
#else
        T2_ROW(0);
        T2_ROW(1);
        T2_ROW(2);
        T2_ROW(3);
        T2_ROW(4);
        T2_ROW(5);
        T2_ROW(6);
        T2_ROW(7);
        T2_ROW(8);
        T2_ROW(9);
        T2_ROW(10);
        T2_ROW(11);
        T2_ROW(12);
        T2_ROW(13);
        T2_ROW(14);
        T2_ROW(15);
#ifdef T2_HALFROWS
        continue;
#endif
        T2_ROW(16);
        T2_ROW(17);
        T2_ROW(18);
        T2_ROW(19);
        T2_ROW(20);
        T2_ROW(21);
        T2_ROW(22);
        T2_ROW(23);
        T2_ROW(24);
        T2_ROW(25);
        T2_ROW(26);
        T2_ROW(27);
        T2_ROW(28);
        T2_ROW(29);
        T2_ROW(30);
        T2_ROW(31);
#endif
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
#undef T2_ROW
// GLU pair (gate in DEST tile 0, up in 1): tile 1 = up * silu(gate). Needs silu_tile_init.
inline void glu_full() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (uint32_t r = 0; r < T; ++r) {
        sfpi::vFloat g = sfpi::dst_reg[r];
        sfpi::vFloat u = sfpi::dst_reg[T + r];
        sfpi::dst_reg[T + r] =
            u * g * _sfpu_sigmoid_<false>(g);  // bf16-accurate exp, one Newton step: the output is bf16
    }
}
}  // namespace ckernel::sfpu
#endif

ALWI void scale_col(uint32_t z, uint32_t m) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::scale_macro(z, m);
    math::clear_dst_reg_addr();
#endif
}

constexpr uint32_t S2F = ckernel::semaphore::UNPACK_MATH_DONE;  // not used by this kernel's LLK calls
// the math thread's wait on S2F must also hold the SEMGET after it (a sync instruction): with STALL_MATH alone the
// SEMGET runs at once on a zero semaphore, the count runs ahead, and the FPU adds into rows the SFPU is still scaling
ALWI void s2f_take() {
#ifdef T2_NOSYNC
    return;
#endif
    MATH((TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SYNC, semaphore::t6_sem(S2F), p_stall::STALL_ON_ZERO)));
    MATH((t6_semaphore_get(S2F)));
}
// pack thread: once the math thread posts FPU_SFPU, DEST tiles r0 .. r0 + cnt - 1 *= the ratios of kb j; then S2F
// (except after the pass's last kb: the next pass starts from the packed, cleared DEST)
ALWI void pk_scale_half(uint32_t j, uint32_t r0, uint32_t cnt, bool signal) {
#ifdef TRISC_PACK
#ifndef T2_NOSYNC
    t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
#endif
    // this thread's SFPU dest base: DEST tile r0 (full-sync DEST); set once its previous pass is done
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, r0 * 64 + ckernel::packer::get_packer_dest_offset());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
#ifndef T2_NOSCALE
    ckernel::sfpu::scale_macro(j, cnt, r0);
#endif
#ifdef T2_NOSYNC
    return;
#endif
    t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    if (signal) {
        t6_semaphore_post<p_stall::WAIT_SFPU>(S2F);
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
    // a 32x32 IEEE fp16 tile (2048 B) in a CB declared Float32 (srcA Tf32 holds fp16 exactly)
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            (uint32_t)DataFormat::Float16, (uint32_t)DataFormat::Tf32, 2048 >> 4, FACE_R_DIM, 4);
    }));
}

void kernel_main() {
    constexpr uint32_t NBLK = get_compile_time_arg_val(0);
    constexpr uint32_t KPC = get_compile_time_arg_val(1);  // J: kbs per DEST pass and per x' chunk
    constexpr uint32_t WSLOT = NBLK * 4;  // t2_writer.cpp's weight slot per kb; its ratio slot per group: NBLK tiles
    const uint32_t KB = get_common_arg_val<uint32_t>(0);
    const uint32_t n_count = get_arg_val<uint32_t>(0);
    const uint32_t m = get_arg_val<uint32_t>(1);
    const uint32_t nblk = get_arg_val<uint32_t>(2);  // the row's block count: every core takes every multicast chunk
    const uint32_t ACC = get_arg_val<uint32_t>(3);   // accumulator tiles per block: the whole ring (m_max x NBLK), so
                                                     // every block starts at the ring's start (tile r * NBLK + c)
    const uint32_t XSLOT = ACC / NBLK * 4 * KPC;     // t2_reader.cpp's x' slot: m_max rows of 4 KPC tiles
    if (m == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_w, cb_x, cb_acc);  // SrcA = in1 (weights), SrcB = in0 (x')
#ifdef T2_NOSPLIT
    const bool split = false;
#else
    const bool split = m >= 2;
#endif
    const uint32_t h = (m + 1) / 2;
    // matmul_block_init's rt fixes the operand reuse (reuse_a = ct >= rt, ct = 1): the pass starts with rt = m, so a
    // half whose reuse differs (1 row, with m > 1: m = 2, 3) re-inits before each half
    const bool reinit = h < 2 || m - h < 2;
    // both handshake semaphores start at 0 (the pack thread consumes FPU_SFPU and produces S2F; it gets here long
    // before the math thread's first post, which waits on the first weights)
    PACK((t6_semaphore_init(semaphore::FPU_SFPU, 0, 15)));
    PACK((t6_semaphore_init(S2F, 0, 15)));
    for (uint32_t blk = 0; blk < nblk; ++blk) {
        const uint32_t c0 = blk * NBLK;
        const uint32_t nb = n_count > c0 ? (n_count - c0 < NBLK ? n_count - c0 : NBLK) : 0;
        if (nb == 0) {  // no columns left here: keep the row's multicast flowing
            for (uint32_t kb = 0; kb < KB; kb += KPC) {
                cb_wait_front(cb_x, XSLOT);
                cb_pop_front(cb_x, XSLOT);
            }
            continue;
        }
        // the SFPU macro setup on the scaling thread (after the previous block's GLU, which uses the SFPU too)
        if (split) {
            silu_tile_init_pack();  // the pack thread's SFPU setup (config register, address modes)
            PACK((ckernel::sfpu::t2_macro_init()));
        } else {
            MATH((ckernel::sfpu::t2_macro_init()));
        }
        cb_reserve_back(cb_acc, ACC);
        pack_reconfig_data_format(cb_acc);
        for (uint32_t g = 0; g < KB; g += KPC) {
            cb_wait_front(cb_x, XSLOT);
            cb_wait_front(cb_w, KPC * WSLOT);
            cb_wait_front(cb_s, NBLK);
            for (uint32_t c = 0; c < nb; ++c) {
                tile_regs_acquire();
                // the column's ratio tile for the whole group (scales.pack_ratio_group) into DEST tile 7
#ifndef T2_NORATIO
                copy_init(cb_s);
                scale_tile_srca();
                copy_tile(cb_s, c, 7);
                reconfig_data_format_srca(cb_w);
#endif
                matmul_block_init(cb_x, cb_w, 0, 1, m, 4 * KPC);
                ternary_srca();
                for (uint32_t j = 0; j < KPC; ++j) {
                    if (split) {
                        if (j > 0) {
                            s2f_take();  // half A of kb j - 1 is scaled
                        }
#ifndef T2_NOMM
                        if (reinit) {
                            matmul_block_init(cb_x, cb_w, 0, 1, h, 4 * KPC);
                            ternary_srca();
                        }
                        for (uint32_t i = 0; i < 4; ++i) {
                            matmul_block(cb_x, cb_w, j * 4 + i, (c * KPC + j) * 4 + i, 0, 0, 1, h, 4 * KPC);
                        }
#endif
#ifndef T2_NOSYNC
                        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
#endif
                        if (j > 0) {
                            s2f_take();  // half B of kb j - 1 is scaled
                        }
#ifndef T2_NOMM
                        if (reinit) {
                            matmul_block_init(cb_x, cb_w, 0, 1, m - h, 4 * KPC);
                            ternary_srca();
                        }
                        for (uint32_t i = 0; i < 4; ++i) {
                            matmul_block(
                                cb_x, cb_w, h * 4 * KPC + j * 4 + i, (c * KPC + j) * 4 + i, h, 0, 1, m - h, 4 * KPC);
                        }
#endif
#ifndef T2_NOSYNC
                        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
#endif
                        pk_scale_half(j, 0, h, j + 1 < KPC);
                        if (j + 1 == KPC) {
                            // the pass's last kb: half A is final, pack it while the FPU computes half B (only
                            // half B's scaling and packing are left after the matmuls)
                            PACK((TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU)));
                            if (c == 0 && g < 2 * KPC) {
                                pack_reconfig_l1_acc(g > 0 ? 1 : 0);
                            }
#ifndef T2_NOPACK
                            for (uint32_t r = 0; r < h; ++r) {
                                pack_tile<true>(r, cb_acc, r * NBLK + c);
                            }
#endif
                        }
                        pk_scale_half(j, h, m - h, j + 1 < KPC);
                        continue;
                    }
                    if (j > 0) {  // the SFPU pass of kb j - 1 finishes before the matmuls add into DEST again
                        MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
                    }
#ifndef T2_NOMM
                    for (uint32_t i = 0; i < 4; ++i) {
                        matmul_block(cb_x, cb_w, j * 4 + i, (c * KPC + j) * 4 + i, 0, 0, 1, m, 4 * KPC);
                    }
#endif
#ifndef T2_NOSCALE
                    scale_col(j, m);
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU)));  // the pack thread's SFPU passes first
                if (!split && c == 0 && g < 2 * KPC) {  // the block's first group overwrites the accumulators
                    pack_reconfig_l1_acc(g > 0 ? 1 : 0);
                }
#ifndef T2_NOPACK
                for (uint32_t r = split ? h : 0; r < m; ++r) {  // split: half A went out during half B's matmuls
                    pack_tile<true>(r, cb_acc, r * NBLK + c);
                }
#endif
                tile_regs_release();
            }
            cb_pop_front(cb_x, XSLOT);
            cb_pop_front(cb_w, KPC * WSLOT);
            cb_pop_front(cb_s, NBLK);
        }
#ifdef RESIDUAL
        // the residual tiles go into the fp32 accumulators by packer L1 accumulation (out = y + residual, one bf16
        // rounding at the output)
        reconfig_data_format_srca(cb_res);
        copy_init(cb_res);
        pack_reconfig_l1_acc(1);
        for (uint32_t r = 0; r < m; ++r) {
            for (uint32_t c = 0; c < nb; ++c) {
                cb_wait_front(cb_res, 1);
                tile_regs_acquire();
                copy_tile(cb_res, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_acc, r * NBLK + c);
                tile_regs_release();
                cb_pop_front(cb_res, 1);
            }
        }
#endif
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_acc, ACC);
        // fp32 accumulators -> bf16 outputs, [row][column]
        cb_wait_front(cb_acc, ACC);
        copy_init(cb_acc);
        reconfig_data_format_srca(cb_acc);
        pack_reconfig_data_format(cb_out);
        for (uint32_t r = 0; r < m; ++r) {
#ifdef GLU_PAIRS
            for (uint32_t c = 0; c < nb; c += 2) {
                tile_regs_acquire();
                copy_tile(cb_acc, r * NBLK + c, 0);
                copy_tile(cb_acc, r * NBLK + c + 1, 1);
#ifndef T2_NOGLU
                silu_tile_init();
                MATH(({
                    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                    ckernel::sfpu::glu_full();
                    math::clear_dst_reg_addr();
                }));
#endif
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_out, 1);
                pack_tile(1, cb_out);
                cb_push_back(cb_out, 1);
                tile_regs_release();
            }
#else
            for (uint32_t c = 0; c < nb; ++c) {
                tile_regs_acquire();
                copy_tile(cb_acc, r * NBLK + c, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_out, 1);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, 1);
                tile_regs_release();
            }
#endif
        }
        cb_pop_front(cb_acc, ACC);
    }
}
