// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// 2D shared-fold ternary prefill linear (linear6.py), compute. Per column block and group of J kbs: fold this core's
// share of the block's units (t6_writer.cpp), W^T = diag(s5) @ t^T per unit (one matmul_block, ct = 4; the ternary
// tiles go through srcA, whose Bfp2 unpack with a forced exponent is exact, transposed), packed as Bfp8_b into c_7
// (exact: each face row holds one output column, one scale); then, once the writer has every core's share in the
// group's c_5 slot, per column the token rows in passes of up to PASS: P[r] = sum over the group's k tiles of x'[r, k]
// @ W[k, c] (matmul_block, transposing the folded tiles back), and the packer adds each pass into the fp32 accumulator
// tile (r, c) in L1 (the block's first group overwrites). The fold of group g + 1 runs before the matmuls of group g.
// After the block every accumulator goes out as bf16 (GLU_PAIRS: up * silu(gate) per (gate, up) column pair).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"  // silu_tile_init

constexpr uint32_t cb_x = tt::CBIndex::c_0;
constexpr uint32_t cb_w = tt::CBIndex::c_1;     // this core's units' ternary tiles (Bfp2_b codes, no exponent section)
constexpr uint32_t cb_wx = tt::CBIndex::c_5;    // every folded tile of a group, [column][k], transposed
constexpr uint32_t cb_dg = tt::CBIndex::c_6;    // this core's units' diag(s5) tiles
constexpr uint32_t cb_fold = tt::CBIndex::c_7;  // this core's folded tiles
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_acc = tt::CBIndex::c_24;  // fp32 accumulators
constexpr uint32_t cb_res = tt::CBIndex::c_4;   // RESIDUAL: tiles added to the outputs
// a group's folded tile (k tile kt, column c) in its c_5 slot: [k][column] when passes take several columns
// (column blocks of CP columns, the last one maybe narrower, each [k][column]: a pass's columns are adjacent)
#define CPB(c) (nb - (c) / CP * CP < CP ? nb - (c) / CP * CP : CP)
#define WIDX(kt, c) ((c) / CP * KT * CP + (kt) * CPB(c) + (c) % CP)
// unit row q (column block q / J, kb q % J): its first c_5 tile
#define QOFF(q) ((q) / J * KT * CP + (q) % J * 4 * CPB((q) / J * CP))

#if defined(TRISC_MATH)
namespace ckernel::sfpu {
// GLU pair (gate in DEST tile 0, up in 1): tile 1 = up * silu(gate). Needs silu_tile_init.
inline void glu_full() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (uint32_t r = 0; r < T; ++r) {
        sfpi::vFloat g = sfpi::dst_reg[r];
        sfpi::vFloat u = sfpi::dst_reg[T + r];
        sfpi::dst_reg[T + r] = u * g * _sfpu_sigmoid_<false>(g);
    }
}
}  // namespace ckernel::sfpu
#endif

ALWI void ternary_srca() {
    // the decode records' ternary tiles: Bfp2_b codes with no exponent section; the unpacker forces exponent 127
    UNPACK(({
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Bfp2_b, (uint32_t)DataFormat::Float16_b, 256 >> 4, FACE_R_DIM, 4);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
        cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(127);
    }));
}

ALWI void folded_srca() {
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0)));
    reconfig_data_format_srca(cb_wx);
}

void kernel_main() {
    constexpr uint32_t NBLK = get_compile_time_arg_val(0);
    constexpr uint32_t J = get_compile_time_arg_val(1);
    constexpr uint32_t UM = get_compile_time_arg_val(2);
    constexpr uint32_t PASS = get_compile_time_arg_val(3);  // token row tiles per DEST pass
    constexpr uint32_t CP = get_compile_time_arg_val(4);    // output columns per DEST pass
    constexpr uint32_t KT = 4 * J;
    constexpr uint32_t SLOT = NBLK * KT;  // c_5 tiles per group
    const uint32_t KB = get_common_arg_val<uint32_t>(0);
    const uint32_t n_count = get_arg_val<uint32_t>(0);
    const uint32_t m = get_arg_val<uint32_t>(1);
    const uint32_t nblk = get_arg_val<uint32_t>(2);
    const uint32_t ACC = get_arg_val<uint32_t>(3);  // m_max x NBLK accumulator tiles
    const uint32_t ri = get_arg_val<uint32_t>(4), R = get_arg_val<uint32_t>(5);
    const uint32_t XSLOT = ACC / NBLK * KT;  // t2_reader.cpp's x' slot: m_max rows of KT tiles
    const uint32_t NG = KB / J;
    if (m == 0) {
        return;
    }
    const uint32_t np = (m + PASS - 1) / PASS;
    compute_kernel_hw_startup(cb_w, cb_x, cb_acc);
    for (uint32_t blk = 0; blk < nblk; ++blk) {
        const uint32_t c0 = blk * NBLK;
        const uint32_t nb = n_count > c0 ? (n_count - c0 < NBLK ? n_count - c0 : NBLK) : 0;
        if (nb == 0) {  // no columns left here: keep the row's multicast flowing
            for (uint32_t g = 0; g < NG; ++g) {
                cb_wait_front(cb_x, XSLOT);
                cb_pop_front(cb_x, XSLOT);
            }
            continue;
        }
        // this core's unit rows (t6_writer.cpp): q = cb J + j, the units (column cb CP + cc, kb j)
        const uint32_t Q = (nb + CP - 1) / CP * J;
        const uint32_t q0 = Q * ri / R, q1 = Q * (ri + 1) / R;
        auto fold = [&]() {
            cb_wait_front(cb_w, UM * 4);
            cb_wait_front(cb_dg, UM);
            cb_reserve_back(cb_fold, UM * 4);
#ifndef T6_NOFOLD
            pack_reconfig_data_format(cb_fold);
            pack_reconfig_l1_acc(0);
            matmul_block_init(cb_dg, cb_w, 1, 4, 1, 1);
            reconfig_data_format_srcb(cb_dg);
            ternary_srca();
            uint32_t ul = 0;
            for (uint32_t q = q0; q < q1; ++q) {
                const uint32_t c1 = q / J * CP;
                for (uint32_t cc = 0; cc < CP && c1 + cc < nb; ++cc, ++ul) {
                    tile_regs_acquire();
                    matmul_block(cb_dg, cb_w, ul, 4 * ul, 0, 1, 4, 1, 1);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t t = 0; t < 4; ++t) {
                        pack_tile<true>(t, cb_fold, QOFF(q) - QOFF(q0) + t * CPB(c1) + cc);
                    }
                    tile_regs_release();
                }
            }
#endif
            cb_push_back(cb_fold, UM * 4);
            cb_pop_front(cb_w, UM * 4);
            cb_pop_front(cb_dg, UM);
        };
        cb_reserve_back(cb_acc, ACC);
        fold();
        for (uint32_t g = 0; g < NG; ++g) {
            if (g + 1 < NG) {
                fold();
            }
            cb_wait_front(cb_x, XSLOT);
            cb_wait_front(cb_wx, SLOT);
            pack_reconfig_data_format(cb_acc);
            pack_reconfig_l1_acc(g > 0 ? 1 : 0);  // the block's first group overwrites the accumulators
            for (uint32_t c0 = 0; c0 < nb; c0 += CP) {
                const uint32_t cp = nb - c0 < CP ? nb - c0 : CP;
                for (uint32_t p = 0, r0 = 0; p < np; ++p) {
                    const uint32_t rp = (m - r0) / (np - p);
                    tile_regs_acquire();
                    // the matmul's replay depends on (rp, cp); the first pass also follows the fold
                    if (c0 == 0 || np > 1 || cp != CP) {
                        matmul_block_init(cb_x, cb_wx, 1, cp, rp, KT);
                        reconfig_data_format_srcb(cb_x);
                        folded_srca();
                    }
#ifndef T6_NOMM
                    for (uint32_t kt = 0; kt < KT; ++kt) {
                        matmul_block(cb_x, cb_wx, r0 * KT + kt, WIDX(kt, c0), 0, 1, cp, rp, KT);
                    }
#endif
                    tile_regs_commit();
                    tile_regs_wait();
#ifndef T6_NOPACK
                    for (uint32_t r = 0; r < rp; ++r) {
                        for (uint32_t cc = 0; cc < cp; ++cc) {
                            pack_tile<true>(r * cp + cc, cb_acc, (r0 + r) * NBLK + c0 + cc);
                        }
                    }
#endif
                    tile_regs_release();
                    r0 += rp;
                }
            }
            cb_pop_front(cb_x, XSLOT);
            cb_pop_front(cb_wx, SLOT);
        }
#ifdef RESIDUAL
        reconfig_data_format_srca(cb_res);
        copy_init(cb_res);
        pack_reconfig_data_format(cb_acc);
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
                silu_tile_init();
                MATH(({
                    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                    ckernel::sfpu::glu_full();
                    math::clear_dst_reg_addr();
                }));
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
