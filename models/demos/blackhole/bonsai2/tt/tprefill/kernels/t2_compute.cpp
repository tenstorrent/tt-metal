// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// 2D exact ternary prefill linear (linear2.py), compute. This core: token row tiles r < m (m <= 7) of its row group x
// its output tile columns, in blocks of up to NBLK. Per block, kb and column c: P[r] = x'[r, kb] @ T[kb, c] for all r
// (4 matmul_blocks, rt = m, ternary in1 with a forced exponent: exact), then the SFPU multiplies every P[r] by the
// column's fp16 scales of the kb (the broadcast-scale tile of scales.py in DEST tile 7: one DEST read per lane vector),
// and the packer adds P[r] into the fp32 accumulator tile (r, c) in L1 (L1 accumulation; the first kb overwrites).
// After the block every accumulator goes out as bf16 (GLU_PAIRS: up * silu(gate) per (gate, up) column pair).
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
constexpr uint32_t cb_s = tt::CBIndex::c_3;  // broadcast-scale tiles (fp16 in a CB declared Float32)
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_acc = tt::CBIndex::c_24;  // fp32 accumulators

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// P[r] (DEST tiles r < m) *= the column's scales: combo Z of the broadcast-scale tile in DEST tile 7 (SFPU rows 4 Z +
// v, v = 2 fc + par). SFPU row q of a tile is face q / 8, column parity q % 2 (module docstring of scales.py).
template <int Z>
inline void scale_p(uint32_t m) {
    constexpr uint32_t T = 32;
    const sfpi::vFloat s0 = sfpi::dst_reg[7 * T + 4 * Z + 0];
    const sfpi::vFloat s1 = sfpi::dst_reg[7 * T + 4 * Z + 1];
    const sfpi::vFloat s2 = sfpi::dst_reg[7 * T + 4 * Z + 2];
    const sfpi::vFloat s3 = sfpi::dst_reg[7 * T + 4 * Z + 3];
    for (uint32_t r = 0; r < m; ++r) {
#pragma GCC unroll 32
        for (uint32_t q = 0; q < T; ++q) {
            const uint32_t v = 2 * ((q / 8) & 1) + (q & 1);
            const sfpi::vFloat sv = v == 0 ? s0 : v == 1 ? s1 : v == 2 ? s2 : s3;
            sfpi::dst_reg[q] = sfpi::dst_reg[q] * sv;
        }
        // next tile: the LLK's face step (_llk_math_eltwise_sfpu_inc_dst_face_addr_) is two inc_dst_addr<8> per face
#pragma GCC unroll 8
        for (uint32_t f = 0; f < 8; ++f) {
            math::inc_dst_addr<8>();
        }
    }
}
// GLU pair (gate in DEST tile 0, up in 1): tile 1 = up * silu(gate). Needs silu_tile_init.
inline void glu_full() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (uint32_t r = 0; r < T; ++r) {
        sfpi::vFloat g = sfpi::dst_reg[r];
        sfpi::vFloat u = sfpi::dst_reg[T + r];
        sfpi::dst_reg[T + r] = u * g * _sfpu_sigmoid_<true>(g);
    }
}
}  // namespace ckernel::sfpu
#endif

#define SP_CASE(Z) \
    case Z: ckernel::sfpu::scale_p<Z>(m); break;
ALWI void scale_col(uint32_t z, uint32_t m) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    switch (z) {
        SP_CASE(0) SP_CASE(1) SP_CASE(2) SP_CASE(3) SP_CASE(4) SP_CASE(5) SP_CASE(6) SP_CASE(7) default : break;
    }
    math::clear_dst_reg_addr();
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
    const uint32_t KB = get_common_arg_val<uint32_t>(0);
    const uint32_t n_count = get_arg_val<uint32_t>(0);
    const uint32_t m = get_arg_val<uint32_t>(1);
    const uint32_t nblk = get_arg_val<uint32_t>(2);  // the row's block count: every core takes every multicast chunk
    const uint32_t ACC = get_arg_val<uint32_t>(3);   // accumulator tiles per block: the whole ring (m_max x NBLK), so
                                                     // every block starts at the ring's start (tile r * NBLK + c)
    if (m == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_w, cb_x, cb_acc);  // SrcA = in1 (weights), SrcB = in0 (x')
    for (uint32_t blk = 0; blk < nblk; ++blk) {
        const uint32_t c0 = blk * NBLK;
        const uint32_t nb = n_count > c0 ? (n_count - c0 < NBLK ? n_count - c0 : NBLK) : 0;
        if (nb == 0) {  // no columns left here: keep the row's multicast flowing
            for (uint32_t kb = 0; kb < KB; ++kb) {
                cb_wait_front(cb_x, m * 4);
                cb_pop_front(cb_x, m * 4);
            }
            continue;
        }
        const uint32_t units = (nb + 7) / 8;
        cb_reserve_back(cb_acc, ACC);
        for (uint32_t kb = 0; kb < KB; ++kb) {
            cb_wait_front(cb_x, m * 4);
            cb_wait_front(cb_w, nb * 4);
            cb_wait_front(cb_s, units);
            for (uint32_t c = 0; c < nb; ++c) {
                tile_regs_acquire();
                copy_init(cb_s);
                scale_tile_srca();
                copy_tile(cb_s, c / 8, 7);
                reconfig_data_format_srca(cb_w);
                matmul_block_init(cb_x, cb_w, 0, 1, m, 4);
                ternary_srca();
                for (uint32_t i = 0; i < 4; ++i) {
                    matmul_block(cb_x, cb_w, i, c * 4 + i, 0, 0, 1, m, 4);
                }
#ifndef T2_NOSCALE
                addcmul_tile_init();
                scale_col(c % 8, m);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_acc);
                pack_reconfig_l1_acc(kb > 0 ? 1 : 0);
                for (uint32_t r = 0; r < m; ++r) {
                    pack_tile<true>(r, cb_acc, r * NBLK + c);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_x, m * 4);
            cb_pop_front(cb_w, nb * 4);
            cb_pop_front(cb_s, units);
        }
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
