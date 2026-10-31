// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse-row op, compute (SFPU, fp32 in DEST). Per segment: s = sum_j V[j] * G[j] in HPCG's column order. SpMV /
// gather mode outputs s. GS mode: d = V[13] (the diagonal), x = G[13] (this row's own x) and outputs
// x_new = (b - s + d*x) / d, HPCG's ComputeSYMGS_ref update.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "gather.h"
#include "compute_gather.h"
#ifdef FASTMAC
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "ckernel_sfpu_recip.h"
namespace ckernel::sfpu {
// one face (8 x 32 lanes): s = v * g (FIRST) or s = s + v * g, with v in DEST tile 1, g in tile 2, s in tile 0.
// The product and the sum are separate statements, each rounded to fp32 as the two-op form was.
template <bool FIRST>
inline void hpcg_mac_face() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat pr = sfpi::dst_reg[1 * T] * sfpi::dst_reg[2 * T];
        if constexpr (FIRST) {
            sfpi::dst_reg[0] = pr;
        } else {
            // the product goes through DEST (rounded to fp32) so the compiler cannot contract it into a fused
            // multiply-add: same two roundings as HPCG's sum += a * x in fp32
            sfpi::dst_reg[1 * T] = pr;
            sfpi::vFloat s = sfpi::dst_reg[0];
            sfpi::vFloat q = sfpi::dst_reg[1 * T];
            sfpi::dst_reg[0] = s + q;
        }
        sfpi::dst_reg++;
    }
}
// one face of the Gauss-Seidel update x = (b - s + d * x) / d, with b in DEST tile 1, s in 0, d in 4, x in 5:
// the same operations and roundings as sub / mul / add / div_binary_tile one after another (the division is
// calculate_sfpu_binary_div's body), but in one pass over only the faces the segment uses
inline void hpcg_gs_face() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat bb = sfpi::dst_reg[1 * T];
        sfpi::vFloat ss = sfpi::dst_reg[0];
        sfpi::vFloat t = bb - ss;
        sfpi::vFloat dd = sfpi::dst_reg[4 * T];
        sfpi::vFloat xx = sfpi::dst_reg[5 * T];
        sfpi::dst_reg[5 * T] = dd * xx;  // d * x, rounded on its own
        sfpi::vFloat u = sfpi::dst_reg[5 * T];
        sfpi::vFloat in0 = t + u;
        sfpi::vFloat in1 = sfpi::dst_reg[4 * T];
        sfpi::vFloat r = sfpu_reciprocal_iter<2>(in1);
        sfpi::vFloat result = in0 * r;
        v_if(sfpi::is_finite(result)) {
            sfpi::vFloat e = in0 - result * in1;
            result = result + e * r;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
inline void hpcg_gs(uint32_t f0, uint32_t f1) {
    _llk_math_eltwise_sfpu_start_(0);
    for (uint32_t f = 0; f < f0; ++f) {
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    for (uint32_t f = f0; f < f1; ++f) {
        ckernel::sfpu::hpcg_gs_face();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}
inline void hpcg_mac(uint32_t f0, uint32_t f1, bool first) {
    _llk_math_eltwise_sfpu_start_(0);
    for (uint32_t f = 0; f < f0; ++f) {
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    for (uint32_t f = f0; f < f1; ++f) {
        if (first) {
            ckernel::sfpu::hpcg_mac_face<true>();
        } else {
            ckernel::sfpu::hpcg_mac_face<false>();
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}
#endif
#endif

void kernel_main() {
    const uint32_t nseg = get_arg_val<uint32_t>(0);
    const uint32_t nseg_gs = get_arg_val<uint32_t>(1);  // the first nseg_gs segments are Gauss-Seidel updates
    const uint32_t nseg_r = get_arg_val<uint32_t>(2);   // RESTF: the next nseg_r (SpMV, block 0) output r - A z
    const uint32_t flags = get_arg_val<uint32_t>(3);
    const uint32_t nph = get_arg_val<uint32_t>(4);  // phases to run, if not 0: a prefix of the table's
    (void)nph;
    const uint32_t ready_n = (flags & F_WSPLIT) ? 2 : 1;  // semaphore 0 at window load: reader (+ writer, WSPLIT)
    (void)nseg_gs;
    (void)nseg_r;
    (void)ready_n;
    compute_kernel_hw_startup(CB_V, CB_OUT);
#if NG > 2 && defined(TRISC_PACK)
    PackGather pg;
    pg.init(ready_n);
    if (nph) {
        pg.NPH = nph;
    }
#endif
#if NG > 3 && defined(TRISC_UNPACK)
    UnpackGather ug;
    ug.init(ready_n);
    if (nph) {
        ug.NPH = nph;
    }
#endif
#if defined(FASTMAC) && defined(TRISC_MATH)
    MathSeg ms;
    ms.init(ready_n);
    if (nph) {
        ms.NPH = nph;
    }
#endif
    for (uint32_t t = 0; t < nseg; ++t) {
#if NG > 2 && defined(TRISC_PACK)
        pg.segment();
#endif
#if NG > 3 && defined(TRISC_UNPACK)
        ug.segment();
#endif
        tile_regs_acquire();
#ifdef GS
        const bool gsseg = t < nseg_gs;
#endif
#ifdef FASTMAC
        uint32_t f0 = 0, f1 = 4;
#ifdef TRISC_MATH
        ms.next();
        f0 = ms.f0;
        f1 = ms.f1;
#endif
        (void)f0;
        (void)f1;
        copy_init(CB_V);  // every CB holds fp32 tiles: one init per segment
        MATH((llk_math_eltwise_binary_sfpu_init<SfpuType::unused>()));
#endif
        for (uint32_t j = 0; j < NSLOT; ++j) {
            const uint32_t ow = owner(j);
            const uint32_t cg = ow == 0 ? CB_G : (ow == 1 ? CB_G2 : (ow == 2 ? CB_G3 : CB_G4));
#ifdef OFFW
            const uint32_t cv = ow == 1 ? CB_V2 : CB_V;  // the writer reads its own slots' matrix values
#else
            const uint32_t cv = CB_V;
#endif
            cb_wait_front(cv, 1);
#if NG > 3 && defined(TRISC_UNPACK)
            if (ow == 3) {
                ug.gather(j);  // into CB_G4's read slot (this thread is producer and consumer)
            } else
#endif
                cb_wait_front(cg, 1);
#ifndef FASTMAC
            copy_init(cv);
#endif
            copy_tile(cv, 0, 1);
#ifdef GS
            if (j == 13 && gsseg) {
                copy_tile(cv, 0, 4);
            }
#endif
#ifndef FASTMAC
            copy_init(cg);
#endif
            copy_tile(cg, 0, 2);
#ifdef GS
            if (j == 13 && gsseg) {
                copy_tile(cg, 0, 5);
            }
#endif
            cb_pop_front(cv, 1);
            cb_pop_front(cg, 1);
#ifdef FASTMAC
            MATH((hpcg_mac(f0, f1, j == 0)));
#else
            mul_binary_tile_init();
            if (j == 0) {
                mul_binary_tile(1, 2, 0);
            } else {
                mul_binary_tile(1, 2, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
#endif
        }
        if (flags & F_ADDB) {  // prolongation: z = z + P zc (b = the output vector's old value)
            cb_wait_front(CB_B, 1);
            copy_init(CB_B);
            copy_tile(CB_B, 0, 1);
            cb_pop_front(CB_B, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
#ifdef GS
        if ((flags & F_RESTF) && !gsseg && t < nseg_gs + nseg_r) {  // restriction input: r - A z on block 0
            cb_wait_front(CB_B, 1);
            copy_init(CB_B);
            copy_tile(CB_B, 0, 1);
            cb_pop_front(CB_B, 1);
            sub_binary_tile_init();
            sub_binary_tile(1, 0, 0);
        }
#endif
#ifdef GS
        if (gsseg) {
            cb_wait_front(CB_B, 1);
            copy_init(CB_B);
            copy_tile(CB_B, 0, 1);
            cb_pop_front(CB_B, 1);
#ifdef FASTMAC
            div_binary_tile_init();  // reciprocal constants + binary SFPU address modes
            MATH((hpcg_gs(f0, f1)));
        }
#else
            sub_binary_tile_init();
            sub_binary_tile(1, 0, 0);  // b - s
            mul_binary_tile_init();
            mul_binary_tile(4, 5, 5);  // d * x
            add_binary_tile_init();
            add_binary_tile(0, 5, 0);  // b - s + d*x
            div_binary_tile_init();
            div_binary_tile(0, 4, 0);  // / d
        }
#endif
#endif
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_OUT, 1);
        pack_tile(0, CB_OUT);
        cb_push_back(CB_OUT, 1);
        tile_regs_release();
    }
}
