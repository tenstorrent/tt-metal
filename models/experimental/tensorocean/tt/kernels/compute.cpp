// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/reg_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/tilize.h"
#include "api/compute/transpose.h"
#include "api/compute/pack_untilize.h"
#include "common.h"
#ifdef FUSED_IN
// HW input rearranging (relayout_fused.h, hw_read / hw_write): per plane unit, tilize the natural rows (lossless
// fp32), transpose each tile, untilize: natural column c becomes row c of CB_TOUT. Units are padded to UNIT_PAGES.
template <uint32_t WT>
inline void hw_unit() {
    tilize_init(CB_RIN, R::WT_MAX, CB_TIL);
    cb_reserve_back(CB_TIL, R::UNIT_PAGES);
    for (uint32_t h = 0; h < R::TR; ++h) {
        cb_wait_front(CB_RIN, R::WT_MAX);
        tilize_block(CB_RIN, WT, CB_TIL, 0, h * WT);
        cb_pop_front(CB_RIN, R::WT_MAX);
    }
    cb_push_back(CB_TIL, R::UNIT_PAGES);
    tilize_uninit(CB_RIN, CB_TIL);
    cb_wait_front(CB_TIL, R::UNIT_PAGES);
    transpose_init(CB_TIL);
    pack_untilize_dest_init<R::TR, R::TR>(CB_TOUT);
    for (uint32_t w = 0; w < WT; ++w) {
        cb_reserve_back(CB_TOUT, R::TR);
        tile_regs_acquire();
        for (uint32_t h = 0; h < R::TR; ++h) {
            transpose_tile(CB_TIL, h * WT + w, h);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest<R::TR, R::TR>(CB_TOUT);
        tile_regs_release();
        cb_push_back(CB_TOUT, R::TR);
    }
    pack_untilize_uninit(CB_TOUT);
    cb_pop_front(CB_TIL, R::UNIT_PAGES);
    if constexpr (WT < R::WT_MAX) {
        cb_reserve_back(CB_TOUT, (R::WT_MAX - WT) * R::TR);
        cb_push_back(CB_TOUT, (R::WT_MAX - WT) * R::TR);
    }
}
#ifdef FUSED_OUT
// output unit: CB_RIN slot = TRO * 32 rows (natural columns) x TR * 32 floats -> CB_TOUT rows = natural output rows
inline void hw_out(uint32_t x, uint32_t nl) {
    for (uint32_t a = x; a < 2 * nl; a += R::NBX) {
        tilize_init(CB_RIN, R::TR, CB_TIL);
        cb_wait_front(CB_RIN, R::UNIT_PAGES);
        cb_reserve_back(CB_TIL, R::UNIT_PAGES);
        for (uint32_t h = 0; h < R::TRO; ++h) {
            tilize_block(CB_RIN, R::TR, CB_TIL, 0, h * R::TR);
            cb_pop_front(CB_RIN, R::TR);
        }
        cb_pop_front(CB_RIN, R::UNIT_PAGES - R::TRO * R::TR);
        cb_push_back(CB_TIL, R::UNIT_PAGES);
        tilize_uninit(CB_RIN, CB_TIL);
        cb_wait_front(CB_TIL, R::UNIT_PAGES);
        transpose_init(CB_TIL);
        pack_untilize_dest_init<R::TRO, R::TRO>(CB_TOUT);
        for (uint32_t w = 0; w < R::TR; ++w) {
            cb_reserve_back(CB_TOUT, R::TRO);
            tile_regs_acquire();
            for (uint32_t h = 0; h < R::TRO; ++h) {
                transpose_tile(CB_TIL, h * R::TR + w, h);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<R::TRO, R::TRO>(CB_TOUT);
            tile_regs_release();
            cb_push_back(CB_TOUT, R::TRO);
        }
        pack_untilize_uninit(CB_TOUT);
        cb_pop_front(CB_TIL, R::UNIT_PAGES);
        cb_reserve_back(CB_TOUT, R::UNIT_PAGES - R::TR * R::TRO);
        cb_push_back(CB_TOUT, R::UNIT_PAGES - R::TR * R::TRO);
    }
}
#endif
inline void hw_transpose(uint32_t x, uint32_t nl) {
    compute_kernel_hw_startup(CB_RIN, CB_TOUT);
    for (uint32_t i = x; i < 10 * nl; i += R::NBX) {  // unit i = (level l0 + i / 10, k = i % 10), as hw_read
        const uint32_t k = i % 10;
        if (k < 2) {
            hw_unit<R::WT_CELL>();
        } else if (k < 6) {
            hw_unit<R::WT_F2>();
        } else {
            hw_unit<R::WT_F1>();
        }
    }
}
#endif
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"
#include "sfpu_shift.h"
using namespace sfpi;
// DEST as 64 chunks of 128 fp32 (4 SFPU vectors each); D(k) = vector of chunk k at the current lane offset
#define D(k) dst_reg[(k) * 4]
// statics: chunks 0..23 (tiles 0-2); level A operands: 24..35; level B: 36..47; F out: 48..63 (tiles 6-7)
#define TAP(i)        \
    gA = D(26 + i);   \
    gB = D(38 + i);   \
    s = D(i);         \
    PA = s * gA + PA; \
    PB = s * gB + PB; \
    s = D(10 + i);    \
    QA = s * gA + QA; \
    QB = s * gB + QB;
template <int G, uint32_t LA>
inline void flux_pair() {
    constexpr int la = P::LOW_A[G], lb = P::LOW_B[G];
#pragma GCC unroll 1
    for (uint32_t v = 0; v < 4; ++v) {
        vFloat gA = D(26), gB = D(38), s = D(0);
        vFloat PA = s * gA, PB = s * gB;
        s = D(10);
        vFloat QA = s * gA, QB = s * gB;
        TAP(1) TAP(2) TAP(3) TAP(4) TAP(5) TAP(6) TAP(7) TAP(8) TAP(9) vFloat f = D(24);
        vFloat fm = f * D(25);
        vFloat r = (copysgn(vFloat(0.25f), f) * QA + PA) * fm;
        if constexpr (la >= 0) {
            r = (f - fm) * D(20) * (vFloat(D(26 + la)) + D(26 + lb)) + r;
        }
        if constexpr (G == 1) {
            D(48 + LA) = vFloat(D(48 + LA)) + r;
        } else {
            D(48 + LA) = r;  // group 1 adds onto group 0
        }
        f = D(36);
        fm = f * D(37);
        r = (copysgn(vFloat(0.25f), f) * QB + PB) * fm;
        if constexpr (la >= 0) {
            r = (f - fm) * D(20) * (vFloat(D(38 + la)) + D(38 + lb)) + r;
        }
        if constexpr (G == 1) {
            D(48 + LA + 1) = vFloat(D(48 + LA + 1)) + r;
        } else {
            D(48 + LA + 1) = r;
        }
        dst_reg++;
    }
}
// output stage: chunks 0..5 F terms level A, 6..11 level B, 12 inverse area; out chunks 16 + li
template <uint32_t LA>
inline void out_pair() {
#pragma GCC unroll 1
    for (uint32_t v = 0; v < 4; ++v) {
        vFloat inv = D(10);
        D(16 + LA) = (vFloat(D(0)) + D(1) + D(2) + D(3) + D(4)) * inv;
        D(16 + LA + 1) = (vFloat(D(5)) + D(6) + D(7) + D(8) + D(9)) * inv;
        dst_reg++;
    }
}
template <int G>
inline void flux_dispatch(uint32_t lp) {
    switch (lp) {
        case 0: flux_pair<G, 0>(); break;
        case 2:
            if constexpr (P::LP > 2) {
                flux_pair<G, 2>();
            }
            break;
        case 4:
            if constexpr (P::LP > 4) {
                flux_pair<G, 4>();
            }
            break;
        case 6:
            if constexpr (P::LP > 6) {
                flux_pair<G, 6>();
            }
            break;
        case 8:
            if constexpr (P::LP > 8) {
                flux_pair<G, 8>();
            }
            break;
        case 10:
            if constexpr (P::LP > 10) {
                flux_pair<G, 10>();
            }
            break;
        case 12:
            if constexpr (P::LP > 12) {
                flux_pair<G, 12>();
            }
            break;
        case 14:
            if constexpr (P::LP > 14) {
                flux_pair<G, 14>();
            }
            break;
    }
}
inline void flux(uint32_t g, uint32_t lp) {
    _llk_math_eltwise_sfpu_start_(0);
    switch (g) {
        case 0: flux_dispatch<0>(lp); break;
        case 1: flux_dispatch<1>(lp); break;
        case 2: flux_dispatch<2>(lp); break;
        case 3: flux_dispatch<3>(lp); break;
        case 4: flux_dispatch<4>(lp); break;
        case 5: flux_dispatch<5>(lp); break;
    }
    _llk_math_eltwise_sfpu_done_();
}
// A' (array 0 shifted by one item) of every level into chunks 24 + li (tiles 3-4, free after the last pair)
inline void make_shifted(uint32_t nl) {
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 1
    for (uint32_t li = 0; li < nl; ++li) {
        sh_chunk<48 * 4, 24 * 4>();
        dst_reg++;
        dst_reg++;
        dst_reg++;
        dst_reg++;
    }
    _llk_math_eltwise_sfpu_done_();
}
// shifted plane copy: tiles 0-2 hold plane p, tiles 3-5 get plane[i + R] (R = 1 or 2) for the CELL_LEN floats
// pair v (64 floats) = vectors 2v (even floats) and 2v + 1 (odd floats); the next pair supplies the carry.
// sh_next<S>: vector S with every row rotated left one lane, lane 7 taken from the next row's lane 0 (the
// last row's from vector S + 2). The next vector is re-read from DEST: a register copy got aliased (v22 debug).
template <uint32_t S>
inline vFloat sh_next() {
    vFloat t1 = dst_reg[S + 2], a = 0.0f, b = 0.0f;
    vFloat y = dst_reg[S];
    subvec_transp(y, t1, a, b);  // t1 = [y.r1, n.r1, ..], a = [y.r2, ..], b = [y.r3, ..]
    vFloat t4 = dst_reg[S + 2];
    subvec_transp(t1, a, b, t4);  // t1 = [y.r1, y.r2, y.r3, n.r0]
    vFloat m = dst_reg[S];
    vFloat lane0 = vFloat(subvec_shflshr1(vFloat(1.0f)));  // 0 at lane 0, 1 elsewhere
    v_if(lane0 == 0.0f) { m = t1; }
    v_endif;
    return sh_rotl1(m);
}
template <uint32_t R>
inline void shift_plane() {
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 1
    for (uint32_t v = 0; v < P::CELL_LEN / 64; ++v) {
        if constexpr (R == 1) {
            dst_reg[96] = vFloat(dst_reg[1]);
            dst_reg[97] = sh_next<0>();
        } else {
            dst_reg[96] = sh_next<0>();
            dst_reg[97] = sh_next<1>();
        }
        dst_reg++;
        dst_reg++;
    }
    _llk_math_eltwise_sfpu_done_();
}
inline void outc(uint32_t lp) {
    _llk_math_eltwise_sfpu_start_(0);
    switch (lp) {
        case 0: out_pair<0>(); break;
        case 2:
            if constexpr (P::LP > 2) {
                out_pair<2>();
            }
            break;
        case 4:
            if constexpr (P::LP > 4) {
                out_pair<4>();
            }
            break;
        case 6:
            if constexpr (P::LP > 6) {
                out_pair<6>();
            }
            break;
        case 8:
            if constexpr (P::LP > 8) {
                out_pair<8>();
            }
            break;
        case 10:
            if constexpr (P::LP > 10) {
                out_pair<10>();
            }
            break;
        case 12:
            if constexpr (P::LP > 12) {
                out_pair<12>();
            }
            break;
        case 14:
            if constexpr (P::LP > 14) {
                out_pair<14>();
            }
            break;
    }
    _llk_math_eltwise_sfpu_done_();
}
#endif

#define CLK() (*(volatile uint32_t*)RISCV_DEBUG_REG_WALL_CLOCK_L)
void kernel_main() {
    uint32_t T[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t t_start = CLK();
    uint32_t t0;
#ifdef FUSED_IN
    hw_transpose(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(0));  // my share of the input rearranging
#endif
    compute_kernel_hw_startup(CB_STAT, CB_F);
    copy_init(CB_STAT);
    const uint32_t nl_core = get_arg_val<uint32_t>(0);
    const uint32_t npass = get_arg_val<uint32_t>(1);
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t nl = nl_core > q * P::LP ? (nl_core - q * P::LP < P::LP ? nl_core - q * P::LP : P::LP) : 0;
        if (nl == 0) {
            continue;
        }
        uint32_t ob_done = 0;
        // shifted plane copies (slots 2..): unpack NTP tiles of the source plane from its L1 slot, shift on the SFPU,
        // pack NTP tiles into the copy's slot (the overrun past CELL_LEN lands in the next slot, written after, or the
        // pad)
        {
            uint32_t pbase = 0;
            UNPACK(pbase = get_local_cb_interface(CB_PLANES).fifo_rd_ptr << 4);
            PACK(pbase = get_local_cb_interface(CB_PLANES).fifo_wr_ptr << 4);
#ifndef HOSTSHIFT
            for (uint32_t li = 0; li < nl; ++li) {
                const uint32_t tok = (li & 1) ? CB_TOK6 : CB_TOK4;  // level li's planes are in L1
                cb_wait_front(tok, 1);
                cb_pop_front(tok, 1);
                for (uint32_t k = 0; k < P::NSHIFT; ++k) {
                    const uint32_t lb = pbase + li * P::LVL_PITCH_B;
                    tile_regs_acquire();
                    UNPACK(get_local_cb_interface(CB_ADDR).fifo_rd_ptr = (lb + P::SHIFT_P[k] * plane_slot_bytes) >> 4);
                    copy_block(CB_ADDR, 0, 0, P::NTP);
                    MATH(if (P::SHIFT_R[k] == 1) shift_plane<1>(); else shift_plane<2>());
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK(get_local_cb_interface(CB_ADDR).fifo_wr_ptr = (lb + (2 + k) * plane_slot_bytes) >> 4);
                    for (uint32_t t = 0; t < P::NTP; ++t) {
                        pack_tile<true>(3 + t, CB_ADDR, t);
                    }
                    tile_regs_release_math_clear();
                }
            }
#endif
            cb_reserve_back(CB_TOK3, 1);
            cb_push_back(CB_TOK3, 1);
            cb_reserve_back(CB_TOK5, 1);
            cb_push_back(CB_TOK5, 1);
        }
#ifdef PHASES
        T[6] = CLK() - t_start;  // prologue: plane shifts (unpack thread view)
#endif
        DeviceZoneScopedN("C1");
        for (uint32_t b = 0; b < P::NBLK; ++b) {
            for (uint32_t g = 0; g < 6; ++g) {
                t0 = CLK();
                cb_wait_front(CB_STAT, 3);
                T[0] += CLK() - t0;
                t0 = CLK();
                if (g != 1) {
                    tile_regs_acquire();
                }
                T[2] += CLK() - t0;  // groups 0 and 1 share one DEST session
                t0 = CLK();
                copy_block(CB_STAT, 0, 0, 3);
                T[3] += CLK() - t0;
                for (uint32_t lp = 0; lp < nl; lp += 2) {
                    const uint32_t lcb = (lp & 2) ? CB_LVL2 : CB_LVL;  // odd pairs come from the writer
                    t0 = CLK();
                    cb_wait_front(lcb, 3);
                    T[1] += CLK() - t0;

#ifndef NOLVLCOPY
                    t0 = CLK();
                    copy_block(lcb, 0, 3, 3);
                    T[5] += CLK() - t0;
#endif

#ifndef NOFLUX
                    t0 = CLK();
                    MATH(flux(g, lp));
                    T[6] += CLK() - t0;
#endif

                    cb_pop_front(lcb, 3);
                }
                {  // the unused rest of this step's level tiles (cores with fewer than LP levels; see LVL_STEP_TILES)
                    const uint32_t npairs = (nl + 1) / 2, rest = LVL_STEP_TILES - 3 * ((npairs + 1) / 2);
                    const uint32_t nodd = npairs / 2, rest2 = nodd ? LVL2_STEP_TILES - 3 * nodd : 0;
                    if (rest) {
                        cb_wait_front(CB_LVL, rest);
                        cb_pop_front(CB_LVL, rest);
                    }
                    if (rest2) {
                        cb_wait_front(CB_LVL2, rest2);
                        cb_pop_front(CB_LVL2, rest2);
                    }
                }
                if (g != 0) {
                    if (g == 1) {
                        MATH(make_shifted(nl));
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    const uint32_t nt = g == 1 ? 4 : 2;  // array 0 also carries its shifted copy A'
                    t0 = CLK();
                    cb_reserve_back(CB_F, nt);
                    T[2] += CLK() - t0;
                    pack_tile(6, CB_F);
                    pack_tile(7, CB_F);
                    if (g == 1) {
                        pack_tile(3, CB_F);
                        pack_tile(4, CB_F);
                    }
                    cb_push_back(CB_F, nt);
                    tile_regs_release_math_clear();  // no DEST zeroing: every slot is rewritten before it is read
                }
                cb_pop_front(CB_STAT, 3);
            }
#ifdef PHASES
            const uint32_t tob = CLK();
#endif
            for (; ob_done < ob_target(b); ++ob_done) {
                for (uint32_t part = 0; part < 2; ++part) {
                    tile_regs_acquire();
                    for (uint32_t lp = 0; lp < nl; lp += 2) {
                        const uint32_t oi =
                            part ? CB_OI2 : CB_OI;  // even rows from the reader, odd rows from the writer
                        t0 = CLK();
                        cb_wait_front(oi, 2);
#ifdef SPLITOI
                        if (part) {
                            T[2] += CLK() - t0;
                        } else
#endif
#ifdef SPLITLAST
                            if (b + 1 == P::NBLK)
                            T[2] += CLK() - t0;
                        else
#endif
                            T[3] += CLK() - t0;
                        t0 = CLK();
                        copy_block(oi, 0, 0, 2);
                        T[4] += CLK() - t0;
                        t0 = CLK();
                        MATH(outc(lp));
                        T[6] += CLK() - t0;
                        cb_pop_front(oi, 2);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    t0 = CLK();
                    cb_reserve_back(CB_OUT, 2);
                    T[4] += CLK() - t0;
                    pack_tile(2, CB_OUT);
                    pack_tile(3, CB_OUT);
                    cb_push_back(CB_OUT, 2);
                    tile_regs_release_math_clear();  // no DEST zeroing: every slot is rewritten before it is read
                }
            }
#ifdef PHASES
            T[1] += CLK() - tob;  // whole output stage (replaces the level-wait counter)
#endif
        }
    }
#ifdef FUSED_OUT
    hw_out(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(0));  // the output units I assemble
#endif
    T[7] = CLK() - t_start;
#ifdef KDEBUG
#ifdef TRISC_MATH
    for (uint32_t i = 0; i < 8; ++i) {
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, T[i]);
    }
#endif
#ifdef TRISC_UNPACK
    {
        volatile uint32_t* w = (volatile uint32_t*)((get_local_cb_interface(CB_RARR).fifo_rd_ptr << 4) + 256);
        for (uint32_t i = 0; i < 8; ++i) {
            w[i] = T[i];
        }
        for (uint32_t i = 0; i < 8; ++i) {
            w[24 + i] = ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
        }
        w[16] = 0xC0FFEE;
    }
#endif
#ifdef TRISC_PACK
    {
        volatile uint32_t* w = (volatile uint32_t*)((get_local_cb_interface(CB_RARR).fifo_wr_ptr << 4) + 256);
        for (uint32_t i = 0; i < 8; ++i) {
            w[8 + i] = T[i];
        }
        w[17] = 0xC0FFEE;
    }
#endif
#endif
}
