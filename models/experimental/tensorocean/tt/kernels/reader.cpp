// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "common.h"

template <typename A>
inline void dram_read(const A& s, uint32_t off, uint32_t l1, uint32_t nbytes) {
    while (nbytes) {
        uint32_t page = off / P::PAGE, in = off % P::PAGE;
        uint32_t n = P::PAGE - in < nbytes ? P::PAGE - in : nbytes;
        noc_async_read(s.get_noc_addr(page) + in, l1, n);
        off += n;
        l1 += n;
        nbytes -= n;
    }
}
#define CLK() (*(volatile uint32_t*)RISCV_DEBUG_REG_WALL_CLOCK_L)

void kernel_main() {
    const uint32_t cell_addr = get_arg_val<uint32_t>(0);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_id = get_arg_val<uint32_t>(2);
    const uint32_t inv_addr = get_arg_val<uint32_t>(3);
    const uint32_t x = get_arg_val<uint32_t>(4);
    const uint32_t l0_core = get_arg_val<uint32_t>(5);
    const uint32_t nl_core = get_arg_val<uint32_t>(6);
    constexpr auto a_cell = TensorAccessorArgs<0>();
    constexpr auto a_fmk = TensorAccessorArgs<a_cell.next_compile_time_args_offset()>();
    constexpr auto a_stat = TensorAccessorArgs<a_fmk.next_compile_time_args_offset()>();
    constexpr auto a_inv = TensorAccessorArgs<a_stat.next_compile_time_args_offset()>();
    const auto CELL = TensorAccessor(a_cell, cell_addr, P::PAGE);
    const auto INV = TensorAccessor(a_inv, inv_addr, P::PAGE);
    const uint32_t fmk_addr = get_arg_val<uint32_t>(7);
    const auto FMK = TensorAccessor(a_fmk, fmk_addr, P::PAGE);
    const uint32_t fmkbuf = get_write_ptr(CB_FMK);  // my own 2-slot f/mask prefetch buffer
    const uint32_t sx = get_arg_val<uint32_t>(8), sy = get_arg_val<uint32_t>(9);  // statics sender of my column
    const uint32_t my_y = get_arg_val<uint32_t>(10);  // my slot in the sender's progress array
    const uint64_t my_prog = get_noc_addr(sx, sy, get_write_ptr(CB_RARR) + my_y * 16);
    volatile tt_l1_ptr uint32_t* valid = (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(11));
    const uint32_t stage = get_write_ptr(CB_STAGE);  // NS-slot ring the sender multicasts statics into
    constexpr uint32_t FMK_SLOT_B = P::LP * 2 * CHB;
    const uint32_t planes = get_write_ptr(CB_PLANES);
    const uint32_t fbuf = get_write_ptr(CB_FBUF);
    uint32_t T[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t t_start = CLK();
    uint32_t t0;

    const uint32_t npass = get_arg_val<uint32_t>(12);  // same for every core of the column (shared statics stream)
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t l0 = l0_core + q * P::LP;
        const uint32_t nl = nl_core > q * P::LP ? (nl_core - q * P::LP < P::LP ? nl_core - q * P::LP : P::LP) : 0;
        if (nl == 0) {  // nothing to compute: just keep the column's statics ring moving
            for (uint32_t k = 0; k < 6 * P::NBLK; ++k) {
                const uint32_t kk = q * 6 * P::NBLK + k;
                while (*valid < kk + 1) {
                    invalidate_l1_cache();
                }
                noc_inline_dw_write(my_prog, kk + 1);
            }
            continue;
        }
        t0 = CLK();
        // all plane slots of my levels (both planes + the host-made shifted copies): one contiguous run per level
        // v26: even levels, one at a time; compute shifts each level as soon as its token arrives (writer: odd levels)
        for (uint32_t li = 0; li < nl; li += 2) {
            dram_read(
                CELL,
                ((l0 + li) * P::NBX + x) * NSLOT_DRAM * plane_slot_bytes,
                planes + li * P::LVL_PITCH_B,
                NSLOT_DRAM * plane_slot_bytes);
            noc_async_read_barrier();
            cb_reserve_back(CB_TOK4, 1);
            cb_push_back(CB_TOK4, 1);
        }
        T[0] += CLK() - t0;
        uint32_t ob_done = 0, toks = 0;
        const uint32_t invbuf = get_write_ptr(CB_INV);  // this band's inverse areas, once per pass
        dram_read(INV, x * P::NOBLK * CHB, invbuf, P::NOBLK * CHB);
        auto fetch_fmk = [&](uint32_t k) {  // f/mask of step k = b * 6 + g, all my levels of this pass, into slot k % 2
            const uint32_t g = k % 6, b = k / 6;
            const uint32_t off = (((g * P::NBX + x) * P::NBLK + b) * P::L + l0) * 2 * CHB;
            for (uint32_t lp = 0; lp < nl;
                 lp += 4) {  // even pairs (levels lp, lp + 1) only: the writer does the odd ones
                dram_read(
                    FMK,
                    off + lp * 2 * CHB,
                    fmkbuf + (k & 1) * FMK_SLOT_B + lp * 2 * CHB,
                    (nl - lp < 2 ? nl - lp : 2) * 2 * CHB);
            }
        };
        // statics of step k go to compute one step ahead of its level tiles
        auto stat_copy = [&](uint32_t k) {
            const uint32_t kk = q * 6 * P::NBLK + k;
            uint32_t t1 = CLK();
            while (*valid < kk + 1) {
                invalidate_l1_cache();
            }
            T[1] += CLK() - t1;
            const uint32_t slot = stage + (kk % P::NS) * STAGE_STAT_B;
            t1 = CLK();
            cb_reserve_back(CB_STAT, 3);
            T[2] += CLK() - t1;
            noc_async_read(get_noc_addr(slot), get_write_ptr(CB_STAT), STAGE_STAT_B);
            noc_async_read_barrier();
            cb_push_back(CB_STAT, 3);
            noc_inline_dw_write(my_prog, kk + 1);  // I have consumed steps 0..kk: their ring slots are free again
        };
        fetch_fmk(0);
        stat_copy(0);
        if (6 * P::NBLK > 1) {
            stat_copy(1);
        }
        t0 = CLK();
        cb_wait_front(CB_TOK3, 1);
        cb_pop_front(CB_TOK3, 1);
        T[0] += CLK() - t0;  // shifted planes written
#ifdef DUMPSHIFT
        if (dbg_addr && core_id == 1) {  // debug: first 2048 floats of the 3 shifted slots of level 0
            for (uint32_t li = 0; li < 3; ++li) {
                for (uint32_t k = 0; k < 3; ++k) {
                    noc_async_write(
                        planes + li * P::LVL_PITCH_B + (2 + k) * plane_slot_bytes,
                        get_noc_addr_from_bank_id<true>(0, dbg_addr) + 8192 + (li * 3 + k) * 8192,
                        8192);
                }
            }
            noc_async_write_barrier();
        }
#endif
        // even-row output operands of output blocks < target; F of blocks < need must be stored by the writer.
        // v25: called early (after step 2 of the next block), CB_OI holds a whole batch so this never blocks compute
        auto out_asm = [&](uint32_t need, uint32_t target) {
            t0 = CLK();
            for (; toks < need; ++toks) {
                cb_wait_front(CB_TOK, 1);
                cb_pop_front(CB_TOK, 1);
            }
            T[5] += CLK() - t0;
            t0 = CLK();
            for (; ob_done < target; ++ob_done) {
                const uint32_t ob = ob_done;
                const uint32_t part = 0;
                for (uint32_t lp = 0; lp < nl; lp += 2) {
                    cb_reserve_back(CB_OI, 2);
                    const uint32_t dst = get_write_ptr(CB_OI);
                    noc_async_read_one_packet_set_state(get_noc_addr(0), CHB);
                    for (uint32_t j = 0; j < 2; ++j) {
                        const uint32_t li = lp + j;
                        if (li >= nl) {
                            break;
                        }
                        for (uint32_t t = 0; t < P::NT; ++t) {
                            const uint32_t g = P::TERM_G[part][t];  // F-array index (0 = ee + eo)
                            const uint32_t start = ob * P::CH + P::TERM_DC[part][t] * P::H + P::TERM_DR[part][t];
                            const uint32_t rho = start & 3;       // 0 or 1
                            const uint32_t s = rho ? P::NFG : g;  // the shifted copy of array 0 follows the NFG arrays
                            const uint32_t a = start - rho, b0 = a / P::CH, o = a % P::CH;
                            noc_async_read_one_packet_with_state(
                                frm_addr(fbuf, s, li, b0 % P::FR) + o * 4, dst + (j * P::NT + t) * CHB);
                        }
                    }
                    noc_async_read_one_packet_with_state(invbuf + ob * CHB, dst + 2 * P::NT * CHB);
                    noc_async_read_barrier();
                    cb_push_back(CB_OI, 2);
                }
            }
            T[6] += CLK() - t0;
        };
        for (uint32_t b = 0; b < P::NBLK; ++b) {
            for (uint32_t g = 0; g < 6; ++g) {
                const uint32_t k = b * 6 + g;
                if (k + 1 < 6 * P::NBLK) {
                    fetch_fmk(k + 1);  // in flight while this step is assembled
                }
                const uint32_t fslot = fmkbuf + (k & 1) * FMK_SLOT_B;
                uint32_t tap_src[10];
                for (uint32_t i = 0; i < 10; ++i) {
                    const uint32_t start = b * P::CH + P::TAP_OFF[g][i];
                    const uint32_t rho = start & 3;
                    tap_src[i] = planes + P::SLOT[P::TAP_P[g][i]][rho] * plane_slot_bytes + (start - rho) * 4;
                }
                noc_async_read_one_packet_set_state(get_noc_addr(0), CHB);
                const uint32_t npairs = (nl + 1) / 2, nmine = (npairs + 1) / 2;
                t0 = CLK();
                cb_reserve_back(CB_LVL, 3 * nmine);
                T[3] += CLK() - t0;
                const uint32_t dst0 = get_write_ptr(CB_LVL);
                t0 = CLK();
                for (uint32_t lp = 0; lp < nl; lp += 4) {  // even pairs
                    const uint32_t dst = dst0 + (lp / 4) * 3 * TILEB;
                    for (uint32_t j = 0; j < 2; ++j) {
                        const uint32_t li = lp + j;
                        if (li >= nl) {
                            break;
                        }
                        const uint32_t base = dst + j * 12 * CHB;
                        const uint32_t lvl_off = li * P::LVL_PITCH_B;
                        noc_async_read_one_packet_with_state(fslot + li * 2 * CHB, base);
                        noc_async_read_one_packet_with_state(fslot + li * 2 * CHB + CHB, base + CHB);
#pragma GCC unroll 10
                        for (uint32_t i = 0; i < 10; ++i) {
                            noc_async_read_one_packet_with_state(tap_src[i] + lvl_off, base + (2 + i) * CHB);
                        }
                    }
                }
                noc_async_read_barrier();
                T[4] += CLK() - t0;
                cb_push_back(CB_LVL, 3 * nmine);
                if (k + 2 < 6 * P::NBLK) {
                    stat_copy(k + 2);
                }
                if (g == 2 && b > 0 && b + 1 < P::NBLK) {
                    out_asm(b, ob_target(b));  // F of blocks < b is stored by now
                }
            }
            if (b + 1 == P::NBLK) {
                out_asm(b + 1, P::NOBLK);  // the last block: everything left, after its own F
            }
        }
    }
    T[7] = CLK() - t_start;
    if (dbg_addr) {
        volatile tt_l1_ptr uint32_t* w = (volatile tt_l1_ptr uint32_t*)get_write_ptr(CB_MCW);
        for (uint32_t i = 0; i < 8; ++i) {
            w[i] = T[i];
        }
        noc_async_write(get_write_ptr(CB_MCW), get_noc_addr_from_bank_id<true>(0, dbg_addr) + core_id * 64, 32);
        w[8] = t_start;
        w[9] = CLK();  // absolute wall clock: kernel start / end (skew check)
        noc_async_write(
            get_write_ptr(CB_MCW) + 32, get_noc_addr_from_bank_id<true>(0, dbg_addr) + 21120 + core_id * 16, 16);
        noc_async_write_barrier();
    }
}
