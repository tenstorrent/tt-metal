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
    uint32_t T[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t t_start = CLK();
    uint32_t t0;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t x = get_arg_val<uint32_t>(1);
    const uint32_t l0_core = get_arg_val<uint32_t>(2);
    const uint32_t nl_core = get_arg_val<uint32_t>(3);
    const uint32_t fmk_addr = get_arg_val<uint32_t>(4);
    const uint32_t stat_addr = get_arg_val<uint32_t>(5);
    const uint32_t nrecv = get_arg_val<uint32_t>(6);
    const uint32_t is_sender = get_arg_val<uint32_t>(7);
    const uint32_t mx0 = get_arg_val<uint32_t>(8), my0 = get_arg_val<uint32_t>(9);
    const uint32_t mx1 = get_arg_val<uint32_t>(10), my1 = get_arg_val<uint32_t>(11);
    const uint32_t sx = get_arg_val<uint32_t>(12), sy = get_arg_val<uint32_t>(13);
    const uint32_t ready_sem = get_semaphore(get_compile_time_arg_val(0));
    const uint32_t valid_sem = get_semaphore(get_compile_time_arg_val(1));
    volatile tt_l1_ptr uint32_t* ready = (volatile tt_l1_ptr uint32_t*)ready_sem;
    volatile tt_l1_ptr uint32_t* valid = (volatile tt_l1_ptr uint32_t*)valid_sem;
    const uint32_t valid_word = get_write_ptr(CB_MCW);
    auto mcast = [&](uint32_t addr) {
        return noc_index == 0 ? get_noc_multicast_addr(mx0, my0, mx1, my1, addr)
                              : get_noc_multicast_addr(mx1, my1, mx0, my0, addr);
    };
    constexpr auto a_out = TensorAccessorArgs<2>();
    constexpr auto a_fmk = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    constexpr auto a_stat = TensorAccessorArgs<a_fmk.next_compile_time_args_offset()>();
    const auto OUT = TensorAccessor(a_out, out_addr, P::PAGE);
    const auto FMK = TensorAccessor(a_fmk, fmk_addr, P::PAGE);
    const auto STAT = TensorAccessor(a_stat, stat_addr, P::PAGE);
    constexpr auto a_inv = TensorAccessorArgs<a_stat.next_compile_time_args_offset()>();
    const auto INV = TensorAccessor(a_inv, get_arg_val<uint32_t>(17), P::PAGE);
    constexpr auto a_cell = TensorAccessorArgs<a_inv.next_compile_time_args_offset()>();
    const auto CELL = TensorAccessor(a_cell, get_arg_val<uint32_t>(18), P::PAGE);
    constexpr uint32_t K = 6 * P::NBLK;  // steps k = b * 6 + g
    uint32_t l0 = 0, nl = 0, kbase = 0, next = 0;
    const uint32_t sstage = get_write_ptr(CB_SSTAGE);  // sender-only 2-slot DRAM prefetch of statics
    const uint32_t stage = get_write_ptr(CB_STAGE);    // the NS-slot ring on every core of the column
    const uint32_t ncol = nrecv + 1;                   // readers that consume each step (incl. mine)
    // v27: statics are fetched SB steps per DRAM barrier, double-buffered (batch j in half j & 1 of sstage)
    auto sread_batch = [&](uint32_t j) {
        for (uint32_t i = 0; i < P::SB; ++i) {
            const uint32_t k = j * P::SB + i;
            if (k >= K) {
                break;
            }
            const uint32_t g = k % 6, b = k / 6;
            dram_read(
                STAT,
                ((g * P::NBX + x) * P::NBLK + b) * 24 * CHB,
                sstage + ((j & 1) * P::SB + i) * STAGE_STAT_B,
                24 * CHB);
        }
    };
    uint32_t next_mc = 0;
    volatile tt_l1_ptr uint32_t* prog =
        (volatile tt_l1_ptr uint32_t*)get_write_ptr(CB_RARR);  // per-reader consumed steps
    auto ready_ok = [&](uint32_t k) {                          // every reader of the column has consumed step kk - NS
        const uint32_t kk = kbase + k;
        if (kk < P::NS) {
            return true;
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i < ncol; ++i) {
            if (prog[i * 4] < kk - P::NS + 1) {
                return false;
            }
        }
        return true;
    };
    auto mc_one = [&](uint32_t k) {  // multicast the statics of step k into ring slot kk % NS on the column
        const uint32_t kk = kbase + k;
        uint32_t tq = CLK();
        while (!ready_ok(k)) {
        }
        T[0] += CLK() - tq;
        tq = CLK();
        if (k % P::SB == 0) {  // batch k / SB has landed; fetch the next one into the other half
            noc_async_read_barrier();
            if ((k / P::SB + 1) * P::SB < K) {
                sread_batch(k / P::SB + 1);
            }
        }
        T[5] += CLK() - tq;
        tq = CLK();
        const uint32_t src = sstage + (((k / P::SB) & 1) * P::SB + k % P::SB) * STAGE_STAT_B;
        const uint32_t dst = stage + (kk % P::NS) * STAGE_STAT_B;
        *(volatile tt_l1_ptr uint32_t*)valid_word = kk + 1;
        if (nrecv) {
            noc_async_write_multicast_loopback_src(src, mcast(dst), 24 * CHB, ncol);
#ifdef MC_BARRIER
            noc_async_write_barrier();  // the data has landed on every core before the flag says so
#endif
            noc_semaphore_set_multicast_loopback_src(valid_word, mcast(valid_sem), ncol);
        } else {
            noc_async_write(src, get_noc_addr(dst), 24 * CHB);
            noc_async_write_barrier();
            *valid = kk + 1;
        }
        noc_async_writes_flushed();
        T[6] += CLK() - tq;
    };
    if (is_sender) {
        for (uint32_t i = 0; i < ncol; ++i) {
            prog[i * 4] = 0;
        }
    }
    const uint32_t planes = get_write_ptr(CB_PLANES);
    const uint32_t fbuf = get_write_ptr(CB_FBUF);

    const uint32_t npass = get_arg_val<uint32_t>(16);
    for (uint32_t q = 0; q < npass; ++q) {
        l0 = l0_core + q * P::LP;
        nl = nl_core > q * P::LP ? (nl_core - q * P::LP < P::LP ? nl_core - q * P::LP : P::LP) : 0;
        kbase = q * K;
        next_mc = 0;
        if (is_sender) {
            sread_batch(0);
        }
        if (nl == 0) {  // no levels this pass: only the multicast duty
            if (is_sender) {
                while (next_mc < K) {
                    mc_one(next_mc++);
                }
            }
            continue;
        }  // overlaps the reader's plane load
        uint32_t ob_done = 0;
        // my half of the plane levels (the reader loads levels [0, nl/2)), then tell the reader
        for (uint32_t li = 1; li < nl; li += 2) {  // odd levels (the reader loads the even ones)
            dram_read(
                CELL,
                ((l0 + li) * P::NBX + x) * NSLOT_DRAM * plane_slot_bytes,
                planes + li * P::LVL_PITCH_B,
                NSLOT_DRAM * plane_slot_bytes);
            noc_async_read_barrier();
            cb_reserve_back(CB_TOK6, 1);
            cb_push_back(CB_TOK6, 1);
        }
        const uint32_t invbuf = get_write_ptr(CB_INV2);
        dram_read(INV, x * P::NOBLK * CHB, invbuf, P::NOBLK * CHB);
        noc_async_read_barrier();
        // v24: the writer assembles the odd level pairs (levels 4m + 2, 4m + 3) of every step into CB_LVL2
        const uint32_t fmkbuf2 = get_write_ptr(CB_FMK2);
        constexpr uint32_t FMK_SLOT_B2 = P::LP * 2 * CHB;
        const uint32_t nodd = ((nl + 1) / 2) / 2;
        auto fetch_fmk2 = [&](uint32_t k) {
            const uint32_t g = k % 6, b = k / 6;
            const uint32_t off = (((g * P::NBX + x) * P::NBLK + b) * P::L + l0) * 2 * CHB;
            for (uint32_t lp = 2; lp < nl; lp += 4) {
                dram_read(
                    FMK,
                    off + lp * 2 * CHB,
                    fmkbuf2 + (k & 1) * FMK_SLOT_B2 + lp * 2 * CHB,
                    (nl - lp < 2 ? nl - lp : 2) * 2 * CHB);
            }
        };
        uint32_t next_asm = 0;
        auto asm_one = [&]() {
            const uint32_t k = next_asm++;
            const uint32_t g = k % 6, b = k / 6;
            if (k + 1 < K) {
                fetch_fmk2(k + 1);
            }
            const uint32_t fslot = fmkbuf2 + (k & 1) * FMK_SLOT_B2;
            uint32_t tap_src[10];
            for (uint32_t i = 0; i < 10; ++i) {
                const uint32_t start = b * P::CH + P::TAP_OFF[g][i];
                const uint32_t rho = start & 3;
                tap_src[i] = planes + P::SLOT[P::TAP_P[g][i]][rho] * plane_slot_bytes + (start - rho) * 4;
            }
            cb_reserve_back(CB_LVL2, 3 * nodd);
            const uint32_t dst0 = get_write_ptr(CB_LVL2);
            noc_async_read_one_packet_set_state(get_noc_addr(0), CHB);
            for (uint32_t lp = 2; lp < nl; lp += 4) {
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
                    for (uint32_t i = 0; i < 10; ++i)
                        noc_async_read_one_packet_with_state(tap_src[i] + lvl_off, base + (2 + i) * CHB);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_LVL2, 3 * nodd);
        };
        auto pump = [&](uint32_t limit) {  // assemble ahead without blocking
            if (is_sender && next_mc < K && ready_ok(next_mc)) {
                mc_one(next_mc++);  // statics first: the column waits on them
            }
            if (nodd && next_asm < K && next_asm < limit && cb_pages_reservable_at_back(CB_LVL2, 3 * nodd)) {
                asm_one();
            }
        };
        cb_wait_front(CB_TOK5, 1);  // the shifted plane copies are written
        cb_pop_front(CB_TOK5, 1);
        if (nodd) {
            fetch_fmk2(0);
            noc_async_read_barrier();
        }  // step 0 is copied right away: its f/m must have landed
        // odd-row output operands of output blocks < target (F of the blocks they need is stored by me already)
        uint32_t ob_w = 0;
        auto out_asm = [&](uint32_t target) {
            for (; ob_done < target; ++ob_done) {
                const uint32_t ob = ob_done;
                // odd-row operand tiles of this output block (the reader does the even rows)
                for (uint32_t lp = 0; lp < nl; lp += 2) {
                    while (!cb_pages_reservable_at_back(CB_OI2, 2)) {
                        pump(next_asm + 1);
                    }
                    cb_reserve_back(CB_OI2, 2);
                    const uint32_t dst = get_write_ptr(CB_OI2);
                    noc_async_read_one_packet_set_state(get_noc_addr(0), CHB);
                    for (uint32_t j = 0; j < 2; ++j) {
                        const uint32_t li = lp + j;
                        if (li >= nl) {
                            break;
                        }
                        for (uint32_t t = 0; t < P::NT; ++t) {
                            const uint32_t g = P::TERM_G[1][t];
                            const uint32_t start = ob * P::CH + P::TERM_DC[1][t] * P::H + P::TERM_DR[1][t];
                            const uint32_t rho = start & 3;
                            const uint32_t s = rho ? P::NFG : g;
                            const uint32_t a = start - rho, b0 = a / P::CH, o = a % P::CH;
                            const uint32_t d = dst + (j * P::NT + t) * CHB;
                            noc_async_read_one_packet_with_state(frm_addr(fbuf, s, li, b0 % P::FR) + o * 4, d);
                        }
                    }
                    noc_async_read_one_packet_with_state(invbuf + ob * CHB, dst + 2 * P::NT * CHB);
                    noc_async_read_barrier();
                    cb_push_back(CB_OI2, 2);
                }
            }
        };
        DeviceZoneScopedN("W1");
        for (uint32_t b = 0; b < P::NBLK; ++b) {
            for (uint32_t g = 0; g < 6; ++g) {
                const uint32_t k = b * 6 + g;
                while (nodd && next_asm <= k) {
                    asm_one();  // compute needs step k's odd pairs before F(k)
                }
                t0 = CLK();
                if (is_sender) {
                    while (next_mc <= k) {
                        mc_one(next_mc++);  // step k must be out
                    }
                    while (next_mc < K && ready_ok(next_mc)) {
                        mc_one(next_mc++);  // and ahead as far as the ring allows
                    }
                }
                T[1] += CLK() - t0;
                if (g == 0) {
                    continue;  // groups 0 and 1 produce one fused F array, packed after group 1
                }
                const uint32_t fg = g - 1;  // F-array index
                t0 = CLK();
                const uint32_t nt = fg == 0 ? 4 : 2;  // array 0 arrives with its shifted copy A' (tiles 2-3)
                while (!cb_pages_available_at_front(CB_F, nt)) {  // keep the column's statics flowing while waiting
                    if (is_sender && next_mc < K && ready_ok(next_mc)) {
                        mc_one(next_mc++);
                    }
                    pump(k + 3);
                }
                cb_wait_front(CB_F, nt);
                T[2] += CLK() - t0;
                t0 = CLK();
                const uint32_t src = get_read_ptr(CB_F);
                const uint32_t sl = b % P::FR;
                const bool mirror = sl == 0;  // slot 0 is also kept in the mirror slot FR
                for (uint32_t li = 0; li < nl; ++li) {
                    noc_async_write(src + li * CHB, get_noc_addr(frm_addr(fbuf, fg, li, sl)), CHB);
                    if (mirror) {
                        noc_async_write(src + li * CHB, get_noc_addr(frm_addr(fbuf, fg, li, P::FR)), CHB);
                    }
                }
                if (fg == 0) {
                    for (uint32_t li = 0; li < nl; ++li) {
                        noc_async_write(src + 2 * TILEB + li * CHB, get_noc_addr(frm_addr(fbuf, P::NFG, li, sl)), CHB);
                        if (mirror) {
                            noc_async_write(
                                src + 2 * TILEB + li * CHB, get_noc_addr(frm_addr(fbuf, P::NFG, li, P::FR)), CHB);
                        }
                    }
                    noc_async_write_barrier();
                    // the last item of the previous block's A' is this block's first A item
                    if (b) {
                        const uint32_t ps = (b - 1) % P::FR;
                        for (uint32_t li = 0; li < nl; ++li) {
                            const uint32_t v = ((volatile uint32_t*)(src + li * CHB))[0];
                            ((volatile uint32_t*)frm_addr(fbuf, P::NFG, li, ps))[P::CH - 1] = v;
                            if (ps == 0) {
                                ((volatile uint32_t*)frm_addr(fbuf, P::NFG, li, P::FR))[P::CH - 1] = v;
                            }
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(CB_F, nt);
                T[3] += CLK() - t0;
                if (g == 2 && b > 0 && b + 1 < P::NBLK) {
                    out_asm(ob_target(b));  // early: CB_OI2 holds the whole batch
                }
            }
            cb_reserve_back(CB_TOK, 1);  // F of block b is stored
            cb_push_back(CB_TOK, 1);
            for (; ob_w < ob_target(b); ++ob_w) {
                if (b + 1 == P::NBLK) {
                    out_asm(ob_w + 1);  // last block: one output block at a time (CB_OUT holds one)
                }
                for (uint32_t part = 0; part < 2; ++part) {
                    t0 = CLK();
                    while (!cb_pages_available_at_front(CB_OUT, 2)) {
                        if (is_sender && next_mc < K && ready_ok(next_mc)) {
                            mc_one(next_mc++);
                        }
                        pump((b + 1) * 6 + 2);
                    }
                    cb_wait_front(CB_OUT, 2);
                    T[4] += CLK() - t0;
                    const uint32_t src = get_read_ptr(CB_OUT);
                    for (uint32_t li = 0; li < nl; ++li) {
                        const uint32_t off = ((((l0 + li) * 2 + part) * P::NBX + x) * P::NOBLK + ob_w) * CHB;
                        noc_async_write(src + li * CHB, OUT.get_noc_addr(off / P::PAGE) + off % P::PAGE, CHB);
                    }
                    noc_async_write_barrier();
                    cb_pop_front(CB_OUT, 2);
                }
            }
        }
    }
    T[7] = CLK() - t_start;
    {
#ifdef KDEBUG
        volatile tt_l1_ptr uint32_t* cw = (volatile tt_l1_ptr uint32_t*)(get_write_ptr(CB_RARR) + 256);
        while (cw[16] != 0xC0FFEE || cw[17] != 0xC0FFEE) {
            invalidate_l1_cache();
        }
        if (get_arg_val<uint32_t>(14)) {
            noc_async_write(
                get_write_ptr(CB_RARR) + 256,
                get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(14)) + 110 * 64 +
                    get_arg_val<uint32_t>(15) * 64,
                64);
        }
        noc_async_write_barrier();
        cw[16] = 0;
        cw[17] = 0;
#endif
        volatile tt_l1_ptr uint32_t* ww = (volatile tt_l1_ptr uint32_t*)get_write_ptr(CB_MCW);
        for (uint32_t i = 0; i < 8; ++i) {
            ww[i] = T[i];
        }
        if (get_arg_val<uint32_t>(14)) {
            noc_async_write(
                get_write_ptr(CB_MCW),
                get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(14)) + get_arg_val<uint32_t>(15) * 64 + 32,
                32);
        }
        noc_async_write_barrier();
    }
}
