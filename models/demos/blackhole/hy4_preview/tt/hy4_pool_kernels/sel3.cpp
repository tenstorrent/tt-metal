// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B36 (HY4_B36_SEL3=1): the stage-3 select of the indexer as ONE flag-consuming program: replaces the push
// wait-assemble (pw_wait), the stock top-2048 over the 8 gathered candidate rows (topk_large_indices) and
// gather_filter_own.
//   Structure used: every source's candidate row (kloc = KL values) is SORTED DESCENDING (output of the stage-2 top-k,
//   gathered in top-k order), so the global top-K of the 8*KL values is, per source, a PREFIX of its list (+ a tie
//   group). The K-th largest key is found by a bisection over the 16-bit monotone key of the bf16 values with per-list
//   binary searches (bounds narrowed as the bisection proceeds); the chip's OWN selected ids are the first m_own
//   entries of its own list. Tie rule at the K-th value (deterministic, the stock xl top-k resolves ties arbitrarily):
//   lower source column first, then lower list position. Output order = list order (value desc). -inf entries are never
//   selected (stock marks them invalid). Same output contract as gather_filter_own: out [1,1,OWN*S,K] u32 = this chip's
//   selected local ids filtered / mapped like id_filter (t < nvalid, owner column == own column, ((t >> 9) << 6) | (t &
//   63)), sentinel tail, replicated to the OWN blocks.
// Work split as gf_own: row s -> NQ cores (x = q, y = s); every core waits for the flags + reads the 8 lists (redundant
// threshold search, ~1-3 us), then filters its slice [q*KQ, (q+1)*KQ) of the own list; the leader (q = 0) concatenates
// in order.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline uint32_t key16(uint32_t v) {
    v &= 0xFFFFu;
    v = (v == 0x8000u) ? 0u : v;  // -0 == +0
    return (v & 0x8000u) ? (~v & 0xFFFFu) : (v | 0x8000u);
}

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t OWN = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t PERROW = get_compile_time_arg_val(4);
    constexpr uint32_t KL = get_compile_time_arg_val(5);
    constexpr uint32_t NQ = get_compile_time_arg_val(6);
    constexpr uint32_t PB = get_compile_time_arg_val(7);
    constexpr uint32_t RP = get_compile_time_arg_val(8);
    constexpr uint32_t NP = get_compile_time_arg_val(9);
    constexpr auto v_args = TensorAccessorArgs<10>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t RB = K * 4;
    constexpr uint32_t KQ = KL / NQ;
    constexpr uint32_t SLOTB = KQ * 4;
    static_assert(PB == KL * 2);
    const uint32_t s = get_arg_val<uint32_t>(0), q = get_arg_val<uint32_t>(1);
    const uint32_t lx = get_arg_val<uint32_t>(2), ly = get_arg_val<uint32_t>(3);
    const uint32_t fx = get_arg_val<uint32_t>(4), fy = get_arg_val<uint32_t>(5);
    // args 6..13: the 8 flag addresses, 14: launch counter address
    const uint32_t a_v = get_common_arg_val<uint32_t>(0), a_g = get_common_arg_val<uint32_t>(1),
                   a_c = get_common_arg_val<uint32_t>(2), a_o = get_common_arg_val<uint32_t>(3);
    Noc noc;
    const auto v_acc = TensorAccessor(v_args, a_v, RP * PB);
    const auto g_acc = TensorAccessor(g_args, a_g, KL * 4);
    const auto c_acc = TensorAccessor(c_args, a_c, 64);
    const auto o_acc = TensorAccessor(o_args, a_o, RB);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    // L1: [vals 8*PB][flag scratch 8*16][lid slice SLOTB][colinfo 64][slots NQ*SLOTB][counts NQ*16][out RB]
    const uint32_t vals_b = base;
    const uint32_t scr = vals_b + 8 * PB;
    const uint32_t lid_b = scr + 128;
    const uint32_t ci_b = lid_b + SLOTB;
    const uint32_t slots_b = ci_b + 64;
    const uint32_t cnt_b = slots_b + NQ * SLOTB;
    const uint32_t out_b = cnt_b + NQ * 16;

    // ---- flag wait (monotonic flags, per-core launch counter) ----
    volatile tt_l1_ptr uint32_t* lcnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(14));
    const uint32_t target = *lcnt + NP;
    *lcnt = target;
    // colinfo + lid slice do not depend on the flags
    noc.async_read(g_acc, cb, SLOTB, {.page_id = s, .offset_bytes = q * SLOTB}, {.offset_bytes = lid_b - base});
    noc.async_read(c_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = ci_b - base});
    bool done;
    {
        DeviceZoneScopedN("GF_WAIT");
        do {
            for (uint32_t c = 0; c < 8; ++c) {
                noc_async_read(get_noc_addr(fx, fy, get_arg_val<uint32_t>(6 + c)), scr + 16 * c, 4);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            done = true;
            for (uint32_t c = 0; c < 8; ++c) {
                done &= ((int32_t)(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 16 * c) - target) >= 0);
            }
        } while (!done);
    }
    {
        DeviceZoneScopedN("GF_RD");
        for (uint32_t c = 0; c < 8; ++c) {
            noc.async_read(
                v_acc, cb, PB, {.page_id = c * NP + s / RP, .offset_bytes = (s % RP) * PB}, {.offset_bytes = c * PB});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
    }

    const uint16_t* __restrict vals = reinterpret_cast<const uint16_t*>(vals_b);
    const uint32_t* __restrict cc = reinterpret_cast<const uint32_t*>(lid_b);
    const uint32_t* __restrict ci = reinterpret_cast<const uint32_t*>(ci_b);
    const uint32_t c0 = ci[0], nvalid = PERROW ? ci[2 + s] : ci[1];

    // ---- selection: how many of the own list are in the global top-K ----
    // pred(v, T): key16(v) >= T for the bf16 bits v, as ONE compare with constants derived from T (mode 0: T > 0x8000
    // positive threshold, signed compare; mode 1: T == 0x8000 (zero); mode 2: negative threshold, unsigned compare
    // against the threshold's bits)
    uint32_t cl[8], ch[8], tmp[8];
    uint32_t sel_lo = 0;  // the own list entries [0, sel_lo) are selected
    {
        DeviceZoneScopedN("GF_SEL");
        // count of entries of list c in [a, b] satisfying pred (descending lists: a prefix), given all j < a do and all
        // j >= b do not
        auto cnt_ge = [&](uint32_t c, uint32_t T, uint32_t a, uint32_t b) -> uint32_t {
            const uint16_t* __restrict l = vals + c * KL;
            if (T > 0x8000u) {
                const int32_t tb = (int32_t)(T - 0x8000u);
                while (a < b) {
                    const uint32_t m = (a + b) >> 1;
                    if ((int32_t)(int16_t)l[m] >= tb) {
                        a = m + 1;
                    } else {
                        b = m;
                    }
                }
            } else if (T == 0x8000u) {
                while (a < b) {
                    const uint32_t m = (a + b) >> 1;
                    if ((uint32_t)l[m] <= 0x8000u) {
                        a = m + 1;
                    } else {
                        b = m;
                    }
                }
            } else {
                const uint32_t nb = (~T) & 0xFFFFu;
                while (a < b) {
                    const uint32_t m = (a + b) >> 1;
                    if ((uint32_t)l[m] <= nb) {
                        a = m + 1;
                    } else {
                        b = m;
                    }
                }
            }
            return a;
        };
        // bracket of the K-th largest key T*: some list has >= K/8 entries >= T*, every list has >= K/8 entries >=
        // min_c key(l_c[K/8-1])
        constexpr uint32_t PIV = K / 8 - 1;
        uint32_t kmin = 0xFFFFu, kmax = 0;
        for (uint32_t c = 0; c < 8; ++c) {
            const uint32_t k = key16(vals[c * KL + PIV]);
            kmin = k < kmin ? k : kmin;
            kmax = k > kmax ? k : kmax;
        }
        uint32_t lo = kmin > 0x80u ? kmin : 0x80u;
        uint32_t tot = 0;
        for (uint32_t c = 0; c < 8; ++c) {
            cl[c] = cnt_ge(c, lo, 0, KL);
            tot += cl[c];
        }
        if (tot <= K && lo == 0x80u) {  // fewer valid entries than K: all selected
            sel_lo = cl[c0];
        } else {
            uint32_t hi = kmax + 1;  // count_ge(hi) < K by construction
            for (uint32_t c = 0; c < 8; ++c) {
                ch[c] = cnt_ge(c, hi, 0, cl[c]);
            }
            while (hi - lo > 1) {
                const uint32_t mid = (lo + hi) >> 1;
                uint32_t t2 = 0;
                for (uint32_t c = 0; c < 8; ++c) {
                    tmp[c] = cnt_ge(c, mid, ch[c], cl[c]);
                    t2 += tmp[c];
                }
                if (t2 >= K) {
                    lo = mid;
                    for (uint32_t c = 0; c < 8; ++c) {
                        cl[c] = tmp[c];
                    }
                } else {
                    hi = mid;
                    for (uint32_t c = 0; c < 8; ++c) {
                        ch[c] = tmp[c];
                    }
                }
            }
            // now: ch[c] = #keys > T*, cl[c] = #keys >= T* for T* = lo (the K-th largest key)
            uint32_t gt = 0;
            for (uint32_t c = 0; c < 8; ++c) {
                gt += ch[c];
            }
            uint32_t rem = K - gt;  // tied entries still to take: lower source first
            for (uint32_t c = 0; c < 8; ++c) {
                const uint32_t g = cl[c] - ch[c];
                const uint32_t take = g < rem ? g : rem;
                rem -= take;
                if (c == c0) {
                    sel_lo = ch[c] + take;
                }
            }
        }
    }

    // ---- filter the slice [q*KQ, (q+1)*KQ) of the own list ----
    uint32_t* __restrict lst = reinterpret_cast<uint32_t*>(slots_b + q * SLOTB);
    uint32_t n = 0;
    const uint32_t j0 = q * KQ;
    const uint32_t jend = sel_lo < j0 + KQ ? sel_lo : j0 + KQ;
    {
        DeviceZoneScopedN("GF_FILT");
        for (uint32_t j = j0; j < jend; ++j) {
            const uint32_t t = cc[j - j0];
            if (t < nvalid && ((t >> 6) & 7u) == c0) {
                lst[n++] = ((t >> 9) << 6) | (t & 63u);
            }
        }
    }

    const uint32_t sem_addr = get_semaphore(0);
    if (q != 0) {
        volatile tt_l1_ptr uint32_t* mycnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_b + q * 16);
        mycnt[0] = n;
        if (n) {
            noc_async_write(slots_b + q * SLOTB, get_noc_addr(lx, ly, slots_b + q * SLOTB), n * 4);
        }
        noc_async_write(cnt_b + q * 16, get_noc_addr(lx, ly, cnt_b + q * 16), 16);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, sem_addr), 1);
        noc_async_atomic_barrier();
        return;
    }
    // leader
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    {
        DeviceZoneScopedN("GF_SEM");
        noc_semaphore_wait(sem, NQ - 1);
        invalidate_l1_cache();
    }
    DeviceZoneScopedN("GF_LEAD");
    uint32_t* __restrict op = reinterpret_cast<uint32_t*>(out_b);
    const uint32_t* __restrict cnts = reinterpret_cast<const uint32_t*>(cnt_b);
    uint32_t tot = 0;
    for (uint32_t j = 0; j < NQ; ++j) {
        const uint32_t nj = (j == 0) ? n : cnts[j * 4];
        const uint32_t* __restrict src = reinterpret_cast<const uint32_t*>(slots_b + j * SLOTB);
        for (uint32_t i = 0; i < nj; ++i) {
            op[tot + i] = src[i];
        }
        tot += nj;
    }
#pragma GCC unroll 8
    for (uint32_t i = tot; i < K; ++i) {
        op[i] = 0xFFFFFFFFu;
    }
    for (uint32_t j = 0; j < OWN; ++j) {
        noc.async_write(cb, o_acc, RB, {.offset_bytes = out_b - base}, {.page_id = j * S + s, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
