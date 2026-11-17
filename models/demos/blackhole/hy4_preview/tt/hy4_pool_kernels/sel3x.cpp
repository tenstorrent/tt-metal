// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// SEL3X = sel3 with the STOCK ORDER emulated (HY4_B36_EXACT=1): the stock xl top-k orders ties by the LLK's index stamp
// (tile row/column coordinate of
//   the element in its 2048-chunk, chunk id in the high bits; direction flips with the sign of the value), so selection
//   at the K-th value and the output order inside runs of equal values follow that stamp instead of the list order.
//   Output = bit-identical to the stock chain.
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

// stamp key of the gathered position p (< 4096, chunk size 2048): the LLK stamps row | tile << 5 | col << 6 | chunk <<
// 11 from the element's place in the tile layout of its chunk; ties are broken by the larger stamp for positive values
// and by the smaller one for negative values (sign-magnitude compare)
static inline uint32_t sk_of(uint32_t p, bool neg) {
    const uint32_t o = p & 2047u;
    const uint32_t tile = o >> 10, w = o & 1023u, face = w >> 8, wf = w & 255u;
    const uint32_t row = ((face >> 1) << 4) | (wf >> 4), col = ((face & 1u) << 4) | (wf & 15u);
    const uint32_t st = row | (tile << 5) | (col << 6) | ((p >> 11) << 11);
    return neg ? 4095u - st : st;
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
    constexpr auto d_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t RB = K * 4;
    constexpr uint32_t KQ = KL / NQ;
    constexpr uint32_t SLOTB =
        KL * 4;  // a core may emit up to a whole list (runs of equal values extend beyond its start range)
    static_assert(PB == KL * 2);
    const uint32_t s = get_arg_val<uint32_t>(0), q = get_arg_val<uint32_t>(1);
    const uint32_t lx = get_arg_val<uint32_t>(2), ly = get_arg_val<uint32_t>(3);
    const uint32_t fx = get_arg_val<uint32_t>(4), fy = get_arg_val<uint32_t>(5);
    // args 6..13: the 8 flag addresses, 14: launch counter address
    const uint32_t a_v = get_common_arg_val<uint32_t>(0), a_g = get_common_arg_val<uint32_t>(1),
                   a_c = get_common_arg_val<uint32_t>(2), a_o = get_common_arg_val<uint32_t>(3),
                   a_d = get_common_arg_val<uint32_t>(4);
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
    const uint32_t ci_b = lid_b + KL * 4;
    const uint32_t slots_b = ci_b + 64;
    const uint32_t cnt_b = slots_b + NQ * SLOTB;
    const uint32_t out_b = cnt_b + NQ * 16;
    const uint32_t sortb = out_b + RB;    // [KL u32] run sort buffer
    const uint32_t skb = sortb + KL * 4;  // [8*KL u16] tie group stamps

    // ---- flag wait (monotonic flags, per-core launch counter) ----
    volatile tt_l1_ptr uint32_t* lcnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(14));
    const uint32_t target = *lcnt + NP;
    *lcnt = target;
    // colinfo + lid slice do not depend on the flags
    noc.async_read(g_acc, cb, KL * 4, {.page_id = s, .offset_bytes = 0}, {.offset_bytes = lid_b - base});
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
    uint32_t st_gtK = 0, st_g = 0, st_nlong = 0, st_maxrun = 0, st_nruns = 0;
    uint32_t jmax = 0;   // own list entries [0, jmax) are >= the K-th value (candidates)
    uint32_t chsel = 0;  // entries [0, chsel) are selected unconditionally, [chsel, jmax) = tie group (selected iff
                         // stamp key >= tsk)
    uint32_t tsk = 0;
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
            jmax = chsel = cl[c0];
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
            st_gtK = 1;
            const uint32_t rem = K - gt;  // tied entries to take: highest stamp key first (stock order)
            uint32_t g = 0;
            for (uint32_t c = 0; c < 8; ++c) {
                g += cl[c] - ch[c];
            }
            st_g = g;
            jmax = cl[c0];
            chsel = ch[c0];
            tsk = 0;
            if (rem < g) {
                // rem-th best stamp key of the tie group = union over the lists c of the contiguous ranges [ch[c],
                // cl[c]) of positions. Closed form (O(256) steps, independent of the group size): in list c the chunk
                // (c >> 2), tile ((c >> 1) & 1) and row block (c & 1) are constant, position j has col = (j >> 8) * 16
                // + (j & 15) and row-in-face (j >> 4) & 15, stamp = chunk << 11 | col << 6 | (c & 3) << 4 | r16.
                const bool neg = lo < 0x8000u;
                auto cnt_col = [&](uint32_t c, uint32_t col, uint32_t& rlo, uint32_t& rhi) -> uint32_t {
                    const int32_t a = (int32_t)ch[c], b = (int32_t)cl[c];
                    const int32_t base = (int32_t)(((col >> 4) << 8) | (col & 15u));
                    const int32_t x = a - base, y = b - 1 - base;
                    if (b <= a || y < 0) {
                        return 0;
                    }
                    rlo = x <= 0 ? 0 : (uint32_t)((x + 15) >> 4);
                    const uint32_t yh = (uint32_t)(y >> 4);
                    rhi = yh > 15 ? 15 : yh;
                    return rhi >= rlo ? rhi - rlo + 1 : 0;
                };
                uint32_t acc = 0, b1chunk = 0, b1col = 0;
                bool found = false;
                for (uint32_t i1 = 0; i1 < 2 && !found; ++i1) {
                    const uint32_t chunk = neg ? i1 : 1 - i1;
                    for (uint32_t i2 = 0; i2 < 32; ++i2) {
                        const uint32_t col = neg ? i2 : 31 - i2;
                        uint32_t cnt = 0, r0, r1;
                        for (uint32_t l = 0; l < 4; ++l) {
                            cnt += cnt_col(chunk * 4 + l, col, r0, r1);
                        }
                        if (acc + cnt >= rem) {
                            found = true;
                            b1chunk = chunk;
                            b1col = col;
                            break;
                        }
                        acc += cnt;
                    }
                }
                // inside the bin (chunk, col): at most 4 x 16 elements, lo6 = (c & 3) << 4 | r16 (unique): 64-bit
                // bitmap, take the (rem - acc)-th in order
                uint32_t m0 = 0, m1 = 0;
                for (uint32_t l = 0; l < 4; ++l) {
                    uint32_t r0 = 0, r1 = 0;
                    if (cnt_col(b1chunk * 4 + l, b1col, r0, r1)) {
                        for (uint32_t r = r0; r <= r1; ++r) {
                            const uint32_t lo6 = (l << 4) | r;
                            if (lo6 < 32) {
                                m0 |= 1u << lo6;
                            } else {
                                m1 |= 1u << (lo6 - 32);
                            }
                        }
                    }
                }
                uint32_t need = rem - acc, lo6f = 0;
                for (uint32_t i = 0; i < 64; ++i) {
                    const uint32_t b6 = neg ? i : 63 - i;
                    const uint32_t bit = b6 < 32 ? (m0 >> b6) & 1u : (m1 >> (b6 - 32)) & 1u;
                    if (bit && --need == 0) {
                        lo6f = b6;
                        break;
                    }
                }
                const uint32_t stamp_t = (b1chunk << 11) | (b1col << 6) | lo6f;
                tsk = neg ? 4095u - stamp_t : stamp_t;
            }
        }
    }

    // ---- emit the own selected entries in stock order: runs of equal values (list order is by value; inside a run by
    // stamp key desc) ---- core q handles the runs whose START lies in [q*KQ, (q+1)*KQ)
    uint32_t* __restrict lst = reinterpret_cast<uint32_t*>(slots_b + q * SLOTB);
    uint32_t n = 0;
    {
        DeviceZoneScopedN("GF_FILT");
        const uint16_t* __restrict own = vals + c0 * KL;
        uint32_t* __restrict sb = reinterpret_cast<uint32_t*>(sortb);
        const uint32_t j0 = q * KQ;
        const uint32_t jlim = jmax < j0 + KQ ? jmax : j0 + KQ;
        uint32_t j = j0;
        while (j < jlim && j > 0 && own[j] == own[j - 1]) {
            ++j;  // continuation of a run started in an earlier slice
        }
        while (j < jlim) {
            const uint32_t v = own[j];
            uint32_t e = j + 1;
            while (e < jmax && own[e] == v) {
                ++e;
            }
            ++st_nruns;
            if (e - j > st_maxrun) {
                st_maxrun = e - j;
            }
            if (e - j > 24) {
                ++st_nlong;
            }
            if (e == j + 1) {
                const uint32_t t = cc[j];
                const bool neg1 = (v & 0x8000u) && v != 0x8000u;
                if ((j < chsel || sk_of(c0 * KL + j, neg1) >= tsk) && t < nvalid && ((t >> 6) & 7u) == c0) {
                    lst[n++] = ((t >> 9) << 6) | (t & 63u);
                }
            } else {
                const bool neg = (v & 0x8000u) && v != 0x8000u;
                uint32_t m = 0;
                if (e - j > 24) {
                    // long run: enumerate the stamp order directly. Inside the list of source c0 the chunk, the tile
                    // and the row block are constant, so the stamp order is (jj >> 8, jj & 15, (jj >> 4) & 15)
                    // lexicographic (col block, col in face, row in face): descending for non-negative values,
                    // ascending for negative ones
                    static_assert(KL == 512, "stamp enumeration assumes 512-entry lists in 2048-chunks");
                    const uint32_t stbase = ((c0 >> 2) << 11) | ((c0 & 3u) << 4);
                    for (uint32_t i1 = 0; i1 < 2; ++i1) {
                        const uint32_t cb = neg ? i1 : 1 - i1;
                        for (uint32_t i2 = 0; i2 < 16; ++i2) {
                            const uint32_t c16 = neg ? i2 : 15 - i2;
                            for (uint32_t i3 = 0; i3 < 16; ++i3) {
                                const uint32_t r16 = neg ? i3 : 15 - i3;
                                const uint32_t jj = (cb << 8) | (r16 << 4) | c16;
                                if (jj >= j && jj < e) {
                                    const uint32_t st =
                                        stbase | (cb << 10) | (c16 << 6) | r16;  // stamp key of position jj of list c0
                                    if (jj < chsel || (neg ? 4095u - st : st) >= tsk) {
                                        sb[m++] = jj;  // already in output order
                                    }
                                }
                            }
                        }
                    }
                    for (uint32_t i = 0; i < m; ++i) {
                        const uint32_t t = cc[sb[i]];
                        if (t < nvalid && ((t >> 6) & 7u) == c0) {
                            lst[n++] = ((t >> 9) << 6) | (t & 63u);
                        }
                    }
                    j = e;
                    continue;
                }
                for (uint32_t jj = j; jj < e; ++jj) {
                    const uint32_t k = sk_of(c0 * KL + jj, neg);
                    if (jj < chsel || k >= tsk) {
                        // insertion into the descending sort buffer
                        const uint32_t item = (k << 16) | jj;
                        uint32_t i = m++;
                        while (i > 0 && sb[i - 1] < item) {
                            sb[i] = sb[i - 1];
                            --i;
                        }
                        sb[i] = item;
                    }
                }
                for (uint32_t i = 0; i < m; ++i) {
                    const uint32_t t = cc[sb[i] & 0xFFFFu];
                    if (t < nvalid && ((t >> 6) & 7u) == c0) {
                        lst[n++] = ((t >> 9) << 6) | (t & 63u);
                    }
                }
            }
            j = e;
        }
    }

    if (a_d !=
        0) {  // debug counters: this core's page of the dbg tensor, accumulated over launches ([0] calls, [1] N>K
              // calls, [2] sum g, [3] max g, [4] sum long runs, [5] max run, [6] sum runs, [7] sum jmax, [8] max jmax)
        const auto d_acc = TensorAccessor(d_args, a_d, 64);
        volatile tt_l1_ptr uint32_t* dbp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 64);
        noc.async_read(d_acc, cb, 64, {.page_id = s * NQ + q, .offset_bytes = 0}, {.offset_bytes = scr + 64 - base});
        noc.async_read_barrier();
        invalidate_l1_cache();
        dbp[0] += 1;
        dbp[1] += st_gtK;
        dbp[2] += st_g;
        if (st_g > dbp[3]) {
            dbp[3] = st_g;
        }
        dbp[4] += st_nlong;
        if (st_maxrun > dbp[5]) {
            dbp[5] = st_maxrun;
        }
        dbp[6] += st_nruns;
        dbp[7] += jmax;
        if (jmax > dbp[8]) {
            dbp[8] = jmax;
        }
        noc.async_write(cb, d_acc, 64, {.offset_bytes = scr + 64 - base}, {.page_id = s * NQ + q, .offset_bytes = 0});
        noc.async_write_barrier();
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
