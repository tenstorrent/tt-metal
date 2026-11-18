// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EP-4 receive-side lists (ep4_layer.build_lists), NCRISC (NOC1: the L1 exchange and posted DRAM writes; no DRAM
// reads - e4_lists_rd.cpp brings the rows). The group's tokens g = home TX + t (C TX of them, the grp rows) are split
// in contiguous ranges over the NC cores (sources); core ids below NOWN also own local experts [EPO me, EPO (me + 1)).
// A token's local pairs: its k with expert id e in this chip's column (e / EPC == CS), local expert le = e - EPC CS.
//   A. Sources: per token its local count; the sum to core 0, which returns the exclusive prefix over the sources (the
//      first O row of the source's range: a token's O rows are contiguous, base + j, j the pair's rank in k order) and
//      writes the total into the ready counts' page (word EPC). Per own home token (home == CS) the row's chips c
//      holding >= 1 of its experts: counted per c into core 0's EXPW words (atomic adds), written to the ready counts'
//      page words EPC + 1 + c - the partials the home pass waits for from chip c.
//   B. Every token's run record (128-B page g of runs: 0 base, 1 count, 2 the last local expert in stream order, 3 g,
//      4 + j the pair's gate weight, 16 + j its local expert), posted. Records (2 words: g | le << 17, O row) to le's
//      owner, plus one ready record (g | last << 17, READY) a token with local pairs to its last expert's owner; the
//      exchange as wsr_exchange.hpp's (counts, offsets, runs into the owners' record buffers, arrivals).
//   C. Owner: per local expert its token list (xbuf rows: piece t / SP's rows from (t / SP) NSP, chip (RB + home)'s
//      from (RB + home) SP, t % SP within; padded with PADT), its O rows (padded with PADR),
//      its ready list (tokens g whose last expert it is, padded with 0xFFFFFFFF), each a page of RS entries
//      (posted); its counts / ready counts into core 0's CNT_ALL, core 0 writes both pages. Over CAP: count PAD.
//   Fence: an acknowledged 4-B write into every DRAM bank after the posted writes.
// CT: 0 K, 1 EPC, 2 NOWN, 3 NC, 4 CAP, 5 RS, 6 TX, 7 EPO, 8 tok_buf, 9 srt, 10 cnt_in, 11 off_in, 12 recbuf, 13 la,
//     14 lb, 15 lc, 16 cnt_all (2 EPC + 16 words), 17 aux (3 x 256), 18 run staging (32 x 128 B), 19 pos (L1), 20 sb
//     (2 x NC words: core 0's sums in, every source's prefix out; then 4 x 16 B EXPW), 21 T, 22 C (chips a row), 23 SP
//     (dispatch piece tokens), 24 NSP (xbuf rows a piece: chips x SP), then TensorAccessorArgs(runs), (idx), (dest),
//     (rdy), (counts), (rcounts), (fence).
// RT: 0 me, 1 first group token, 2 tokens, 3 RB (the row's first chip index r C), 4 CS, 5 PADT, 6 PADR, 7 runs, 8 idx,
// 9 dest, 10 rdy, 11 counts,
//     12 rcounts, 13 fence, 14 stamps, 15 + k core k's NoC x | y << 8.
// Semaphores: 0 arrivals, 1 done, 2 start count (core 0), 3 start go, 4 the reader's rows in.
#include "api/dataflow/dataflow_api.h"

namespace {
template <typename T>
FORCE_INLINE volatile tt_l1_ptr T* p(uint32_t a) {
    return reinterpret_cast<volatile tt_l1_ptr T*>(a);
}
FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
constexpr uint32_t PAD = 0xFFFFFFFFu, READY = 0xFFFFFFFEu;
}  // namespace

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t EPC = get_compile_time_arg_val(1);
    constexpr uint32_t NOWN = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t CAP = get_compile_time_arg_val(4);
    constexpr uint32_t RS = get_compile_time_arg_val(5);
    constexpr uint32_t TX = get_compile_time_arg_val(6);
    constexpr uint32_t EPO = get_compile_time_arg_val(7);
    constexpr uint32_t tok_buf = get_compile_time_arg_val(8);
    constexpr uint32_t srt = get_compile_time_arg_val(9);
    constexpr uint32_t cnt_in = get_compile_time_arg_val(10);
    constexpr uint32_t off_in = get_compile_time_arg_val(11);
    constexpr uint32_t recbuf = get_compile_time_arg_val(12);
    constexpr uint32_t la = get_compile_time_arg_val(13);
    constexpr uint32_t lb = get_compile_time_arg_val(14);
    constexpr uint32_t lc = get_compile_time_arg_val(15);
    constexpr uint32_t cnt_all = get_compile_time_arg_val(16);
    constexpr uint32_t aux = get_compile_time_arg_val(17);
    constexpr uint32_t rstage = get_compile_time_arg_val(18);
    constexpr uint32_t pos = get_compile_time_arg_val(19);
    constexpr uint32_t sb = get_compile_time_arg_val(20);
    constexpr uint32_t T = get_compile_time_arg_val(21);
    constexpr uint32_t CG = get_compile_time_arg_val(22);
    constexpr uint32_t SP = get_compile_time_arg_val(23);
    constexpr uint32_t NSP = get_compile_time_arg_val(24);
    constexpr uint32_t expw = (sb + 2 * NC * 4 + 63) / 64 * 64;  // (16 B words)
    constexpr auto rargs = TensorAccessorArgs<25>();
    constexpr auto iargs = TensorAccessorArgs<rargs.next_compile_time_args_offset()>();
    constexpr auto dargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr auto yargs = TensorAccessorArgs<dargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<yargs.next_compile_time_args_offset()>();
    constexpr auto qargs = TensorAccessorArgs<cargs.next_compile_time_args_offset()>();
    constexpr auto fargs = TensorAccessorArgs<qargs.next_compile_time_args_offset()>();
    static_assert(NOWN <= 256 && EPO <= 8 && EPC < 256 && 4 * TX <= (1u << 17));

    const uint32_t me = get_arg_val<uint32_t>(0), a = get_arg_val<uint32_t>(1), nt = get_arg_val<uint32_t>(2);
    const uint32_t RB = get_arg_val<uint32_t>(3), CS = get_arg_val<uint32_t>(4);
    const uint32_t PADT = get_arg_val<uint32_t>(5), PADR = get_arg_val<uint32_t>(6);
    const auto rs = TensorAccessor(rargs, get_arg_val<uint32_t>(7), 128);
    const auto is = TensorAccessor(iargs, get_arg_val<uint32_t>(8), RS * 4);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(9), RS * 4);
    const auto ys = TensorAccessor(yargs, get_arg_val<uint32_t>(10), RS * 4);
    const auto cs = TensorAccessor(cargs, get_arg_val<uint32_t>(11), EPC * 4);
    const auto qs = TensorAccessor(qargs, get_arg_val<uint32_t>(12), (EPC + 16) * 4);
    const auto fs = TensorAccessor(fargs, get_arg_val<uint32_t>(13), 64);
    volatile tt_l1_ptr uint32_t* stamps = p<uint32_t>(get_arg_val<uint32_t>(14));
    auto noc_at = [&](uint32_t k, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(15 + k);
        return get_noc_addr(v & 0xFF, v >> 8, addr);
    };
    stamps[0] = wall();

    {  // the start: state words zeroed, then every core's
        for (uint32_t k = 0; k < NC; ++k) {
            p<uint32_t>(cnt_in)[k] = 0;
        }
        for (uint32_t o = 0; o < NOWN; ++o) {
            p<uint32_t>(off_in)[o] = 0;
        }
        for (uint32_t k = 0; k < 2 * NC; ++k) {
            p<uint32_t>(sb)[k] = 0;
        }
        for (uint32_t c = 0; c < CG; ++c) {
            p<uint32_t>(expw)[4 * c] = 0;
        }
        const uint32_t cnt = get_semaphore(2), go = get_semaphore(3);
        noc_semaphore_inc(noc_at(0, cnt), 1);
        noc_async_atomic_barrier();
        if (me == 0) {
            noc_semaphore_wait(p<uint32_t>(cnt), NC);
            for (uint32_t k = 0; k < NC; ++k) {
                noc_inline_dw_write(noc_at(k, go), 1);
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait(p<uint32_t>(go), 1);
    }
    noc_semaphore_wait(p<uint32_t>(get_semaphore(4)), 1);  // (the reader's rows in L1)
    stamps[1] = wall();

    // ---- A: local counts, per-owner record counts
    volatile tt_l1_ptr uint32_t* ocnt = p<uint32_t>(aux);
    volatile tt_l1_ptr uint32_t* start = p<uint32_t>(aux + 256 * 4);
    volatile tt_l1_ptr uint32_t* fill = p<uint32_t>(aux + 512 * 4);
    volatile tt_l1_ptr uint32_t* ps = p<uint32_t>(pos);
    for (uint32_t o = 0; o < NOWN; ++o) {
        ocnt[o] = 0;
    }
    uint32_t mysum = 0, exp_c[4] = {0, 0, 0, 0};
    const uint32_t e_lo = EPC * CS;
    for (uint32_t i = 0; i < nt; ++i) {
        const uint32_t g = a + i;
        if (g % TX >= T) {
            continue;
        }
        volatile tt_l1_ptr uint32_t* row = p<uint32_t>(tok_buf + i * 128);
        if (g / TX == CS) {
            uint32_t cm = 0;
            for (uint32_t k = 0; k < K; ++k) {
                cm |= 1u << (row[16 + k] / EPC);
            }
            for (uint32_t c = 0; c < CG; ++c) {
                exp_c[c] += (cm >> c) & 1;
            }
        }
        uint32_t n = 0, last = 0, lastp = 0;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t le = row[16 + k] - e_lo;
            if (le < EPC) {
                ++ocnt[le / EPO];
                if (n == 0 || ps[le] >= lastp) {
                    lastp = ps[le];
                    last = le;
                }
                ++n;
            }
        }
        if (n) {
            ++ocnt[last / EPO];
        }
        mysum += n;
    }
    for (uint32_t c = 0; c < CG; ++c) {
        if (exp_c[c]) {
            noc_semaphore_inc(noc_at(0, expw + 16 * c), exp_c[c]);
        }
    }
    noc_async_atomic_barrier();
    // the exclusive prefix of the sums over the sources (core 0)
    noc_inline_dw_write(noc_at(0, sb + me * 4), mysum + 1);
    if (me == 0) {
        uint32_t acc = 0;
        for (uint32_t k = 0; k < NC; ++k) {
            do {
                invalidate_l1_cache();
            } while (p<uint32_t>(sb)[k] == 0);
            const uint32_t v = p<uint32_t>(sb)[k] - 1;
            p<uint32_t>(sb)[k] = 0;
            noc_inline_dw_write(noc_at(k, sb + NC * 4), acc + 1);  // (into source k's word NC)
            acc += v;
        }
        p<uint32_t>(cnt_all)[2 * EPC] = acc;  // (the O rows used: word EPC of the ready counts' page)
    }
    do {
        invalidate_l1_cache();
    } while (p<uint32_t>(sb)[NC] == 0);
    uint32_t orow = p<uint32_t>(sb)[NC] - 1;
    p<uint32_t>(sb)[NC] = 0;

    // ---- B: records sorted by owner (2 words each, every owner's run padded to an even count: 16 B), run pages
    start[0] = 0;
    for (uint32_t o = 0; o < NOWN; ++o) {
        start[o + 1] = start[o] + ((ocnt[o] + 1) & ~1u);
        fill[o] = start[o];
    }
    auto put = [&](uint32_t w0, uint32_t w1, uint32_t o) {
        const uint32_t at = fill[o]++;
        p<uint32_t>(srt)[2 * at] = w0;
        p<uint32_t>(srt)[2 * at + 1] = w1;
    };
    for (uint32_t i = 0; i < nt; ++i) {
        const uint32_t g = a + i;
        volatile tt_l1_ptr uint32_t* row = p<uint32_t>(tok_buf + i * 128);
        volatile tt_l1_ptr uint32_t* rr = p<uint32_t>(rstage + (i % 32) * 128);
        uint32_t n = 0, last = 0, lastp = 0;
        if (g % TX < T) {
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t le = row[16 + k] - e_lo;
                if (le < EPC) {
                    put(g | le << 17, orow + n, le / EPO);
                    rr[4 + n] = row[k];
                    rr[16 + n] = le;
                    if (n == 0 || ps[le] >= lastp) {
                        lastp = ps[le];
                        last = le;
                    }
                    ++n;
                }
            }
        }
        if (n) {
            put(g | last << 17, READY, last / EPO);
        }
        rr[0] = orow;
        rr[1] = n;
        rr[2] = last;
        rr[3] = g;
        for (uint32_t j = n; j < 12; ++j) {
            rr[4 + j] = 0;
            rr[16 + j] = PAD;
        }
        for (uint32_t j = 28; j < 32; ++j) {
            rr[j] = 0;
        }
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(rstage + (i % 32) * 128, rs.get_noc_addr(g), 128);
        if (i % 32 == 31) {
            noc_async_posted_writes_flushed();  // (the staging slots are rewritten)
        }
        orow += n;
    }
    noc_async_posted_writes_flushed();
    for (uint32_t o = 0; o < NOWN; ++o) {
        for (uint32_t r = fill[o]; r < start[o + 1]; ++r) {
            p<uint32_t>(srt)[2 * r] = PAD;
            p<uint32_t>(srt)[2 * r + 1] = PAD;
        }
        noc_inline_dw_write(noc_at(o, cnt_in + me * 4), start[o + 1] - start[o] + 1);
    }
    // owners: offsets (sources in core order)
    uint32_t total = 0;
    if (me < NOWN) {
        for (uint32_t k = 0; k < NC; ++k) {
            do {
                invalidate_l1_cache();
            } while (p<uint32_t>(cnt_in)[k] == 0);
        }
        for (uint32_t k = 0; k < NC; ++k) {
            noc_inline_dw_write(noc_at(k, off_in + me * 4), total + 1);
            total += p<uint32_t>(cnt_in)[k] - 1;
            p<uint32_t>(cnt_in)[k] = 0;
        }
    }
    for (uint32_t o = 0; o < NOWN; ++o) {
        do {
            invalidate_l1_cache();
        } while (p<uint32_t>(off_in)[o] == 0);
        const uint32_t off = p<uint32_t>(off_in)[o] - 1, n = start[o + 1] - start[o];
        p<uint32_t>(off_in)[o] = 0;
        if (n && off + n <= CAP) {
            noc_async_write(srt + start[o] * 8, noc_at(o, recbuf + off * 8), n * 8);
        }
    }
    noc_async_write_barrier();
    const uint32_t arr = get_semaphore(0), done = get_semaphore(1);
    for (uint32_t o = 0; o < NOWN; ++o) {
        noc_semaphore_inc(noc_at(o, arr), 1);
    }
    noc_async_atomic_barrier();
    stamps[2] = wall();

    // ---- C: owners
    if (me < NOWN) {
        noc_semaphore_wait(p<uint32_t>(arr), NC);
        const uint32_t e0 = me * EPO, ne = EPC - e0 < EPO ? EPC - e0 : EPO;
        const bool over = total > CAP;
        uint32_t m[EPO], q[EPO], base[EPO + 1], qbase[EPO + 1], ef[EPO], qf[EPO];
        for (uint32_t j = 0; j < EPO; ++j) {
            m[j] = q[j] = 0;
        }
        if (!over) {
            for (uint32_t i = 0; i < total; ++i) {
                const uint32_t w0 = p<uint32_t>(recbuf)[2 * i], w1 = p<uint32_t>(recbuf)[2 * i + 1];
                if (w0 != PAD) {
                    ++(w1 == READY ? q : m)[(w0 >> 17) - e0];
                }
            }
        }
        base[0] = qbase[0] = 0;
        for (uint32_t j = 0; j < ne; ++j) {
            base[j + 1] = base[j] + ((m[j] + 31) & ~31u);
            qbase[j + 1] = qbase[j] + ((q[j] + 31) & ~31u);
            ef[j] = base[j];
            qf[j] = qbase[j];
        }
        if (!over) {
            for (uint32_t i = 0; i < total; ++i) {
                const uint32_t w0 = p<uint32_t>(recbuf)[2 * i], w1 = p<uint32_t>(recbuf)[2 * i + 1];
                if (w0 == PAD) {
                    continue;
                }
                const uint32_t j = (w0 >> 17) - e0, g = w0 & 0x1FFFF;
                if (w1 == READY) {
                    p<uint32_t>(lc)[qf[j]++] = g;
                } else {
                    const uint32_t t = g % TX;
                    p<uint32_t>(la)[ef[j]] = (t / SP) * NSP + (RB + g / TX) * SP + t % SP;
                    p<uint32_t>(lb)[ef[j]] = w1;
                    ++ef[j];
                }
            }
            for (uint32_t j = 0; j < ne; ++j) {
                for (uint32_t i = ef[j]; i < base[j + 1]; ++i) {
                    p<uint32_t>(la)[i] = PADT;
                    p<uint32_t>(lb)[i] = PADR;
                }
                for (uint32_t i = qf[j]; i < qbase[j + 1]; ++i) {
                    p<uint32_t>(lc)[i] = PAD;
                }
                const uint32_t bytes = (base[j + 1] - base[j]) * 4, qbytes = (qbase[j + 1] - qbase[j]) * 4;
                if (bytes) {
                    noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                        la + base[j] * 4, is.get_noc_addr(e0 + j), bytes);
                    noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                        lb + base[j] * 4, ds.get_noc_addr(e0 + j), bytes);
                }
                if (qbytes) {
                    noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                        lc + qbase[j] * 4, ys.get_noc_addr(e0 + j), qbytes);
                }
            }
        }
        for (uint32_t j = 0; j < ne; ++j) {
            noc_inline_dw_write(noc_at(0, cnt_all + (e0 + j) * 4), over ? PAD : m[j]);
            noc_inline_dw_write(noc_at(0, cnt_all + (EPC + e0 + j) * 4), over ? PAD : q[j]);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(noc_at(0, done), 1);
        noc_async_atomic_barrier();
    }
    stamps[3] = wall();
    if (me == 0) {
        noc_semaphore_wait(p<uint32_t>(done), NOWN);
        for (uint32_t j = 2 * EPC + 1; j < 2 * EPC + 16; ++j) {
            p<uint32_t>(cnt_all)[j] = 0;
        }
        for (uint32_t c = 0; c < CG; ++c) {  // (every source's atomic adds landed: they precede its sum, read above)
            p<uint32_t>(cnt_all)[2 * EPC + 1 + c] = p<uint32_t>(expw)[4 * c];
        }
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(cnt_all, cs.get_noc_addr(0), EPC * 4);
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(cnt_all + EPC * 4, qs.get_noc_addr(0), (EPC + 16) * 4);
    }
    // ---- fence
    noc_async_posted_writes_flushed();
    for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
        noc_async_write(rstage, fs.get_noc_addr(i), 4);
    }
    noc_async_write_barrier();
    stamps[63] = wall();
}
