// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert head (ws_router.py), NCRISC (NOC1: the group's multicasts and unicasts, the exchange, posted DRAM writes;
// no DRAM reads). Core c of a router group of GS cores (a column segment: one NOC1 multicast rectangle, its hub h at
// the rectangle's NOC1 start end). Group row tile i's owner is core i % GS. x comes in pieces s (BK K tiles of a row
// tile, stream order), each tilized by its producer and relayed into the hub's relay ring (wsr2_reader.cpp). A
// non-blocking loop over:
//   QX x ring: push piece s (CB 1) once slot s % NX's flag reads s + 1; the count of pieces the compute popped goes
//      into my word of the hub's XPROG;
//   QH (the hub) piece s from relay slot s % NR, once the slot's flag reads s + 1, into slot s % NX of every group
//      core's ring, linked, then its flag s + 1 as a 4-B multicast write through the same command buffer (it lands
//      after the payload and closes the path): piece s waits until every group core has popped piece s - NX and until
//      piece s - 2 has landed here (at most two in flight). Then the relay slot is free for piece s + NR: s + NR + 1
//      into its producer's RGRANT (downward on NOC1);
//   QR my run tile of each row tile i (CB 4) into slot (i / GS) % NG of the owner's gather ring at tile c, once the
//      owner's credit word in my GCRED reads i / GS + 1; once acknowledged, + 1 on the owner's GARR;
//   QG (owner) credits: my n-th owned row tile's gather slot is free once the compute popped my row tile n - NG: then
//      n + 1 into my word of every group core's GCRED; arrivals: GARR at GS (n + 1) -> push the slot (CB 5);
//   QA (owner) per owned row tile: the top-K keys (CB 7) and exp of their logits (CB 8), a token a column: w (t, k) =
//      e_k / sum_k e_k (soft float), each token's w row ([16] fp32) to w page t (posted); records t | k << 16 | e
//      << 20. WSR_SIDE (lane groups G): k's result row is t S + slot, the slot chosen so the row's DRAM bank lies on
//      the side of the expert's lane group (e % G >= G / 2: the upper WSR_SIDE_BANKS / 2 banks), the rest in order; w
//      and the record carry the slot instead of k. QA_TOK tokens a pass, so the soft float does not hold up the other
//      queues;
//   (WSR_DEBUG_LOGITS) QL my logits tiles (CB 9) to the logits^T tensor (posted).
// A launch starts with this core's state words zeroed (flags, credits, the exchange's counts and offsets: their L1 is
// another program's between launches), then a barrier over the chip (semaphores 5, 6; the BRISC waits for it too)
// before anything is sent. At the end a barrier over the chip (every core's multicasts and gathers done, the compute
// done with W_r^T, whose L1 the exchange's record buffers reuse), then the exchange (wsr_exchange.hpp).
// RT: 0 me, 1 the group's first row tile, 2 its row tiles, 3 c, 4 T, 5 w, 6 idx, 7 dest, 8 counts, 9 fence, 10 stamps
//     (2: loop done, 3: B done, 4: C done, 63: end; wall-clock cycles: 20 x multicasts issue to sent, 21 tilized x
//     pieces in the relay waiting for credits / the stream order (the hub), 22 runs waiting for the owner's credit,
//     23 in QA),
//     11 logits^T addr, 12-15 the group's NOC1 multicast rectangle, 16 h, 17 + m: group core m's NoC x | y << 8,
//     17 + GS + k: core k's.
// CT: wsr_exchange.hpp's, then 0 BK, 1 NX, 2 KT, 3 GS, 4 NG, 5 xring, 6 xflags, 7 xprog, 8 gcred, 9 garr, 10 gather
//     ring, 11 id mask, 12 row tiles of x (the logits^T pages a row), 13 NR, 14 relay ring, 15 relay flags, 16 RGRANT,
//     then TensorAccessorArgs(logits^T).
// Semaphores: 0, 1 (the exchange), 2 (the barrier's count, core 0), 3 (the barrier's go), 4 (the flag value source),
// 5 (the start barrier's count, core 0), 6 (its go).
// WSR_EP4 (EP-4, the experts split over the mesh row's chips; WSR_EP4 = experts a chip column): the w output is the
// token's 128-B topk row instead (words 0-15 w (t, k), 16 + k the expert id, 28 the route word; 4-KB pages of 32
// tokens, token t at page t / 32, offset (t % 32) 128: the group exchange gathers whole pages), every row tile's 32
// dispatch route words go to the route tensor (one page, 128 B a row tile; tokens >= T 0), and there is no exchange.
// Route word: bit RB + e / WSR_EP4 for each of the token's experts e not on this chip's column MC (sparse_dispatch_2d's
// mask, chip index r C + c; order 0: X first), link (t & 1) << 17. The route words go in pieces of WSR_EP4_SP tokens
// (the dispatch's calls), piece p a tensor of its own ([1, 1, 1, SP], one page). RT 17 + GS + NC: RB (r C), MC, then
// the pieces' addresses; CT after the logits': TensorAccessorArgs(a piece).
#include "wsr_exchange.hpp"

void kernel_main() {
    using namespace wsr;
    constexpr uint32_t o = ct_end;
    constexpr uint32_t BK = get_compile_time_arg_val(o + 0);
    constexpr uint32_t NX = get_compile_time_arg_val(o + 1);
    constexpr uint32_t KT = get_compile_time_arg_val(o + 2);
    constexpr uint32_t GS = get_compile_time_arg_val(o + 3);
    constexpr uint32_t NG = get_compile_time_arg_val(o + 4);
    constexpr uint32_t xring = get_compile_time_arg_val(o + 5);
    constexpr uint32_t xflags = get_compile_time_arg_val(o + 6);
    constexpr uint32_t xprog = get_compile_time_arg_val(o + 7);
    constexpr uint32_t gcred = get_compile_time_arg_val(o + 8);
    constexpr uint32_t garr = get_compile_time_arg_val(o + 9);
    constexpr uint32_t gring = get_compile_time_arg_val(o + 10);
    constexpr uint32_t IDM = get_compile_time_arg_val(o + 11);
    constexpr uint32_t RTX = get_compile_time_arg_val(o + 12);
    constexpr uint32_t NR = get_compile_time_arg_val(o + 13);
    constexpr uint32_t relay = get_compile_time_arg_val(o + 14);
    constexpr uint32_t rflags = get_compile_time_arg_val(o + 15);
    constexpr uint32_t rgrant = get_compile_time_arg_val(o + 16);
    constexpr auto largs = TensorAccessorArgs<o + 17>();
#ifdef WSR_EP4
    constexpr auto rargs = TensorAccessorArgs<largs.next_compile_time_args_offset()>();
    constexpr uint32_t WROW = 128, WPAGE = 4096;
#else
    constexpr uint32_t WROW = 64, WPAGE = 64;
#endif
    constexpr uint32_t NB = KT / BK, XT = 2048, FT = 4096, EPC = 2;
    constexpr uint32_t xr_cb = tt::CBIndex::c_1, run_cb = tt::CBIndex::c_4;
    constexpr uint32_t ga_cb = tt::CBIndex::c_5, be_cb = tt::CBIndex::c_7, ee_cb = tt::CBIndex::c_8;
    constexpr uint32_t lg_cb = tt::CBIndex::c_9;
    constexpr uint32_t R_TRID = 0xC;

    const uint32_t me = get_arg_val<uint32_t>(0), R0 = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    const uint32_t c = get_arg_val<uint32_t>(3), T = get_arg_val<uint32_t>(4);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(5), WPAGE);
    volatile tt_l1_ptr uint32_t* stamps = p<uint32_t>(get_arg_val<uint32_t>(10));
    const auto ls = TensorAccessor(largs, get_arg_val<uint32_t>(11), FT);
    const uint32_t mx0 = get_arg_val<uint32_t>(12), my0 = get_arg_val<uint32_t>(13);
    const uint32_t mx1 = get_arg_val<uint32_t>(14), my1 = get_arg_val<uint32_t>(15);
    const uint32_t h = get_arg_val<uint32_t>(16);
    auto member = [&](uint32_t m, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(17 + m);
        return get_noc_addr(v & 0xFF, v >> 8, addr);
    };
    auto noc_at = [&](uint32_t k, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(17 + GS + k);
        return get_noc_addr(v & 0xFF, v >> 8, addr);
    };
    auto word = [](uint32_t base, uint32_t i) -> volatile tt_l1_ptr uint32_t& {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + i * 16);
    };
    const uint32_t xscr_addr = get_semaphore(4);
#ifdef WSR_EP4
    const uint32_t RB = get_arg_val<uint32_t>(17 + GS + NC), MC = get_arg_val<uint32_t>(18 + GS + NC);
    constexpr uint32_t SP = WSR_EP4_SP;
    static_assert(SP % 32 == 0);
    auto route_at = [&](uint32_t r) {  // (row tile r's 32 route words: piece r 32 / SP, offset r 32 % SP)
        const auto rs = TensorAccessor(rargs, get_arg_val<uint32_t>(19 + GS + NC + r * 32 / SP), SP * 4);
        return rs.get_noc_addr(0, (r * 32 % SP) * 4);
    };
    const uint32_t rstage = wst + 32 * WROW;  // (the row tile's 32 route words, after the topk rows' staging)
#endif
    volatile tt_l1_ptr uint32_t* xscr = p<uint32_t>(xscr_addr);

    {  // the start: state words zeroed, then every core's
        for (uint32_t i = 0; i < NX; ++i) {
            word(xflags, i) = 0;
        }
        for (uint32_t m = 0; m < GS; ++m) {
            word(xprog, m) = 0;
            word(gcred, m) = 0;
        }
        word(garr, 0) = 0;
        for (uint32_t i = 0; i < NR; ++i) {
            word(rflags, i) = 0;
        }
        word(rgrant, 0) = 0;
        for (uint32_t k = 0; k < NC; ++k) {
            p<uint32_t>(cnt_in)[k] = 0;
        }
        for (uint32_t o = 0; o < NOWN; ++o) {
            p<uint32_t>(off_in)[o] = 0;
        }
        const uint32_t cnt = get_semaphore(5), go = get_semaphore(6);
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
    const uint32_t nblk = n * NB;
    const uint32_t n_own = n > c ? (n - c + GS - 1) / GS : 0;
    // QX
    uint32_t b_s = 0, x_cred = 0, x_popped = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    // QH
    uint32_t h_s = c == h ? 0 : nblk;
    // QR
    uint32_t r_i = 0;
    bool r_fly = false;
    // QG
    uint32_t g_cr = 0, g_push = 0, g_popped = 0;
    const uintptr_t gacked = (uintptr_t)get_cb_tiles_acked_ptr(ga_cb);
    uint16_t g_last = (uint16_t)reg_read(gacked);
    // QA
    constexpr uint32_t QA_TOK = 4;
    uint32_t a_n = 0, a_col = 0, nrec = 0, t_qa = 0;
    // QL
#ifdef WSR_DEBUG_LOGITS
    uint32_t l_i = 0;
#else
    uint32_t l_i = n;
#endif

    uint32_t t_mc = 0, t_cr = 0, t_rc = 0, last = wall();
    bool cr_wait = false, rc_wait = false;
    while (b_s < nblk || x_cred < nblk || h_s < nblk || r_i < n || g_push < n_own || a_n < n_own || l_i < n) {
        {
            const uint32_t now = wall();
            t_cr += cr_wait ? now - last : 0;
            t_rc += rc_wait ? now - last : 0;
            last = now;
            cr_wait = rc_wait = false;
        }
        // ---- QX
        if (b_s < nblk && cb_pages_reservable_at_back(xr_cb, BK)) {
            invalidate_l1_cache();
            if (word(xflags, b_s % NX) >= b_s + 1) {
                cb_reserve_back(xr_cb, BK);
                cb_push_back(xr_cb, BK);
                ++b_s;
            }
        }
        {
            const uint16_t a = (uint16_t)reg_read(xacked);
            x_popped += (uint16_t)(a - x_last);
            x_last = a;
            if (x_cred < x_popped / BK) {
                x_cred = x_popped / BK;
                noc_inline_dw_write(member(h, xprog + c * 16), x_cred);
            }
        }
        // ---- QH
        if (h_s < nblk) {
            const uint32_t s = h_s;
            invalidate_l1_cache();
            if (word(rflags, s % NR) >= s + 1) {
                bool ok = s < 2 || word(xflags, (s - 2) % NX) >= s - 1;
                if (ok && s >= NX) {
                    for (uint32_t m = 0; m < GS && ok; ++m) {
                        ok = word(xprog, m) >= s - NX + 1;
                    }
                }
                cr_wait = !ok;
                if (ok) {
                    const uint32_t a = wall();
                    const uint32_t src = relay + (s % NR) * BK * XT, dst = xring + (s % NX) * BK * XT;
                    if constexpr (GS == 1) {
                        noc_async_write(src, get_noc_addr(my_x[noc_index], my_y[noc_index], dst), BK * XT);
                        noc_async_write_barrier();
                        word(xflags, s % NX) = s + 1;
                    } else {
                        *xscr = s + 1;
                        noc_async_write_multicast_loopback_src(
                            src, get_noc_multicast_addr(mx0, my0, mx1, my1, dst), BK * XT, GS, true /*linked*/);
                        noc_async_write_multicast_loopback_src(
                            xscr_addr,
                            get_noc_multicast_addr(mx0, my0, mx1, my1, xflags + (s % NX) * 16),
                            4,
                            GS,
                            false);
                        noc_async_writes_flushed();
                    }
                    if (s + NR < nblk) {
                        noc_inline_dw_write(member(((s + NR) % NB) % GS, rgrant), s + NR + 1);
                    }
                    t_mc += wall() - a;
                    ++h_s;
                }
            }
        }
        // ---- QR
        if (r_fly) {
            if (ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, R_TRID)) {
                noc_semaphore_inc(member(r_i % GS, garr), 1);
                cb_pop_front(run_cb, 1);
                ++r_i;
                r_fly = false;
            }
        } else if (r_i < n && cb_pages_available_at_front(run_cb, 1)) {
            const uint32_t ow = r_i % GS, nn = r_i / GS;
            invalidate_l1_cache();
            rc_wait = word(gcred, ow) < nn + 1;
            if (!rc_wait) {
                noc_async_write_one_packet_with_trid(
                    get_read_ptr(run_cb), member(ow, gring + ((nn % NG) * GS + c) * FT), FT, R_TRID);
                noc_async_write_set_trid(0);
                r_fly = true;
            }
        }
        // ---- QG
        {
            const uint16_t a = (uint16_t)reg_read(gacked);
            g_popped += (uint16_t)(a - g_last);
            g_last = a;
            while (g_cr < n_own && g_cr < g_popped / GS + NG) {
                for (uint32_t m = 0; m < GS; ++m) {
                    noc_inline_dw_write(member(m, gcred + c * 16), g_cr + 1);
                }
                ++g_cr;
            }
            if (g_push < g_cr) {
                invalidate_l1_cache();
                if (word(garr, 0) >= GS * (g_push + 1)) {
                    cb_reserve_back(ga_cb, GS);
                    cb_push_back(ga_cb, GS);
                    ++g_push;
                }
            }
        }
        // ---- QA
        if (a_n < n_own && cb_pages_available_at_front(be_cb, 1) && cb_pages_available_at_front(ee_cb, 1)) {
            const uint32_t a = wall();
            const uint32_t r = R0 + c + a_n * GS;
            volatile tt_l1_ptr uint32_t* key = p<uint32_t>(get_read_ptr(be_cb));
            volatile tt_l1_ptr uint32_t* e = p<uint32_t>(get_read_ptr(ee_cb));
            const uint32_t col_end = a_col + QA_TOK < 32 ? a_col + QA_TOK : 32;
            for (uint32_t col = a_col; col < col_end && r * 32 + col < T; ++col) {
                const uint32_t t = r * 32 + col;
                float v[K], sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    v[k] = __builtin_bit_cast(float, e[at(k, col)]);
                    sum += v[k];
                }
                const float inv = 1.0f / sum;
                volatile tt_l1_ptr uint32_t* wr = p<uint32_t>(wst + col * WROW);
#if defined(WSR_EP4)
                uint32_t mask = 0;
                for (uint32_t k = 0; k < 16; ++k) {
                    wr[k] = k < K ? __builtin_bit_cast(uint32_t, v[k] * inv) : 0;
                }
                for (uint32_t k = 0; k < K; ++k) {
                    const uint32_t ek = key[at(k, col)] & IDM, ck = ek / WSR_EP4;
                    wr[16 + k] = ek;
                    mask |= ck != MC ? 1u << (RB + ck) : 0u;
                }
                for (uint32_t k = 16 + K; k < 32; ++k) {
                    wr[k] = 0;
                }
                wr[28] = mask | (t & 1) << 17;
                p<uint32_t>(rstage)[col] = wr[28];
#elif defined(WSR_SIDE)
                // (bank side: pick k's slot j among the token's result rows t S + j so that the row's DRAM bank, (t S + j)
                // % WSR_SIDE_BANKS, is on the side of k's expert's lane group (e % WSR_SIDE >= WSR_SIDE / 2: the upper
                // half of the banks), the rest in order; w and the record carry the slot, so the lists' rows, the
                // fold's rows and its weights stay t S + j / w (t, j))
                constexpr uint32_t SR = get_compile_time_arg_val(2);  // (S: result rows a token)
                uint32_t slot[K] = {};
                uint32_t up = 0, freem = (1u << K) - 1, pend = 0;  // (slots in the upper banks, slots free, k unplaced)
                const uint32_t base = (t * SR) % WSR_SIDE_BANKS;
                for (uint32_t j = 0; j < K; ++j) {
                    up |= (((base + j) % WSR_SIDE_BANKS) >= WSR_SIDE_BANKS / 2 ? 1u : 0u) << j;
                }
                for (uint32_t k = 0; k < K; ++k) {
                    const bool es = ((key[at(k, col)] & IDM) % WSR_SIDE) >= WSR_SIDE / 2;
                    const uint32_t cand = freem & (es ? up : ~up);
                    if (cand) {
                        slot[k] = __builtin_ctz(cand);
                        freem &= ~(1u << slot[k]);
                    } else {
                        pend |= 1u << k;
                    }
                }
                while (pend) {  // (the rest in order of the free slots)
                    const uint32_t k = __builtin_ctz(pend);
                    pend &= pend - 1;
                    slot[k] = __builtin_ctz(freem);
                    freem &= freem - 1;
                }
                for (uint32_t k = K; k < 16; ++k) {
                    wr[k] = 0;
                }
                for (uint32_t k = 0; k < K; ++k) {
                    wr[slot[k]] = __builtin_bit_cast(uint32_t, v[k] * inv);
                }
#else
                for (uint32_t k = 0; k < 16; ++k) {
                    wr[k] = k < K ? __builtin_bit_cast(uint32_t, v[k] * inv) : 0;
                }
#endif
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    wst + col * WROW, ws.get_noc_addr(t * WROW / WPAGE, t * WROW % WPAGE), WROW);
#ifndef WSR_EP4
                for (uint32_t k = 0; k < K; ++k) {
#if defined(WSR_SIDE)
                    p<uint32_t>(rec)[nrec++] = t | slot[k] << 16 | (key[at(k, col)] & IDM) << 20;
#else
                    p<uint32_t>(rec)[nrec++] = t | k << 16 | (key[at(k, col)] & IDM) << 20;
#endif
                }
#endif
            }
            a_col = col_end;
            if (a_col == 32 || r * 32 + a_col >= T) {
#ifdef WSR_EP4
                for (uint32_t col = T - r * 32 < 32 ? T - r * 32 : 32; col < 32; ++col) {  // (tokens >= T: nowhere)
                    p<uint32_t>(rstage)[col] = 0;
                }
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(rstage, route_at(r), 128);
#endif
                noc_async_posted_writes_flushed();  // (the w staging rows are rewritten for the next row tile)
                cb_pop_front(be_cb, 1);
                cb_pop_front(ee_cb, 1);
                ++a_n;
                a_col = 0;
            }
            t_qa += wall() - a;
        }
#ifdef WSR_DEBUG_LOGITS
        // ---- QL
        if (l_i < n && cb_pages_available_at_front(lg_cb, EPC)) {
            for (uint32_t e = 0; e < EPC; ++e) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    get_read_ptr(lg_cb) + e * FT, ls.get_noc_addr((c * EPC + e) * RTX + R0 + l_i), FT);
            }
            noc_async_posted_writes_flushed();
            cb_pop_front(lg_cb, EPC);
            ++l_i;
        }
#endif
    }
    noc_async_write_barrier();
    noc_async_posted_writes_flushed();
    stamps[2] = wall();
    stamps[20] = t_mc;
    stamps[21] = t_cr;
    stamps[22] = t_rc;
    stamps[23] = t_qa;

    // ---- the barrier: every core's group traffic done
    const uint32_t bar = get_semaphore(2), go = get_semaphore(3);
    noc_semaphore_inc(noc_at(0, bar), 1);
    noc_async_atomic_barrier();
    if (me == 0) {
        noc_semaphore_wait(p<uint32_t>(bar), NC);
        for (uint32_t k = 0; k < NC; ++k) {
            noc_inline_dw_write(noc_at(k, go), 1);
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(p<uint32_t>(go), 1);
    for (uint32_t i = 0; i < NX; ++i) {
        word(xflags, i) = 0;
    }
    for (uint32_t m = 0; m < GS; ++m) {
        word(xprog, m) = 0;
        word(gcred, m) = 0;
    }
    word(garr, 0) = 0;
    for (uint32_t i = 0; i < NR; ++i) {
        word(rflags, i) = 0;
    }
    word(rgrant, 0) = 0;

#ifndef WSR_EP4
    exchange(
        me,
        nrec,
        T,
        noc_at,
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9),
        stamps);
#else
    noc_async_write_barrier();
    noc_async_posted_writes_flushed();
    stamps[63] = wall();
#endif
}
