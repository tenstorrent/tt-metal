// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear: stream every second output-tile record of this core into `cb` (NCRISC the even ones into
// c_1, BRISC the odd ones into c_6: two issuing RISCs read the 4-KB interleaved pages at 95% of the 512 GB/s peak,
// one at 82%, bonsai/mb/dram_bw.py). One 4-KB DRAM page per read (16 CB pages of 256 B), up to NB reads in flight,
// each with its own transaction id. The CB holds exactly two records and every record starts a new half, so a read
// never wraps around the CB's end.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// X_HUB (decode_linear3.py rotated()): x' arrives from the fused rotation's cores. They write their blocks into the
// hub's (core 0's) L1 buffer and bump its counter; the hub multicasts the whole x' and a flag to every core (loopback);
// every core then copies its buffer into c_0. Polled between weight-page reads so the stream never waits for it.
struct XHub {
    uint32_t buf, xbytes, nblk, is_hub, nrect, rect_arg0, cb_x, x_pages, r_cb, r_pages;
    bool sent, done;
};
FORCE_INLINE void xhub_poll(XHub& h, bool block) {
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h.buf + h.xbytes);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h.buf + h.xbytes + 16);
    volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h.buf + h.xbytes + 32);
    do {
        invalidate_l1_cache();
        if (h.is_hub && !h.sent && *cnt >= h.nblk) {
            *cnt = 0;
            *one = 1;
            for (uint32_t q = 0; q < h.nrect; ++q) {
                const uint32_t a = h.rect_arg0 + 5 * q;
                const uint32_t x0 = get_common_arg_val<uint32_t>(a), y0 = get_common_arg_val<uint32_t>(a + 1);
                const uint32_t x1 = get_common_arg_val<uint32_t>(a + 2), y1 = get_common_arg_val<uint32_t>(a + 3);
                const uint32_t nd = get_common_arg_val<uint32_t>(a + 4);
                const uint64_t d = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, h.buf)
                                                  : get_noc_multicast_addr(x1, y1, x0, y0, h.buf);
                const uint64_t f = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, h.buf + h.xbytes)
                                                  : get_noc_multicast_addr(x1, y1, x0, y0, h.buf + h.xbytes);
                noc_async_write_multicast_loopback_src(h.buf, d, h.xbytes, nd);
                noc_semaphore_set_multicast_loopback_src(h.buf + h.xbytes + 32, f, nd);
            }
            h.sent = true;
        }
        if (!h.done && (!h.is_hub || h.sent) && *flag != 0) {
            *flag = 0;
            noc_async_read(get_noc_addr(h.buf), get_write_ptr(h.cb_x), h.xbytes);
            noc_async_read_barrier();  // (also the weight pages in flight)
            cb_push_back(h.cb_x, h.x_pages);
            if (h.r_cb != 0xFFFFFFFFu) {
                cb_push_back(h.r_cb, h.r_pages);
            }
            h.done = true;
        }
    } while (block && !h.done);
}

// W_STAGED: this core's weight pages were copied into L1 (at `base`, page page0 first) by tdec3_stage.cpp in the
// previous op; the stream reads them locally instead of from DRAM.
struct StagedW {
    uint32_t base, page0;
    FORCE_INLINE uint64_t get_noc_addr(uint32_t page) const { return ::get_noc_addr(base + (page - page0) * 4096); }
};

template <uint32_t ppr, uint32_t NB, typename Accessor>
FORCE_INLINE void stream_records(
    uint32_t cb,
    const Accessor& w,
    uint32_t first_nt,
    uint32_t num_nt,
    uint32_t parity,
    uint32_t bank = 0,
    uint32_t x_cb = 0xFFFFFFFFu,
    uint32_t x_pages = 0,
    uint32_t r_cb = 0xFFFFFFFFu,
    uint32_t r_pages = 0,
    XHub* hub = nullptr) {
    // x_cb (X_PAGED early start): the caller issued its x' (and residual, r_cb) reads from L1 without waiting; once
    // the first weight pages are in flight, a barrier lands them and they are pushed
    constexpr uint32_t PAGE = 4096;
    constexpr uint32_t CBP = PAGE / 256;
    const uint32_t cb_base = get_write_ptr(cb);
    const uint32_t cb_end = cb_base + 2 * ppr * PAGE;
    uint32_t l1 = cb_base;
    uint32_t head = 0, count = 0, trid = 0;
    for (uint32_t i = parity; i < num_nt; i += 2) {
        const uint32_t page0 = (first_nt + i) * ppr;
        for (uint32_t j = 0; j < ppr; ++j) {
            if (count == NB) {
                noc_async_read_barrier_with_trid(head + 1);
                cb_push_back(cb, CBP);
                head = (head + 1) % NB;
                --count;
            }
            if (hub != nullptr && !hub->done) {
                // the compute waits for x' before it frees CB pages: keep the hub moving while the CB is full
                while (!cb_pages_reservable_at_back(cb, (count + 1) * CBP)) {
                    xhub_poll(*hub, false);
                    if (hub->done) {
                        break;
                    }
                }
            }
            cb_reserve_back(cb, (count + 1) * CBP);
            noc_async_read_set_trid(trid + 1);
#ifndef FAKE_STREAM
#ifdef BANK_TEST
            // timing only: every page from one DRAM bank (pages bank, bank + 8, ...); the data is wrong
            noc_async_read(w.get_noc_addr(bank + 8 * ((page0 + j) / 8)), l1, PAGE);
#else
            noc_async_read(w.get_noc_addr(page0 + j), l1, PAGE);
#endif
#endif
            trid = (trid + 1) % NB;
            ++count;
            l1 += PAGE;
            if (l1 >= cb_end) {
                l1 = cb_base;
            }
            if (hub != nullptr && !hub->done) {
                xhub_poll(*hub, false);
            }
            if (x_cb != 0xFFFFFFFFu && count == NB) {
                noc_async_read_barrier();
                cb_push_back(x_cb, x_pages);
                if (r_cb != 0xFFFFFFFFu) {
                    cb_push_back(r_cb, r_pages);
                }
                x_cb = 0xFFFFFFFFu;
            }
        }
    }
    if (hub != nullptr) {
        xhub_poll(*hub, true);
    }
    if (x_cb != 0xFFFFFFFFu) {
        noc_async_read_barrier();
        cb_push_back(x_cb, x_pages);
        if (r_cb != 0xFFFFFFFFu) {
            cb_push_back(r_cb, r_pages);
        }
    }
    while (count > 0) {
        noc_async_read_barrier_with_trid(head + 1);
        cb_push_back(cb, CBP);
        head = (head + 1) % NB;
        --count;
    }
    noc_async_read_set_trid(0);
}

#ifdef W_STREAM
#include "../../stream/kernels/rx_stream.hpp"
// W_STREAM: this core's records come from its weight-stream ring (stream.py), in order, RX_SLOT / 4096 pages per slot
// (the last slot padded); the reader takes them all, record i into cb0 (even i) or cb1 (odd i), as the two RISCs'
// DRAM streams would. x_cb / hub: as stream_records.
template <uint32_t ppr>
FORCE_INLINE void stream_ring(
    uint32_t cb0,
    uint32_t cb1,
    uint32_t num_nt,
    uint32_t ring,
    uint32_t rcount,
    uint32_t freed,
    uint32_t x_cb = 0xFFFFFFFFu,
    uint32_t x_pages = 0,
    uint32_t r_cb = 0xFFFFFFFFu,
    uint32_t r_pages = 0,
    XHub* hub = nullptr) {
    constexpr uint32_t PAGE = 4096, CBP = PAGE / 256, PPS = RX_SLOT / PAGE;
    RxRing rx = rx_open(ring, rcount, freed);
    const uint32_t pages = num_nt * ppr;
    for (uint32_t q0 = 0; q0 < pages; q0 += PPS) {
        while (!rx_ready(rx)) {
            if (hub != nullptr && !hub->done) {
                xhub_poll(*hub, false);
            }
        }
        const uint32_t src = rx.ring + (rx.c % RX_RING_SLOTS) * RX_SLOT;
        const uint32_t n = pages - q0 < PPS ? pages - q0 : PPS;
        // the slot's pages: one read per page, one barrier; a CB's pages in one slot belong to one record, so they
        // are contiguous from its write pointer (records start at a half of the two-record CB)
        uint32_t cnt[2] = {0, 0};
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t q = q0 + k;
            const uint32_t side = (q / ppr) & 1;
            const uint32_t cb = side ? cb1 : cb0;
            if (hub != nullptr && !hub->done) {
                while (!cb_pages_reservable_at_back(cb, (cnt[side] + 1) * CBP)) {
                    xhub_poll(*hub, false);
                    if (hub->done) {
                        break;
                    }
                }
            }
            cb_reserve_back(cb, (cnt[side] + 1) * CBP);
            noc_async_read(get_noc_addr(src + k * PAGE), get_write_ptr(cb) + cnt[side] * PAGE, PAGE);
            ++cnt[side];
        }
        noc_async_read_barrier();
        if (cnt[0]) {
            cb_push_back(cb0, cnt[0] * CBP);
        }
        if (cnt[1]) {
            cb_push_back(cb1, cnt[1] * CBP);
        }
        rx_release(rx);
        if (x_cb != 0xFFFFFFFFu) {  // x' (and the residual) issued by the caller: landed with the first slot
            cb_push_back(x_cb, x_pages);
            if (r_cb != 0xFFFFFFFFu) {
                cb_push_back(r_cb, r_pages);
            }
            x_cb = 0xFFFFFFFFu;
        }
        if (hub != nullptr && !hub->done) {
            xhub_poll(*hub, false);
        }
    }
    if (hub != nullptr) {
        xhub_poll(*hub, true);
    }
    if (x_cb != 0xFFFFFFFFu) {
        noc_async_read_barrier();
        cb_push_back(x_cb, x_pages);
        if (r_cb != 0xFFFFFFFFu) {
            cb_push_back(r_cb, r_pages);
        }
    }
    rx_close(rx);
}
#endif

// PAGE_SPLIT: the two RISCs take alternate 4-KB pages of this core's records (NCRISC the even ones, BRISC the odd
// ones), so both stream for any number of output tiles per core (whole records by parity left one RISC idle on a
// core with one tile: 285 GB/s). The compute walks the pages in order, alternating between c_1 and c_6.
template <uint32_t ppr, uint32_t NB, typename Accessor>
FORCE_INLINE void stream_pages(uint32_t cb, const Accessor& w, uint32_t first_nt, uint32_t num_nt, uint32_t parity) {
    constexpr uint32_t PAGE = 4096;
    constexpr uint32_t CBP = PAGE / 256;
    const uint32_t cb_base = get_write_ptr(cb);
    const uint32_t cb_end = cb_base + 2 * ppr * PAGE;
    uint32_t l1 = cb_base;
    uint32_t head = 0, count = 0, trid = 0;
    const uint32_t page0 = first_nt * ppr, pages = num_nt * ppr;
    for (uint32_t q = parity; q < pages; q += 2) {
        if (count == NB) {
            noc_async_read_barrier_with_trid(head + 1);
            cb_push_back(cb, CBP);
            head = (head + 1) % NB;
            --count;
        }
        cb_reserve_back(cb, (count + 1) * CBP);
        noc_async_read_set_trid(trid + 1);
#ifndef FAKE_STREAM
        noc_async_read(w.get_noc_addr(page0 + q), l1, PAGE);
#endif
        trid = (trid + 1) % NB;
        ++count;
        l1 += PAGE;
        if (l1 >= cb_end) {
            l1 = cb_base;
        }
    }
    while (count > 0) {
        noc_async_read_barrier_with_trid(head + 1);
        cb_push_back(cb, CBP);
        head = (head + 1) % NB;
        --count;
    }
    noc_async_read_set_trid(0);
}
