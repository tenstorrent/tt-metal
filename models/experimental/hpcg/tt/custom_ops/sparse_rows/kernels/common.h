// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse-row op: metadata and DRAM helpers shared by the reader and the writer.
// META row (uint32 words) per core:
//   [0] R window pieces  [1] NPH phases  [2] NIN incs expected per phase (neighbours + self)  [3] NNB neighbours
//   [4] NB out blocks
//   [5 .. 5+3R)  window pieces: (DRAM q lo, words, L1 byte offset in the window) -- the x values this core reads
//   then per out block b: seg_off, nseg, exch_off, nexch, own_a, own_e, own_l1 (L1 byte offset of own_a)  (7 words)
//   then NNB x (noc_x, noc_y, my slot there), then NPH phase -> out block ids, then 2 words (import table, unused)
//   segments: (q0, q1) pairs; exchange entries: (noc_x, noc_y, qs, qe, L1 byte offset of qs in that core's window)
#pragma once
#include <cstdint>
struct Meta {
    const volatile uint32_t* m;
    uint32_t R, NPH, NIN, NNB, NB;
    uint32_t lo(uint32_t r) const { return m[5 + 3 * r]; }
    uint32_t len(uint32_t r) const { return m[5 + 3 * r + 1]; }
    uint32_t l1(uint32_t r) const { return m[5 + 3 * r + 2]; }
    const volatile uint32_t* blk(uint32_t b) const { return m + 5 + 3 * R + 7 * b; }
    const volatile uint32_t* nb() const { return m + 5 + 3 * R + 7 * NB; }  // NNB x (noc_x, noc_y, my slot there)
    uint32_t phase(uint32_t p) const { return m[5 + 3 * R + 7 * NB + 3 * NNB + p]; }
    // phase id = out block b (0..7): a Gauss-Seidel phase; 8 + b: an SpMV pass over block b (RESID launches)
    uint32_t blkid(uint32_t p) const { return phase(p) & 7; }
    bool sp(uint32_t p) const { return (phase(p) >> 3) != 0; }
    uint32_t ngs() const {
        uint32_t c = 0;
        for (uint32_t p = 0; p < NPH; ++p) {
            c += !sp(p);
        }
        return c;
    }
    void init(uint32_t addr) {
        m = (const volatile uint32_t*)addr;
        R = m[0];
        NPH = m[1];
        NIN = m[2];
        NNB = m[3];
        NB = m[4];
    }
};
// write address `ahead` pages past this CB's write pointer (pages reserved for earlier in-flight units that are not
// pushed yet; cb_reserve_back does not move the write pointer), wrapping at the end of the CB
inline uint32_t cb_ahead_ptr(uint32_t cb, uint32_t ahead, uint32_t page_bytes) {
    const LocalCBInterface& c = get_local_cb_interface(cb);
    uint32_t a = c.fifo_wr_ptr + ahead * page_bytes;
    if (a >= c.fifo_limit) {
        a -= c.fifo_size;
    }
    return a;
}
// read words [q, q + n) of a q-ordered vector (4 KB pages) into L1
template <typename A>
inline void read_q(const A& acc, uint32_t q, uint32_t l1, uint32_t n, uint8_t noc = noc_index) {
    while (n) {
        const uint32_t in = q & 1023, k = (1024 - in) < n ? (1024 - in) : n;
        noc_async_read(acc.get_noc_addr(q >> 10, 0, noc) + in * 4, l1, k * 4, noc);
        q += k;
        l1 += k * 4;
        n -= k;
    }
}
template <typename A>
inline void write_q(const A& acc, uint32_t q, uint32_t l1, uint32_t n) {
    while (n) {
        const uint32_t in = q & 1023, k = (1024 - in) < n ? (1024 - in) : n;
        noc_async_write(l1, acc.get_noc_addr(q >> 10) + in * 4, k * 4);
        q += k;
        l1 += k * 4;
        n -= k;
    }
}
// progress slots: semaphore 1 + i, i = 0 this core's writer, i = 1..NNB the neighbours (max semantics, so a core
// that runs ahead can never stand in for a slow one). Value v = rounds done: 1 = window loaded, q + 2 = phase q done.
inline void wait_slots(uint32_t sem_base, uint32_t nslots, uint32_t target) {
    for (uint32_t i = 0; i < nslots; ++i) {
        volatile tt_l1_ptr uint32_t* s = (volatile tt_l1_ptr uint32_t*)(sem_base + (1 + i) * 16);
        while (*s < target) {
            invalidate_l1_cache();
        }
    }
    invalidate_l1_cache();
}
inline void wait_sem(uint32_t sem_addr, uint32_t target) {
    volatile tt_l1_ptr uint32_t* s = (volatile tt_l1_ptr uint32_t*)sem_addr;
    while (*s < target) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
}
