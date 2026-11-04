// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b model head (tt/mk_head.py): device top-K of the logits on the lm_head readers' activation mover
// (kernels/mk/reader_a.cpp, MK_TOPK). Every lm_head phase's NR bf16 output tiles are scanned in L1 before they are
// written out: per token row (< T <= 32) the reader keeps its K best (key, vocab id) (ties: the lower id, i.e. the
// first in vocab order, as torch.argmax) as a heap in its L1 scratch MK_TOPK_CB (row r at r * K * 8 bytes: K x (key,
// id); then 32 counts); after the last phase every row is sorted best-first and the rows go to this reader's page of
// the DRAM landing tensor, then the select core's semaphore is incremented (kernels/head/topk_select.cpp merges). bf16
// order key: (v & 0x8000) ? ~v & 0xffff : v | 0x8000 (larger key = larger value). Runtime args at MK_TOPK_RT: T, cw
// (valid columns per phase), vocab id of phase 0 column 0, select core NoC x, y, the landing tensor's address (one
// MK_TOPK_PAGE-byte page per reader), this reader's page. Defines: MK_TOPK_K, MK_TOPK_SEM, MK_TOPK_CB (32 * K * 8 + 128
// bytes), MK_TOPK_PAGE (>= 32 * K * 8).
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace mktopk {

constexpr uint32_t K = MK_TOPK_K;
constexpr uint32_t ROWS = 32;
constexpr uint32_t MAXP = 15;  // scanned phases per launch (MkHead asserts splits <= 15, tiles per phase <= 128)

FORCE_INLINE volatile tt_l1_ptr uint32_t* base() {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(MK_TOPK_CB));
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* counts() { return base() + ROWS * K * 2; }

// While scanning, a row's kept candidates are a binary min-HEAP of 32-bit composites (key << 16) | (0xffff - local),
// local = (phase slot << 12) | column within this reader's tiles of the phase: one unsigned compare orders "larger key,
// then lower vocab id" (a reader's columns grow with local). The root is the row's WORST kept candidate, i.e. the
// rejection threshold; send() sorts best-first and expands to (key, vocab id) pairs (ph_vb: the slot's first id).
uint32_t ph_vb[MAXP];
uint32_t nph;
// per row, a lower bound of its K-th best value from the first scanned phase (prepass): composite (fk << 16) - 1, i.e.
// a value enters only if its key is >= fk (0: no bound)
uint32_t row_floor[ROWS];

FORCE_INLINE uint32_t bf16_key(uint32_t v) { return v ^ (0x8000u | ((0u - (v >> 15)) & 0x7fffu)); }

// every row empty, called once per launch before the first scan (empty slots are written by send)
FORCE_INLINE void init() {
    for (uint32_t r = 0; r < ROWS; ++r) {
        counts()[r] = 0;
        row_floor[r] = 0;
    }
    nph = 0;
}

FORCE_INLINE void sift_down(volatile tt_l1_ptr uint32_t* h, uint32_t m, uint32_t i, uint32_t x) {
    while (true) {  // the hole at i takes x where no child is worse (smaller) than it
        uint32_t c = 2 * i + 1;
        if (c >= m) {
            break;
        }
        uint32_t cv = h[c];
        if (c + 1 < m) {
            const uint32_t dv = h[c + 1];
            if (dv < cv) {
                c = c + 1;
                cv = dv;
            }
        }
        if (cv >= x) {
            break;
        }
        h[i] = cv;
        i = c;
    }
    h[i] = x;
}

// the accept path, out of line (the scan loop stays small and register-resident): x is better than the row's root
// (or the row is not full); returns the row's new root, 0 while it is not full (every composite is > 0: local <=
// 0xefff)
__attribute__((noinline)) uint32_t take(volatile tt_l1_ptr uint32_t* h, volatile tt_l1_ptr uint32_t* cnt, uint32_t x) {
    const uint32_t n = *cnt;
    if (n < K) {  // push: sift up while the new candidate is worse than its parent
        uint32_t i = n;
        while (i > 0) {
            const uint32_t p = (i - 1) >> 1;
            const uint32_t pv = h[p];
            if (pv <= x) {
                break;
            }
            h[i] = pv;
            i = p;
        }
        h[i] = x;
        *cnt = n + 1;
        return n + 1 == K ? h[0] : 0u;
    }
    sift_down(h, K, 0, x);  // the new candidate replaces the root
    return h[0];
}

// the fast reject bound of a root: once the row's worst kept value is >= +0 (key >= 0x8000), a bf16 value can only
// enter if, read as int16, it exceeds the root's raw bits t (non-negative bf16 orders like int16, every negative one is
// < 0 <= t); for a word of two values, max((int32) w, (int32) (w << 16)) > (t << 16) | 0xffff tests both lanes in one
// compare (equal keys never enter: the earlier column wins). Otherwise INT32_MIN: every word takes the exact path
// (max(w, w << 16) is never INT32_MIN: w = 0x80000000 gives w << 16 = 0)
FORCE_INLINE int32_t bound(uint32_t root) {
    return root >= 0x80000000u ? static_cast<int32_t>((root & 0x7fff0000u) | 0xffffu) : INT32_MIN;
}

// the k-th largest of a[0..n) (1 <= k <= n; a is reordered): Hoare quickselect
FORCE_INLINE uint32_t kth_largest(uint32_t* a, int32_t n, int32_t k) {
    int32_t lo = 0, hi = n - 1;
    const int32_t want = n - k;  // its index in ascending order
    while (lo < hi) {
        const uint32_t pivot = a[(lo + hi) >> 1];
        int32_t i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot) {
                ++i;
            }
            while (a[j] > pivot) {
                --j;
            }
            if (i <= j) {
                const uint32_t x = a[i];
                a[i] = a[j];
                a[j] = x;
                ++i;
                --j;
            }
        }
        if (want <= j) {
            hi = j;
        } else if (want >= i) {
            lo = i;
        } else {
            return a[want];  // between the partitions: equal to the pivot
        }
    }
    return a[want];
}

// the first scanned phase, before its scan: per row the order-key maximum of each face row (16 values) of its whole
// tiles; at least K such groups -> their K-th largest maximum fk bounds the row's K-th best value from below (K groups
// hold a value >= fk each), so values with key < fk can never be in the row's top-K: row_floor. It cuts the heap's
// accepts of a 26-tile phase ~3x (host emulation: 135 -> 45 per row at TP8) for ~one extra pass over the words.
FORCE_INLINE void prepass(uint32_t l1, uint32_t NR, uint32_t n0, uint32_t cw, uint32_t tile_bytes, uint32_t T) {
    uint32_t g[64];
    for (uint32_t r = 0; r < T; ++r) {
        const uint32_t rbase = ((r >> 4) << 1) * 256u + (r & 15u) * 16u;
        int32_t ng = 0;
        for (uint32_t j = 0; j < NR && ng + 2 <= 64; ++j) {
            if ((n0 + j) * 32 + 32 > cw) {
                break;  // whole tiles only (a partial tile's values are still scanned, just not used for the bound)
            }
            const uint32_t* w = reinterpret_cast<const uint32_t*>(l1 + j * tile_bytes) + (rbase >> 1);
            for (uint32_t f = 0; f < 2; ++f) {
                uint32_t mx = 0;
#pragma GCC unroll 8
                for (uint32_t q = 0; q < 8; ++q) {
                    const uint32_t x = w[f * 128u + q];
                    const uint32_t sb = (x >> 15) & 0x00010001u;               // each lane's sign bit at its bit 0
                    const uint32_t k = x ^ (0x80008000u | ((sb << 15) - sb));  // both lanes' bf16_key at once
                    const uint32_t hi = k >> 16, lo = k & 0xffffu;
                    const uint32_t m2 = hi > lo ? hi : lo;
                    mx = mx > m2 ? mx : m2;
                }
                g[ng++] = mx;
            }
        }
        if (ng >= static_cast<int32_t>(K)) {
            const uint32_t fk = kth_largest(g, ng, static_cast<int32_t>(K));
            row_floor[r] = fk ? (fk << 16) - 1u : 0u;
        }
    }
}

// phase P's NR bf16 output tiles at l1 (columns n0 * 32 ..) -> the running top-K of every row < T
FORCE_INLINE void scan(uint32_t P, uint32_t l1, uint32_t NR, uint32_t n0, uint32_t tile_bytes) {
    const uint32_t T = get_arg_val<uint32_t>(MK_TOPK_RT + 0);
    const uint32_t cw = get_arg_val<uint32_t>(MK_TOPK_RT + 1);
    ph_vb[nph] = get_arg_val<uint32_t>(MK_TOPK_RT + 2) + P * cw + n0 * 32;
    const uint32_t lbase = 0xffffu - (nph << 12);  // 0xffff - local of this phase's column 0
    asm volatile("" ::: "memory");                 // the tiles' plain loads below stay after the CB wait
    if (nph == 0) {
        prepass(l1, NR, n0, cw, tile_bytes, T);
    }
    ++nph;
    volatile tt_l1_ptr uint32_t* b = base();
    for (uint32_t j = 0; j < NR; ++j) {
        const uint32_t col0 = (n0 + j) * 32;
        if (col0 >= cw) {
            break;  // DS padding columns (zero weights) are not vocab
        }
        const uint32_t ncol = cw - col0 < 32 ? cw - col0 : 32;
        const uint32_t lb = lbase - j * 32;  // 0xffff - local of the tile's column 0
        volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1 + j * tile_bytes);
        for (uint32_t r = 0; r < T; ++r) {
            volatile tt_l1_ptr uint32_t* h = b + r * K * 2;
            volatile tt_l1_ptr uint32_t* cnt = counts() + r;
            const uint32_t fl = row_floor[r];
            uint32_t root = *cnt == K ? h[0] : fl;  // full: h[0] >= fl (only keys >= the floor enter)
            int32_t bnd = bound(root);
            const uint32_t rbase = ((r >> 4) << 1) * 256u + (r & 15u) * 16u;  // face row r & 15 of face 2 (r >> 4)
            if (ncol == 32) {  // a whole tile: the row's two face rows as 2 x 8 words (two bf16 per word, column order)
                // the row's 16 words read through a plain pointer (after the CB wait: the loads may be scheduled ahead)
                const uint32_t* w0 = reinterpret_cast<const uint32_t*>(l1 + j * tile_bytes) + (rbase >> 1);
                int32_t m = INT32_MIN;  // max over the 16 words of max(w, w << 16): one branch rejects all 32 values
#pragma GCC unroll 8
                for (uint32_t q = 0; q < 8; ++q) {
                    const uint32_t a = w0[q], c2 = w0[128 + q];
                    const int32_t ma = static_cast<int32_t>(a) > static_cast<int32_t>(a << 16)
                                           ? static_cast<int32_t>(a)
                                           : static_cast<int32_t>(a << 16);
                    const int32_t mc = static_cast<int32_t>(c2) > static_cast<int32_t>(c2 << 16)
                                           ? static_cast<int32_t>(c2)
                                           : static_cast<int32_t>(c2 << 16);
                    m = m > ma ? m : ma;
                    m = m > mc ? m : mc;
                }
                if (m <= bnd) {
                    continue;  // none of the row's 32 values enters (the scan's common case)
                }
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t* w = w0 + f * 128u;
#pragma GCC unroll 8
                    for (uint32_t q = 0; q < 8; ++q) {
                        const uint32_t word = w[q];
                        const int32_t hi = static_cast<int32_t>(word), lo = static_cast<int32_t>(word << 16);
                        if ((hi > lo ? hi : lo) <= bnd) {
                            continue;  // neither value of this word enters
                        }
                        const uint32_t c = f * 16u + 2u * q;
                        uint32_t x = (bf16_key(word & 0xffffu) << 16) | (lb - c);
                        if (x > root) {
                            root = take(h, cnt, x);
                            root = root < fl ? fl : root;
                            bnd = bound(root);
                        }
                        x = (bf16_key(word >> 16) << 16) | (lb - c - 1u);
                        if (x > root) {
                            root = take(h, cnt, x);
                            root = root < fl ? fl : root;
                            bnd = bound(root);
                        }
                    }
                }
            } else {
                for (uint32_t c = 0; c < ncol; ++c) {
                    const uint32_t x = (bf16_key(t[rbase + (c >> 4) * 256u + (c & 15u)]) << 16) | (lb - c);
                    if (x > root) {
                        root = take(h, cnt, x);
                        root = root < fl ? fl : root;
                    }
                }
            }
        }
    }
}

// a row's heap of n composites -> best-first order in place (heapsort: the worst goes to the back each round), then
// expanded in place (back to front: entry i lands at 2 i >= i) to K (key, vocab id) pairs, the empty slots (0, ~0)
FORCE_INLINE void finish_row(volatile tt_l1_ptr uint32_t* h, uint32_t n) {
    for (uint32_t m = n; m > 1; --m) {
        const uint32_t worst = h[0];
        const uint32_t last = h[m - 1];
        h[m - 1] = worst;
        sift_down(h, m - 1, 0, last);
    }
    for (uint32_t i = K; i-- > 0;) {
        if (i >= n) {
            h[2 * i] = 0u;
            h[2 * i + 1] = 0xffffffffu;
            continue;
        }
        const uint32_t x = h[i];
        const uint32_t local = 0xffffu - (x & 0xffffu);
        h[2 * i + 1] = ph_vb[local >> 12] + (local & 0xfffu);
        h[2 * i] = x >> 16;
    }
}

// the rows -> this reader's landing page (+ the select core's semaphore)
FORCE_INLINE void send() {
    const uint32_t sx = get_arg_val<uint32_t>(MK_TOPK_RT + 3), sy = get_arg_val<uint32_t>(MK_TOPK_RT + 4);
    const uint32_t land = get_arg_val<uint32_t>(MK_TOPK_RT + 5);
    const uint32_t slot = get_arg_val<uint32_t>(MK_TOPK_RT + 6);
    const InterleavedAddrGen<true> pages{.bank_base_address = land, .page_size = MK_TOPK_PAGE};
    volatile tt_l1_ptr uint32_t* b = base();
    for (uint32_t r = 0; r < ROWS; ++r) {  // the heaps -> best-first (key, id) rows (empty slots (0, ~0) behind them)
        finish_row(b + r * K * 2, counts()[r]);
    }
    noc_async_write(get_write_ptr(MK_TOPK_CB), pages.get_noc_addr(slot), ROWS * K * 8);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(MK_TOPK_SEM)), 1);
    noc_async_atomic_barrier();
}

}  // namespace mktopk
