// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Per-row top-16 of bf16 logits, fix 4b (M6b; d_topk.hpp is fix 4): the same lists (order key of the bf16 value, the
// larger key first, equal keys the smaller vocab index first; per row 32 words (key, index) x 16), computed with
//   - a prefilter: the compute engine's row maximum of every 32-column logits tile (d_streamer_compute_v2.cpp, column 0
//     of tile t of `tmax`). The 16 largest tile maxima are 16 distinct values >= the 16th tile maximum `thr`, so the
//     16th list entry's key is >= key(thr), and a tile whose maximum's key is < key(thr) holds no entry: only the tiles
//     at or above it are scanned (exact; in increasing column order, so equal keys keep the smaller index);
//   - only the rows the mask selects (the drafting rows; mask == nullptr: every row);
//   - plain (non-volatile) loads and lists in registers / the stack, written out once per row.
// The caller invalidates the RISC's L1 cache after the CB waits.
#pragma once

#include <stdint.h>

namespace dtop2 {

constexpr uint32_t kTop = 16;
constexpr uint32_t kRowWords = 2 * kTop;
constexpr uint32_t kRowBytes = 4 * kRowWords;

FORCE_INLINE uint32_t key16(uint32_t b) { return (b & 0x8000) ? (~b & 0xffff) : (b | 0x8000); }
FORCE_INLINE bool better(uint32_t k, uint32_t i, uint32_t bk, uint32_t bi) { return k > bk || (k == bk && i < bi); }

struct List {
    uint32_t k[kTop] = {};
    uint32_t i[kTop] = {};
    uint32_t n = 0;

    FORCE_INLINE void insert(uint32_t key, uint32_t idx) {
        uint32_t p = n < kTop ? n : kTop - 1;
        if (n == kTop && !better(key, idx, k[p], i[p])) {
            return;
        }
        while (p > 0 && better(key, idx, k[p - 1], i[p - 1])) {
            if (p < kTop) {
                k[p] = k[p - 1];
                i[p] = i[p - 1];
            }
            p--;
        }
        k[p] = key;
        i[p] = idx;
        n += n < kTop ? 1 : 0;
    }
};

// the 16 largest of `count` keys: returns the 16th (0 when fewer)
FORCE_INLINE uint32_t sixteenth(const uint32_t* keys, uint32_t count) {
    uint32_t top[kTop] = {};
    uint32_t n = 0;
    for (uint32_t t = 0; t < count; t++) {
        const uint32_t key = keys[t];
        if (n == kTop && key <= top[kTop - 1]) {
            continue;
        }
        uint32_t p = n < kTop ? n : kTop - 1;
        while (p > 0 && key > top[p - 1]) {
            if (p < kTop) {
                top[p] = top[p - 1];
            }
            p--;
        }
        top[p] = key;
        n += n < kTop ? 1 : 0;
    }
    return n == kTop ? top[kTop - 1] : 0;
}

// rows [r0, r1) (those with mask[r] != 0) of the logits (nv 32 x 32 bf16 tiles at `base`; their row maxima at `tmax`)
FORCE_INLINE void top16_rows(
    uint32_t base,
    uint32_t tmax,
    uint32_t nv,
    uint32_t first_index,
    uint32_t r0,
    uint32_t r1,
    const uint32_t* mask,
    volatile tt_l1_ptr uint32_t* rec) {
    uint32_t tkey[64];
    for (uint32_t r = r0; r < r1; r++) {
        if (mask != nullptr && mask[r] == 0) {
            continue;
        }
        const uint32_t roff = (r / 16) * 1024 + (r % 16) * 32;
        for (uint32_t t = 0; t < nv; t++) {
            tkey[t] = key16(*reinterpret_cast<const uint16_t*>(tmax + t * 2048 + roff));
        }
        const uint32_t thr = sixteenth(tkey, nv);
        List L;
        uint32_t lthr = 0;
        for (uint32_t t = 0; t < nv; t++) {
            if (tkey[t] < thr) {
                continue;
            }
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t* row = reinterpret_cast<const uint32_t*>(base + t * 2048 + roff + f * 512);
                const uint32_t col0 = first_index + t * 32 + f * 16;
                uint32_t w[8];
                for (uint32_t j = 0; j < 8; j++) {
                    w[j] = row[j];
                }
                for (uint32_t j = 0; j < 8; j++) {
                    const uint32_t kv = w[j] ^ ((((w[j] >> 15) & 0x10001u) * 0x7FFFu) | 0x80008000u);
                    const uint32_t k0 = kv & 0xffff, k1 = kv >> 16;
                    if (L.n < kTop || k0 > lthr) {
                        L.insert(k0, col0 + 2 * j);
                        lthr = L.k[kTop - 1];
                    }
                    if (L.n < kTop || k1 > lthr) {
                        L.insert(k1, col0 + 2 * j + 1);
                        lthr = L.k[kTop - 1];
                    }
                }
            }
        }
        for (uint32_t e = 0; e < kTop; e++) {
            rec[r * kRowWords + 2 * e] = L.k[e];
            rec[r * kRowWords + 2 * e + 1] = L.i[e];
        }
    }
}

}  // namespace dtop2
