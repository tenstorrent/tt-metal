// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Per-row top-16 of bf16 logits (M6b fix 4; DFlash2's selector takes the unary top-16 of the target head): the order
// key of a bf16 value (monotone in the value: negative values bit-inverted, positive values with the sign bit set) and
// a sorted list of (key, vocab index) per row, the larger key first and, on equal keys, the smaller index first.
// Records: per row 32 words, (key, index) x 16.
#pragma once

#include <stdint.h>

namespace dtop {

constexpr uint32_t kTop = 16;
constexpr uint32_t kRowWords = 2 * kTop;  // one row's record
constexpr uint32_t kRowBytes = 4 * kRowWords;

// a < b in the list order: (key, index) ranks before when the key is larger or, equal, the index smaller
FORCE_INLINE bool better(uint32_t k, uint32_t i, uint32_t bk, uint32_t bi) { return k > bk || (k == bk && i < bi); }

// insert (k, i) into the sorted list (n entries, at most kTop kept); returns the new n
FORCE_INLINE uint32_t insert(volatile tt_l1_ptr uint32_t* list, uint32_t n, uint32_t k, uint32_t i) {
    uint32_t p = n < kTop ? n : kTop - 1;
    if (n == kTop && !better(k, i, list[2 * p], list[2 * p + 1])) {
        return n;
    }
    while (p > 0 && better(k, i, list[2 * (p - 1)], list[2 * (p - 1) + 1])) {
        if (p < kTop) {
            list[2 * p] = list[2 * (p - 1)];
            list[2 * p + 1] = list[2 * (p - 1) + 1];
        }
        p--;
    }
    list[2 * p] = k;
    list[2 * p + 1] = i;
    return n < kTop ? n + 1 : n;
}

// rows [r0, r1) of the logits (nv 32 x 32 bf16 tiles at `base`, R = 32 face layout): each row's top-16 into
// rec[r * kRowWords ..] (global vocab indices: first_index + column). Values are scanned in increasing column order,
// so within a row equal keys keep the earlier (smaller) index; a candidate enters only above the 16th key.
FORCE_INLINE void top16_rows(
    uint32_t base, uint32_t nv, uint32_t first_index, uint32_t r0, uint32_t r1, volatile tt_l1_ptr uint32_t* rec) {
    for (uint32_t r = r0; r < r1; r++) {
        volatile tt_l1_ptr uint32_t* list = rec + r * kRowWords;
        uint32_t n = 0, thr = 0;
        for (uint32_t t = 0; t < nv; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    base + t * 2048 + (r / 16) * 1024 + f * 512 + (r % 16) * 32);
                const uint32_t col0 = first_index + t * 32 + f * 16;
                for (uint32_t w = 0; w < 8; w++) {
                    const uint32_t v = row[w];
                    // both halves' keys at once: sign set -> invert all 16 bits, else set the sign bit
                    const uint32_t kv = v ^ ((((v >> 15) & 0x10001u) * 0x7FFFu) | 0x80008000u);
                    const uint32_t k0 = kv & 0xffff, k1 = kv >> 16;
                    if (n < kTop || k0 > thr) {
                        n = insert(list, n, k0, col0 + 2 * w);
                        thr = list[2 * (kTop - 1)];
                    }
                    if (n < kTop || k1 > thr) {
                        n = insert(list, n, k1, col0 + 2 * w + 1);
                        thr = list[2 * (kTop - 1)];
                    }
                }
            }
        }
    }
}

}  // namespace dtop
