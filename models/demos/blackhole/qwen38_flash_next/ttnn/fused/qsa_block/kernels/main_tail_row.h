// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail one-row staging core (QWEN38_QSA_MT_ROW=1): building the new row from the pack tiles.

#pragma once

#include <cstdint>

#include "main_tail_cbs.h"

namespace main_tail {
// What the default path stores for a bf16 value x of the new row (found by running all 65536 bit patterns through it):
// the RISC patch and the SFPU multiply by 1.0 give  +0.0 for an exponent field of 0 (zero, denormal, -0.0, either
// sign);  Inf (the sign kept) for an exponent field of 255 (a NaN becomes Inf);  x for an exponent field of 9..254;
// and, for the exponent fields 1..8 (|x| < 2^-118), the bf16 RNE of x + 2^-127 (a denormal result becomes +0.0).
inline uint16_t canonical(uint16_t x) {
    const uint32_t e = (x >> 7) & 0xFF;
    if (e >= 9 && e != 0xFF) {
        return x;
    }
    if (e == 0) {
        return 0;
    }
    if (e == 0xFF) {
        return x & 0xFF80;
    }
    const uint32_t s = x >> 15;
    // |x| in units of 2^-149 (exact), then x + 2^-127 (units: 2^22)
    uint32_t z = ((128u + (x & 0x7F)) << 16) << (e - 1);
    if (s == 0) {
        z += 1u << 22;
    } else {
        z -= 1u << 22;
        if (z < (1u << 23)) {
            return 0;
        }
    }
    uint32_t n = 0;
    for (uint32_t t = z; t != 0; t >>= 1) {
        ++n;
    }
    const uint32_t sh = n - 8;
    uint32_t q = z >> sh;
    const uint32_t rem = z & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (q & 1))) {
        ++q;
    }
    return static_cast<uint16_t>((s << 15) + ((n - 23) << 7) + (q - 128));
}

// two bf16 per word: a word whose halves both have an exponent field in 9..254 is stored unchanged
inline uint32_t canonical_word(uint32_t w) {
    const uint32_t lo = w & 0xFFFF, hi = w >> 16;
    if (((lo >> 7) & 0xFF) - 9 < 246u && ((hi >> 7) & 0xFF) - 9 < 246u) {
        return w;
    }
    return static_cast<uint32_t>(canonical(lo)) | (static_cast<uint32_t>(canonical(hi)) << 16);
}

// tiles first .. first + count - 1 of the pack tiles at pack_l1 (row `lane` of each tile) -> count * 64 row-major bytes
// at dst
inline void build_tiles(uint32_t pack_l1, uint32_t first, uint32_t count, uint32_t lane, uint32_t dst) {
    for (uint32_t t = 0; t < count; ++t) {
        for (uint32_t half = 0; half < 2; ++half) {
            const volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                pack_l1 + (first + t) * TILE_BYTES + chunk_offset(lane, half));
            volatile tt_l1_ptr uint32_t* d =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + t * 2 * ROW_BYTES + half * ROW_BYTES);
            for (uint32_t k = 0; k < ROW_BYTES / 4; ++k) {
                d[k] = canonical_word(s[k]);
            }
        }
    }
}
}  // namespace main_tail
