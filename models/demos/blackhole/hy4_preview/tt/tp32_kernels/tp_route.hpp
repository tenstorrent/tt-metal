// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// X40 TP32: route page = uint32 [T][16] per chip (replicated): words 0..7 expert ids, 8..15 routing scores (bf16 bits
// in the low 16 bits). Every core builds the same ascending list of the distinct experts of the T tokens.
#pragma once
#include <stdint.h>

FORCE_INLINE uint32_t tp_build_plan(volatile tt_l1_ptr uint32_t* route, uint32_t T, uint16_t* active) {
    uint32_t bm[8];
    for (uint32_t w = 0; w < 8; ++w) {
        bm[w] = 0;
    }
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t e = route[t * 16 + k] & 255u;
            bm[e >> 5] |= (1u << (e & 31u));
        }
    }
    uint32_t na = 0;
    for (uint32_t w = 0; w < 8; ++w) {
        uint32_t v = bm[w];
        for (uint32_t b = 0; v != 0 && b < 32; ++b, v >>= 1) {
            if (v & 1u) {
                active[na++] = (uint16_t)(w * 32 + b);
            }
        }
    }
    return na;
}

// tp24 row-independent grouping (verify-round track): the experts of group `grp` = the distinct experts e of the T
// tokens with e % 3 == grp, ascending. The old plan took active[3 q + grp] of the union list, so which core/partial sum
// an expert landed in (hence the bf16/fp32 association of each token's sum over its experts) depended on the experts of
// ALL other rows. A per-expert group is a function of the expert id only.
FORCE_INLINE uint32_t
tp_build_plan_grp(volatile tt_l1_ptr uint32_t* route, uint32_t T, uint16_t* active, uint32_t grp) {
    uint32_t bm[8];
    for (uint32_t w = 0; w < 8; ++w) {
        bm[w] = 0;
    }
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t e = route[t * 16 + k] & 255u;
            bm[e >> 5] |= (1u << (e & 31u));
        }
    }
    uint32_t na = 0;
    for (uint32_t w = 0; w < 8; ++w) {
        uint32_t v = bm[w];
        for (uint32_t b = 0; v != 0 && b < 32; ++b, v >>= 1) {
            if (v & 1u) {
                const uint32_t e = w * 32 + b;
                if (e % 3u == grp) {
                    active[na++] = (uint16_t)e;
                }
            }
        }
    }
    return na;
}
