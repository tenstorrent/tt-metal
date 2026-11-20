// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The cross-chip top-k merge of one user on one chip (fused/merges/topk_merge.py), for a data-movement kernel on
// the core that holds the user's chip-local survivor: 2048 fused words [bf16 score | u16 stamp] in rank order as
// the index-key scan leaves them (fused/indexer_scan/CONTRACT.md).
//
//     hy4_topk::Result r = hy4_topk::merge<BITS0, BITS1>(words, payload, use0, use1, chip, topk);
//
// Global order of the entries of all chips: BF16 score (descending), then chip, then the chip's own rank. A chip's
// winners are therefore its first `keep` entries, and the union over the chips is exactly the global top-k.
//
// Two rounds, each an all-gather of one count vector per chip (collective.hpp: hand_over / wait_landed / finish;
// slot c of the landing is chip c's vector). With key = the 16-bit order key of a score:
//   round 0   thresholds t_j = j << (16 - BITS0); vector[j] = this chip's entries with key >= t_j, j = 0 .. 2^BITS0.
//             Every chip sums the vectors and finds the last bucket j* whose count reaches topk.
//   round 1   thresholds t_i = (j* << BITS1) + i inside that bucket; vector[i] likewise, i = 0 .. 2^BITS1.
//             The last i* whose global count reaches topk is the cut key T; from the same landing every chip reads
//             every chip's count above T and at T, gives the entries equal to T out in chip order and keeps its own.
// Entries the query does not see (NEG rows, -inf padding: keys below KEY_REAL_MIN) are never counted; a user with
// fewer than topk visible entries in total keeps all of them (ALL).
//
// Nothing but counts crosses chips: (2^BITS + 1) uint16 per chip, user and round.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "collective.hpp"

namespace hy4_topk {

constexpr uint32_t LIST = 2048;            // entries of a survivor
constexpr uint32_t KEY_REAL_MIN = 0x0F80;  // order key of -(2^98); the scan's cut counts keys above 0x0F7F as seen

constexpr uint32_t FLAG_ALL = 1;      // fewer than topk visible entries in total: every chip keeps all of its own
constexpr uint32_t FLAG_TIMEOUT = 2;  // a landing wait ran out: the result is not valid
constexpr uint32_t FLAG_RECOUNT = 4;  // the cut's bucket was not the one counted ahead: round 1 was counted again

// BF16 bits -> an unsigned key in the order of the values.
FORCE_INLINE uint32_t order_key(uint32_t bits) { return (bits & 0x8000) ? 0xFFFF - bits : bits | 0x8000; }

// Word offset, in the survivor's two raw tiles, of the entry of rank `rank` (0 = best).
FORCE_INLINE uint32_t rank_word(uint32_t rank) { return ((rank & 127) << 4) | (rank >> 7); }

FORCE_INLINE uint32_t key_at(const volatile tt_l1_ptr uint32_t* words, uint32_t rank) {
    return order_key(words[rank_word(rank)] >> 16);
}

// The number of entries with key >= t, given that every rank below `lo` has key >= t and none from `hi` on has:
// a galloping search from lo, since the caller's thresholds come in descending order and the runs between them
// are short when the list is dense.
inline uint32_t count_ge(const volatile tt_l1_ptr uint32_t* words, uint32_t t, uint32_t lo, uint32_t hi) {
    uint32_t step = 1;
    while (lo < hi) {
        uint32_t probe = lo + step - 1;
        if (probe >= hi) {
            probe = hi - 1;
        }
        if (key_at(words, probe) >= t) {
            lo = probe + 1;
            step <<= 1;
        } else {
            hi = probe;
            break;
        }
    }
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (key_at(words, mid) >= t) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// out[i] = entries with key >= base + (i << shift), i = 0 .. n, as uint16. Ranks below `lo` are the entries at or
// above the top threshold (i = n); ranks from `hi` on are below `base` or not visible.
inline void count_vector(
    const volatile tt_l1_ptr uint32_t* words,
    volatile tt_l1_ptr uint16_t* out,
    uint32_t base,
    uint32_t shift,
    uint32_t n,
    uint32_t lo,
    uint32_t hi) {
    uint32_t r = lo;
    out[n] = static_cast<uint16_t>(lo);
    uint32_t i = n - 1;
    while (i >= 1) {
        if (r >= hi) {
            out[i--] = static_cast<uint16_t>(r);
            continue;
        }
        const uint32_t bucket = (key_at(words, r) - base) >> shift;  // of the best entry not yet counted, <= i
        while (i > bucket) {
            out[i--] = static_cast<uint16_t>(r);
        }
        if (i == 0) {
            break;
        }
        r = count_ge(words, base + (i << shift), r, hi);
        out[i--] = static_cast<uint16_t>(r);
    }
    out[0] = static_cast<uint16_t>(hi);
}

// The landing of one round: slot c holds chip c's vector.
struct Landing {
    uint32_t base;
    uint32_t stride;
    uint32_t members;
    FORCE_INLINE uint32_t at(uint32_t chip, uint32_t i) const {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + chip * stride)[i];
    }
    inline uint32_t total(uint32_t i) const {
        uint32_t sum = 0;
        for (uint32_t c = 0; c < members; ++c) {
            sum += at(c, i);
        }
        return sum;
    }
    // The last i in [0, n) whose total reaches `topk`; total(0) must reach it and total(n) must not.
    inline uint32_t last_reaching(uint32_t n, uint32_t topk) const {
        uint32_t lo = 0, hi = n;
        while (hi - lo > 1) {
            const uint32_t mid = (lo + hi) >> 1;
            if (total(mid) >= topk) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
};

struct Result {
    uint32_t keep;  // this chip's entries in the global top-k: its ranks 0 .. keep - 1
    uint32_t key;   // the cut key T: the key of the last entry kept globally (0 with FLAG_ALL)
    uint32_t flags;
    uint32_t above;      // entries above T, all chips
    uint32_t before;     // entries equal to T on the chips before this one
    uint32_t stamps[6];  // timing: counts 0, handed 0, landed 0, counts 1, handed 1, landed 1
};

FORCE_INLINE Landing landing_of(const hy4_collective::Use& use, uint32_t parity) {
    invalidate_l1_cache();
    return Landing{use.landing + parity * use.parity_stride, use.slot_stride, use.members};
}

// `words`: the survivor in L1. `payload`: two vectors of scratch, each 2 * (2^max(BITS) + 1) bytes rounded up to
// 16 (use.bytes).
// `chip`: this chip's slot. The two uses are of one site and in this order in the sender's table.
template <uint32_t BITS0, uint32_t BITS1>
inline Result merge(
    const volatile tt_l1_ptr uint32_t* words,
    uint32_t payload,
    const hy4_collective::Use& use0,
    const hy4_collective::Use& use1,
    uint32_t chip,
    uint32_t topk) {
    static_assert(BITS0 + BITS1 == 16 && BITS0 >= 1 && BITS1 >= 1, "the two rounds resolve the 16 key bits");
    constexpr uint32_t N0 = 1u << BITS0;
    constexpr uint32_t N1 = 1u << BITS1;
    volatile tt_l1_ptr uint16_t* vector = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(payload);
    Result result{};
    uint32_t parity = 0;

    // ---- round 0 ----
    const uint32_t seen = count_ge(words, KEY_REAL_MIN, 0, LIST);
    count_vector(words, vector, 0, BITS1, N0, 0, seen);
    result.stamps[0] = hy4_collective::now();
    hy4_collective::hand_over(use0, payload);
    // While round 0 is on the fabric: the round 1 vector of the bucket this chip expects the cut in (that of its
    // own entry of rank topk / members: the cut's, when the chips' lists are alike), ready if the landing agrees.
    volatile tt_l1_ptr uint16_t* ahead = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(payload + use0.bytes);
    uint32_t guess = N0;
    if (seen != 0) {
        const uint32_t share = topk / use0.members;
        guess = key_at(words, share < seen ? share : seen - 1) >> BITS1;
        count_vector(words, ahead, guess << BITS1, 0, N1, vector[guess + 1], vector[guess]);
    }
    result.stamps[1] = hy4_collective::now();
    if (!hy4_collective::wait_landed(use0, parity)) {
        result.flags |= FLAG_TIMEOUT;
    }
    result.stamps[2] = hy4_collective::now();
    uint32_t bucket = 0;
    {
        const Landing landed = landing_of(use0, parity);
        if (landed.total(0) < topk) {
            result.flags |= FLAG_ALL;
        } else {
            bucket = landed.last_reaching(N0, topk);
        }
    }
    const uint32_t own_above = vector[bucket + 1];  // this chip's entries above the bucket
    const uint32_t own_from = vector[bucket];       // and from its lower edge up
    hy4_collective::finish(use0);

    // ---- round 1 ---- (a user that keeps everything still takes part: the sender waits for every hand-over)
    const uint32_t base = bucket << BITS1;
    uint32_t round1 = payload + use0.bytes;
    if (bucket != guess) {
        count_vector(words, vector, base, 0, N1, own_above, own_from);
        round1 = payload;
        result.flags |= FLAG_RECOUNT;
    }
    result.stamps[3] = hy4_collective::now();
    hy4_collective::hand_over(use1, round1);
    result.stamps[4] = hy4_collective::now();
    if (!hy4_collective::wait_landed(use1, parity)) {
        result.flags |= FLAG_TIMEOUT;
    }
    result.stamps[5] = hy4_collective::now();
    if (result.flags & FLAG_ALL) {
        result.keep = seen;
    } else {
        const Landing landed = landing_of(use1, parity);
        const uint32_t i = landed.last_reaching(N1, topk);
        uint32_t above = 0, before = 0;
        for (uint32_t c = 0; c < landed.members; ++c) {
            const uint32_t a = landed.at(c, i + 1);
            above += a;
            if (c < chip) {
                before += landed.at(c, i) - a;
            }
        }
        const uint32_t own_a = landed.at(chip, i + 1);
        const uint32_t own_tied = landed.at(chip, i) - own_a;
        const uint32_t left = topk - above;  // entries equal to T that the global top-k takes, > 0
        const uint32_t take = before >= left ? 0 : (left - before < own_tied ? left - before : own_tied);
        result.keep = own_a + take;
        result.key = base + i;
        result.above = above;
        result.before = before;
    }
    hy4_collective::finish(use1);
    return result;
}

}  // namespace hy4_topk
