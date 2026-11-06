// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared device-side bookkeeping of the per-expert dispatch ops (custom_ops/expert_stream/op.py, phase 2):
// the per-expert row counts come in as a BF16 tile row (ttnn.sum(routing != 0) over the tokens), from which every
// kernel derives the same pass enumeration (expert ascending, 32 rows per pass) and the same expert -> core-group
// assignment, so the active set stays data-dependent on device (trace-safe, no host counts).

#pragma once

#include <stdint.h>

#include "routing_scan.hpp"

// BF16 bit pattern of a small non-negative integer -> the integer.
inline uint32_t bf16_to_uint(uint16_t b) {
    if (b == 0) {
        return 0;
    }
    const uint32_t exp = (b >> 7) & 0xFF;
    const uint32_t mant = (b & 0x7F) | 0x80;
    if (exp < 127) {
        return 0;
    }
    const int32_t sh = static_cast<int32_t>(exp) - 127 - 7;
    return sh >= 0 ? (mant << sh) : (mant >> (-sh));
}

// counts: E_TILES BF16 tiles back to back in L1 whose row 0 holds the per-expert token counts -> n[e].
template <uint32_t E>
inline void read_counts(const volatile tt_l1_ptr uint16_t* counts, uint16_t* n) {
    for (uint32_t e = 0; e < E; ++e) {
        n[e] = static_cast<uint16_t>(bf16_to_uint(counts[(e >> 5) * 1024 + tile_elem(0, e & 31)]));
    }
}

// passes[e] = ceil(n[e] / 32), base[e] = the global index of expert e's first pass; returns the total pass count.
template <uint32_t E>
inline uint32_t pass_bases(const uint16_t* n, uint16_t* passes, uint16_t* base) {
    uint32_t k = 0;
    for (uint32_t e = 0; e < E; ++e) {
        passes[e] = static_cast<uint16_t>((n[e] + 31) / 32);
        base[e] = static_cast<uint16_t>(k);
        k += passes[e];
    }
    return k;
}

// The pass budget of the dispatch ops (op.py ``max_passes``): sum_e ceil(n_e / 32) <= MAX_PICKS * T / 32 + E holds
// whenever every routing row has at most MAX_PICKS non-zeros (the top-8 router), and the act / table / resident-h
// buffers are sized by it. A routing matrix that breaks the top-k contract (a NaN / Inf row spread over every
// selected column by the fp32 selector matmul, stale rows past seq_len, a corrupted row gather) breaks the bound,
// and the 2026-10-27 (2,4) block hang was exactly that: the down writer clamped its own loop to P_MAX while its
// compute consumed the unclamped metadata page, so the op deadlocked. Every kernel that derives the enumeration
// applies clamp_passes right after pass_bases (the computes take their per-expert counts from the writers'
// metadata page / mailboxes, i.e. the clamped values), so the readers' expert set, the writers' activation / table
// loops and the computes' pass totals agree whatever the counts say: an over-budget routing yields a truncated,
// bounded result and the gate|up writer flags it in the pass table (TABLE_FLAG_WORD) instead of hanging.
constexpr uint32_t DISPATCH_MAX_PICKS = 8;  // op.py MAX_PICKS (tests/test_dispatch_pass_budget_host.py pins the pair)
constexpr uint32_t dispatch_max_passes(uint32_t T, uint32_t E) { return DISPATCH_MAX_PICKS * T / 32 + E; }
constexpr uint32_t TABLE_FLAG_WORD = 63;  // pass-table pad word (uint16) = 1 when the enumeration was clamped

// Truncates the enumeration to ``p_max`` passes in expert order: the expert crossing the budget keeps its first
// passes (n[e] cut to the rows they hold), later experts get 0 passes and n[e] = 0 (inactive for every kernel);
// base[] is rewritten for the truncated counts. Returns the clamped total; ``clamped`` reports whether the budget
// was hit. A function of the counts alone: identical on every core and in every kernel of both ops.
template <uint32_t E>
inline uint32_t clamp_passes(uint16_t* n, uint16_t* passes, uint16_t* base, uint32_t p_max, bool& clamped) {
    uint32_t k = 0;
    clamped = false;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t room = p_max - k;
        if (passes[e] > room) {
            passes[e] = static_cast<uint16_t>(room);
            n[e] = static_cast<uint16_t>(room * 32);
            clamped = true;
        }
        base[e] = static_cast<uint16_t>(k);
        k += passes[e];
    }
    return k;
}

// Longest-processing-time assignment of the active experts to `groups` core groups by pass count (experts sorted by
// passes descending, id ascending; each goes to the least loaded group, lowest id on ties). Fills `mine` with the
// experts of `group` in assignment order and returns their count. Identical on every core.
template <uint32_t E, uint32_t MAX_GROUPS>
inline uint32_t assign_group(const uint16_t* passes, uint32_t groups, uint32_t group, uint16_t* mine) {
    uint16_t order[E];
    uint32_t na = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (passes[e] == 0) {
            continue;
        }
        uint32_t i = na;
        while (i > 0 && passes[order[i - 1]] < passes[e]) {
            order[i] = order[i - 1];
            --i;
        }
        order[i] = static_cast<uint16_t>(e);
        ++na;
    }
    uint32_t load[MAX_GROUPS];
    for (uint32_t g = 0; g < MAX_GROUPS; ++g) {
        load[g] = 0;
    }
    uint32_t n_mine = 0;
    for (uint32_t i = 0; i < na; ++i) {
        uint32_t g = 0;
        for (uint32_t j = 1; j < groups; ++j) {
            if (load[j] < load[g]) {
                g = j;
            }
        }
        load[g] += passes[order[i]];
        if (g == group) {
            mine[n_mine++] = order[i];
        }
    }
    return n_mine;
}

// EP mode: the routing comes as the top-k index list idx [T, K] (BF16 integers, global expert ids, TILE layout:
// T/32 tiles of 32x32 with columns 0..K-1 valid) and the weights w [T, K] (BF16, same layout); this chip owns the
// experts [base, base + E). Per-expert token counts:
template <uint32_t E, uint32_t T, uint32_t K>
inline void count_from_indices(const volatile tt_l1_ptr uint16_t* idx, uint32_t base, uint16_t* n) {
    for (uint32_t e = 0; e < E; ++e) {
        n[e] = 0;
    }
    for (uint32_t t = 0; t < T; ++t) {
        const volatile tt_l1_ptr uint16_t* tile = idx + (t >> 5) * 1024;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = bf16_to_uint(tile[tile_elem(t & 31, k)]);
            if (e >= base && e < base + E) {
                ++n[e - base];
            }
        }
    }
}

// The tokens routed to local expert e (ascending) and their weights (raw BF16 bits); returns the count.
template <uint32_t T, uint32_t K>
inline uint32_t rows_from_indices(
    const volatile tt_l1_ptr uint16_t* idx,
    const volatile tt_l1_ptr uint16_t* w,
    uint32_t global_e,
    uint16_t* rows,
    uint16_t* wbits) {
    uint32_t nr = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t off = (t >> 5) * 1024;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t el = off + tile_elem(t & 31, k);
            if (bf16_to_uint(idx[el]) == global_e) {
                rows[nr] = static_cast<uint16_t>(t);
                wbits[nr] = w[el];
                ++nr;
                break;  // one slot per (token, expert)
            }
        }
    }
    return nr;
}

// EP mode with row-major routing (EP = 2, the mesh path): idx = the all_to_all_dispatch metadata [t_valid, K] UINT16
// (global expert ids, exact: BF16 cannot hold 257..287) packed in L1 at K elements per token; w = the routing weights
// [t_valid, K] BF16 packed at K elements per token (the tail of every dispatched token row). Rows >= t_valid (the
// tile padding of a short batch) are never scanned.
template <uint32_t E, uint32_t K>
inline void count_from_rows(const volatile tt_l1_ptr uint16_t* idx, uint32_t t_valid, uint32_t base, uint16_t* n) {
    for (uint32_t e = 0; e < E; ++e) {
        n[e] = 0;
    }
    for (uint32_t t = 0; t < t_valid; ++t) {
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = idx[t * K + k];
            if (e >= base && e < base + E) {
                ++n[e - base];
            }
        }
    }
}

template <uint32_t K>
inline uint32_t rows_from_rows(
    const volatile tt_l1_ptr uint16_t* idx,
    const volatile tt_l1_ptr uint16_t* w,
    uint32_t t_valid,
    uint32_t global_e,
    uint16_t* rows,
    uint16_t* wbits) {
    uint32_t nr = 0;
    for (uint32_t t = 0; t < t_valid; ++t) {
        for (uint32_t k = 0; k < K; ++k) {
            if (idx[t * K + k] == global_e) {
                rows[nr] = static_cast<uint16_t>(t);
                wbits[nr] = w[t * K + k];
                ++nr;
                break;  // one slot per (token, expert)
            }
        }
    }
    return nr;
}

// Work units of the gate|up dispatch: (expert, column pair) for every active expert and each of the bank's PPB
// pairs, LPT-assigned to the core groups by pass count (units ordered by passes descending, expert ascending, pair
// ascending; each to the least loaded group, lowest id on ties: with PPB = 1 exactly assign_group). Calls
// f(e, q) for this group's units in that order (no unit list in memory: the bound would be E x PPB).
template <uint32_t E, uint32_t PPB, uint32_t MAX_GROUPS, typename F>
inline void for_each_unit(const uint16_t* passes, uint32_t groups, uint32_t group, F&& f) {
    uint16_t order[E];
    uint32_t na = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (passes[e] == 0) {
            continue;
        }
        uint32_t i = na;
        while (i > 0 && passes[order[i - 1]] < passes[e]) {
            order[i] = order[i - 1];
            --i;
        }
        order[i] = static_cast<uint16_t>(e);
        ++na;
    }
    uint32_t load[MAX_GROUPS];
    for (uint32_t g = 0; g < MAX_GROUPS; ++g) {
        load[g] = 0;
    }
    for (uint32_t i = 0; i < na; ++i) {
        const uint32_t e = order[i];
        for (uint32_t q = 0; q < PPB; ++q) {
            uint32_t g = 0;
            for (uint32_t j = 1; j < groups; ++j) {
                if (load[j] < load[g]) {
                    g = j;
                }
            }
            load[g] += passes[e];
            if (g == group) {
                f(e, q);
            }
        }
    }
}

// Chunked work units of the gate|up dispatch (decode ledger item (d), 2026-10-26): (expert, chunk) with U = PPB /
// CHUNKS pairs per unit, LPT-assigned like for_each_unit (units ordered by passes descending, expert ascending, chunk
// ascending). U = 1 is exactly for_each_unit (one pair per unit, 8 cores per bank per expert at PPB = 8); U = PPB is
// one unit per expert (the per-expert assignment, 1 core per bank per expert). Calls f(e, c) for this group's units.
template <uint32_t E, uint32_t PPB, uint32_t U, uint32_t MAX_GROUPS, typename F>
inline void for_each_unit_chunk(const uint16_t* passes, uint32_t groups, uint32_t group, F&& f) {
    static_assert(U >= 1 && PPB % U == 0, "the pairs per unit must divide the pairs per bank");
    constexpr uint32_t CHUNKS = PPB / U;
    uint16_t order[E];
    uint32_t na = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (passes[e] == 0) {
            continue;
        }
        uint32_t i = na;
        while (i > 0 && passes[order[i - 1]] < passes[e]) {
            order[i] = order[i - 1];
            --i;
        }
        order[i] = static_cast<uint16_t>(e);
        ++na;
    }
    uint32_t load[MAX_GROUPS];
    for (uint32_t g = 0; g < MAX_GROUPS; ++g) {
        load[g] = 0;
    }
    for (uint32_t i = 0; i < na; ++i) {
        const uint32_t e = order[i];
        for (uint32_t c = 0; c < CHUNKS; ++c) {
            uint32_t g = 0;
            for (uint32_t j = 1; j < groups; ++j) {
                if (load[j] < load[g]) {
                    g = j;
                }
            }
            load[g] += passes[e];
            if (g == group) {
                f(e, c);
            }
        }
    }
}

// Rotated row-gather order (GATHER_ROTATE, hypothesis H1 of custom_ops/expert_stream/plan.py): when many cores
// gather the same pass rows in the same order their reads convoy on one source L1 bank at a time; starting every
// core at a different row spreads the reads over the rows' banks. Returns the j-th row index to issue.
inline uint32_t rotated_row(uint32_t j, uint32_t n, uint32_t rot) {
    const uint32_t r = j + (n ? rot % n : 0);
    return r >= n ? r - n : r;
}

// Byte offset of row `row` (its first 16 columns) inside a BF16 32x32 tile; the next 16 columns are 512 B further.
inline uint32_t tile_row_bytes(uint32_t row) { return ((row >> 4) * 2) * 512 + (row & 15) * 32; }
