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

// Byte offset of row `row` (its first 16 columns) inside a BF16 32x32 tile; the next 16 columns are 512 B further.
inline uint32_t tile_row_bytes(uint32_t row) { return ((row >> 4) * 2) * 512 + (row & 15) * 32; }
