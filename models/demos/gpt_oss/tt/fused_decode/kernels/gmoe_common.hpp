// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped (sparse) MoE of the batch-32 DFlash verify (dflash/grouped_moe.py): the shared slot plan.
//
// Every row is routed to K_SEL (on a BF16 tie K_SEL + 1) of the E experts (gmoe_route.cpp). Expert e's routed rows (in
// row order) fill tiles[e] = ceil(count / 32) consecutive 32-row slots starting at slot base[e] (slots in expert
// order, at most ROWS * (K_SEL + 1) / 32 + E = 72). Every op of the grouped MoE
// gets this plan from the route op's table (gmoe_route.cpp; recomputing it from the ids takes ~13 us on a RISC).
//
// Work units: unit u = e * NU + j (expert e, column tile j of NU); a unit of an expert with slots costs wc + slots * cc
// (weight read + per-slot compute), others 0. Core c of NC takes the units whose start cost lies in
// [c * total / NC, (c + 1) * total / NC).

#pragma once

#include <stdint.h>

namespace gmoe {
constexpr uint32_t E = 32;
#ifndef GMOE_ROWS
#define GMOE_ROWS 256
#endif
constexpr uint32_t ROWS = GMOE_ROWS;  // rows of the routed pass (gmoe_route.cpp; 32 .. 256)
constexpr uint32_t K_SEL = 4;
constexpr uint32_t K_IDS = K_SEL + 1;  // the router's ids per row (top-k + 1: a BF16 tie routes k + 1 experts)

// expert e's first column tile in the dense [*, E * 720] layout (720 columns = 22.5 tiles per expert)
inline uint32_t g0(uint32_t e) { return (45 * e) / 2; }

struct Plan {
    uint32_t tiles[E];
    uint32_t base[E];
};

// the plan table (gmoe_route.cpp: [tiles[E], base[E]] words) at l1 -> p
inline void read_table(uint32_t l1, Plan& p) {
    const uint32_t* t = reinterpret_cast<const uint32_t*>(l1);
    for (uint32_t e = 0; e < E; ++e) {
        p.tiles[e] = t[e];
        p.base[e] = t[E + e];
    }
}

inline uint32_t unit_cost(const Plan& p, uint32_t e, uint32_t wc, uint32_t cc) {
    return p.tiles[e] ? wc + p.tiles[e] * cc : 0;
}

// the first unit whose start cost is >= T (E * NU when none)
inline uint32_t find_unit(const Plan& p, uint32_t NU, uint32_t wc, uint32_t cc, uint32_t T) {
    uint32_t P = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t uc = unit_cost(p, e, wc, cc);
        if (T <= P) {
            return e * NU;
        }
        if (uc) {
            const uint32_t j = (T - P + uc - 1) / uc;
            if (j < NU) {
                return e * NU + j;
            }
        }
        P += NU * uc;
    }
    return E * NU;
}

// core c's unit range [u0, u1) of NC cores
inline void unit_range(
    const Plan& p, uint32_t NU, uint32_t wc, uint32_t cc, uint32_t c, uint32_t NC, uint32_t& u0, uint32_t& u1) {
    uint32_t total = 0;
    for (uint32_t e = 0; e < E; ++e) {
        total += NU * unit_cost(p, e, wc, cc);
    }
    u0 = find_unit(p, NU, wc, cc, static_cast<uint32_t>((static_cast<uint64_t>(total) * c) / NC));
    u1 = c + 1 == NC ? E * NU
                     : find_unit(p, NU, wc, cc, static_cast<uint32_t>((static_cast<uint64_t>(total) * (c + 1)) / NC));
}

// the meta page a reader passes to its compute / writer: [u0, u1, tiles[E], base[E]]
constexpr uint32_t META_WORDS = 2 + 2 * E;
inline void put_meta(uint32_t l1, uint32_t u0, uint32_t u1, const Plan& p) {
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    m[0] = u0;
    m[1] = u1;
    for (uint32_t e = 0; e < E; ++e) {
        m[2 + e] = p.tiles[e];
        m[2 + E + e] = p.base[e];
    }
}
}  // namespace gmoe
