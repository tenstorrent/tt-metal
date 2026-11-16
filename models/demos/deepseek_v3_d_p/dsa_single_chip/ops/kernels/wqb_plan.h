// Work plan of the wq_b matmul, derived on every RISC from the core's logical coordinates and the host's plan table.
// K is split in two halves of KH tiles: rows [0, HROWS) of the grid ("top") take K half 0, the other rows ("bottom")
// K half 1. Within a half, core rank r = x * HROWS + y % HROWS; the NT x KH weight tiles of the half are numbered
// column-major (u = n * KH + kk) and rank r covers units [PLAN[r], PLAN[r + 1]) (common runtime args at ARG_PLAN,
// each range at least KH long), the same in both halves, so top rank r and bottom rank r cover the same columns.
// A rank's range covers up to four column segments: TAIL (kk in [kb, KH), kb > 0; the column's head is on rank r - 1),
// FULL (all KH), HEAD (kk in [0, ke), ke < KH; the column's tail is on rank r + 1).
// Reduction: the top core holding a column's TAIL/FULL segment finalizes it (adds the other partials and writes the
// output). Partials go to it as: bottom same segment -> slot 0; top HEAD of rank r - 1 -> slot 1; bottom HEAD of rank
// r - 1 -> slot 2. A finalizer's non-HEAD segments are indexed fi in natural order (a TAIL is always fi 0).
// Processing order: HEAD first (its partial is needed by another core), then FULL, then TAIL.
#pragma once
#include <cstdint>

extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;

namespace wqb {

enum Kind : uint32_t { HEAD = 0, FULL = 1, TAIL = 2 };

struct Seg {
    uint32_t col, kb, ke, kind, fi;
};

struct Plan {
    uint32_t n, count, half, rank;
    Seg s[4];
};

template <uint32_t KH, uint32_t NT, uint32_t NCOLS, uint32_t HROWS, uint32_t ARG_PLAN>
inline Plan make_plan() {
    const uint32_t y = my_logical_y_;
    const uint32_t half = y >= HROWS ? 1 : 0;
    const uint32_t rank = static_cast<uint32_t>(my_logical_x_) * HROWS + (y - half * HROWS);
    const uint32_t first = get_common_arg_val<uint32_t>(ARG_PLAN + rank);
    const uint32_t count = get_common_arg_val<uint32_t>(ARG_PLAN + rank + 1) - first;
    Seg nat[4];
    uint32_t nn = 0, nfi = 0;
    for (uint32_t u = first; u < first + count;) {
        const uint32_t col = u / KH, kb = u - col * KH;
        const uint32_t left = first + count - u;
        const uint32_t ke = left < KH - kb ? kb + left : KH;
        const uint32_t kind = kb > 0 ? TAIL : (ke < KH ? HEAD : FULL);
        nat[nn++] = {col, kb, ke, kind, kind == HEAD ? 0u : nfi++};
        u += ke - kb;
    }
    Plan p{0, count, half, rank, {}};
    for (uint32_t want = HEAD; want <= TAIL; ++want) {
        for (uint32_t i = 0; i < nn; ++i) {
            if (nat[i].kind == want) {
                p.s[p.n++] = nat[i];
            }
        }
    }
    return p;
}

// words per RISC in the weight-chunk claim table (wqb_wstream.h)
constexpr uint32_t TABLE_STRIDE = 32;

// partials a finalizer receives for its segment
inline uint32_t expected_partials(const Seg& s) { return s.kind == TAIL ? 3 : 1; }

}  // namespace wqb
