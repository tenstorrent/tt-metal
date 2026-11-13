// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert dynamic-M plan: one expert's schedule for one core (cluster k, core j), computed on device from M alone
// (every core of every cluster computes the same split, so neighbours agree without talking).
//
// Split: R = ceil(M / 32) row tiles over NA = min(NCL, max(1, R / 2)) active clusters (>= 2 rows of cost each: a row
// is never split across three clusters). The flattened (row, K-tile) work is cut at cost targets c T / NA, T = R (KT G
// + D) (G tile products a K-tile, D a row's down, paid by the row's owner). A boundary at cost target t lands in row
// r = t / (KT G + D) with rest q: the split row goes to the left cluster at K offset (q - D) / G, or to the right one
// at q / G - whichever keeps the offset inside the row (left first).
// Program (fixed 4-row h ring order): send pieces, then full rows, then pieces that wait for a partial; a DN op
// after each produced row. x blocks: the gate/up pieces' K ranges in program order, BK K-tiles at a time; block l
// belongs to core l % GC.
#pragma once

#include <cstdint>

namespace ws_plan {

constexpr uint32_t PRE_L = 1, PRE_R = 2, SEND_L = 4, SEND_R = 8, DN = 16;
constexpr uint32_t MAXOP = 96, MAXOWN = 40, MAXMY = 40;
// plan slot layout (uint32 words)
constexpr uint32_t H_READY = 0, H_R = 1, H_M = 2, H_NOP = 3, H_NOWN = 4, H_NBE = 5, H_NMY = 6;
constexpr uint32_t OFF_OP = 8;                        // MAXOP x (k0, k1, flags)
constexpr uint32_t OFF_OWN = OFF_OP + 3 * MAXOP;      // MAXOWN owned rows (production order)
constexpr uint32_t OFF_MY = OFF_OWN + MAXOWN;         // MAXMY x (local block index, r, k, n)
constexpr uint32_t SLOT_WORDS = 512;
static_assert(OFF_MY + 4 * MAXMY <= SLOT_WORDS);
constexpr uint32_t NSLOT = 3;
// control words after the slots: done counters (expert + 1) of the plan's readers
constexpr uint32_t CTRL = NSLOT * SLOT_WORDS;
constexpr uint32_t DONE_BR = 0, DONE_NC = 1, DONE_T0 = 2, DONE_T1 = 3, DONE_T2 = 4, NDONE = 5;
constexpr uint32_t PLAN_WORDS = CTRL + 16;

struct Geo {
    uint32_t KT, G, D, NCL, GC, BK;
};

// boundary c (1 <= c < na) of the split: position (flattened K index) and owner of its split row (0 left, 1 right)
inline void boundary(const Geo& g, uint32_t R, uint32_t na, uint32_t c, uint32_t& pos, uint32_t& right) {
    const uint32_t rowcost = g.KT * g.G + g.D;
    const uint32_t T = R * rowcost;
    const uint32_t t = (c * T) / na;
    const uint32_t r = t / rowcost, q = t - r * rowcost;
    right = 0;
    if (q >= g.D + g.G && (q - g.D) / g.G <= g.KT - 1) {  // left owns, splits inside the row
        pos = r * g.KT + (q - g.D) / g.G;
    } else if (q / g.G >= 1 && q / g.G <= g.KT - 1) {  // right owns
        pos = r * g.KT + q / g.G;
        right = 1;
    } else if (q < g.G) {  // at the row's start
        pos = r * g.KT;
    } else {  // the whole row to the left
        pos = (r + 1) * g.KT;
    }
}

inline uint32_t active_clusters(const Geo& g, uint32_t R) {
    uint32_t na = R / 2;
    if (na < 1) {
        na = 1;
    }
    return na < g.NCL ? na : g.NCL;
}

// fill one plan slot (words p) for cluster k, core j
inline void make(const Geo& g, uint32_t M, uint32_t k, uint32_t j, volatile uint32_t* p) {
    const uint32_t R = (M + 31) / 32;
    p[H_R] = R;
    p[H_M] = M;
    uint32_t nop = 0, nown = 0, nbe = 0, nmy = 0;
    const uint32_t na = R ? active_clusters(g, R) : 0;
    if (k < na) {
        uint32_t s = 0, e = R * g.KT, s_right = 0, e_right = 0;  // (owner of the split row at s / at e)
        if (k > 0) {
            boundary(g, R, na, k, s, s_right);
        }
        if (k + 1 < na) {
            boundary(g, R, na, k + 1, e, e_right);
        }
        // program: sends, full rows, receivers (3 passes over the pieces in K order, regenerated each pass: no
        // arrays on the small BRISC stack); a DN op after each produced row
        for (uint32_t pass = 0; pass < 3; ++pass) {
            for (uint32_t f = s; f < e;) {
                const uint32_t r = f / g.KT;
                uint32_t end = (r + 1) * g.KT;
                if (end > e) {
                    end = e;
                }
                const uint32_t k0 = f - r * g.KT, k1 = end - r * g.KT;
                f = end;
                uint32_t fl = 0;
                if (k0 > 0) {  // the row began in the left cluster
                    fl |= s_right ? PRE_L : SEND_L;
                }
                if (k1 < g.KT) {  // the row goes on in the right cluster
                    fl |= e_right ? SEND_R : PRE_R;
                }
                const uint32_t cls = (fl & (SEND_L | SEND_R)) ? 0 : ((fl & (PRE_L | PRE_R)) ? 2 : 1);
                if (cls != pass) {
                    continue;
                }
                p[OFF_OP + 3 * nop] = k0;
                p[OFF_OP + 3 * nop + 1] = k1;
                p[OFF_OP + 3 * nop + 2] = fl;
                ++nop;
                for (uint32_t kk = k0; kk < k1; kk += g.BK) {  // x blocks of this piece
                    if (nbe % g.GC == j) {
                        p[OFF_MY + 4 * nmy] = nbe;
                        p[OFF_MY + 4 * nmy + 1] = r;
                        p[OFF_MY + 4 * nmy + 2] = kk;
                        p[OFF_MY + 4 * nmy + 3] = k1 - kk < g.BK ? k1 - kk : g.BK;
                        ++nmy;
                    }
                    ++nbe;
                }
                if (pass > 0) {
                    p[OFF_OWN + nown++] = r;
                    p[OFF_OP + 3 * nop] = 0;
                    p[OFF_OP + 3 * nop + 1] = 0;
                    p[OFF_OP + 3 * nop + 2] = DN;
                    ++nop;
                }
            }
        }
    }
    p[H_NOP] = nop;
    p[H_NOWN] = nown;
    p[H_NBE] = nbe;
    p[H_NMY] = nmy;
}

}  // namespace ws_plan
