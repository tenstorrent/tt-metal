// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Column reduce-scatter microbench for a K-split gate/up (one group column, NC cores, each owning one (gate, up) pair
// for every row): every core produces its partials of row q at t0 + q * T and sends each other core o the (gate, up)
// tile pair o owns into o's landing slot q % D, contributor r (payload, then a 4-byte flag q + 1 from the same command
// buffer). The owner's NCRISC consumes slot q % D once all NC - 1 contributions have landed and credits the
// contributors (inline write of its consumed count into their credit word for it); a contributor sends row q to o
// once o has consumed row q - D.
//   MODE 0 (direction split): BRISC sends both tiles to the rows below on NOC0, NCRISC to the rows above on NOC1
//   MODE 1 (gate / up split, moe_fused_swiglu's): BRISC sends every gate tile on NOC0, NCRISC every up tile on NOC1
//     (each NoC reaches half its targets by wrapping around the column); two flags a contribution
// Credits: BRISC to the rows below (NOC0), NCRISC to the rows above (NOC1), both modes.
// Stamps: [0, 1] BRISC start / end, [2, 3] NCRISC start / end, [64 + q] slot q complete at the owner (q < 400).
// CT: 0 R, 1 D, 2 TB, 3 MODE, 4 land, 5 flags, 6 cred, 7 stamps, 8 T, 9 cons word, 10 NC
// RT: 0 r (my index), 1..NC NoC x of core i, NC + 1..2 NC NoC y of core i
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t R = get_compile_time_arg_val(0), D = get_compile_time_arg_val(1);
constexpr uint32_t TB = get_compile_time_arg_val(2), MODE = get_compile_time_arg_val(3);
constexpr uint32_t LAND = get_compile_time_arg_val(4), FLAGS = get_compile_time_arg_val(5);
constexpr uint32_t CRED = get_compile_time_arg_val(6), T = get_compile_time_arg_val(8);
constexpr uint32_t CONS = get_compile_time_arg_val(9), NC = get_compile_time_arg_val(10);
#ifdef COMPILE_FOR_BRISC
constexpr bool BR = true;
#else
constexpr bool BR = false;
#endif
constexpr uint32_t SCR = FLAGS + (D * NC * 2 + (BR ? 0 : 1)) * 16;  // this RISC's flag source word (16 B offset 0)

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
FORCE_INLINE volatile tt_l1_ptr uint32_t* w(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
FORCE_INLINE uint32_t land(uint32_t s, uint32_t k, uint32_t t) { return LAND + ((s * NC + k) * 2 + t) * TB; }
FORCE_INLINE uint32_t flag(uint32_t s, uint32_t k, uint32_t f) { return FLAGS + ((s * NC + k) * 2 + f) * 16; }

void kernel_main() {
    volatile tt_l1_ptr uint32_t* st = w(get_compile_time_arg_val(7));
    const uint32_t r = get_arg_val<uint32_t>(0);
    uint32_t xs[NC], ys[NC];
    for (uint32_t i = 0; i < NC; ++i) {
        xs[i] = get_arg_val<uint32_t>(1 + i);
        ys[i] = get_arg_val<uint32_t>(1 + NC + i);
    }
    // my send targets and creditees (rows below for BRISC, above for NCRISC)
    uint32_t tg[NC], ntg = 0, cd[NC], ncd = 0;
    for (uint32_t o = 0; o < NC; ++o) {
        if (o == r) {
            continue;
        }
        const bool below = o > r;
        if (MODE == 1 || below == BR) {
            tg[ntg++] = o;
        }
        if (below == BR) {
            cd[ncd++] = o;
        }
    }
    const uint32_t t0 = wall();
    st[BR ? 0 : 2] = t0;
    if constexpr (!BR) {
        *w(CONS) = 0;
    }
    uint32_t q = 0, ti = 0, cred_sent = 0, cons_q = 0;
    *w(SCR) = 1;
    while (q < R || cred_sent < R || (!BR && cons_q < R)) {
        invalidate_l1_cache();
        // consume (NCRISC): slot cons_q % D complete?
        if (!BR && cons_q < R) {
            const uint32_t s = cons_q % D;
            bool all = true;
            for (uint32_t k = 0; k < NC && all; ++k) {
                if (k != r) {
                    all = *w(flag(s, k, 0)) >= cons_q + 1 && (MODE == 0 || *w(flag(s, k, 1)) >= cons_q + 1);
                }
            }
            if (all) {
                if (cons_q < 400) {
                    st[64 + cons_q] = wall();
                }
                *w(CONS) = ++cons_q;
            }
        }
        // credits
        const uint32_t c = *w(CONS);
        if (c > cred_sent) {
            for (uint32_t i = 0; i < ncd; ++i) {
                noc_inline_dw_write(get_noc_addr(xs[cd[i]], ys[cd[i]], CRED + r * 16), c);
            }
            cred_sent = c;
        }
        // sends of row q
        if (q < R && wall() - t0 >= q * T) {
            while (ti < ntg) {
                const uint32_t o = tg[ti];
                if (q >= D && *w(CRED + o * 16) < q - D + 1) {
                    break;
                }
                const uint32_t s = q % D;
                if (MODE == 0) {
                    noc_async_write(land(s, r, 0), get_noc_addr(xs[o], ys[o], land(s, r, 0)), 2 * TB);
                    noc_async_write(SCR, get_noc_addr(xs[o], ys[o], flag(s, r, 0)), 4);
                } else {
                    const uint32_t t = BR ? 0 : 1;
                    noc_async_write(land(s, r, t), get_noc_addr(xs[o], ys[o], land(s, r, t)), TB);
                    noc_async_write(SCR, get_noc_addr(xs[o], ys[o], flag(s, r, t)), 4);
                }
                ++ti;
            }
            if (ti == ntg) {
                noc_async_writes_flushed();
                ++q;
                ti = 0;
                *w(SCR) = q + 1;
            }
        }
    }
    noc_async_write_barrier();
    st[BR ? 1 : 3] = wall();
}
