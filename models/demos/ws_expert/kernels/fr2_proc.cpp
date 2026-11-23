// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric ring microbench, RISC split (fr2_*): a core a lane, the processor RISC. Examines every landed packet in order
// (forwards left 0: retire, else decrement and forward), publishes the decisions to the sender RISC (act[] per slot,
// then READY), takes the piggy-backed credit from the newest landed packet into the partner core's CPB (it is the
// partner lane's downstream credit), and passes this lane's freed prefix and done flag to the partner core (its
// FREED_O / DONE_O). All writes to other cores go from here, so the sender's command buffers keep their send state.
// CT: 0 S, 1 PB, 2 TOT (arrivals), 3 ring (L1), 4 words, 5 act (L1, S words)
// RT: 0 partner core x | y << 8
#include "api/dataflow/dataflow_api.h"
#include "fr2_words.hpp"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0), PB = get_compile_time_arg_val(1);
    constexpr uint32_t TOT = get_compile_time_arg_val(2), ring = get_compile_time_arg_val(3);
    constexpr uint32_t W = get_compile_time_arg_val(4), act = get_compile_time_arg_val(5);
    using namespace fr2;
    auto w = [](uint32_t i) -> volatile tt_l1_ptr uint32_t& { return word(W, i); };
    volatile tt_l1_ptr uint32_t* acts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act);
    const uint32_t pxy = get_arg_val<uint32_t>(0);
    auto put = [&](uint32_t i, uint32_t v) { noc_inline_dw_write(get_noc_addr(pxy & 0xFF, pxy >> 8, W + 16 * i), v); };

    uint32_t r = 0, cmax = 0, last_f = 0, sent_done = 0, busy = 0, n_put = 0;
    while (true) {
        invalidate_l1_cache();
        const uint32_t a = w(ARR);
        const uint32_t t0 = wall();
        if (r < a) {
            const uint32_t c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring + ((a - 1) % S) * PB + PB - 16)[1];
            if (c > cmax) {  // (monotonic: packets carry non-decreasing counts)
                cmax = c;
                put(CPB, c);
                ++n_put;
            }
            for (; r < a; ++r) {
                volatile tt_l1_ptr uint32_t* m =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring + (r % S) * PB + PB - 16);
                const uint32_t fl = m[0];
                if (fl == 0) {
                    acts[r % S] = ACT_RETIRE;
                } else {
                    m[0] = fl - 1;
                    acts[r % S] = ACT_FWD;
                }
            }
            w(READY) = r;
            busy += wall() - t0;
        }
        const uint32_t f = w(FREED);
        if (f != last_f) {
            put(FREED_O, f);
            last_f = f;
            ++n_put;
        }
        if (!sent_done && w(DONE)) {
            put(DONE_O, 1);
            sent_done = 1;
        }
        if (sent_done && w(TDONE) && last_f == w(FREED)) {
            break;
        }
    }
    w(ST_P + 0) = busy;
    w(ST_P + 1) = n_put;
    w(ST_P + 2) = r;
    noc_async_write_barrier();
}
