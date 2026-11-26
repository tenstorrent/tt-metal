// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 14 fabric all-reduce, lane core NCRISC = landing bookkeeping + result hand-off (tt/fabric_ar.py). The fabric
// routers write the peers' packets straight into this core's landing CBs (persistent L1 tensors at the same address on
// every chip); this kernel turns the per-source packet counters into CB pushes so TRISC reduces chunk by chunk as the
// data lands, then hands the packed result to the producers (RESULT_READY) and to BRISC (RES_DONE).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "far_common.hpp"

constexpr uint32_t PT = get_compile_time_arg_val(0);
constexpr uint32_t PKT_T = get_compile_time_arg_val(1);
constexpr uint32_t N1 = get_compile_time_arg_val(2);
constexpr uint32_t N2 = get_compile_time_arg_val(3);
constexpr uint32_t CB_L1 = get_compile_time_arg_val(4);  // CB_L1 + q: round-1 slot q (q = p1: the producers' data)
constexpr uint32_t CB_L2 = get_compile_time_arg_val(5);  // CB_L2 + q: round-2 slot q (q != p2)
constexpr uint32_t CB_RES = get_compile_time_arg_val(6);
constexpr uint32_t GRP = get_compile_time_arg_val(7);     // packets per counter increment (lane_send.cpp)
constexpr uint32_t CB_OWN = get_compile_time_arg_val(8);  // the producers' packed partial (own slot of the first round)
constexpr uint32_t NPKT = (PT + PKT_T - 1) / PKT_T;
constexpr uint32_t NSIG = (NPKT + GRP - 1) / GRP;  // counter increments per source per iteration

void kernel_main() {
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t p1 = get_arg_val<uint32_t>(a++);
    const uint32_t p2 = get_arg_val<uint32_t>(a++);
    const uint32_t n_prod = get_arg_val<uint32_t>(a++);
    const uint32_t iters = get_arg_val<uint32_t>(a++);
    // result consumers: +1 on a word at cons_addr of each (standalone: the producers' RESULT_READY counter; megakernel,
    // FAR_CONS_SEM defined: the consumer phase's hand-off semaphore, which it waits for and resets)
#if defined(FAR_CONS_SEM_RT)
    // the runtime arg = program semaphore id of even iterations; FAR_CONS_SEM2 (optional) = the id of odd iterations
    // (a layer program with two all-reduces: the attention consumers' and the FFN consumers' hand-off semaphores)
    const uint32_t cons_sem0 = get_arg_val<uint32_t>(a++);
#elif defined(FAR_CONS_SEM)
    const uint32_t cons_addr = get_semaphore(FAR_CONS_SEM);
    a++;
#else
    const uint32_t cons_addr = get_arg_val<uint32_t>(a++);
#endif
    const uint32_t n_cons = get_arg_val<uint32_t>(a++);
    const uint32_t cons_xy = a;  // n_cons x (x, y)
    volatile tt_l1_ptr uint32_t* gen = far::word(sync, far::GEN_NCRISC);
    uint32_t g = *gen;
    for (uint32_t it = 0; it < iters; it++) {
        g++;
        far::wait_ge(far::word(sync, far::OWN_READY), g * n_prod);
        // own partial (written by the producers) = round-1 slot p1, or the round-2 own slot without round 1
        cb_reserve_back(CB_OWN, PT);
        cb_push_back(CB_OWN, PT);
        if (it == 0) {
            far::probe(sync, far::PROBE_NC + 0, get_timestamp());
        }
        if constexpr (N1 > 1) {
            for (uint32_t j = 0; j < NSIG; j++) {
                // tiles of packets j*GRP .. min((j+1)*GRP, NPKT) - 1
                const uint32_t t0 = j * GRP * PKT_T, t1 = (j + 1) * GRP * PKT_T < PT ? (j + 1) * GRP * PKT_T : PT;
                for (uint32_t q = 0; q < N1; q++) {
                    if (q != p1) {
                        far::wait_ge(far::word(sync, far::ARR1 + q), (g - 1) * NSIG + j + 1);
                        cb_reserve_back(CB_L1 + q, t1 - t0);
                        cb_push_back(CB_L1 + q, t1 - t0);
                    }
                }
            }
            if (it == 0) {
                far::probe(sync, far::PROBE_NC + 1, get_timestamp());
            }
        }
        if constexpr (N2 > 1) {
            for (uint32_t j = 0; j < NSIG; j++) {
                // tiles of packets j*GRP .. min((j+1)*GRP, NPKT) - 1
                const uint32_t t0 = j * GRP * PKT_T, t1 = (j + 1) * GRP * PKT_T < PT ? (j + 1) * GRP * PKT_T : PT;
                for (uint32_t q = 0; q < N2; q++) {
                    if (q != p2) {
                        far::wait_ge(far::word(sync, far::ARR2 + q), (g - 1) * NSIG + j + 1);
                        cb_reserve_back(CB_L2 + q, t1 - t0);
                        cb_push_back(CB_L2 + q, t1 - t0);
                    }
                }
            }
            if (it == 0) {
                far::probe(sync, far::PROBE_NC + 2, get_timestamp());
            }
        }
        cb_wait_front(CB_RES, PT);
        if (it == 0) {
            far::probe(sync, far::PROBE_NC + 3, get_timestamp());
        }
#if defined(FAR_CONS_SEM_RT)
#if defined(FAR_CONS_SEM2)
        const uint32_t cons_addr = get_semaphore((it & 1) ? FAR_CONS_SEM2 : cons_sem0);
#else
        const uint32_t cons_addr = get_semaphore(cons_sem0);
#endif
#endif
        for (uint32_t k = 0; k < n_cons; k++) {
            const uint32_t x = get_arg_val<uint32_t>(cons_xy + 2 * k);
            const uint32_t y = get_arg_val<uint32_t>(cons_xy + 2 * k + 1);
            noc_semaphore_inc(get_noc_addr(x, y, cons_addr), 1);
        }
        // the consumers read the result before the producers can write the next iteration's partial (standalone: the
        // producers are the consumers; megakernel: the next partial depends on the consumer phase's output), and
        // TRISC packs the next result only after that partial: popping now is safe
        cb_pop_front(CB_RES, PT);
        noc_async_atomic_barrier();
        *far::word(sync, far::RES_DONE) = g;
    }
    *gen = g;
    far::probe(sync, far::PROBE_NC + 4, get_timestamp());
}
