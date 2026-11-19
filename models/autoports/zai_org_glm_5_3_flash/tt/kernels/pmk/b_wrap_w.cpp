// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 16 M-B: program B's receiver BRISC (moe_brisc.cpp: plan, weight-chunk reader, combine writer) as a phase body
// of the fused layer program (kernels/pmk/b_wrap.hpp: the guard words, the A -> B semaphore hand-over). Every header of
// the wrapped kernel FIRST (the family hoists them: a header first seen inside the namespace would land there); the
// wrapped kernel's kernel_main is bw::phase_main (the family's
// "#define kernel_main phase_main" covers this file).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/b_wrap.hpp"

namespace bw {
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_brisc.cpp"
}  // namespace bw

#ifdef PMK_PLAN_EARLY
#if !defined(PMK_PLAN_SKIP) || !defined(PMK_PLAN_P)
#error "PMK_PLAN_EARLY: with PMK_PLAN_SKIP (B skips its plan) on a plan-core prefix (PMK_PLAN_P)"
#endif
#define PMK_BW_MARKS 5  // + B.eplan
#else
#define PMK_BW_MARKS 4
#endif

void kernel_main() {
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        for (uint32_t k = 0; k < PMK_BW_MARKS;
             ++k) {  // GLM53_PMK_PHASE_PROBE: [B.eplan] / B.handover / plan / sweep / comb
            PMK_PP_MARK();
        }
        return;
    }
#ifdef PMK_PLAN_EARLY
    // Stage 14 B2.3 step (2): program B's routing plan on the first PMK_PLAN_P B cores (the idle MM-only BRISCs) BEFORE
    // the hand-over -- here right after their window entry, i.e. while the members still run P13. It waits for P12's
    // ids (every member's router-tail writer +1s PMK_PLAN_SIG_SEM after its rows landed), barriers on PMK_PLAN_SEM_A /
    // _B (ids that A / P12 / P13 never touch on these cores: zero at launch, used once), and writes the tables before
    // this core's round-1 arrival, so the hand-over is the plan-done barrier. The tables are partition-invariant:
    // bitwise. A barrier id may also carry the P12 signal (ids are scarce on MM-only: at TP8 id 4 is P15's SEM_P15_TP8
    // on the stage-1 reducers): its wait then adds PMK_PLAN_SIG_N. Exact because a later barrier's +1s reach a core
    // only after EVERY plan core passed its P12 wait (they come after barrier A, which needs every plan core's
    // arrival).
    constexpr uint32_t BASE_A = (PMK_PLAN_SEM_A == PMK_PLAN_SIG_SEM ? PMK_PLAN_SIG_N : 0);
    constexpr uint32_t BASE_B =
        (PMK_PLAN_SEM_B == PMK_PLAN_SIG_SEM ? PMK_PLAN_SIG_N : 0) + (PMK_PLAN_SEM_B == PMK_PLAN_SEM_A ? PMK_PLAN_P : 0);
    static_assert(
        PMK_PLAN_SEM_A != PMK_PLAN_SIG_SEM,
        "barrier A must not share the signal's id (its +1s could pass the P12 wait)");
    auto plan_barrier = [](uint32_t sid, uint32_t base) {
        for (uint32_t r = 0; r < PMK_PLAN_P; ++r) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(PMK_RT_CORES + 2 * r),
                    get_arg_val<uint32_t>(PMK_RT_CORES + 2 * r + 1),
                    get_semaphore(sid)),
                1);  // fmt: skip
        }
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sid)), base + PMK_PLAN_P);
    };
    if (get_arg_val<uint32_t>(PMK_RT_COUNT + 3) < PMK_PLAN_P) {  // plan_count's core index: a plan core
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(PMK_PLAN_SIG_SEM)), PMK_PLAN_SIG_N);
        bw::plan_count::phase_main();
        plan_barrier(PMK_PLAN_SEM_A, BASE_A);
        bw::plan_dest::phase_main();
        plan_barrier(PMK_PLAN_SEM_B, BASE_B);
        bw::plan_runs::phase_main();
        noc_async_write_barrier();  // the tables landed before this core's round-1 arrival
    }
    PMK_PP_MARK();  // GLM53_PMK_PHASE_PROBE: B.eplan (the early plan done on a plan core; immediate elsewhere)
#endif
    pmk_bw::handover<true>(ph_rt0, PMK_RT_CORES, PMK_P, PMK_RT_STREAMERS);
    PMK_PP_MARK();           // GLM53_PMK_PHASE_PROBE: B.handover (the A -> B semaphore hand-over done)
#if !defined(PMK_BW_NOBODY)  // the bisect knob GLM53_PMK_B_NOBODY=1: the hand-over / reclaim only
    bw::phase_main();
#else
    for (uint32_t k = 0; k < 3; ++k) {
        PMK_PP_MARK();
    }
#endif
}
