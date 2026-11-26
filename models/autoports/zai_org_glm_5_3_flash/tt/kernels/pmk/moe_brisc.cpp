// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK MoE phase set -- BRISC (NoC0) composite (tt/pmk_moe.py, moe_common.hpp): [wait H8] -> plan step 1 (counts)
// -> barrier -> plan step 2 (dest / starts / staging) -> barrier -> plan step 3 (disp runs) -> barrier -> the expert
// FFN weight streamer (moe_xffn_reader: this core's units, gate|up / down chunks into the double-buffered rings) ->
// the combine writer (y tiles; H9 per unit). Defines (host): PMK_CT_COUNT / DEST / RUNS / FFN / COMB and the PMK_RT_*
// runtime bases, PMK_RT_CORES (the P MM cores' NoC (x, y) for the barriers), PMK_P, optional PMK_MOE_WAIT_SEM (+
// PMK_RT_WAIT: the count), optional PMK_H9 (PMK_RT_H9: n consumers, then (x, y, base) each; +1 at base + 16 * block).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"

using namespace pmk_moe;

#if defined(PMK_H9) && defined(PMK_H9_TABLE)
// H9 TABLE mode (non-B lever (2), P15 streamed behind the combine): PMK_RT_H9 = [n targets, (x, y, sync base) x n,
// then per 32-row block (target index, word)]: the unit of row block `block` raises ONE word -- the readiness word of
// the lane / reducer that sends or reduces that block (fs_common.hpp LREADY + i / OWNRDY + i); a block is complete at
// H_SPLIT units (the reader waits g x FS_RDY_MULT)
FORCE_INLINE void pmk_h9_signal(uint32_t block) {
    const uint32_t n = get_arg_val<uint32_t>(PMK_RT_H9);
    const uint32_t t = get_arg_val<uint32_t>(PMK_RT_H9 + 1 + 3 * n + 2 * block);
    const uint32_t w = get_arg_val<uint32_t>(PMK_RT_H9 + 2 + 3 * n + 2 * block);
    noc_semaphore_inc(
        get_noc_addr(
            get_arg_val<uint32_t>(PMK_RT_H9 + 1 + 3 * t),
            get_arg_val<uint32_t>(PMK_RT_H9 + 2 + 3 * t),
            get_arg_val<uint32_t>(PMK_RT_H9 + 3 + 3 * t) + 16 * w),
        1);
    noc_async_atomic_barrier();
}
#define PMK_H9_SIGNAL(block) pmk_h9_signal(block)
#elif defined(PMK_H9)
FORCE_INLINE void pmk_h9_signal(uint32_t block) {
    const uint32_t n = get_arg_val<uint32_t>(PMK_RT_H9);
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t x = get_arg_val<uint32_t>(PMK_RT_H9 + 1 + 3 * r);
        const uint32_t y = get_arg_val<uint32_t>(PMK_RT_H9 + 2 + 3 * r);
        const uint32_t base = get_arg_val<uint32_t>(PMK_RT_H9 + 3 + 3 * r);
        noc_semaphore_inc(get_noc_addr(x, y, base + 16 * block), 1);
    }
    noc_async_atomic_barrier();
}
#define PMK_H9_SIGNAL(block) pmk_h9_signal(block)
#else
#define PMK_H9_SIGNAL(block) \
    do {                     \
    } while (0)
#endif

namespace plan_count {
constexpr uint32_t CT0 = PMK_CT_COUNT;
constexpr uint32_t RT0 = PMK_RT_COUNT;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/plan_count.hpp"
}  // namespace plan_count
namespace plan_dest {
constexpr uint32_t CT0 = PMK_CT_DEST;
constexpr uint32_t RT0 = PMK_RT_DEST;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/plan_dest.hpp"
}  // namespace plan_dest
namespace plan_runs {
constexpr uint32_t CT0 = PMK_CT_RUNS;
constexpr uint32_t RT0 = PMK_RT_RUNS;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/plan_runs.hpp"
}  // namespace plan_runs
namespace ffn {
constexpr uint32_t CT0 = PMK_CT_FFN;
constexpr uint32_t RT0 = PMK_RT_FFN;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/ffn_reader.hpp"
}  // namespace ffn
namespace comb {
constexpr uint32_t CT0 = PMK_CT_COMB;
constexpr uint32_t RT0 = PMK_RT_COMB;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/comb_writer.hpp"
}  // namespace comb
#ifdef PMK_COMB_SPLIT
namespace combr {  // perf5 PMK_COMB_SPLIT: the combine's odd slots rebuilt here (the NCRISC writes y)
constexpr uint32_t CT0 = PMK_CT_COMBR;
constexpr uint32_t RT0 = PMK_RT_COMBR;
constexpr uint32_t HALF = 1;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/comb_reader.hpp"
}  // namespace combr
#endif

void kernel_main() {
    PMK_B_PROLOGUE();  // GLM53_PMK_ARGS=table: this core's arg row -> rta_l1_base (moe_common.hpp)
#if defined(PMK_MCAST)
    {  // v2 lever 3: the READY / VAL words (CB_TAB prefix) start at 0 for this launch; receivers credit after barrier 3
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_TAB));
        for (uint32_t k = 0; k < SYNC_WORDS_B / 4; ++k) {
            w[k] = 0;
        }
    }
#endif
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 1  // the bisect's stage 0 (Stage 14 B2.3): the launch alone, no body
    return;
#endif
#if defined(PMK_MOE_WAIT_SEM)
    wait_min(PMK_MOE_WAIT_SEM, get_arg_val<uint32_t>(PMK_RT_WAIT));  // H8: x-hat_f store + routing ids landed
#endif
#if defined(PMK_PLAN_SKIP)
    // Stage 14 B2.3 step (2): the plan ran BEFORE B's hand-over on the plan cores (b_wrap_w.cpp PMK_PLAN_EARLY); its
    // tables landed before those cores' round-1 arrival, so after the hand-over every core reads them -- no plan here
#elif defined(PMK_PLAN_P)
#if defined(PMK_MCAST_EXT)
#error "PMK_PLAN_P: the self-streamed program B (no streamer cores waiting on SEM_PLAN3)"
#endif
    // Stage 14 B2.3 PMK_PLAN_P: the plan on the first PMK_PLAN_P cores of the core list only (its cost is linear in its
    // core count; its tables are partition-invariant -- the single-core fill order -- so the output is bitwise);
    // barriers 1 / 2 among the plan cores; the plan cores' last step counts SEM_PLAN3 on EVERY core, all wait >=
    // PMK_PLAN_P
    static_assert(PMK_PLAN_P <= PMK_P, "plan cores are a prefix of the core list");
    if (get_arg_val<uint32_t>(PMK_RT_COUNT + 3) < PMK_PLAN_P) {  // plan_count's core index
        plan_count::phase_main();
        barrier_all(SEM_PLAN1, PMK_RT_CORES, PMK_PLAN_P);
        plan_dest::phase_main();
        barrier_all(SEM_PLAN2, PMK_RT_CORES, PMK_PLAN_P);
        plan_runs::phase_main();
        noc_async_write_barrier();  // this core's tables landed before its count
        for (uint32_t r = 0; r < PMK_P; ++r) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(PMK_RT_CORES + 2 * r),
                    get_arg_val<uint32_t>(PMK_RT_CORES + 2 * r + 1),
                    get_semaphore(SEM_PLAN3)),
                1);
        }
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PLAN3)), PMK_PLAN_P);
#else
    plan_count::phase_main();
    barrier_all(SEM_PLAN1, PMK_RT_CORES, PMK_P);
    plan_dest::phase_main();
    barrier_all(SEM_PLAN2, PMK_RT_CORES, PMK_P);
    plan_runs::phase_main();
    barrier_all(SEM_PLAN3, PMK_RT_CORES, PMK_P);
#endif
#if defined(PMK_MCAST_EXT)
    {  // program B: the expert streamers wait SEM_PLAN3 >= P too (PMK_RT_STREAMERS: n, then NoC (x, y) each)
        const uint32_t n = get_arg_val<uint32_t>(PMK_RT_STREAMERS);
        for (uint32_t r = 0; r < n; ++r) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(PMK_RT_STREAMERS + 1 + 2 * r),
                    get_arg_val<uint32_t>(PMK_RT_STREAMERS + 2 + 2 * r),
                    get_semaphore(SEM_PLAN3)),
                1);
        }
        noc_async_atomic_barrier();
    }
#endif
    PMK_PP_MARK();                             // GLM53_PMK_PHASE_PROBE: B.plan (the 3 plan steps + barriers done)
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 2  // the M-B1 bisect (pmk_b_window GLM53_PMK_B_STAGE): plan only
    return;
#endif
    ffn::phase_main();  // the weight stream; its CB_CTL / CB_WCTL sentinels end the compute's and writer's FFN loops
    PMK_PP_MARK();      // GLM53_PMK_PHASE_PROBE: B.sweep (this core's weight stream done)
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 5  // the M-B1 bisect: no combine (2) / no COUT consumer (3 / 4)
    return;
#endif
#ifdef PMK_COMB_SPLIT
    wait_min(
        SEM_FFN, PMK_P);  // every core's partials landed (the NCRISCs' barrier_all raises it to P; no increment here)
    combr::phase_main();  // the odd slots of every unit (the NCRISC: the even ones, y_s, the y write)
#else
    comb::phase_main();  // consumes CB_COUT: starts when the NCRISC's combine reader (after SEM_FFN) feeds the compute
#endif
    PMK_PP_MARK();  // GLM53_PMK_PHASE_PROBE: B.comb (this core's combine output written)
}
