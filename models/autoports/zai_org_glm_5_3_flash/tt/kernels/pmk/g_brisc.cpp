// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK W7 step 5 -- program G (tt/pmk_gang.py: the EP routed MoE on GANGS) -- BRISC (NoC0) composite on the P gang
// cores: [wait H8] -> plan step 1 (counts) -> barrier -> plan step 2 (dest / starts / staging) -> barrier -> plan step
// 3 (disp runs) -> barrier -> the gang weight reader (gang_reader.hpp = 14's moe_xffn_gang_reader.cpp: this gang's
// units, W slice s of every local expert) -> the combine writer (y tiles; H9 per unit). moe_brisc.cpp's composition
// with the gang FFN in place of xffn; every core of every gang runs all phases (plan / combine over the same P cores).
// Defines (host): PMK_CT_COUNT / DEST / RUNS / GANG / COMB and the PMK_RT_* bases, PMK_RT_CORES, PMK_P, optional
// PMK_MOE_WAIT_SEM (+ PMK_RT_WAIT), optional PMK_H9 (PMK_RT_H9); the gang's XFFN_GANG / XFFN_GANG_NG / XFFN_SHARD_BASES
// / XFFN_X_ON_NOC1 / XFFN_VALID_ROWS; PMK_EP_SENTINEL (plan_dest: another chip's pair -> the RM_SKIP combine's zero
// part).
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
namespace gang {
constexpr uint32_t CT0 = PMK_CT_GANG;
constexpr uint32_t RT0 = PMK_RT_GANG;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/gang_reader.hpp"
}  // namespace gang
namespace comb {
constexpr uint32_t CT0 = PMK_CT_COMB;
constexpr uint32_t RT0 = PMK_RT_COMB;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/comb_writer.hpp"
}  // namespace comb

void kernel_main() {
    PMK_B_PROLOGUE();  // GLM53_PMK_ARGS=table: this core's arg row -> rta_l1_base (moe_common.hpp)
#if defined(PMK_MOE_WAIT_SEM)
    wait_min(PMK_MOE_WAIT_SEM, get_arg_val<uint32_t>(PMK_RT_WAIT));  // H8: x-hat_f store (x_ext) + routing ids landed
#endif
    plan_count::phase_main();
    barrier_all(SEM_PLAN1, PMK_RT_CORES, PMK_P);
    plan_dest::phase_main();
    barrier_all(SEM_PLAN2, PMK_RT_CORES, PMK_P);
    plan_runs::phase_main();
    barrier_all(SEM_PLAN3, PMK_RT_CORES, PMK_P);  // disp / cnt / start landed on every gang core
    PMK_PP_MARK();                                // GLM53_PMK_PHASE_PROBE: G.plan
    gang::phase_main();  // its CB_CTL / CB_WCTL sentinels end the compute's and the writer's gang loops
    PMK_PP_MARK();       // GLM53_PMK_PHASE_PROBE: G.gang (this core's weight reads done)
    comb::phase_main();  // consumes CB_COUT: starts when the NCRISC's combine reader (after SEM_FFN) feeds the compute
    PMK_PP_MARK();       // GLM53_PMK_PHASE_PROBE: G.comb
}
