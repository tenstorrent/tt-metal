// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK W7 step 5 -- program G (tt/pmk_gang.py) -- NCRISC (NoC1) composite on the P gang cores: the gang's activation
// mover and ring (gang_writer.hpp = 14's moe_xffn_gang_writer.cpp: the group's x rows from x_ext through the plan's
// disp rows, the in-gang y ring reduce-scatter on CB_RIN / SEM_RING_*, the owner's y rows into out_rm, the end-of-body
// credit drain) -> barrier SEM_FFN (every gang core's rows landed) -> the combine reader (comb_reader.hpp; PMK_RM_SKIP:
// other chips' pairs / weight-0 pairs read the zero part). Defines as g_brisc.cpp (PMK_CT_GANG / COMB, PMK_RT_GANG /
// COMB, PMK_RT_CORES, PMK_P).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"

using namespace pmk_moe;

namespace gang {
constexpr uint32_t CT0 = PMK_CT_GANG;
constexpr uint32_t RT0 = PMK_RT_GANG;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/gang_writer.hpp"
}  // namespace gang
namespace comb {
constexpr uint32_t CT0 = PMK_CT_COMB;
constexpr uint32_t RT0 = PMK_RT_COMB;
constexpr uint32_t HALF = 0;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/comb_reader.hpp"
}  // namespace comb

void kernel_main() {
    PMK_B_PROLOGUE();  // GLM53_PMK_ARGS=table: this core's arg row -> rta_l1_base (moe_common.hpp)
    gang::phase_main();
    noc_async_write_barrier();                  // this core's out_rm rows are in DRAM before it counts
    PMK_PP_MARK();                              // GLM53_PMK_PHASE_PROBE: G.ffnw (ring + out_rm rows done)
    barrier_all(SEM_FFN, PMK_RT_CORES, PMK_P);  // every gang core's expert output rows are in out_rm
    PMK_PP_MARK();                              // GLM53_PMK_PHASE_PROBE: G.ffn_barrier (the lockstep wait)
    comb::phase_main();
    PMK_PP_MARK();  // GLM53_PMK_PHASE_PROBE: G.combr
}
