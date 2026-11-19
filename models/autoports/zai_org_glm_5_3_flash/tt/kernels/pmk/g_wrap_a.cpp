// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// W7 step 5 / G v2: program G (tt/pmk_gang.py) on a gang core -- after Stage 16 M-B's program B wrapper: NCRISC
// (g_ncrisc.cpp: x-row gather / out rows, combine reader) as a phase body of the fused layer program
// (kernels/pmk/b_wrap.hpp: the guard words, the A -> B semaphore hand-over). Every header of the wrapped kernel FIRST
// (the family hoists them: a header first seen inside the namespace would land there); the wrapped kernel's kernel_main
// is bw::phase_main (the family's
// "#define kernel_main phase_main" covers this file).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/b_wrap.hpp"

namespace bw {
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/g_ncrisc.cpp"
}  // namespace bw

void kernel_main() {
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        for (uint32_t k = 0; k < 4;
             ++k) {  // GLM53_PMK_PHASE_PROBE: the 4 marks G.handover / ffnw / ffn_barrier / combr
            PMK_PP_MARK();
        }
        return;
    }
    pmk_bw::wait_go(ph_rt0);
    PMK_PP_MARK();           // GLM53_PMK_PHASE_PROBE: G.handover (the round-2 release seen)
#if !defined(PMK_BW_NOBODY)  // the bisect knob GLM53_PMK_B_NOBODY=1: the hand-over / reclaim only
    bw::phase_main();
#else
    for (uint32_t k = 0; k < 3; ++k) {
        PMK_PP_MARK();
    }
#endif
}
