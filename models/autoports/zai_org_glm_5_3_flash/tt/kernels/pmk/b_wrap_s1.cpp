// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 16 M-B: program B's expert streamer NCRISC (moe_stream_ncrisc.cpp, PMK_MCAST_DUAL) as a phase body of the fused
// layer program (kernels/pmk/b_wrap.hpp: the guard words, the A -> B semaphore hand-over). Every header of the wrapped
// kernel FIRST (the family hoists them: a header first seen inside the namespace would land there); the wrapped
// kernel's kernel_main is bw::phase_main (the family's
// "#define kernel_main phase_main" covers this file).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/b_wrap.hpp"

namespace bw {
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_stream_ncrisc.cpp"
}  // namespace bw

void kernel_main() {
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        return;
    }
    pmk_bw::wait_go(ph_rt0);
#if !defined(PMK_BW_NOBODY) && !(defined(PMK_BW_STAGE) && PMK_BW_STAGE < 2)  // bisect: no body / plan-only stage
    bw::phase_main();
#endif
}
