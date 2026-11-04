// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel root core (whole-layer program) — NCRISC (NoC1): mhc_pre's root sync role twice: P1 (the
// attention sublayer) then P7 (the FFN sublayer, MK_P7_CT / MK_P7_RT), the root's CBs rewound in between (the FFN
// instance reuses the attention instance's CB indices; mhc_pre's semaphores are reset after every wait, sync.cpp).
#define MK_PHASED 1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if defined(MHC_DMA_ZERO)  // Stage 14c: mhc_pre sync.cpp zero fills (NoC API, at global scope before any phase)
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#endif
#include "api/debug/kernel_profiler.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"

namespace ph1 {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/sync.cpp"
#undef kernel_main
}  // namespace ph1

namespace ph7 {
constexpr uint32_t PH_CT0 = MK_P7_CT;
uint32_t ph_rt0 = MK_P7_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/sync.cpp"
#undef kernel_main
}  // namespace ph7

void kernel_main() {
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {  // stack program (kernels/mk/stack.hpp)
        if (layer > 0) {
            mkstack::load_args(layer);
            mk::cb_rewind_range(0, 2);
            mk::cb_rewind_range(14, 7);
        }
        ph1::phase_main();
        mk::cb_rewind_range(0, 2);   // GATHER, MSG
        mk::cb_rewind_range(14, 7);  // RED .. DBG (root only)
        ph7::phase_main();
    }  // layer
}
