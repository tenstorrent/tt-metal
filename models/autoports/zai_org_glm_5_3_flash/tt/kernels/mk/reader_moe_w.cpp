// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel MoE FFN (tt/mk_moe.py) — reader core, weight streamer (BRISC, NoC0): the dense reader phases
// (mk/reader_w.cpp: router, shared gate_up | shared down) with the routed-expert phase (mkx/xw.cpp) run between dense
// phase MK_SPLIT - 1 and MK_SPLIT (the dense kernel's MK_BETWEEN hook). Argument bases: the dense phases at 0 / 0,
// the expert phase at MK_X_CT / MK_X_RT.
#define MK_PHASED 1
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/mkx_common.hpp"

namespace experts {
constexpr uint32_t PH_CT0 = MK_X_CT;
uint32_t ph_rt0 = MK_X_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/xw.cpp"
#undef kernel_main
}  // namespace experts

#define MK_BETWEEN() ::experts::phase_main()

namespace dense {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/reader_w.cpp"
#undef kernel_main
}  // namespace dense

void kernel_main() { dense::phase_main(); }
