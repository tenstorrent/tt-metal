// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK program B, PMK_MCAST_DUAL -- the expert STREAMER core's NCRISC (kernels/pmk/exp_streamer_nc.hpp): the second
// half of every chunk on NoC1. Defines: PMK_CT_STREAM / PMK_RT_STREAM (the BRISC's lists).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"

namespace exstream1 {
constexpr uint32_t CT0 = PMK_CT_STREAM;
constexpr uint32_t RT0 = PMK_RT_STREAM;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/exp_streamer_nc.hpp"
}  // namespace exstream1

void kernel_main() {
    PMK_B_PROLOGUE();  // GLM53_PMK_ARGS=table (moe_common.hpp)
    exstream1::phase_main();
}
