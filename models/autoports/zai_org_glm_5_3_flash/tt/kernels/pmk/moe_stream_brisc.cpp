// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK program B -- the expert STREAMER core's BRISC (kernels/pmk/exp_streamer.hpp): the only kernel on a bank-adjacent
// streamer core (its NCRISC / TRISC are idle in this program). Defines: PMK_CT_STREAM / PMK_RT_STREAM.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"

namespace exstream {
constexpr uint32_t CT0 = PMK_CT_STREAM;
constexpr uint32_t RT0 = PMK_RT_STREAM;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/exp_streamer.hpp"
}  // namespace exstream

void kernel_main() {
    PMK_B_PROLOGUE();  // GLM53_PMK_ARGS=table (moe_common.hpp)
    exstream::phase_main();
}
