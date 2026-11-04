// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK W7 step 5 -- program G (tt/pmk_gang.py) -- compute composite on the P gang cores: the gang FFN compute
// (gang_compute.hpp = 14's moe_xffn_gang_compute.cpp: tilize, gate|up with fp32 L1 accumulation, clamped SwiGLU, down
// of W slice s, + the predecessor's ring block from CB_RIN) until the reader's sentinel, then the combine compute
// (comb_compute.hpp). compute_kernel_hw_startup runs once; the second phase reconfigures the data formats only.
// Defines: PMK_CT_GANG / PMK_CT_COMB, PMK_RT_GANG / PMK_RT_COMB.
#include <cstdint>

// every header of the phase files FIRST (pragma once: their includes inside the phase namespaces are then no-ops)
#include "api/compute/bcast.h"
#include "api/compute/clamped_silu_glu.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/pmk_fid.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"

using namespace pmk_moe;

namespace pmkc {
inline bool g_started = false;
template <SrcOrder O = SrcOrder::Regular>
ALWI void hw_startup(uint32_t a, uint32_t b, uint32_t c) {
    if (!g_started) {
        g_started = true;
        ::compute_kernel_hw_startup<O>(a, b, c);
    } else {
        reconfig_data_format<O>(a, b);
        pack_reconfig_data_format(c);
    }
}
}  // namespace pmkc
#define compute_kernel_hw_startup pmkc::hw_startup

namespace gang {
constexpr uint32_t CT0 = PMK_CT_GANG;
constexpr uint32_t RT0 = PMK_RT_GANG;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/gang_compute.hpp"
}  // namespace gang
namespace comb {
constexpr uint32_t CT0 = PMK_CT_COMB;
constexpr uint32_t RT0 = PMK_RT_COMB;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/comb_compute.hpp"
}  // namespace comb

void kernel_main() {
    gang::phase_main();
    comb::phase_main();
}
