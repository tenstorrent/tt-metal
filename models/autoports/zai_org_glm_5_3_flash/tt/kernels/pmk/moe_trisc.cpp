// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK MoE phase set -- compute composite: the expert FFN compute (moe_xffn_compute: tilize, gate|up with fp32 L1
// accumulation, clamped SwiGLU, down packed untilized) until the reader's sentinel, then the combine compute
// (moe_combine_rm_compute: the K weighted row parts summed in fp32, tilized). compute_kernel_hw_startup runs once;
// the second phase reconfigures the data formats only (the owner composites' mkc::hw_startup pattern).
// Defines: PMK_CT_FFN / PMK_CT_COMB, PMK_RT_FFN / PMK_RT_COMB.
#include <cstdint>

// every header of the phase files FIRST (pragma once: their includes inside the phase namespaces are then no-ops --
// an API header first seen inside a namespace would land there)
#include "api/compute/bcast.h"
#include "api/compute/clamped_silu_glu.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/pmk_fid.hpp"  // the combine's PMK_COMB_FID helpers
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/pmk_fid.hpp"  // PMK_FFN_FID / PMK_PART_FID
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

namespace ffn {
constexpr uint32_t CT0 = PMK_CT_FFN;
constexpr uint32_t RT0 = PMK_RT_FFN;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/ffn_compute.hpp"
}  // namespace ffn
namespace comb {
constexpr uint32_t CT0 = PMK_CT_COMB;
constexpr uint32_t RT0 = PMK_RT_COMB;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/comb_compute.hpp"
}  // namespace comb

void kernel_main() {
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 2  // the M-B1 bisect: plan only (the BRISC's)
    return;
#endif
    ffn::phase_main();
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 3  // the M-B1 bisect: no combine
    return;
#endif
    comb::phase_main();
}
