// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel MoE FFN (tt/mk_moe.py) — reader core compute: the dense reader phases (mk/reader_c.cpp) with the
// routed-expert compute (mkx/xc.cpp) between dense phase MK_SPLIT - 1 and MK_SPLIT. compute_kernel_hw_startup runs
// once (mkc::hw_startup); argument bases: dense 0 / 0, experts MK_X_CT / MK_X_RT.
#define MK_PHASED 1
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
// (global scope: reader_c.cpp's deferred-1/rms(h) pack uses it from inside namespace dense; #pragma once there)
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/mkx_common.hpp"

namespace mkc {
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
ALWI void hw_startup(uint32_t a, uint32_t c) { hw_startup<SrcOrder::Regular>(a, a, c); }
}  // namespace mkc
#define compute_kernel_hw_startup mkc::hw_startup

namespace experts {
constexpr uint32_t PH_CT0 = MK_X_CT;
uint32_t ph_rt0 = MK_X_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/xc.cpp"
#undef kernel_main
}  // namespace experts

#define MK_BETWEEN() ::experts::phase_main()

namespace dense {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/reader_c.cpp"
#undef kernel_main
}  // namespace dense

void kernel_main() { dense::phase_main(); }
