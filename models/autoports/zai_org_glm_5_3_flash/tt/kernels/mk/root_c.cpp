// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel root core (whole-layer program) — compute: mhc_pre's root compute twice: P1 (attention sublayer) then P7
// (FFN sublayer, MK_P7_CT), the root's CBs rewound in between. compute_kernel_hw_startup runs once (mkc::hw_startup).
#define MK_PHASED 1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/compute/common.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_sfpu.hpp"

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

namespace ph1 {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/compute.cpp"
#undef kernel_main
}  // namespace ph1

namespace ph7 {
constexpr uint32_t PH_CT0 = MK_P7_CT;
uint32_t ph_rt0 = MK_P7_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/compute.cpp"
#undef kernel_main
}  // namespace ph7

void kernel_main() {
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {  // stack program (kernels/mk/stack.hpp)
        if (layer > 0) {
            mk::cb_rewind_range(0, 2);
            mk::cb_rewind_range(14, 7);
        }
        ph1::phase_main();
        mk::cb_rewind_range(0, 2);
        mk::cb_rewind_range(14, 7);
        ph7::phase_main();
    }  // layer
}
