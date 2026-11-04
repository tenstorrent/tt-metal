// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel FFN-half owner core — compute (tt/mk_ffn.py): P7 mhc_pre compute (the FFN sublayer, K role) -> P8 swiglu
// compute -> P10 mhc_post compute (the layer output). One kernel, each phase's file included in its own namespace with
// its own argument bases (MK_P8_CT / MK_P10_CT; kernels/mk/phase_args.hpp); compute_kernel_hw_startup once
// (mkc::hw_startup).
#define MK_PHASED 1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include <utility>

// every header the included phase kernels use, at global scope first (an API header first seen inside a phase
// namespace would put its definitions into that namespace)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_pre/kda_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_sfpu.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/kda_rec_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_v2_sfpu.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/swiglu_common.hpp"
#include "api/compute/eltwise_unary/clamp.h"

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

namespace ph8 {
constexpr uint32_t PH_CT0 = MK_P8_CT;
uint32_t ph_rt0 = MK_P8_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/compute.cpp"
#undef kernel_main
}  // namespace ph8

namespace ph10 {
constexpr uint32_t PH_CT0 = MK_P10_CT;
uint32_t ph_rt0 = MK_P10_RT;
#define kernel_main phase_main
#if defined(MK_POST_PACKED)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/compute_p.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/compute.cpp"
#endif
#undef kernel_main
}  // namespace ph10

void kernel_main() {
    ph1::phase_main();
    ph8::phase_main();
    ph10::phase_main();
}
