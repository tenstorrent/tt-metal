// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MkMla (the whole attention half, ONE program) owner core — compute (tt/mk_mla.py): mhc_pre (compute.cpp, K role; the
// root core runs mhc_pre's root kernels) -> the attention phase (kernels/mla/att_c.cpp, its DSA path under
// MLA_DSA: flash-decoding over the paged latent cache, head group g, row chunk c) -> mhc_post (packed decode
// compute: the attention output a from the o_proj readers + the stacked residual -> s1). Each phase file is included in
// its own namespace with its own argument bases (kernels/mk/phase_args.hpp; MK_PA_CT / MK_PA_RT = the attention's,
// MK_PP_CT / MK_PP_RT = mhc_post's).
#define MK_PHASED 1
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"  // mhc_post/compute.cpp (MLA_POST_TILE)
#include "api/debug/kernel_profiler.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_sfpu.hpp"

// one compute_kernel_hw_startup per kernel: the later phases reconfigure instead (kernels/mk/owner_ffn_c.cpp)
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
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"

namespace ph1 {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/compute.cpp"
#undef kernel_main
}  // namespace ph1

namespace pha {
constexpr uint32_t PH_CT0 = MK_PA_CT;
uint32_t ph_rt0 = MK_PA_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_c.cpp"  // DSA under MLA_DSA
#undef kernel_main
}  // namespace pha

namespace php {
constexpr uint32_t PH_CT0 = MK_PP_CT;
uint32_t ph_rt0 = MK_PP_RT;
#define kernel_main phase_main
#if defined(MLA_POST_TILE)  // t > 4 (tt/mk_mla.py): the per-tile kernels, flat residual / s1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/compute.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/compute_p.cpp"
#endif
#undef kernel_main
}  // namespace php

void kernel_main() {
    ph1::phase_main();
    pha::phase_main();
    php::phase_main();
}
