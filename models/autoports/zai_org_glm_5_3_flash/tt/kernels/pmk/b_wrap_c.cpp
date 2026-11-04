// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 16 M-B: program B's receiver TRISC (moe_trisc.cpp: FFN compute, combine compute; the matmuls / multiplies at
// PMK_PART_FID) as a phase body of the fused layer program (kernels/pmk/b_wrap.hpp: the guard words, the A -> B
// semaphore hand-over). Every header of the wrapped kernel FIRST (the family hoists them: a header first seen inside
// the namespace would land there); the wrapped kernel's kernel_main is bw::phase_main (the family's
// "#define kernel_main phase_main" covers this file).
#include <cstdint>

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

namespace bw {
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_trisc.cpp"
}  // namespace bw
// (cf3h-k, 2026-10-26) the wrapped kernel's compute_kernel_hw_startup -> bw::pmkc::hw_startup macro must not leak
// into the parts after this one in the TRISC composite (P15's RS8 reduces call compute_kernel_hw_startup: 'pmkc' has
// not been declared in their phase namespace); they take the real one, as P8's RS8 reduce does before it
#undef compute_kernel_hw_startup

void kernel_main() {
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        return;
    }
#if !defined(PMK_BW_NOBODY)  // the bisect knob GLM53_PMK_B_NOBODY=1: the hand-over / reclaim only
    bw::phase_main();
#endif
}
