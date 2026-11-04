// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MkMlaB owner core — BRISC (NoC0) (tt/mk_mla.py): the attention phase (kernels/mla/att_dm0.cpp: flash-decoding over
// the paged latent cache, head group g, row chunk c) -> mhc_post (packed decode reader: the attention output a from the
// o_proj readers + the stacked residual -> s1). Each phase file is included in its own namespace with its own argument
// bases (kernels/mk/phase_args.hpp; MK_PP_CT / MK_PP_RT = mhc_post's).
#define MK_PHASED 1
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"

namespace pha {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_dm0.cpp"
#undef kernel_main
}  // namespace pha

namespace php {
constexpr uint32_t PH_CT0 = MK_PP_CT;
uint32_t ph_rt0 = MK_PP_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader_p.cpp"
#undef kernel_main
}  // namespace php

void kernel_main() {
    pha::phase_main();
    php::phase_main();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
