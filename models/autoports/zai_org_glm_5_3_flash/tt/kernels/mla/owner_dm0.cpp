// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MkMla (the whole attention half, ONE program) owner core — BRISC (NoC0) (tt/mk_mla.py): mhc_pre (stream.cpp, K role;
// the root core runs mhc_pre's root kernels) -> the attention phase (kernels/mla/att_dm0.cpp, with DSA (MLA_DSA)
// att_dsa_dm0.cpp: flash-decoding over the paged latent cache, head group g, row chunk c) -> mhc_post (packed decode
// reader: the attention output a from the o_proj readers + the stacked residual -> s1). Each phase file is included in
// its own namespace with its own argument bases (kernels/mk/phase_args.hpp; MK_PA_CT / MK_PA_RT = the attention's,
// MK_PP_CT / MK_PP_RT = mhc_post's).
#define MK_PHASED 1
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"

namespace ph1 {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/stream.cpp"
#undef kernel_main
}  // namespace ph1

namespace pha {
constexpr uint32_t PH_CT0 = MK_PA_CT;
uint32_t ph_rt0 = MK_PA_RT;
#define kernel_main phase_main
#if defined(MLA_DSA_SHARD)  // the sequence-sharded split-K variant (kernels/mla/dsa_shard.hpp)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_dsa_shard_dm0.cpp"
#elif defined(MLA_DSA)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_dsa_dm0.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_dm0.cpp"
#endif
#undef kernel_main
}  // namespace pha

namespace php {
constexpr uint32_t PH_CT0 = MK_PP_CT;
uint32_t ph_rt0 = MK_PP_RT;
#define kernel_main phase_main
#if defined(MLA_POST_TILE)  // t > 4 (tt/mk_mla.py): the per-tile kernels, flat residual / s1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader_p.cpp"
#endif
#undef kernel_main
}  // namespace php

void kernel_main() {
    ph1::phase_main();
    pha::phase_main();
    php::phase_main();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
