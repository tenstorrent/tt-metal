// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel MoE FFN (tt/mk_moe.py) — owner core dm0: P1 mhc_pre (FFN) -> P5 router tail (owners 0 .. T-1, one token
// row each; the others skip it) -> P8 swiglu (shared expert) -> P10 mhc_post (routed + shared). Defines: MK_P5_CT /
// MK_P5_RT, MK_P8_*, MK_P10_* + the phases'.
#define MK_PHASED 1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/swiglu_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/router_tail_common.hpp"

namespace ph1 {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/stream.cpp"
#undef kernel_main
}  // namespace ph1

namespace ph5 {
constexpr uint32_t PH_CT0 = MK_P5_CT;
uint32_t ph_rt0 = MK_P5_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/reader.cpp"
#undef kernel_main
}  // namespace ph5

namespace ph8 {
constexpr uint32_t PH_CT0 = MK_P8_CT;
uint32_t ph_rt0 = MK_P8_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/reader.cpp"
#undef kernel_main
}  // namespace ph8

namespace ph10 {
constexpr uint32_t PH_CT0 = MK_P10_CT;
uint32_t ph_rt0 = MK_P10_RT;
#define kernel_main phase_main
#if defined(MK_POST_PACKED)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader_p.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader.cpp"
#endif
#undef kernel_main
}  // namespace ph10

void kernel_main() {
    ph1::phase_main();
    MK_MARK();  // mhc_pre (FFN) streamed
    ph5::phase_main();
    MK_MARK();  // router tail row done
    ph8::phase_main();
    MK_MARK();  // swiglu inputs read (incl. the gate_up hand-off wait)
    ph10::phase_main();
    MK_MARK();  // mhc_post inputs read (incl. the down hand-off wait)
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
