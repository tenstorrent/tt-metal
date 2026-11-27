// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel owner core — NCRISC (NoC1): P1 mhc_pre sync (K role: leader or member, by MHC_LEADER) -> P3 kda_pre
// writer (it starts on the compute's outputs, so no hand-off of its own; signals the kda_rec head cores when
// MK_KDA_SIGNAL_SEM) [-> P4 kda_rec writer, when MK_P4_CT] [-> P6 mhc_post writer, when MK_P6_CT]. Defines: MK_P3_CT /
// MK_P3_RT, MK_P4_* + the phases'.
#define MK_PHASED 1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if defined(MHC_DMA_ZERO)  // Stage 14c: mhc_pre sync.cpp zero fills (NoC API, at global scope before any phase)
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#endif
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_pre/kda_pre_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/kda_rec_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/swiglu_common.hpp"

namespace ph1 {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/sync.cpp"
#undef kernel_main
}  // namespace ph1

namespace ph3 {
constexpr uint32_t PH_CT0 = MK_P3_CT;
uint32_t ph_rt0 = MK_P3_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_pre/writer.cpp"
#undef kernel_main
}  // namespace ph3

#if defined(MK_P4_CT)
namespace ph4 {
constexpr uint32_t PH_CT0 = MK_P4_CT;
uint32_t ph_rt0 = MK_P4_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/writer.cpp"
#undef kernel_main
}  // namespace ph4
#endif

#if defined(MK_P6_CT)
namespace ph6 {
constexpr uint32_t PH_CT0 = MK_P6_CT;
uint32_t ph_rt0 = MK_P6_RT;
#define kernel_main phase_main
#if defined(MK_POST_PACKED)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/writer_p.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/writer.cpp"
#endif
#undef kernel_main
}  // namespace ph6
#endif

// whole layer (MK_P7_CT): the FFN sublayer after the attention one — P7 mhc_pre (FFN; the attention instance's CB
// indices, rewound), P8 swiglu, P10 mhc_post (FFN; the attention instance's CB indices, rewound)
#if defined(MK_P7_CT)
namespace ph7 {
constexpr uint32_t PH_CT0 = MK_P7_CT;
uint32_t ph_rt0 = MK_P7_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/sync.cpp"
#undef kernel_main
}  // namespace ph7
namespace ph8 {
constexpr uint32_t PH_CT0 = MK_P8_CT;
uint32_t ph_rt0 = MK_P8_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/writer.cpp"
#undef kernel_main
}  // namespace ph8
namespace ph10 {
constexpr uint32_t PH_CT0 = MK_P10_CT;
uint32_t ph_rt0 = MK_P10_RT;
#define kernel_main phase_main
#if defined(MK_POST_PACKED)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/writer_p.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/writer.cpp"
#endif
#undef kernel_main
}  // namespace ph10
#endif

void kernel_main() {
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {  // stack program (kernels/mk/stack.hpp)
        MK_LAYER_BEGIN(layer);
        if (layer > 0) {
            mkstack::load_args(layer);
            mk::cb_rewind_range(0, 29);
            mk::cb_rewind_range(32, 32);
        }
        ph1::phase_main();  // probe: mhc_pre sync's own marks first (role-dependent count)
        MK_MARK();          // mhc_pre done (sync)
        ph3::phase_main();
        MK_MARK();  // kda_pre outputs written (+ head cores signalled)
#if defined(MK_P4_CT)
        ph4::phase_main();
        MK_MARK();  // kda_rec outputs written (+ o_proj readers signalled)
#endif
#if defined(MK_P6_CT)
        ph6::phase_main();
        MK_MARK();  // mhc_post output written
#endif
#if defined(MK_P7_CT)
        mk::signal_self(MK_S1_SEM);        // P6's s1 writes landed (writer barrier) -> the BRISC's P7 may read them
        mk::cb_rewind_range(0, 14);        // mhc_pre (K role) CBs back to their base for the FFN instance
        mk::cb_rewind_range(POST_CB0, 8);  // mhc_post CBs likewise
        ph7::phase_main();
        MK_MARK();  // FFN mhc_pre done (sync)
        ph8::phase_main();
        MK_MARK();  // m written (+ down readers signalled)
        ph10::phase_main();
        MK_MARK();  // layer output written
#if defined(MK_X_SEM)
        if (layer + 1 < MK_LAYERS) {
            mk::signal_self(
                MK_X_SEM);  // P10's residual writes landed (writer barrier) -> the next layer's P1 may read them
        }
#endif
#endif
#if defined(MK_PROBE_ARG)
        MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
    }  // layer
}
