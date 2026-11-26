// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel owner core — BRISC (NoC0), the owner phases in order, each the unmodified phase kernel included in its
// own namespace with its own argument bases (PH_CT0 / ph_rt0; kernels/mk/phase_args.hpp):
//   P1 mhc_pre stream (K role)  ->  [wait: the readers' qkv + low-rank hand-off]  ->  P3 kda_pre reader
//   [-> P4 kda_rec reader (it waits for the kda_pre writers' hand-off itself), when MK_P4_CT is defined]
//   [-> P6 mhc_post reader (attention post-mix; waits for the o_proj readers before reading y), when MK_P6_CT].
// Defines (host): MK_P3_CT / MK_P3_RT, MK_P4_CT / MK_P4_RT (argument bases), MK_SEM_QKV / MK_QKV_COUNT (hand-off),
// plus the phases' own defines (MHC_K, KDA_CB0, REC_CB0, ...).
#define MK_PHASED 1
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_pre/kda_pre_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/kda_rec_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/swiglu_common.hpp"

// Stage 14 (MK_PRE_PREFETCH): the layer index for P1's residual hand-off hook (mhc_pre/stream.cpp MK_STREAM_XWAIT)
uint32_t g_mk_layer = 0;

namespace ph1 {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#if defined(MK_PRE_PREFETCH) && defined(MK_X_SEM)
#define MK_STREAM_XWAIT()                  \
    do {                                   \
        if (g_mk_layer > 0) {              \
            mk::wait_handoff(MK_X_SEM, 1); \
        }                                  \
    } while (0)
#elif defined(MK_PRE_PREFETCH)
#define MK_STREAM_XWAIT() \
    do {                  \
    } while (0)  // a single-layer program: the input x is in place at launch
#endif
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/stream.cpp"
#if defined(MK_STREAM_XWAIT)
#undef MK_STREAM_XWAIT
#endif
#undef kernel_main
}  // namespace ph1

namespace ph3 {
constexpr uint32_t PH_CT0 = MK_P3_CT;
uint32_t ph_rt0 = MK_P3_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_pre/reader.cpp"
#undef kernel_main
}  // namespace ph3

#if defined(MK_P4_CT)
namespace ph4 {
constexpr uint32_t PH_CT0 = MK_P4_CT;
uint32_t ph_rt0 = MK_P4_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/reader.cpp"
#undef kernel_main
}  // namespace ph4
#endif

#if defined(MK_P6_CT)
namespace ph6 {
constexpr uint32_t PH_CT0 = MK_P6_CT;
uint32_t ph_rt0 = MK_P6_RT;
#define kernel_main phase_main
#if defined(MK_POST_PACKED)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader_p.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader.cpp"
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
#if defined(MK_PRE_PREFETCH)
#define MK_STREAM_XWAIT() mk::wait_handoff(MK_S1_SEM, 1)  // this core's s1 columns written by its NCRISC (P6)
#endif
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/stream.cpp"
#if defined(MK_STREAM_XWAIT)
#undef MK_STREAM_XWAIT
#endif
#undef kernel_main
}  // namespace ph7
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
#define MK_POST_SECOND 1  // Stage 14: the FFN instance (its own wait count MK_POST2_WAIT_COUNT, if given)
#if defined(MK_POST_PACKED)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader_p.cpp"
#else
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/reader.cpp"
#endif
#undef MK_POST_SECOND
#undef kernel_main
}  // namespace ph10
#endif

void kernel_main() {
#if defined(MK_PARK_SEM)
    // forwarding ring: publish this core's park region (MK_PARK_CB: never touched by the owner phases) to the readers
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MK_PARK_SEM)) = get_write_ptr(MK_PARK_CB);
#endif
    // stack program (MK_LAYERS > 1, kernels/mk/stack.hpp): the whole layer below once per layer; layer l > 0 starts
    // when this core's NCRISC wrote its residual columns of layer l - 1 (P10 -> res, MK_X_SEM), with layer l's
    // runtime args and every CB of the owner phases back at its base (all empty at a layer boundary)
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {
        MK_LAYER_BEGIN(layer);
        g_mk_layer = layer;
        if (layer > 0) {
            mkstack::load_args(layer);
#if defined(MK_X_SEM) && !defined(MK_PRE_PREFETCH)
            mk::wait_handoff(MK_X_SEM, 1);
#endif
            // MK_PRE_PREFETCH: rewound BEFORE the residual hand-off — this RISC has no read in flight into any of these
            // CBs (P10's reads were pushed) and P1 fills only CB_FN / CB_W before the hand-off (P7, their last user,
            // was consumed before P10's down hand-off could pass); P1 waits MK_X_SEM itself before its x reads
            mk::cb_rewind_range(0, 29);
            mk::cb_rewind_range(32, 32);
        }
        ph1::phase_main();  // probe marks 0..2 (mhc_pre stream: start, inputs in L1, hn written)
#if !defined(KDA_PREFETCH)  // (Stage 14 KDA_PREFETCH: the kda_pre reader waits itself, after its constant reads)
        mk::wait_handoff(MK_SEM_QKV, MK_QKV_COUNT);
#endif
        MK_MARK();  // 3: qkv + low-rank landed (KDA_PREFETCH: kda_pre starts)
        ph3::phase_main();
        MK_MARK();  // 4: kda_pre inputs read
#if defined(MK_P4_CT)
        ph4::phase_main();
        MK_MARK();  // 5: kda_rec inputs read (incl. the kda_pre hand-off wait)
#endif
#if defined(MK_P6_CT)
        ph6::phase_main();
        MK_MARK();  // 6: mhc_post inputs read (incl. the o_proj hand-off wait)
#endif
#if defined(MK_P7_CT)
        mk::cb_rewind_range(0, 14);        // mhc_pre (K role) CBs back to their base for the FFN instance
        mk::cb_rewind_range(POST_CB0, 8);  // mhc_post CBs likewise
#if !defined(MK_PRE_PREFETCH)
        mk::wait_handoff(MK_S1_SEM, 1);  // this core's s1 columns written by its NCRISC (P6) -> P7 reads them
#endif
        // (MK_PRE_PREFETCH: P7's stream waits MK_S1_SEM itself, after its s1-independent reads; P7 fills CB_FN / CB_W
        // first, whose last user, the attention P1, was consumed before the qkv hand-off)
        ph7::phase_main();
        MK_MARK();  // 7: FFN mhc_pre streamed
        ph8::phase_main();
        MK_MARK();  // 8: swiglu inputs read (incl. the gate_up hand-off wait)
        ph10::phase_main();
        MK_MARK();  // 9: FFN mhc_post inputs read (incl. the down hand-off wait)
#endif
#if defined(MK_PROBE_ARG)
        MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
    }  // layer
}
