// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel MoE FFN (tt/mk_moe.py) — reader core, activation mover (NCRISC, NoC1): the dense reader phases
// (mk/reader_a.cpp: router, shared gate_up | shared down) with the routed-expert phase (mkx/xa.cpp) run between dense
// phase MK_SPLIT - 1 and MK_SPLIT (the dense kernel's MK_BETWEEN hook). Argument bases: the dense phases at 0 / 0,
// the expert phase at MK_X_CT / MK_X_RT.
#define MK_PHASED 1
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/mkx_common.hpp"

#if defined(MKX_XROWS)
#define MKX_XROWS_PRE 1  // the expert phase's x zero fill runs as this composite's prologue (mkx/xa.cpp)
#endif
namespace experts {
constexpr uint32_t PH_CT0 = MK_X_CT;
uint32_t ph_rt0 = MK_X_RT;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/xa.cpp"
#undef kernel_main
}  // namespace experts

#define MK_BETWEEN() ::experts::phase_main()
#if defined(MK_QKV_ALIAS) && defined(MKX_XROWS_PRE)
// 7a GLM53_MK_QKV_ALIAS: the qkv weight ring shares its L1 with the experts' CB_X (mk_sparse.py), so CB_X's zero fill
// waits until the qkv phase (dense phase 0) is consumed. It is issued once this RISC wrote the low-rank phase's (1)
// outputs: then it runs at the start of the o_proj phase's long kda_rec hand-off wait instead of between qkv and
// low-rank (after phase 0 it cost ~11 us of the B=1 qkv path, card1 z07 vs z05). The experts (MK_SPLIT) come later.
#define MK_AFTER_PHASE 1
#define MK_AFTER_HOOK() ::experts::xrows_prologue()
#endif

namespace dense {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/reader_a.cpp"
#undef kernel_main
}  // namespace dense

void kernel_main() {
#if defined(MKX_XROWS_PRE) && !defined(MK_QKV_ALIAS)
    ::experts::xrows_prologue();
#endif
    dense::phase_main();
}
