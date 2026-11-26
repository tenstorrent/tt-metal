// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b stack program (tt/mk_stack.py): the router tail's reader (router_tail/reader.cpp, the standalone kernel of
// 7a's MkSparseLayer tail cores) as a phase inside the layer loop: per layer its own runtime args (the router bias
// differs per layer: kernels/mk/stack.hpp) and the tail CBs RTAIL_CB0 .. + 4 back at their bases.
#define MK_PHASED 1
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/router_tail_common.hpp"

namespace rt {
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#define kernel_main phase_main
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/reader.cpp"
#undef kernel_main
}  // namespace rt

void kernel_main() {
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {
        if (layer > 0) {
            mkstack::load_args(layer);
            mk::cb_rewind_range(RTAIL_CB0, 5);
        }
        rt::phase_main();
    }
}
