// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// swiglu writer (NCRISC / NoC1): the core's NT m tiles into pages c0 .. of the [1,1,B,I] bf16 down-projection input;
// megakernel (MK_SWI_SIGNAL_SEM): then +1 on each consumer (the down readers) after the writes landed.
// Compile-time args (PH_CT0 + i): NT, TensorAccessorArgs of m.
// Runtime args (ph_rt0 + i): m address, c0, consumers n, then n NoC (x, y).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/swiglu_common.hpp"

void kernel_main() {
    using namespace swiglu;
    constexpr uint32_t NT = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr auto m_args = TensorAccessorArgs<PH_CT0 + 1>();
    const auto m = TensorAccessor(m_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const uint32_t c0 = get_arg_val<uint32_t>(ph_rt0 + 1);
    for (uint32_t j = 0; j < NT; ++j) {
        cb_wait_front(CB_M, 1);
        noc_async_write_page(c0 + j, m, get_read_ptr(CB_M));
        noc_async_write_barrier();
        cb_pop_front(CB_M, 1);
    }
#if defined(MK_SWI_SIGNAL_SEM)
    mk::signal_handoff(MK_SWI_SIGNAL_SEM, ph_rt0 + 3, get_arg_val<uint32_t>(ph_rt0 + 2));
#endif
}
