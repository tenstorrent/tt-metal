// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// swiglu reader (BRISC / NoC0): waits for the gate_up producers (MK_SWI_WAIT_SEM, count = runtime arg), then reads the
// core's NT gate tiles (columns c0 ..) and NT up tiles (I/32 + c0 ..) of the [1,1,B,2I] bf16 gate_up output.
// Compile-time args (PH_CT0 + i): NT, I_TILES (I / 32), TensorAccessorArgs of gate_up.
// Runtime args (ph_rt0 + i): gate_up address, c0, hand-off count.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/swiglu_common.hpp"

void kernel_main() {
    using namespace swiglu;
    constexpr uint32_t NT = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t I_TILES = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr auto gu_args = TensorAccessorArgs<PH_CT0 + 2>();
    const auto gu = TensorAccessor(gu_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const uint32_t c0 = get_arg_val<uint32_t>(ph_rt0 + 1);
#if defined(MK_SWI_WAIT_SEM)
    mk::wait_handoff(MK_SWI_WAIT_SEM, get_arg_val<uint32_t>(ph_rt0 + 2));
#endif
    cb_reserve_back(CB_G, NT);
    cb_reserve_back(CB_U, NT);
    const uint32_t page = get_tile_size(CB_G);
    for (uint32_t j = 0; j < NT; ++j) {
        noc_async_read_page(c0 + j, gu, get_write_ptr(CB_G) + j * page);
        noc_async_read_page(I_TILES + c0 + j, gu, get_write_ptr(CB_U) + j * page);
    }
    noc_async_read_barrier();
    cb_push_back(CB_G, NT);
    cb_push_back(CB_U, NT);
}
