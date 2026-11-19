// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 prefill mHC post-mix — writer (mhc_post_pf_common.hpp): the unit's 4 output tiles into the flat residual
// [1, 1, T, 4 * 4096] (stream s, column tile c of row tile r = tile r * 4 CT + s CT + c). Compile-time: CT. Runtime:
// out address, first unit, unit count.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_pf_common.hpp"

using namespace mpp;

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    if (n == 0) {
        return;
    }
    const InterleavedAddrGen<true> og{.bank_base_address = out_addr, .page_size = FP32_TILE};
    for (uint32_t u = u0; u < u0 + n; ++u) {
        const uint32_t r = u / CT, c = u % CT;
        cb_wait_front(CB_OUT, N_OUT);
        const uint32_t rd = get_read_ptr(CB_OUT);
        for (uint32_t s = 0; s < N_OUT; ++s) {  // MPP_PRE: h tile r CT + c; post: stream s of the flat residual
            noc_async_write(rd + s * FP32_TILE, og.get_noc_addr(r * N_OUT * CT + s * CT + c), FP32_TILE);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, N_OUT);
    }
}
