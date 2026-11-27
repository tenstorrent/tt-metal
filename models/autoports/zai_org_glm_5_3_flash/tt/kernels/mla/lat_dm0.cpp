// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel latent core — BRISC (NoC0): wait for the q_a | kv_a readers' hand-off, then pull the KX kv_a tiles
// (column tiles [KOFF, KOFF + KX) of the projection output) into CB_X and the RMSNorm weight tiles (fp32, row 0 = w)
// into CB_NW for the compute (kernels/mla/lat_c.cpp).
// Compile-time args: CB_X, CB_NW, KX, KOFF, WAIT_SEM, WAIT_COUNT, ON (output tiles per core: its norm-weight tiles),
//   then the TensorAccessorArgs of x and of the weight.
// Runtime args: x address, weight address, OOFF (this core's output slice [OOFF, OOFF + ON) of the KX tiles).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"

void kernel_main() {
    constexpr uint32_t CB_X = get_compile_time_arg_val(0);
    constexpr uint32_t CB_NW = get_compile_time_arg_val(1);
    constexpr uint32_t KX = get_compile_time_arg_val(2);
    constexpr uint32_t KOFF = get_compile_time_arg_val(3);
    constexpr uint32_t WAIT_SEM = get_compile_time_arg_val(4);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(5);
    constexpr uint32_t ON = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const uint32_t OOFF = get_arg_val<uint32_t>(2);
    MK_MARK();
    // the weight does not depend on the hand-off: in L1 before the wait
    if constexpr (ON > 0) {  // ON = 0: the inv-only q-norm core (no weight)
        cb_reserve_back(CB_NW, ON);
        const uint32_t wl1 = get_write_ptr(CB_NW);
        for (uint32_t k = 0; k < ON; ++k) {
            noc_async_read_page(OOFF + k, w, wl1 + k * get_tile_size(CB_NW));
        }
        noc_async_read_barrier();
        cb_push_back(CB_NW, ON);
    }
    mk::wait_handoff(WAIT_SEM, WAIT_COUNT);
    MK_MARK();
    cb_reserve_back(CB_X, KX);
    const uint32_t l1 = get_write_ptr(CB_X);
    for (uint32_t k = 0; k < KX; ++k) {
        noc_async_read_page(KOFF + k, x, l1 + k * get_tile_size(CB_X));
    }
    noc_async_read_barrier();
    cb_push_back(CB_X, KX);
    MK_MARK();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
