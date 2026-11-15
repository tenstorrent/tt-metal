// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Rows-phase reduce, reducer core (BRISC): writes the reduced T rows of Q tile columns into rows 0..T-1 of the output
// [32 x 6144] TILE DRAM tensor (the input of the column one-shot all-reduce).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
    constexpr uint32_t Q = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t TILEB = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(0);
    const uint32_t off = get_arg_val<uint32_t>(1);  // tile row of this chip's token block inside the wire tile (0 or 1)
    const auto out_acc = TensorAccessor(out_args, out_addr, 2048);
    {
        DeviceZoneScopedN("P_WAITOUT");
        cb_wait_front(CB_OUT, Q);
    }
    const uint32_t src = get_read_ptr(CB_OUT);
    for (uint32_t i = 0; i < Q; ++i) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write(src + i * TILEB + f * 32 * H + off * 32, out_acc.get_noc_addr(p * Q + i, f * 512), 32 * T);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, Q);
}
