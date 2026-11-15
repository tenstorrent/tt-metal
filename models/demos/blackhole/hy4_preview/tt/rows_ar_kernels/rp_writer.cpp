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
    constexpr uint32_t NSIG = get_compile_time_arg_val(5);  // FAR cores to signal after the output is written (0: none)
    constexpr uint32_t SIGSEM = get_compile_time_arg_val(6);  // local semaphore id on those cores
    constexpr auto out_args = TensorAccessorArgs<7>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(0);
    const uint32_t off = get_arg_val<uint32_t>(1);  // tile row of this chip's token block inside the wire tile (0 or 1)
    const auto out_acc = TensorAccessor(out_args, out_addr, 2048);
#ifdef RP_PK
    constexpr uint32_t NTC = 1;
#else
    constexpr uint32_t NTC = Q;
#endif
    {
        DeviceZoneScopedN("P_WAITOUT");
        cb_wait_front(CB_OUT, NTC);
    }
    const uint32_t src = get_read_ptr(CB_OUT);
#ifdef RP_PK
    for (uint32_t i = 0; i < Q; ++i) {
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t y = i * T + t;
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(
                    src + ((y >> 4) * 2 + f) * 512 + (y & 15) * 32,
                    out_acc.get_noc_addr(p * Q + i, f * 512 + t * 32),
                    32);
            }
        }
    }
#else
    for (uint32_t i = 0; i < Q; ++i) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write(src + i * TILEB + f * 32 * H + off * 32, out_acc.get_noc_addr(p * Q + i, f * 512), 32 * T);
        }
    }
#endif
    noc_async_write_barrier();
    for (uint32_t k = 0; k < NSIG;
         ++k) {  // the DRAM output of this chunk is complete: release the one-shot all-reduce readers of this program
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(2 + 2 * k), get_arg_val<uint32_t>(3 + 2 * k), get_semaphore(SIGSEM)), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(CB_OUT, NTC);
}
