// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// One-token GatedDeltaNet step, writer (BRISC): head j's new recurrent state (16 bf16 tiles, c_21; the compute also
// reads them, so neither side pops) and its 4 output tiles (c_16).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t st_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t j = get_arg_val<uint32_t>(2);
    constexpr uint32_t HT = get_compile_time_arg_val(0);
    constexpr auto st_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 2048;
    const auto st = TensorAccessor(st_args, st_addr, TB);
    const auto out = TensorAccessor(out_args, out_addr, TB);
    constexpr uint32_t cb_snew = 21, cb_out = 16;

    cb_wait_front(cb_snew, HT * HT);
    const uint32_t s = get_read_ptr(cb_snew);
    for (uint32_t t = 0; t < HT * HT; ++t) {
        noc_async_write(s + t * TB, st.get_noc_addr(j * HT * HT + t), TB);
    }
    cb_wait_front(cb_out, HT);
    const uint32_t o = get_read_ptr(cb_out);
    for (uint32_t i = 0; i < HT; ++i) {
        noc_async_write(o + i * TB, out.get_noc_addr(j * HT + i), TB);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, HT);
}
