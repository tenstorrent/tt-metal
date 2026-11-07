// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Spins until `ticks` wall-clock ticks have passed and writes [t_start, t_end] to page 0 of the stats tensor.
// The host times two spin lengths to get the wall-clock frequency.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t ticks = get_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr auto st_args = TensorAccessorArgs<0>();
    const auto st = TensorAccessor(st_args, stats_addr);

    // 32-bit modular arithmetic: the 64-bit read is not atomic across a low-word wrap.
    const uint32_t t_start = get_timestamp_32b();
    uint32_t t_end = t_start;
    while (t_end - t_start < ticks) {
        t_end = get_timestamp_32b();
    }
    cb_reserve_back(cb_stats, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stats));
    for (uint32_t i = 0; i < 16; ++i) {
        s[i] = 0;
    }
    s[0] = t_start;
    s[2] = t_end;
    noc_async_write(get_write_ptr(cb_stats), st.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
