// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert end pass, writer: each token's y row (the first H bf16 of its 4 packed tiles in CB 16) to page t of the
// row-major y. Stamps (RT 3): 5 cycles waiting for a packed row, 6 for the writes to leave.
// RT: 0 y addr, 1 first token, 2 tokens, 3 stamps. CT: 0 row bytes, then TensorAccessorArgs(y).
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr auto yargs = TensorAccessorArgs<1>();
    constexpr uint32_t out_cb = tt::CBIndex::c_16, TPR = 4;
    const uint32_t t0 = get_arg_val<uint32_t>(1), nt = get_arg_val<uint32_t>(2);
    const auto ys = TensorAccessor(yargs, get_arg_val<uint32_t>(0), row_bytes);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    uint32_t t_wait = 0, t_flush = 0;
    for (uint32_t t = t0; t < t0 + nt; ++t) {
        uint32_t a = wall();
        cb_wait_front(out_cb, TPR);
        t_wait += wall() - a;
        noc_async_write_page(t, ys, get_read_ptr(out_cb));
        a = wall();
        noc_async_writes_flushed();
        t_flush += wall() - a;
        cb_pop_front(out_cb, TPR);
    }
    noc_async_write_barrier();
    stamps[5] = t_wait;
    stamps[6] = t_flush;
}
