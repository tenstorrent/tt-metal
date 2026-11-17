// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert fold, end core writer (NCRISC, NOC1): each of this end core's tokens' y row (the first H bf16 of its 4
// packed tiles in CB out) to page t of the row-major y, posted, in wf_reader.cpp's token order; fenced at the end by an
// acknowledged 4-B write into every DRAM bank (the fence tensor's pages) on the same NoC and VC.
// RT: 0 y addr, 1 c, 2 e, 3 stamps (5: cycles waiting for a packed row, 6: for the writes to leave), 4 fence addr.
// CT: 0 row bytes, 1 NE, 2 NCL, 3 row tiles, 4 T, 5 CB out, then TensorAccessorArgs(y), (fence).
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t NE = get_compile_time_arg_val(1);
    constexpr uint32_t NCL = get_compile_time_arg_val(2);
    constexpr uint32_t R = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr auto yargs = TensorAccessorArgs<6>();
    constexpr auto fargs = TensorAccessorArgs<yargs.next_compile_time_args_offset()>();
    constexpr uint32_t TPR = 4;
    const uint32_t c = get_arg_val<uint32_t>(1), e = get_arg_val<uint32_t>(2);
    const auto ys = TensorAccessor(yargs, get_arg_val<uint32_t>(0), row_bytes);
    const auto fs = TensorAccessor(fargs, get_arg_val<uint32_t>(4), 64);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    uint32_t t_wait = 0, t_flush = 0;
    for (uint32_t k = 0; c + NCL * (e + NE * k) < R; ++k) {
        const uint32_t t0 = (c + NCL * (e + NE * k)) * 32;
        for (uint32_t t = t0; t < t0 + 32 && t < T; ++t) {
            uint32_t a = wall();
            cb_wait_front(out_cb, TPR);
            t_wait += wall() - a;
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(get_read_ptr(out_cb), ys.get_noc_addr(t), row_bytes);
            a = wall();
            noc_async_posted_writes_flushed();
            t_flush += wall() - a;
            cb_pop_front(out_cb, TPR);
        }
    }
    // the fence: every posted y row landed when its acks are back
    for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
        noc_async_write(get_read_ptr(out_cb), fs.get_noc_addr(i), 4);
    }
    noc_async_write_barrier();
    stamps[5] = t_wait;
    stamps[6] = t_flush;
}
