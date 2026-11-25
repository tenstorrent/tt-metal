// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused short conv, writer: fp32 x -> ring slot (dense pages), output row (row 0 of a row-form tensor).
#include "rowio.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t row_out = get_named_compile_time_arg_val("row_out");  // 0: dense twin only
    constexpr uint32_t cb_xf = 16;

    const uint32_t slot = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_tile = get_arg_val<uint32_t>(2);
    const uint32_t out_es = get_arg_val<uint32_t>(3);
    const uint32_t out_dense = get_arg_val<uint32_t>(4);  // dense twin of the output (0: none)

    {
        // inputs this RISC reads while the reader gathers x and reads taps / ring. Args 5..: the reader's x args,
        // ring slots at lag 1, 2, 3, taps, resid args.
        constexpr uint32_t has_resid = get_named_compile_time_arg_val("has_resid");
        constexpr uint32_t cb_x = 0, cb_resid = 3, cb_stage = 26,
                           cb_sync = 25;  // own stage: the reader stages tiles too
        uint32_t a = 5;
        const uint32_t x_addr = get_arg_val<uint32_t>(a++);
        const uint32_t x_tile = get_arg_val<uint32_t>(a++);
        const uint32_t x_es = get_arg_val<uint32_t>(a++);
        const uint32_t x_row = get_arg_val<uint32_t>(a++);
        const uint32_t x_dense = get_arg_val<uint32_t>(a++);
        a += 4;  // ring slots, taps: read by the reader
        const uint32_t res_addr = get_arg_val<uint32_t>(a++);
        const uint32_t res_tile = get_arg_val<uint32_t>(a++);
        const uint32_t res_es = get_arg_val<uint32_t>(a++);
        const uint32_t res_row = get_arg_val<uint32_t>(a++);
        const uint32_t res_dense = get_arg_val<uint32_t>(a++);
        const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), rowio::DRAM_RD_ALIGN);

        DeviceZoneScopedN("W_inputs");
        cb_reserve_back(cb_sync, 1);
        if (x_dense == 0) {  // second half of x, into the reader's (not yet pushed) cb_x pages
            rowio::gather_row(
                x_addr, x_tile, x_es, ntc, x_row, get_write_ptr(cb_x), T * 1024 * x_es, stage, ntc / 2, ntc);
        } else if (x_dense == 2) {  // row form in L1
            rowio::gather_row<false>(
                x_addr, x_tile, x_es, ntc, x_row, get_write_ptr(cb_x), T * 1024 * x_es, stage, ntc / 2, ntc);
        }
        cb_push_back(cb_sync, 1);
        if constexpr (has_resid) {
            cb_reserve_back(cb_resid, T);
            if (res_dense) {
                rowio::read_pages(res_addr, res_tile, res_row, T, get_write_ptr(cb_resid));
            } else {
                rowio::gather_row(
                    res_addr, res_tile, res_es, ntc, res_row, get_write_ptr(cb_resid), T * 1024 * res_es, stage);
            }
        }
        noc_async_read_barrier();
        if constexpr (has_resid) {
            cb_push_back(cb_resid, T);
        }
    }
    cb_wait_front(cb_xf, T);
    rowio::write_pages(slot, 4096, 0, T, get_read_ptr(cb_xf));
    {
        DeviceZoneScopedN("W_wait_out");
        cb_wait_front(cb_out, T);
    }
    {
        DeviceZoneScopedN("W_scatter");
        if (out_dense) {
            rowio::write_pages(out_dense, out_tile, 0, T, get_read_ptr(cb_out));
        }
        if constexpr (row_out) {
            rowio::scatter_row(out_addr, out_tile, out_es, ntc, 0, get_read_ptr(cb_out), ntc / 2, ntc);  // reader: rest
        }
        noc_async_write_barrier();
    }
    if constexpr (get_named_compile_time_arg_val("sig")) {
        // joined program (common.py joined): the next part reads this output; its writes have landed (barrier above)
        const uint64_t s = get_noc_addr(
            get_named_compile_time_arg_val("sig_x"), get_named_compile_time_arg_val("sig_y"), get_semaphore(0));
        noc_semaphore_inc(s, 1);
        noc_async_atomic_barrier();
    }
    cb_pop_front(cb_xf, T);
    cb_pop_front(cb_out, T);
}
