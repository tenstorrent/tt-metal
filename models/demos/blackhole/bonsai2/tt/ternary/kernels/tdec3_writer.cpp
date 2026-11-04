// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear v3, writer (BRISC): streams the odd output-tile weight records into c_6 (tdec_stream.hpp),
// then writes this core's output tiles.
#include "tdec_stream.hpp"
#include "core_span.hpp"
#ifndef OUT_PER_NT
#define OUT_PER_NT 1
#endif
// SPLIT_NT > 0: output tile columns from SPLIT_NT on go to a second tensor (runtime arg 5), at column - SPLIT_NT.
#ifndef SPLIT_NT
#define SPLIT_NT 0
#endif

void kernel_main() {
#ifdef COMMON_ARGS
    // common args: w, out, Nt, ncores, grid_x, unit, out2
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    uint32_t first_nt, num_nt;
    core_span(
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        first_nt,
        num_nt);
#define OUT2_ADDR get_common_arg_val<uint32_t>(6)
#else
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_nt = get_arg_val<uint32_t>(2);
    const uint32_t num_nt = get_arg_val<uint32_t>(3);
#define OUT2_ADDR get_arg_val<uint32_t>(5)
#endif
    constexpr uint32_t ppr = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr auto w_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
#if SPLIT_NT > 0
    constexpr auto out2_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // No weight reads until this core's x' and scales have landed (NCRISC pushes c_2 after both): with every core's
    // stream in flight (~8 MB queued at the banks) the small reads waited 17 us behind them.
#ifdef INLINE_SCALES
    cb_wait_front(tt::CBIndex::c_0, 1);  // x' landed (NCRISC pushes c_0 first)
#else
    cb_wait_front(tt::CBIndex::c_2, num_nt * S_PER_NT);
#endif
    const auto w = TensorAccessor(w_args, w_addr, 4096);
#ifdef PAGE_SPLIT
    stream_pages<ppr, NB>(tt::CBIndex::c_6, w, first_nt, num_nt, 1);
#else
    stream_records<ppr, NB>(tt::CBIndex::c_6, w, first_nt, num_nt, 1);
#endif

    const uint32_t bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, bytes);
#if SPLIT_NT > 0
    const auto out2 = TensorAccessor(out2_args, OUT2_ADDR, bytes);
#endif
#ifdef GLU_PAIRS
    // One output tile per (gate, up) record pair.
    for (uint32_t i = 0; i < num_nt / 2; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t nt = first_nt / 2 + i;
#else
    for (uint32_t i = 0; i < num_nt * OUT_PER_NT; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t nt = first_nt + i;
#endif
#if SPLIT_NT > 0
        const uint64_t dst = nt < SPLIT_NT ? out.get_noc_addr(nt) : out2.get_noc_addr(nt - SPLIT_NT);
#else
        const uint64_t dst = out.get_noc_addr(nt);
#endif
        noc_async_write(get_read_ptr(cb_out), dst, bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
