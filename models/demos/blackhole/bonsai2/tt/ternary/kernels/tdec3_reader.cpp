// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear v3, reader (NCRISC). Loads the rotated activation x' (K bf16 = Kt tiles of [1,32], 64 B each,
// one contiguous read of this core's copy: x' is stored once per DRAM bank, as 130 cores reading one page from one
// bank took 18.7 us) into c_0 and this core's fp16 scale records (KB tiles of [1,32] per output tile column) into
// c_2, then streams the even output-tile weight records into c_1 (tdec_stream.hpp).
#include "tdec_stream.hpp"
#include "core_span.hpp"

void kernel_main() {
#ifdef COMMON_ARGS
    // common args: x, s, w, Nt, ncores, grid_x, unit, copies
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    uint32_t first_nt, num_nt;
    core_span(
        get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        get_common_arg_val<uint32_t>(6),
        first_nt,
        num_nt);
    const uint32_t x_copy = core_index(get_common_arg_val<uint32_t>(5)) % get_common_arg_val<uint32_t>(7);
#else
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t first_nt = get_arg_val<uint32_t>(3);
    const uint32_t num_nt = get_arg_val<uint32_t>(4);
    const uint32_t x_copy = get_arg_val<uint32_t>(5);  // which DRAM-bank copy of x' this core reads
#endif
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t ppr = get_compile_time_arg_val(2);
    constexpr uint32_t NB = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_s = tt::CBIndex::c_2;
    constexpr uint32_t X_BYTES = Kt * 64;
#ifdef S_RECORD_BYTES
    constexpr uint32_t S_BYTES = S_RECORD_BYTES;  // scale record per output tile column
    constexpr uint32_t S_PAGES = S_RECORD_BYTES / 2048;
#else
    constexpr uint32_t S_BYTES = KB * S_TILE_BYTES;
    constexpr uint32_t S_PAGES = KB;
#endif

#ifdef X_PAGED
    // x' as a [K / 1024, 1024] L1 tensor: its 2-KB pages from the cores' L1 (no DRAM bank queue, one copy)
    const auto xs = TensorAccessor(x_args, x_addr, 2048);
    cb_reserve_back(cb_x, Kt);
    (void)x_copy;
    for (uint32_t p = 0; p < Kt / 32; ++p) {
        noc_async_read(xs.get_noc_addr(p), get_write_ptr(cb_x) + p * 2048, 2048);
    }
#else
    const auto xs = TensorAccessor(x_args, x_addr, X_BYTES);
    cb_reserve_back(cb_x, Kt);
    noc_async_read(xs.get_noc_addr(x_copy), get_write_ptr(cb_x), X_BYTES);
#endif
#ifdef RESIDUAL
    {
        // RESIDUAL: this core's residual tiles (same layout as the output) into c_3; the compute adds them to the
        // accumulators before the one bf16 rounding (decode_linear3.py residual=)
        constexpr auto r_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
        const auto rs = TensorAccessor(r_args, get_common_arg_val<uint32_t>(8), 2048);
        cb_reserve_back(tt::CBIndex::c_3, num_nt);
        for (uint32_t i = 0; i < num_nt; ++i) {
            noc_async_read(rs.get_noc_addr(first_nt + i), get_write_ptr(tt::CBIndex::c_3) + i * 2048, 2048);
        }
    }
#endif
#ifdef INLINE_SCALES
#ifndef X_PAGED
    noc_async_read_barrier();
    cb_push_back(cb_x, Kt);
#ifdef RESIDUAL
    cb_push_back(tt::CBIndex::c_3, num_nt);
#endif
#endif
#else
    const auto ss = TensorAccessor(s_args, s_addr, S_BYTES);
    cb_reserve_back(cb_s, num_nt * S_PAGES);
    uint32_t l1 = get_write_ptr(cb_s);
    for (uint32_t i = 0; i < num_nt; ++i) {
        noc_async_read(ss.get_noc_addr(first_nt + i), l1, S_BYTES);
        l1 += S_BYTES;
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, Kt);
    cb_push_back(cb_s, num_nt * S_PAGES);  // also BRISC's go signal (tdec3_writer.cpp)
#endif

    const auto w = TensorAccessor(w_args, w_addr, 4096);
#ifdef PAGE_SPLIT
    stream_pages<ppr, NB>(tt::CBIndex::c_1, w, first_nt, num_nt, 0);
#else
#if defined(X_PAGED) && defined(INLINE_SCALES)
    // x' (and the residual) come from L1: start the weight stream at once, push them when they land
#ifdef RESIDUAL
    stream_records<ppr, NB>(tt::CBIndex::c_1, w, first_nt, num_nt, 0, x_copy, cb_x, Kt, tt::CBIndex::c_3, num_nt);
#else
    stream_records<ppr, NB>(tt::CBIndex::c_1, w, first_nt, num_nt, 0, x_copy, cb_x, Kt);
#endif
#else
    stream_records<ppr, NB>(tt::CBIndex::c_1, w, first_nt, num_nt, 0, x_copy);
#endif
#endif
}
