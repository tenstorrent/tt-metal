// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear v3, reader (NCRISC). Loads the rotated activation x' (K bf16 = Kt tiles of [1,32], 64 B each,
// one contiguous read of this core's copy: x' is stored once per DRAM bank, as 130 cores reading one page from one
// bank took 18.7 us) into c_0 and this core's fp16 scale records (KB tiles of [1,32] per output tile column) into
// c_2, then streams the even output-tile weight records into c_1 (tdec_stream.hpp).
#include "tdec_stream.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t first_nt = get_arg_val<uint32_t>(3);
    const uint32_t num_nt = get_arg_val<uint32_t>(4);
    const uint32_t x_copy = get_arg_val<uint32_t>(5);  // which DRAM-bank copy of x' this core reads
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

    const auto xs = TensorAccessor(x_args, x_addr, X_BYTES);
    cb_reserve_back(cb_x, Kt);
    noc_async_read(xs.get_noc_addr(x_copy), get_write_ptr(cb_x), X_BYTES);
#ifdef INLINE_SCALES
    noc_async_read_barrier();
    cb_push_back(cb_x, Kt);
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
    stream_records<ppr, NB>(tt::CBIndex::c_1, w, first_nt, num_nt, 0);
}
