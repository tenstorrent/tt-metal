// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47a XPUSH merge reader: as hy4_merge_kernels/mrg_reader.cpp (TR = 1) but the 8 source partials come from the push
// landing tensor (DRAM, 4 KiB pages, page c*9 + p = tiles 2p, 2p+1 of source c; page c*9+8 = ms, l) after the
// per-source flags reached this core's launch counter + 9 (flags are never reset).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t TPC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(2);
    constexpr uint32_t CB_A = get_compile_time_arg_val(3);
    constexpr uint32_t CB_ONE = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5, 2>();
    constexpr auto s_args =
        TensorAccessorArgs<x_args.next_compile_time_args_offset(), x_args.next_common_runtime_args_offset()>();
    constexpr uint32_t NC = 8, TILE_B = 2048, NPK = 9;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const uint32_t fx = get_arg_val<uint32_t>(2), fy = get_arg_val<uint32_t>(3);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(4);
    const auto xa = TensorAccessor(x_args, x_addr, 2 * TILE_B);
    const auto sa = TensorAccessor(s_args, s_addr, TILE_B);
    CircularBuffer st(CB_ST), a(CB_A), one(CB_ONE);
    one.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* po = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one.get_write_ptr());
        for (uint32_t i = 0; i < TILE_B / 4; ++i) {
            po[i] = 0x3F803F80u;  // bf16 1.0 x 2
        }
    }
    one.push_back(1);
    st.reserve_back(1 + 2 * NC);
    uint32_t p = st.get_write_ptr();
    noc_async_read(sa.get_noc_addr(s), p, TILE_B);
    // flag wait: the sink read is in flight; poll the 8 source flags (scratch = the last ST tile slot, rewritten by the
    // reads below)
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_addr);
    const uint32_t target = *cnt + NPK;
    *cnt = target;
    noc_async_read_barrier();
    const uint32_t scr = p + (2 * NC) * TILE_B;
    volatile tt_l1_ptr uint32_t* scr_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    for (uint32_t c = 0; c < NC; ++c) {
        const uint64_t flag_noc = get_noc_addr(fx, fy, get_arg_val<uint32_t>(5 + c));
        do {
            noc_async_read(flag_noc, scr, 4);
            noc_async_read_barrier();
        } while ((int32_t)(scr_p[0] - target) < 0);
    }
    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t pg = c * NPK + 8;
        noc_async_read(xa.get_noc_addr(pg), p + (1 + c) * TILE_B, TILE_B);
        noc_async_read(xa.get_noc_addr(pg, TILE_B), p + (1 + NC + c) * TILE_B, TILE_B);
    }
    noc_async_read_barrier();
    st.push_back(1 + 2 * NC);
    a.reserve_back(NC * TPC);
    uint32_t pa = a.get_write_ptr();
    for (uint32_t j = 0; j < TPC; ++j) {
        const uint32_t t = j0 + j;
        for (uint32_t c = 0; c < NC; ++c) {
            noc_async_read(xa.get_noc_addr(c * NPK + (t >> 1), (t & 1) * TILE_B), pa + (j * NC + c) * TILE_B, TILE_B);
        }
    }
    noc_async_read_barrier();
    a.push_back(NC * TPC);
}
