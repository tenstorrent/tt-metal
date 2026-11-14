// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode cross-column merge, reader. x = a2a output [8 src, S, 32, 576] TILE (18 tiles per (src, user): 16 o tiles, tile 16 = scaled max in col 0, tile 17 = local sum in col 0).
// sink [S] tiles (col 0 = sink logit of the head row). This core: user s, output tile columns j0 .. j0+TPC-1.
// CB_ST gets [sink, ms_0..7, l_0..7] (17 tiles), CB_A gets the o tiles ordered (j, src).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t TPC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(2);
    constexpr uint32_t CB_A = get_compile_time_arg_val(3);
    constexpr uint32_t CB_ONE = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5, 2>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset(), x_args.next_common_runtime_args_offset()>();
    constexpr uint32_t NC = 8, TILE_B = 2048, TPR = 18;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const auto xa = TensorAccessor(x_args, x_addr, TILE_B);
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
    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t base = (c * S + s) * TPR;
        noc_async_read(xa.get_noc_addr(base + 16), p + (1 + c) * TILE_B, TILE_B);
        noc_async_read(xa.get_noc_addr(base + 17), p + (1 + NC + c) * TILE_B, TILE_B);
    }
    noc_async_read_barrier();
    st.push_back(1 + 2 * NC);
    a.reserve_back(NC * TPC);
    uint32_t pa = a.get_write_ptr();
    for (uint32_t j = 0; j < TPC; ++j) {
        for (uint32_t c = 0; c < NC; ++c) {
            noc_async_read(xa.get_noc_addr((c * S + s) * TPR + j0 + j), pa + (j * NC + c) * TILE_B, TILE_B);
        }
    }
    noc_async_read_barrier();
    a.push_back(NC * TPC);
}
