// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFlash2 fused grouped dynamic conv -- reader (BRISC, NoC0): the constants once (J[2 side], J[2 side + 1], SH, a zero
// tile), then per column tile c: X[c], CF[q0(tap 0) / 32], CF[q0(tap 1) / 32], E[q0(tap 0) % 32], E[q0(tap 1) % 32],
// B[c]. See df2_conv_common.hpp. DF2_CONV_RES: X[c] is read as the fp32 partial into CB_X32 (the compute rounds it
// into CB_X) and the residual tile RES[c] into CB_RES (runtime arg 10 = its address, its accessor after SH's).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "df2_conv_common.hpp"

using namespace df2conv;

void kernel_main() {
    constexpr uint32_t GPT = get_compile_time_arg_val(0);  // groups per column tile (32 / G)
    constexpr uint32_t NG = get_compile_time_arg_val(1);   // groups (H / G)
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto cf_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<cf_args.next_compile_time_args_offset()>();
    constexpr auto j_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<j_args.next_compile_time_args_offset()>();
    constexpr auto sh_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
#if defined(DF2_CONV_RES)
    constexpr auto res_args = TensorAccessorArgs<sh_args.next_compile_time_args_offset()>();
    const auto res = TensorAccessor(res_args, get_arg_val<uint32_t>(10));  // arg 10 = the residual (bf16)
#endif
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));  // (DF2_CONV_RES: the fp32 partial, 4-KiB pages)
    const auto cf = TensorAccessor(cf_args, get_arg_val<uint32_t>(1));
    const auto e = TensorAccessor(e_args, get_arg_val<uint32_t>(2));
    const auto j = TensorAccessor(j_args, get_arg_val<uint32_t>(3));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(4));
    const auto sh = TensorAccessor(sh_args, get_arg_val<uint32_t>(5));
    // arg 6 = out (the writer's)
    const uint32_t c0 = get_arg_val<uint32_t>(7);
    const uint32_t n = get_arg_val<uint32_t>(8);
    const uint32_t side = get_arg_val<uint32_t>(9);

    // the constants of this call
    cb_reserve_back(CB_J, 2);
    noc_async_read_page(2 * side, j, get_write_ptr(CB_J));
    noc_async_read_page(2 * side + 1, j, get_write_ptr(CB_J) + TILE_BYTES);
    cb_reserve_back(CB_SH, 1);
    noc_async_read_page(0, sh, get_write_ptr(CB_SH));
    cb_reserve_back(CB_ZERO, 1);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, get_write_ptr(CB_ZERO) + off, MEM_ZEROS_SIZE);
        }
    }
    noc_async_read_barrier();
    cb_push_back(CB_J, 2);
    cb_push_back(CB_SH, 1);
    cb_push_back(CB_ZERO, 1);

    for (uint32_t c = c0; c < c0 + n; ++c) {
        cb_reserve_back(CB_X, 1);
        cb_reserve_back(CB_CF, 2);
        cb_reserve_back(CB_E, 2);
        cb_reserve_back(CB_B, 1);
#if defined(DF2_CONV_RES)
        cb_reserve_back(CB_X32, 1);
        cb_reserve_back(CB_RES, 1);
        noc_async_read_page(c, x, get_write_ptr(CB_X32));
        noc_async_read_page(c, res, get_write_ptr(CB_RES));
#else
        noc_async_read_page(c, x, get_write_ptr(CB_X));
#endif
        noc_async_read_page(c, b, get_write_ptr(CB_B));
        for (uint32_t t = 0; t < 2; ++t) {
            const uint32_t q0 = side * 2 * NG + t * NG + c * GPT;
            noc_async_read_page(q0 >> 5, cf, get_write_ptr(CB_CF) + t * TILE_BYTES);
            noc_async_read_page(q0 & 31u, e, get_write_ptr(CB_E) + t * TILE_BYTES);
        }
        noc_async_read_barrier();
#if defined(DF2_CONV_RES)
        cb_push_back(CB_X32, 1);
        cb_push_back(CB_RES, 1);
#else
        cb_push_back(CB_X, 1);
#endif
        cb_push_back(CB_CF, 2);
        cb_push_back(CB_E, 2);
        cb_push_back(CB_B, 1);
    }
}
