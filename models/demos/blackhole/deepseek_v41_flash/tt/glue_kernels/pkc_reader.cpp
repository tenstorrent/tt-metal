// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// T19 packed-mHC collapse (+ sum of squares), reader.  Same streaming as pk_comb_reader.cpp (NCA = 4 pre tiles, no post
// / y) and, additionally, the "ones in column 0" tile of the sum-of-squares matmul.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_i = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ct = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(3);
    constexpr uint32_t NCA = get_compile_time_arg_val(4);
    constexpr uint32_t CT = get_compile_time_arg_val(5);  // column tiles per stream
    constexpr uint32_t BLK = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto a_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;
    constexpr uint32_t XT = 4 * CT;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto a_acc = TensorAccessor(a_args, a_addr, TILE);
    experimental::CB cbi(cb_i), cbct(cb_ct), cbx(cb_x), cbo(cb_ones);

    cbi.reserve_back(1);
    noc.async_write_zeros(cbi, TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbi.get_write_ptr());
        for (uint32_t k = 0; k < 32; ++k) {
            const uint32_t face = (k >= 16) ? 3 : 0;
            o[face * 256 + (k % 16) * 16 + (k % 16)] = 0x3F800000u;
        }
    }
    cbi.push_back(1);

    cbo.reserve_back(1);
    noc.async_write_zeros(cbo, TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
        for (uint32_t k = 0; k < 32; ++k) {
            const uint32_t face = (k >= 16) ? 2 : 0;  // column 0
            o[face * 256 + (k % 16) * 16] = 0x3F800000u;
        }
    }
    cbo.push_back(1);

    cbct.reserve_back(NCA);
    for (uint32_t q = 0; q < NCA; ++q) {
        noc.async_read(a_acc, cbct, TILE, {.page_id = r * NCA + q, .offset_bytes = 0}, {.offset_bytes = q * TILE});
    }
    noc.async_read_barrier();
    cbct.push_back(NCA);

    for (uint32_t b = 0; b < ncols; b += BLK) {
        cbx.reserve_back(4 * BLK);
        for (uint32_t ci = 0; ci < BLK; ++ci) {
            const uint32_t c = c0 + b + ci;
            for (uint32_t i = 0; i < 4; ++i) {
                noc.async_read(
                    x_acc,
                    cbx,
                    TILE,
                    {.page_id = r * XT + i * CT + c, .offset_bytes = 0},
                    {.offset_bytes = (ci * 4 + i) * TILE});
            }
        }
        noc.async_read_barrier();
        cbx.push_back(4 * BLK);
    }
}
