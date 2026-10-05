// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Packed-layout mHC combine, reader.  x [1,1,N,4C] fp32 (stream i at tile columns i*CT..), optional y / y2 [1,1,N,C]
// fp32. A core owns token tile row r and column tiles [c0, c0+ncols).  Builds an identity tile, loads the NCA+NCB
// coefficient tiles of row r (column 0 of tile q = coefficient q of every token), then streams BLK column tiles per
// block: 4 x tiles (+ y, y2 tiles).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_i = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ct = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = get_compile_time_arg_val(2);
    constexpr uint32_t cb_y = get_compile_time_arg_val(3);
    constexpr uint32_t cb_y2 = get_compile_time_arg_val(4);
    constexpr uint32_t NCA = get_compile_time_arg_val(5);
    constexpr uint32_t NCB = get_compile_time_arg_val(6);
    constexpr uint32_t CT = get_compile_time_arg_val(7);  // column tiles per stream
    constexpr uint32_t BLK = get_compile_time_arg_val(8);
    constexpr uint32_t HAS_Y = get_compile_time_arg_val(9);
    constexpr uint32_t HAS_Y2 = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto a_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto y2_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;
    constexpr uint32_t XT = 4 * CT;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t y2_addr = get_common_arg_val<uint32_t>(4);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto a_acc = TensorAccessor(a_args, a_addr, TILE);
    const auto b_acc = TensorAccessor(b_args, b_addr, TILE);
    const auto y_acc = TensorAccessor(y_args, y_addr, TILE);
    const auto y2_acc = TensorAccessor(y2_args, y2_addr, TILE);
    experimental::CB cbi(cb_i), cbct(cb_ct), cbx(cb_x), cby(cb_y), cby2(cb_y2);

    // identity tile: diagonal entry (k,k) lives in face (k>=16 ? 3 : 0)
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

    cbct.reserve_back(NCA + NCB);
    for (uint32_t q = 0; q < NCA; ++q) {
        noc.async_read(a_acc, cbct, TILE, {.page_id = r * NCA + q, .offset_bytes = 0}, {.offset_bytes = q * TILE});
    }
    for (uint32_t q = 0; q < NCB; ++q) {
        noc.async_read(
            b_acc, cbct, TILE, {.page_id = r * NCB + q, .offset_bytes = 0}, {.offset_bytes = (NCA + q) * TILE});
    }
    noc.async_read_barrier();
    cbct.push_back(NCA + NCB);

    for (uint32_t b = 0; b < ncols; b += BLK) {
        cbx.reserve_back(4 * BLK);
        if constexpr (HAS_Y) {
            cby.reserve_back(BLK);
        }
        if constexpr (HAS_Y2) {
            cby2.reserve_back(BLK);
        }
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
            if constexpr (HAS_Y) {
                noc.async_read(
                    y_acc, cby, TILE, {.page_id = r * CT + c, .offset_bytes = 0}, {.offset_bytes = ci * TILE});
            }
            if constexpr (HAS_Y2) {
                noc.async_read(
                    y2_acc, cby2, TILE, {.page_id = r * CT + c, .offset_bytes = 0}, {.offset_bytes = ci * TILE});
            }
        }
        noc.async_read_barrier();
        cbx.push_back(4 * BLK);
        if constexpr (HAS_Y) {
            cby.push_back(BLK);
        }
        if constexpr (HAS_Y2) {
            cby2.push_back(BLK);
        }
    }
}
