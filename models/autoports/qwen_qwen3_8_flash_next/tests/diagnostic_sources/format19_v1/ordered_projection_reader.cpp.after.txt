// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Native BF16 tile face layout: TL, TR, BL, BR; sixteen rows per face.
constexpr uint32_t tile_element(uint32_t row, uint32_t column) {
    return (row >> 4) * 512 + (column >> 4) * 256 + (row & 15u) * 16 + (column & 15u);
}

void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(xa, get_arg_val<uint32_t>(0), 2048);
    const auto weight = TensorAccessor(wa, get_arg_val<uint32_t>(1), 2048);
    const uint32_t output_column_tile = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_left = 2, cb_right = 3;
    // These immutable scratch reservations belong only to the reader. They
    // are never handed to compute and are not recycled during this program.
    cb_reserve_back(cb_x, 80);
    cb_reserve_back(cb_w, 80);
    const uint32_t x_address = get_write_ptr(cb_x);
    const uint32_t w_address = get_write_ptr(cb_w);
    for (uint32_t tile = 0; tile < 80; ++tile) {
        noc_async_read(input.get_noc_addr(tile), x_address + tile * 2048, 2048);
        noc_async_read(weight.get_noc_addr(tile * 20 + output_column_tile), w_address + tile * 2048, 2048);
    }
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint16_t* x = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(x_address);
    const volatile tt_l1_ptr uint16_t* w = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(w_address);
    for (uint32_t pair = 0; pair < 1280; ++pair) {
        // Intel VDPBF16PS consumes the high member first. Do not pre-sum pairs.
        for (uint32_t member = 0; member < 2; ++member) {
            const uint32_t k = pair * 2 + (1 - member);
            const uint32_t k_tile = k >> 5;
            const uint32_t k_local = k & 31u;
            cb_reserve_back(cb_left, 1);
            cb_reserve_back(cb_right, 1);
            volatile tt_l1_ptr uint16_t* left = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_left));
            volatile tt_l1_ptr uint16_t* right =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_right));
            for (uint32_t row = 0; row < 32; ++row) {
                const uint16_t value = row < count ? x[k_tile * 1024 + tile_element(row, k_local)] : 0;
                for (uint32_t column = 0; column < 32; ++column) {
                    const uint32_t target = tile_element(row, column);
                    left[target] = value;
                    right[target] = w[k_tile * 1024 + tile_element(k_local, column)];
                }
            }
            cb_push_back(cb_left, 1);
            cb_push_back(cb_right, 1);
        }
    }
}
