// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/moreh_layernorm_backward/kernels/utils.hpp"

void kernel_main() {
    const auto input_addr = get_arg_val<uint32_t>(0);
    const bool input_is_dram = get_arg_val<uint32_t>(1) == 1;
    const auto num_tiles = get_arg_val<uint32_t>(2);
    const auto decimal = get_arg_val<uint32_t>(3);
    const auto origin_h = get_arg_val<uint32_t>(4);
    const auto origin_w = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_input = 0;
    constexpr uint32_t cb_id_one = 1;
    constexpr uint32_t cb_id_decimal = 2;
    constexpr uint32_t cb_id_mask_h_w = 3;

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_decimal, decimal);
    fill_cb_with_value(cb_id_one, scaler.u);
    generate_mask_h_w_if_needed(cb_id_mask_h_w, origin_h, origin_w);

    constexpr uint32_t onetile = 1;

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_reserve_back(cb_id_input, onetile);
        if (input_is_dram) {
            noc_async_read_tile(tile_idx, dram_input_addrg, input_l1_write_ptr);
        } else {
            noc_async_read_tile(tile_idx, l1_input_addrg, input_l1_write_ptr);
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_input, onetile);
    }

}  // void kernel_main()
