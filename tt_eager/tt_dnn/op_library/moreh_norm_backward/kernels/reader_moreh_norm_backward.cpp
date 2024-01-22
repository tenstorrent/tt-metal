// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_norm/kernel_utils/common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const bool output_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto output_grad_addr = get_arg_val<uint32_t>(i++);
    const bool output_grad_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto decimal = get_arg_val<uint32_t>(i++);

    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);

    const auto input_n = get_arg_val<uint32_t>(i++);
    const auto input_c = get_arg_val<uint32_t>(i++);
    const auto input_origin_h = get_arg_val<uint32_t>(i++);
    const auto input_origin_w = get_arg_val<uint32_t>(i++);

    const auto output_n = get_arg_val<uint32_t>(i++);
    const auto output_c = get_arg_val<uint32_t>(i++);
    const auto output_origin_h = get_arg_val<uint32_t>(i++);
    const auto output_origin_w = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_output = cb_id++;
    const auto cb_id_output_grad = cb_id++;
    const auto cb_id_decimal = cb_id++;

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    // output_grad
    const uint32_t output_grad_tile_bytes = get_tile_size(cb_id_output_grad);
    const auto output_grad_data_format = get_dataformat(cb_id_output_grad);

    const InterleavedAddrGenFast<true> dram_output_grad_addrg = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_bytes,
        .data_format = output_grad_data_format};

    const InterleavedAddrGenFast<false> l1_output_grad_addrg = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_bytes,
        .data_format = output_grad_data_format};

    fill_cb_with_value(cb_id_decimal, decimal);

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    const auto output_l1_write_ptr = get_write_ptr(cb_id_output);
    const auto output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const auto N = input_n;
    const auto C = input_c;
    const auto Ht = (input_origin_h + TILE_H - 1) / TILE_H;
    const auto Wt = (input_origin_w + TILE_W - 1) / TILE_H;

    const auto oN = output_n;
    const auto oC = output_c;
    const auto oHt = (output_origin_h + TILE_H - 1) / TILE_H;
    const auto oWt = (output_origin_w + TILE_W - 1) / TILE_W;

    const bool need_to_bcast_n = (N != oN);
    const bool need_to_bcast_c = (C != oC);
    const bool need_to_bcast_ht = (Ht != oHt);
    const bool need_to_bcast_wt = (Wt != oWt);

    for (uint32_t input_tile_idx = tile_offset; input_tile_idx < tile_offset + num_input_tiles_per_core;
         ++input_tile_idx) {
        // input
        // n * C * Ht * Wt + c * Ht * Wt + ht * Wt + wt
        const auto input_n_idx = input_tile_idx / (C * Ht * Wt);
        const auto input_c_idx = (input_tile_idx / (Ht * Wt)) % C;
        const auto input_ht_idx = (input_tile_idx / Wt) % Ht;
        const auto input_wt_idx = input_tile_idx % Wt;

        // input_grad
        const auto input_grad_tile_idx = input_tile_idx;

        // output
        const auto output_n_idx = need_to_bcast_n ? 0 : input_n_idx;
        const auto output_c_idx = need_to_bcast_c ? 0 : input_c_idx;
        const auto output_ht_idx = need_to_bcast_ht ? 0 : input_ht_idx;
        const auto output_wt_idx = need_to_bcast_wt ? 0 : input_wt_idx;

        const auto output_tile_idx =
            output_n_idx * oC * oHt * oWt + output_c_idx * oHt * oWt + output_ht_idx * oWt + output_wt_idx;

        // output
        const auto output_grad_tile_idx = output_tile_idx;

        cb_reserve_back(cb_id_input, 1);
        cb_reserve_back(cb_id_output, 1);
        cb_reserve_back(cb_id_output_grad, 1);

        if (input_is_dram) {
            noc_async_read_tile(input_tile_idx, dram_input_addrg, input_l1_write_ptr);
        } else {
            noc_async_read_tile(input_tile_idx, l1_input_addrg, input_l1_write_ptr);
        }

        if (output_is_dram) {
            noc_async_read_tile(output_tile_idx, dram_output_addrg, output_l1_write_ptr);
        } else {
            noc_async_read_tile(output_tile_idx, l1_output_addrg, output_l1_write_ptr);
        }

        if (output_grad_is_dram) {
            noc_async_read_tile(output_grad_tile_idx, dram_output_grad_addrg, output_grad_l1_write_ptr);
        } else {
            noc_async_read_tile(output_grad_tile_idx, l1_output_grad_addrg, output_grad_l1_write_ptr);
        }

        noc_async_read_barrier();
        cb_push_back(cb_id_input, 1);
        cb_push_back(cb_id_output, 1);
        cb_push_back(cb_id_output_grad, 1);
    }

}  // void kernel_main()
