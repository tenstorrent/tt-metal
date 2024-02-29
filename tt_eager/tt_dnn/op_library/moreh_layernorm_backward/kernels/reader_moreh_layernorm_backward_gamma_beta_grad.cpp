// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    const auto output_grad_addr = get_arg_val<uint32_t>(0);
    const auto input_addr = get_arg_val<uint32_t>(1);
    const auto mean_addr = get_arg_val<uint32_t>(2);
    const auto rstd_addr = get_arg_val<uint32_t>(3);

    const auto num_cols_per_core = get_arg_val<uint32_t>(4);
    const auto NCHt = get_arg_val<uint32_t>(5);
    const auto Wt = get_arg_val<uint32_t>(6);
    const auto tile_offset = get_arg_val<uint32_t>(7);
    const auto mask_h = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_output_grad = 0;
    constexpr uint32_t cb_id_input = 1;
    constexpr uint32_t cb_id_mean = 2;
    constexpr uint32_t cb_id_rstd = 3;
    constexpr uint32_t cb_id_scaler = 4;
    constexpr uint32_t cb_id_mask_h = 5;

    const uint32_t output_grad_tile_bytes = get_tile_size(cb_id_output_grad);
    const auto output_grad_data_format = get_dataformat(cb_id_output_grad);

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const uint32_t mean_tile_bytes = get_tile_size(cb_id_mean);
    const auto mean_data_format = get_dataformat(cb_id_mean);

    const uint32_t rstd_tile_bytes = get_tile_size(cb_id_rstd);
    const auto rstd_data_format = get_dataformat(cb_id_rstd);

    constexpr bool output_grad_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool mean_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool rstd_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool gamma_grad_has_value = get_compile_time_arg_val(4) == 1;
    constexpr bool do_mask_h = get_compile_time_arg_val(5) == 1;

    const InterleavedAddrGenFast<output_grad_is_dram> output_grad_addrg = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_bytes,
        .data_format = output_grad_data_format};

    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const InterleavedAddrGenFast<mean_is_dram> mean_addrg = {
        .bank_base_address = mean_addr, .page_size = mean_tile_bytes, .data_format = mean_data_format};

    const InterleavedAddrGenFast<rstd_is_dram> rstd_addrg = {
        .bank_base_address = rstd_addr, .page_size = rstd_tile_bytes, .data_format = rstd_data_format};

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_scaler, scaler.u);

    if (do_mask_h) {
        generate_mask_h(cb_id_mask_h, mask_h);
    }

    const uint32_t start_tile_idx = tile_offset;

    for (uint32_t w_idx = 0; w_idx < num_cols_per_core; w_idx++) {
        for (uint32_t h_idx = 0; h_idx < NCHt; h_idx++) {
            // output_grad (N, C, H, W)
            const uint32_t dy_tile_idx = Wt * h_idx + w_idx + start_tile_idx;
            cb_reserve_back(cb_id_output_grad, onetile);
            uint32_t output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);
            noc_async_read_tile(dy_tile_idx, output_grad_addrg, output_grad_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_output_grad, onetile);

            if (gamma_grad_has_value) {
                // input (N, C, H, W)
                const uint32_t x_tile_idx = Wt * h_idx + w_idx + start_tile_idx;
                cb_reserve_back(cb_id_input, onetile);
                uint32_t input_l1_write_ptr = get_write_ptr(cb_id_input);
                noc_async_read_tile(x_tile_idx, input_addrg, input_l1_write_ptr);
                noc_async_read_barrier();
                cb_push_back(cb_id_input, onetile);

                // mean (N, C, H, 1)
                const uint32_t mean_tile_idx = h_idx;
                cb_reserve_back(cb_id_mean, onetile);
                uint32_t mean_l1_write_ptr = get_write_ptr(cb_id_mean);
                noc_async_read_tile(mean_tile_idx, mean_addrg, mean_l1_write_ptr);
                noc_async_read_barrier();
                cb_push_back(cb_id_mean, onetile);

                // rstd (N, C, H, 1)
                const uint32_t rstd_tile_idx = h_idx;
                cb_reserve_back(cb_id_rstd, onetile);
                uint32_t rstd_l1_write_ptr = get_write_ptr(cb_id_rstd);
                noc_async_read_tile(rstd_tile_idx, rstd_addrg, rstd_l1_write_ptr);
                noc_async_read_barrier();
                cb_push_back(cb_id_rstd, onetile);
            }  // gamma_grad_has_value

        }  // NCHt loop
    }      // Wt loop
}  // void kernel_main()
