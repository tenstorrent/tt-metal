// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    const auto output_grad_addr = get_arg_val<uint32_t>(0);
    const auto input_addr = get_arg_val<uint32_t>(1);
    const auto mean_addr = get_arg_val<uint32_t>(2);
    const auto rstd_addr = get_arg_val<uint32_t>(3);
    const auto gamma_addr = get_arg_val<uint32_t>(4);

    const auto num_rows_per_core = get_arg_val<uint32_t>(5);
    const auto Wt = get_arg_val<uint32_t>(6);
    const auto tile_offset = get_arg_val<uint32_t>(7);
    const auto n = get_arg_val<uint32_t>(8);
    const auto recip_n = get_arg_val<uint32_t>(9);
    const auto mask_h = get_arg_val<uint32_t>(10);
    const auto mask_w = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_id_output_grad = 0;
    constexpr uint32_t cb_id_input = 1;
    constexpr uint32_t cb_id_mean = 2;
    constexpr uint32_t cb_id_rstd = 3;
    constexpr uint32_t cb_id_scaler = 4;
    constexpr uint32_t cb_id_n_recip_n = 5;
    constexpr uint32_t cb_id_gamma = 6;
    constexpr uint32_t cb_id_mask_h_w = 7;

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
    constexpr bool gamma_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool gamma_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool do_mask_h = get_compile_time_arg_val(6) == 1;
    constexpr bool do_mask_w = get_compile_time_arg_val(7) == 1;

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

    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto gamma_data_format = get_dataformat(cb_id_gamma);
    const InterleavedAddrGenFast<gamma_is_dram> gamma_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_scaler, scaler.u);
    fill_cb_with_value(cb_id_n_recip_n, n);
    fill_cb_with_value(cb_id_n_recip_n, recip_n);

    if (do_mask_h || do_mask_w) {
        generate_mask_h_w(cb_id_mask_h_w, mask_h, mask_w);
    }

    uint32_t offs = 0;
    const uint32_t NCHt = num_rows_per_core;
    constexpr uint32_t onetile = 1;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // mean (N, C, H, 1)
        const uint32_t mean_l1_write_ptr = get_write_ptr(cb_id_mean);
        cb_reserve_back(cb_id_mean, onetile);
        noc_async_read_tile(ncht + (tile_offset / Wt), mean_addrg, mean_l1_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_id_mean, onetile);

        // rstd (N, C, H, 1)
        const uint32_t rstd_l1_write_ptr = get_write_ptr(cb_id_rstd);
        cb_reserve_back(cb_id_rstd, onetile);
        noc_async_read_tile(ncht + (tile_offset / Wt), rstd_addrg, rstd_l1_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_id_rstd, onetile);

        // For Sum[dy] and Sum[y * dy]
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // input (N, C, H, W)
            const uint32_t input_l1_write_ptr = get_write_ptr(cb_id_input);
            cb_reserve_back(cb_id_input, onetile);
            noc_async_read_tile(offs + wt + tile_offset, input_addrg, input_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_input, onetile);

            // output_grad (N, C, H, W)
            const uint32_t output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);
            cb_reserve_back(cb_id_output_grad, onetile);
            noc_async_read_tile(offs + wt + tile_offset, output_grad_addrg, output_grad_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_output_grad, onetile);

            if (gamma_has_value) {
                // gamma (1, 1, 1, W)
                const uint32_t gamma_l1_write_ptr = get_write_ptr(cb_id_gamma);
                cb_reserve_back(cb_id_gamma, onetile);
                noc_async_read_tile(wt, gamma_addrg, gamma_l1_write_ptr);
                noc_async_read_barrier();
                cb_push_back(cb_id_gamma, onetile);
            }  // gamma_has_value
        }      // wt loop

        // For ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (1.0 / (n * rstd))
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // output_grad (N, C, H, W)
            const uint32_t output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);
            cb_reserve_back(cb_id_output_grad, onetile);
            noc_async_read_tile(offs + wt + tile_offset, output_grad_addrg, output_grad_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_output_grad, onetile);

            if (gamma_has_value) {
                // gamma (1, 1, 1, W)
                const uint32_t gamma_l1_write_ptr = get_write_ptr(cb_id_gamma);
                cb_reserve_back(cb_id_gamma, onetile);
                noc_async_read_tile(wt, gamma_addrg, gamma_l1_write_ptr);
                noc_async_read_barrier();
                cb_push_back(cb_id_gamma, onetile);
            }  // gamma_has_value

            // input (N, C, H, W)
            const uint32_t input_l1_write_ptr = get_write_ptr(cb_id_input);
            cb_reserve_back(cb_id_input, onetile);
            noc_async_read_tile(offs + wt + tile_offset, input_addrg, input_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_input, onetile);
        }  // wt loop

        offs += Wt;
    }  // ncht loop
}  // void kernel_main()
