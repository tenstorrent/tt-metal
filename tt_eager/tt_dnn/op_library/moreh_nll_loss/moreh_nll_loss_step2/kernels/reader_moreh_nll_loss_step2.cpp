// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t target_addr = get_arg_val<uint32_t>(1);
    uint32_t weight_addr = get_arg_val<uint32_t>(2);
    uint32_t divisor_addr = get_arg_val<uint32_t>(3);
    uint32_t ignore_index = get_arg_val<uint32_t>(4);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(5);
    uint32_t start_id = get_arg_val<uint32_t>(6);
    uint32_t C = get_arg_val<uint32_t>(7);
    uint32_t HtWt = get_arg_val<uint32_t>(8);
    uint32_t origin_h = get_arg_val<uint32_t>(9);
    uint32_t origin_w = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_input = tt::CB::c_in0;
    constexpr uint32_t cb_target = tt::CB::c_in1;
    constexpr uint32_t cb_weight = tt::CB::c_in2;
    constexpr uint32_t cb_divisor = tt::CB::c_in3;
    constexpr uint32_t cb_one = tt::CB::c_in4;

    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;
    constexpr uint32_t cb_tmp_input = tt::CB::c_intermed1;

    // ublocks size defined in tiles
    const uint32_t input_tile_bytes = get_tile_size(cb_input);
    const DataFormat input_data_format = get_dataformat(cb_input);

    const uint32_t target_tile_bytes = get_compile_time_arg_val(0);

    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    const DataFormat weight_data_format = get_dataformat(cb_weight);

    const uint32_t divisor_tile_bytes = get_tile_size(cb_divisor);
    const DataFormat divisor_data_format = get_dataformat(cb_divisor);

    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool target_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool weight_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool divisor_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool weight_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool divisor_has_value = get_compile_time_arg_val(6) == 1;

    const InterleavedAddrGenFast<input_is_dram> addrg_input = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    const InterleavedAddrGen<target_is_dram> addrg_target = {
        .bank_base_address = target_addr, .page_size = target_tile_bytes};
    const InterleavedAddrGenFast<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr, .page_size = weight_tile_bytes, .data_format = weight_data_format};
    const InterleavedAddrGenFast<divisor_is_dram> addrg_divisor = {
        .bank_base_address = divisor_addr, .page_size = divisor_tile_bytes, .data_format = divisor_data_format};

    constexpr uint32_t onetile = 1;

    union {
        float f;
        uint32_t u;
    } one, zero;
    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

    fill_cb_with_value(cb_one, one.u);

    volatile tt_l1_ptr uint16_t* weight_l1_ptr;
    if (weight_has_value) {
        uint32_t weight_num_tile = (C + weight_tile_bytes - 1) / weight_tile_bytes;
        cb_reserve_back(cb_weight, weight_num_tile);
        uint32_t l1_write_addr_weight = get_write_ptr(cb_weight);
        weight_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_weight);

        for (uint32_t i = 0; i < weight_num_tile; ++i) {
            noc_async_read_tile(i, addrg_weight, l1_write_addr_weight);
            noc_async_read_barrier();
            l1_write_addr_weight += weight_tile_bytes;
        }
    }

    if (divisor_has_value) {
        cb_reserve_back(cb_divisor, onetile);
        uint32_t l1_write_addr_divisor = get_write_ptr(cb_divisor);
        noc_async_read_tile(0, addrg_divisor, l1_write_addr_divisor);
        noc_async_read_barrier();
        cb_push_back(cb_divisor, onetile);
    }

    // read ublocks from src0 to CB0, then push ublocks to compute (unpacker)
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t n = i / HtWt;
        uint32_t htwt = i % HtWt;

        cb_reserve_back(cb_target, onetile);
        uint32_t l1_write_addr_target = get_write_ptr(cb_target);
        volatile tt_l1_ptr uint32_t* target_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr_target);
        uint64_t target_noc_addr = get_noc_addr(n * HtWt + htwt, addrg_target);
        noc_async_read(target_noc_addr, l1_write_addr_target, target_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_target, onetile);

        cb_reserve_back(cb_tmp_weight, onetile);
        uint32_t l1_write_addr_lsum = get_write_ptr(cb_tmp_weight);
        volatile tt_l1_ptr uint16_t* lsum_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_lsum);

        for (uint32_t j = 0; j < 32; ++j) {
            for (uint32_t k = 0; k < 32; ++k) {
                uint32_t index = target_l1_ptr[j * 32 + k];
                if (index != ignore_index) {
                    if (0 <= index && index < C) {
                        if (weight_has_value) {
                            uint16_t value = weight_l1_ptr[index];
                            lsum_l1_ptr[j * 32 + k] = value;
                        } else {
                            lsum_l1_ptr[j * 32 + k] = u16_one;
                        }
                    } else {
                        lsum_l1_ptr[j * 32 + k] = u16_zero;
                    }
                } else {
                    lsum_l1_ptr[j * 32 + k] = u16_zero;
                }
            }
        }
        mask_tile_if_need(l1_write_addr_lsum, origin_h, origin_w);
        cb_push_back(cb_tmp_weight, onetile);

        cb_reserve_back(cb_tmp_input, onetile);
        uint32_t l1_write_addr_tmp_input = get_write_ptr(cb_tmp_input);
        volatile tt_l1_ptr uint16_t* tmp_input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_tmp_input);

        for (uint32_t j = 0; j < 32; ++j) {
            for (uint32_t k = 0; k < 32; ++k) {
                uint32_t c = target_l1_ptr[j * 32 + k];
                if (c != ignore_index) {
                    if (0 <= c && c < C) {
                        cb_reserve_back(cb_input, onetile);
                        uint32_t l1_write_addr_input = get_write_ptr(cb_input);
                        volatile tt_l1_ptr uint16_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_input);
                        noc_async_read_tile(n * C * HtWt + c * HtWt + htwt, addrg_input, l1_write_addr_input);
                        noc_async_read_barrier();

                        uint16_t value = input_l1_ptr[j * 32 + k];
                        tmp_input_l1_ptr[j * 32 + k] = value;
                    } else {
                        tmp_input_l1_ptr[j * 32 + k] = u16_zero;
                    }
                } else {
                    tmp_input_l1_ptr[j * 32 + k] = u16_zero;
                }
            }
        }
        mask_tile_if_need(l1_write_addr_tmp_input, origin_h, origin_w);
        cb_push_back(cb_tmp_input, onetile);

        cb_wait_front(cb_target, onetile);
        cb_pop_front(cb_target, onetile);
    }
}
