// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    auto input_addr = get_arg_val<uint32_t>(i++);
    auto target_addr = get_arg_val<uint32_t>(i++);
    auto weight_addr = get_arg_val<uint32_t>(i++);
    auto divisor_addr = get_arg_val<uint32_t>(i++);
    auto ignore_index = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto N = get_arg_val<uint32_t>(i++);
    auto C = get_arg_val<uint32_t>(i++);
    auto Wt = get_arg_val<uint32_t>(i++);
    auto num_inner_tile = get_arg_val<uint32_t>(i++);
    auto weight_num_tile = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CB::c_in0;
    constexpr uint32_t cb_target = tt::CB::c_in1;
    constexpr uint32_t cb_weight = tt::CB::c_in2;
    constexpr uint32_t cb_divisor = tt::CB::c_in3;

    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;
    constexpr uint32_t cb_tmp_input = tt::CB::c_intermed1;

    constexpr uint32_t cb_output = tt::CB::c_out0;

    // ublocks size defined in tiles
    const uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input_data_format = get_dataformat(cb_input);

    const uint32_t target_tile_bytes = get_tile_size(cb_target);

    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    const auto weight_data_format = get_dataformat(cb_weight);

    const uint32_t divisor_tile_bytes = get_tile_size(cb_divisor);
    const auto divisor_data_format = get_dataformat(cb_divisor);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool target_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool weight_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool divisor_is_dram = get_compile_time_arg_val(3) == 1;

    uint32_t target_element_size = 4;  // sizeof(int32)

    const InterleavedAddrGen<input_is_dram> addrg_input = {
        .bank_base_address = input_addr,
        .page_size = 1024 * element_size,
    };

    const InterleavedAddrGen<target_is_dram> addrg_target = {
        .bank_base_address = target_addr, .page_size = 1024 * target_element_size};

    const InterleavedAddrGen<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr,
        .page_size = 1024 * element_size,
    };

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

#if defined(DIVISOR)
    cb_reserve_back(cb_divisor, onetile);
    uint32_t l1_write_addr_divisor = get_write_ptr(cb_divisor);
    volatile tt_l1_ptr uint16_t* target_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_divisor);

    noc_async_read_tile(0, addrg_divisor, l1_write_addr_divisor);
    noc_async_read_barrier();

    cb_push_back(cb_divisor, onetile);
#endif

#if defined(WEIGHT)
    cb_reserve_back(cb_weight, weight_num_tile);
    uint32_t l1_write_addr_weight = get_write_ptr(cb_weight);
    volatile tt_l1_ptr uint16_t* weight_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_weight);

    for (uint32_t i = 0; i < weight_num_tile * 2; ++i) {
        uint32_t noc_id = i / 2;
        uint32_t noc_offset = 0;
        if (noc_id * 2 != i) {
            noc_offset += 256 * element_size;
        }
        uint64_t src_noc_addr = get_noc_addr(noc_id, addrg_weight, noc_offset);
        noc_async_read(src_noc_addr, l1_write_addr_weight, NOC_MINIMUM_READ_SIZE);
        noc_async_read_barrier();
        l1_write_addr_weight += NOC_MINIMUM_READ_SIZE;
    }
#endif

    cb_reserve_back(cb_input, onetile);
    uint32_t l1_write_addr_input = get_write_ptr(cb_input);
    volatile tt_l1_ptr uint16_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_input);

    // iterate from start_id to end_id
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_target, onetile);
        uint32_t l1_write_addr_target = get_write_ptr(cb_target);
        volatile tt_l1_ptr int32_t* target_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_write_addr_target);
        uint32_t target_noc_id = i;
        uint64_t target_noc_addr = get_noc_addr(target_noc_id, addrg_target);
        noc_async_read(target_noc_addr, l1_write_addr_target, target_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_target, onetile);

#if defined(WEIGHT)
        cb_reserve_back(cb_tmp_weight, onetile);

#if defined(FP32_DEST_ACC_EN)
        uint32_t l1_write_addr_tmp_weight = get_write_ptr(cb_tmp_weight);
        volatile tt_l1_ptr float* tmp_weight_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_addr_tmp_weight);
#else
        uint32_t l1_write_addr_tmp_weight = get_write_ptr(cb_tmp_weight);
        volatile tt_l1_ptr uint16_t* tmp_weight_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_tmp_weight);
#endif

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t tilized_idx = get_tilized_idx(h, w);
                int32_t target_val = target_l1_ptr[tilized_idx];
                if (target_val != ignore_index) {
                    if (0 <= target_val && target_val < static_cast<int32_t>(C)) {
#if defined(FP32_DEST_ACC_EN)
                        tmp_weight_l1_ptr[tilized_idx] = bfloat16_to_float(weight_l1_ptr[target_val]);
#else
                        tmp_weight_l1_ptr[tilized_idx] = weight_l1_ptr[target_val];
#endif
                        continue;
                    }
                }
#if defined(FP32_DEST_ACC_EN)
                tmp_weight_l1_ptr[tilized_idx] = 0.0f;
#else
                tmp_weight_l1_ptr[tilized_idx] = u16_zero;
#endif
            }
        }
        cb_push_back(cb_tmp_weight, onetile);
#endif

        cb_reserve_back(cb_tmp_input, onetile);
#if defined(FP32_DEST_ACC_EN)
        uint32_t l1_write_addr_tmp_input = get_write_ptr(cb_tmp_input);
        volatile tt_l1_ptr float* tmp_input_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_addr_tmp_input);
#else
        uint32_t l1_write_addr_tmp_input = get_write_ptr(cb_tmp_input);
        volatile tt_l1_ptr uint16_t* tmp_input_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_tmp_input);
#endif

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t tilized_idx = get_tilized_idx(h, w);
                int32_t target_val = target_l1_ptr[tilized_idx];

                if (target_val != ignore_index) {
                    if (0 <= target_val && target_val < static_cast<int32_t>(C)) {
                        uint32_t n = i / num_inner_tile;
                        uint32_t inner = i % num_inner_tile;

                        // read input
                        // input: (N, C, d1,d2,... dn)
                        // noc_id: n * C * num_inner_tile + c * num_inner_tile + inner
                        uint32_t noc_offset;
                        get_noc_offset(h, w, element_size, noc_offset);

                        uint32_t noc_id = (n * C * num_inner_tile) + target_val * num_inner_tile + inner;
                        uint64_t src_noc_addr = get_noc_addr(noc_id, addrg_input, noc_offset);
                        noc_async_read(src_noc_addr, l1_write_addr_input, NOC_MINIMUM_READ_SIZE);
                        noc_async_read_barrier();

#if defined(FP32_DEST_ACC_EN)
                        tmp_input_l1_ptr[tilized_idx] = bfloat16_to_float(input_l1_ptr[w % 16]);
#else
                        tmp_input_l1_ptr[tilized_idx] = input_l1_ptr[w % 16];
#endif
                        continue;
                    }
                }

#if defined(FP32_DEST_ACC_EN)
                tmp_input_l1_ptr[tilized_idx] = 0.0f;
#else
                tmp_input_l1_ptr[tilized_idx] = u16_zero;
#endif
            }
        }

        cb_push_back(cb_tmp_input, onetile);
        cb_pop_front(cb_target, onetile);
    }
}
