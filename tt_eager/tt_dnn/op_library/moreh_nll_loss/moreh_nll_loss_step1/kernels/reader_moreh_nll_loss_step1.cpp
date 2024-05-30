// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    auto target_addr = get_arg_val<uint32_t>(i++);
    auto weight_addr = get_arg_val<uint32_t>(i++);
    auto ignore_index = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    auto num_units_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto N = get_arg_val<uint32_t>(i++);
    auto C = get_arg_val<uint32_t>(i++);
    auto weight_num_tile = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);
    auto target_element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_target = tt::CB::c_in0;
    constexpr uint32_t cb_weight = tt::CB::c_in1;

    constexpr uint32_t cb_output = tt::CB::c_out0;

    // ublocks size defined in tiles
    const uint32_t target_tile_bytes = get_tile_size(cb_target);

    constexpr bool target_is_dram = get_compile_time_arg_val(0) == 1;
#if defined(WEIGHT)
    constexpr bool weight_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool weight_has_value = get_compile_time_arg_val(2) == 1;
#endif

    const InterleavedAddrGen<target_is_dram> addrg_target = {
        .bank_base_address = target_addr, .page_size = target_tile_bytes};

#if defined(WEIGHT)
    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    const DataFormat weight_data_format = get_dataformat(cb_weight);
    const InterleavedAddrGen<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr,
        .page_size = 1024 * element_size,
    };
#endif

    constexpr uint32_t onetile = 1;

    union {
        float f;
        uint32_t u;
    } one, zero;
    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

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

    uint32_t end_id = start_id + num_units_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        // target: (N, d1, d2, .. dk)
        cb_reserve_back(cb_target, onetile);
        uint32_t l1_write_addr_target = get_write_ptr(cb_target);
        volatile tt_l1_ptr int32_t* target_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_write_addr_target);
        uint32_t target_noc_id = i;
        uint64_t target_noc_addr = get_noc_addr(target_noc_id, addrg_target);
        noc_async_read(target_noc_addr, l1_write_addr_target, target_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_target, onetile);

        cb_reserve_back(cb_output, onetile);

        uint32_t l1_write_addr_output = get_write_ptr(cb_output);
        volatile tt_l1_ptr uint16_t* output_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_output);

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t inout_idx = h * TILE_WIDTH + w;
                int32_t target_val = target_l1_ptr[inout_idx];
                if (target_val != ignore_index) {
                    if (0 <= target_val && target_val < static_cast<int32_t>(C)) {
#if defined(WEIGHT)
                        uint32_t target_idx = target_val;
                        output_l1_ptr[inout_idx] = weight_l1_ptr[target_idx];
#else
                        output_l1_ptr[inout_idx] = u16_one;
#endif
                    } else {
                        output_l1_ptr[inout_idx] = u16_zero;
                    }
                } else {
                    output_l1_ptr[inout_idx] = u16_zero;
                }
            }
        }
        cb_push_back(cb_output, onetile);

        cb_pop_front(cb_target, onetile);
    }
}
