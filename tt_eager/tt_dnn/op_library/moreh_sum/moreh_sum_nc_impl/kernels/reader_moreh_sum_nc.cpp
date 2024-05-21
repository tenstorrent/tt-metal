// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

inline uint32_t get_read_tile_id(uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size) {
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

void kernel_main() {
    ArgFetcher arg_fetcher;
    const auto input_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_input_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_output_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto input_is_dram = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const auto dim = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto reduce_tile_size = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto inner_tile_size = arg_fetcher.get_next_arg_val<uint32_t>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    uint32_t l1_write_addr_in0;
    uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const auto input_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        auto read_tile_id = (dim == 0) ? (i) : (get_read_tile_id(i, reduce_tile_size, inner_tile_size));
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            if (input_is_dram) {
                noc_async_read_tile(read_tile_id, dram_input_addrg, l1_write_addr_in0);
            } else {
                noc_async_read_tile(read_tile_id, l1_input_addrg, l1_write_addr_in0);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += inner_tile_size;
        }
    }
}
