// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"


template <bool input_DRAM, bool weights_DRAM, bool out_DRAM>
inline __attribute__((always_inline))
void embeddings_(
                    const uint32_t embeddings_per_core,
                    const uint32_t input_offset,
                    const uint32_t weight_offset,
                    const uint32_t num_output_rows,
                    const uint32_t page_size,
                    uint32_t input_l1_addr,
                    uint32_t weight_l1_addr,
                    const InterleavedAddrGen<input_DRAM>& input,
                    const InterleavedAddrGen<weights_DRAM>& weights,
                    const InterleavedAddrGen<out_DRAM>& out

) {


    for (uint32_t i = 0; i < num_output_rows; i++) {
        auto noc_input_src_addr = get_noc_addr(i+input_offset, input);
        noc_async_read(noc_input_src_addr, input_l1_addr, sizeof(uint32_t));
        noc_async_read_barrier();
        uint32_t row = ((uint32_t *)input_l1_addr)[0];

        if(row>=weight_offset && row < (weight_offset+embeddings_per_core)){
            auto noc_src_addr = get_noc_addr(row, weights);
            auto noc_dst_addr = get_noc_addr(i+input_offset, out);
            noc_async_read(noc_src_addr, weight_l1_addr, page_size);
            noc_async_read_barrier();
            noc_async_write(weight_l1_addr, noc_dst_addr, page_size);
            noc_async_write_barrier();
        }
    }



}

void kernel_main() {
    std::uint32_t input_offset = get_arg_val<uint32_t>(0);
    std::uint32_t weight_offset = get_arg_val<uint32_t>(1);
    std::uint32_t embeddings_per_core = get_arg_val<uint32_t>(2);
    std::uint32_t num_output_rows      = get_arg_val<uint32_t>(3);
    std::uint32_t input_dram_buffer_src_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t weights_dram_buffer_src_addr  = get_arg_val<uint32_t>(5);
    std::uint32_t output_dram_buffer_dst_addr      = get_arg_val<uint32_t>(6);


    #define in_is_dram get_compile_time_arg_val(0) == 1
    #define weights_is_dram get_compile_time_arg_val(1) == 1
    #define out_is_dram get_compile_time_arg_val(2) == 1
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_inter_weights = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_inter_input = get_compile_time_arg_val(5);

    const InterleavedAddrGen<in_is_dram> input_rows_0 = {
        .bank_base_address = input_dram_buffer_src_addr, .page_size = sizeof(uint32_t)};
    const InterleavedAddrGen<weights_is_dram> weights_0 = {
        .bank_base_address = weights_dram_buffer_src_addr , .page_size = page_size};
    const InterleavedAddrGen<out_is_dram> out_0 = {
        .bank_base_address = output_dram_buffer_dst_addr , .page_size = page_size};
    cb_reserve_back(cb_id_inter_input, 1);
    uint32_t inter_input_l1_addr = get_write_ptr(cb_id_inter_input);
    cb_reserve_back(cb_id_inter_weights, 1);
    uint32_t inter_weights_l1_addr = get_write_ptr(cb_id_inter_weights);

    embeddings_<in_is_dram, weights_is_dram, out_is_dram>(
                            embeddings_per_core,
                            input_offset,
                            weight_offset,
                            num_output_rows,
                            page_size,
                            inter_input_l1_addr,
                            inter_weights_l1_addr,
                            input_rows_0,
                            weights_0,
                            out_0
    );



}
