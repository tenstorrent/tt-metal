// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/moreh_sum_backward/kernels/utils.hpp"

template <typename T>
inline T get_next_arg_val()
{
    static int arg_idx = 0;
    return get_arg_val<T> (arg_idx++);
}

void kernel_main() {
    const auto output_grad_addr = get_next_arg_val<uint32_t>();
    const auto num_output_tiles = get_next_arg_val<uint32_t>();
    const auto start_id = get_next_arg_val<uint32_t>();
    const auto output_grad_is_dram = (get_next_arg_val<uint32_t>() == 1);

    const auto output_grad_n = get_next_arg_val<uint32_t>();
    const auto output_grad_c = get_next_arg_val<uint32_t>();
    const auto output_grad_ht = get_next_arg_val<uint32_t>();
    const auto output_grad_wt = get_next_arg_val<uint32_t>();

    const auto input_grad_n = get_next_arg_val<uint32_t>();
    const auto input_grad_c = get_next_arg_val<uint32_t>();
    const auto input_grad_ht = get_next_arg_val<uint32_t>();
    const auto input_grad_wt = get_next_arg_val<uint32_t>();

    const auto n_need_bcast = get_next_arg_val<uint32_t>();
    const auto c_need_bcast = get_next_arg_val<uint32_t>();
    const auto ht_need_bcast = get_next_arg_val<uint32_t>();
    const auto wt_need_bcast = get_next_arg_val<uint32_t>();
    const auto num_dim = get_next_arg_val<uint32_t>();

    const auto output_grad_HtWt = output_grad_ht * output_grad_wt;
    const auto output_grad_CHtWt = output_grad_c * output_grad_HtWt;

    const auto input_grad_HtWt = input_grad_ht * input_grad_wt;
    const auto input_grad_CHtWt = input_grad_c * input_grad_HtWt;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;

    // zero tile
    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    scaler.f = 1.0f / num_dim;
    fill_cb_with_value(cb_id_in2, scaler.u, 1);

    uint32_t l1_write_addr_in0;
    uint32_t output_grad_tile_bytes = get_tile_size(cb_id_in0);
    const auto output_grad_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<true> dram_output_grad_addrg = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_bytes,
        .data_format = output_grad_data_format};
    const InterleavedAddrGenFast<false> l1_output_grad_addrg = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_bytes,
        .data_format = output_grad_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        const auto cur_n = i / input_grad_CHtWt;
        const auto cur_c = (i / input_grad_HtWt) % input_grad_c;
        const auto cur_ht = (i / input_grad_wt) % input_grad_ht;
        const auto cur_wt = i % input_grad_wt;

        const auto target_n = (n_need_bcast) ? (0) : (cur_n);
        const auto target_c = (c_need_bcast) ? (0) : (cur_c);
        const auto target_ht = (ht_need_bcast) ? (0) : (cur_ht);
        const auto target_wt = (wt_need_bcast) ? (0) : (cur_wt);

        auto read_tile_id =
            target_n * output_grad_CHtWt + target_c * output_grad_HtWt + target_ht * output_grad_wt + target_wt;

        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        if (output_grad_is_dram) {
            noc_async_read_tile(read_tile_id, dram_output_grad_addrg, l1_write_addr_in0);
        } else {
            noc_async_read_tile(read_tile_id, l1_output_grad_addrg, l1_write_addr_in0);
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
