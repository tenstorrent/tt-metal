// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr  = get_arg_val<uint32_t>(0);
    const uint32_t k_addr  = get_arg_val<uint32_t>(1);
    const uint32_t v_addr         = get_arg_val<uint32_t>(2);
    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);

    const InterleavedAddrGenFast<is_dram> q_reader = {
        .bank_base_address = q_addr,
        .page_size = q_tile_bytes,
        .data_format = q_data_format
    };

    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr,
        .page_size = k_tile_bytes,
        .data_format = k_data_format
    };

    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr,
        .page_size = v_tile_bytes,
        .data_format = v_data_format
    };

    // Read just one tile for Q, K and V.

    // Read Q chunk
    cb_reserve_back(cb_q_in, 1);
    uint32_t q_write_ptr = get_write_ptr(cb_q_in);
    noc_async_read_tile(0, q_reader, q_write_ptr);
    noc_async_read_barrier();
    cb_push_back(cb_q_in, 1);

    // Read K chunk transposed
    cb_reserve_back(cb_k_in, 1);
    uint32_t k_write_ptr = get_write_ptr(cb_k_in);
    noc_async_read_tile(0, k_reader, k_write_ptr);
    noc_async_read_barrier();
    cb_push_back(cb_k_in, 1);

    // Read V chunk
    cb_reserve_back(cb_v_in, 1);
    uint32_t v_write_ptr = get_write_ptr(cb_v_in);
    noc_async_read_tile(0, v_reader, v_write_ptr);
    noc_async_read_barrier();
    cb_push_back(cb_v_in, 1);
}
