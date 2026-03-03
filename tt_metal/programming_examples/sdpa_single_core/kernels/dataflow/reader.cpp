// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(3);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_chunks = get_compile_time_arg_val(5);

    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);

    const uint32_t q_tile_size_bytes = get_tile_size(tt::CBIndex::c_0);
    const uint32_t kv_tile_size_bytes = get_tile_size(tt::CBIndex::c_1);

    const auto q_accessor = TensorAccessor(q_args, q_addr, q_tile_size_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, kv_tile_size_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, kv_tile_size_bytes);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_3;

    constexpr uint32_t num_q_tiles = Sq_chunk_t * head_dim_t;
    constexpr uint32_t num_kt_tiles = head_dim_t * Sk_chunk_t;
    constexpr uint32_t num_v_tiles = Sv_chunk_t * head_dim_t;

    for (uint32_t q = 0; q < num_q_chunks; q++) {
        // Read Q chunk once per Q iteration
        cb_reserve_back(cb_q_in, num_q_tiles);
        uint32_t q_l1_addr = get_write_ptr(cb_q_in);
        uint32_t q_tile_offset = q * num_q_tiles;
        for (uint32_t t = 0; t < num_q_tiles; t++) {
            noc_async_read_tile(q_tile_offset + t, q_accessor, q_l1_addr);
            q_l1_addr += q_tile_size_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_in, num_q_tiles);

        // Read K and V for each K chunk
        for (uint32_t k = 0; k < num_k_chunks; k++) {
            // Read K tiles from DRAM and transpose into CB
            // K in DRAM: [Sk_chunk_t × head_dim_t] per chunk (row-major tiles)
            // CB after:  [head_dim_t × Sk_chunk_t] (transposed layout for compute)
            cb_reserve_back(cb_kt_in, num_kt_tiles);
            uint32_t base_kt_l1_addr = get_write_ptr(cb_kt_in);
            uint32_t k_tile_id = k * num_kt_tiles;
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                uint32_t write_ptr = base_kt_l1_addr + row * kv_tile_size_bytes;
                for (uint32_t col = 0; col < head_dim_t; ++col) {
                    noc_async_read_tile(k_tile_id++, k_accessor, write_ptr);
                    write_ptr += kv_tile_size_bytes * Sk_chunk_t;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_kt_in, num_kt_tiles);

            // Read V tiles from DRAM
            cb_reserve_back(cb_v_in, num_v_tiles);
            uint32_t v_l1_addr = get_write_ptr(cb_v_in);
            uint32_t v_tile_offset = k * num_v_tiles;
            for (uint32_t t = 0; t < num_v_tiles; t++) {
                noc_async_read_tile(v_tile_offset + t, v_accessor, v_l1_addr);
                v_l1_addr += kv_tile_size_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_v_in, num_v_tiles);
        }
    }
}
