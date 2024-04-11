// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t q_heads_per_group               = get_compile_time_arg_val(0); // number of Q heads in the group, n
    constexpr uint32_t k_heads_per_group               = get_compile_time_arg_val(1); // number of K heads in the group, expecting 1
    constexpr uint32_t v_heads_per_group               = get_compile_time_arg_val(2); // number of V heads in the group, expecting 1


    constexpr uint32_t q_out_block_wt_size_bytes       = get_compile_time_arg_val(3); // size of a Q head in bytes
    constexpr uint32_t k_out_block_wt_size_bytes       = get_compile_time_arg_val(4); // size of a K head `` ``
    constexpr uint32_t v_out_block_wt_size_bytes       = get_compile_time_arg_val(5); // size of a V head `` ``

    constexpr uint32_t block_wt_size_bytes             = get_compile_time_arg_val(6); // size of each group, used for skipping each group
    constexpr uint32_t block_ht                        = get_compile_time_arg_val(7); // number of tiles along the seq_len*batch dimension
    constexpr uint32_t block_groups_wt                 = get_compile_time_arg_val(8); // number of tiles inside one nQ K V group

    constexpr uint32_t q_num_tiles                     = get_compile_time_arg_val(9); // total number of Q pages, used for CB reservation
    constexpr uint32_t k_num_tiles                     = get_compile_time_arg_val(10); // total number of K pages
    constexpr uint32_t v_num_tiles                     = get_compile_time_arg_val(11); // total number of V pages

    constexpr uint32_t q_size_per_group                = get_compile_time_arg_val(12); // total size of all n Q heads in a group, used to find start of first K head
    constexpr uint32_t q_k_size_per_group              = get_compile_time_arg_val(13); // total size of all n Q heads and K heads (expecting 1) in a group

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_out0 = tt::CB::c_out0;
    constexpr uint32_t cb_out1 = tt::CB::c_out1;
    constexpr uint32_t cb_out2 = tt::CB::c_out2;


    // copy one entire head_dim tile, then go to next sequence*batch tile and do another head_dim.
    // after that, go to next head (head_dim > sequence * batch > head)
    // since Q's heads are shuffled and n Q heads are interleaved with a KV, have to do 3 loops


    const uint32_t single_tile_size_bytes = get_tile_size(cb_in0);
    const DataFormat data_format = get_dataformat(cb_in0);

    /**
     * Iterate over number of heads in each group (n Q, 1 K, 1 V) where total number of groups = total number of KV heads
     * block_ht is the number of tiles along the batch * seq_len dimension shard
    */
    uint64_t src_noc_addr = get_noc_addr(get_read_ptr(cb_in0));

    // re-order q
    cb_reserve_back(cb_out0, q_num_tiles);
    uint32_t l1_write_addr_out0 = get_write_ptr(cb_out0);
    uint32_t src_noc_addr_offset_outer = 0;
    for (uint32_t k = 0; k < block_groups_wt; k++) { // number of groups, divided into tiles
        uint32_t l1_read_addr_offset = 0; // start at 0 since Q is always first
        for (uint32_t j = 0; j < q_heads_per_group; j++) { // go to next Q heads in the group (0 to n-1 for the nQ per KV group)
            for (uint32_t i = 0; i < block_ht; i++) { // iterate across seq_len dimension tiles
                uint64_t q_src_noc_addr = src_noc_addr + l1_read_addr_offset + src_noc_addr_offset_outer;
                noc_async_read(q_src_noc_addr, l1_write_addr_out0, q_out_block_wt_size_bytes); // read one head worth of tiles
                l1_write_addr_out0 += q_out_block_wt_size_bytes; // read in one Q head
                l1_read_addr_offset += block_wt_size_bytes; // go to next group along sequence
            }
            src_noc_addr_offset_outer += q_out_block_wt_size_bytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_out0, q_num_tiles);

    // re-order k
    cb_reserve_back(cb_out1, k_num_tiles);
    uint32_t l1_write_addr_out1 = get_write_ptr(cb_out1);
    src_noc_addr_offset_outer = 0;
    for (uint32_t k = 0; k < block_groups_wt; ++k) { // m kv head tiles
        uint32_t l1_read_addr_offset = q_size_per_group; // skip the first n*Q since we're writing K
        for (uint32_t j = 0; j < k_heads_per_group; j++) { // 1 k tile per kv head
            for (uint32_t i = 0; i < block_ht; i++) { // seq_len
                uint64_t k_src_noc_addr = src_noc_addr + l1_read_addr_offset + src_noc_addr_offset_outer;
                noc_async_read(k_src_noc_addr, l1_write_addr_out1, k_out_block_wt_size_bytes);
                l1_write_addr_out1 += k_out_block_wt_size_bytes;
                l1_read_addr_offset += block_wt_size_bytes;
            }
            src_noc_addr_offset_outer += k_out_block_wt_size_bytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_out1, k_num_tiles);


    // re-order v
    cb_reserve_back(cb_out2, v_num_tiles);
    uint32_t l1_write_addr_out2 = get_write_ptr(cb_out2);
    src_noc_addr_offset_outer = 0;
    for (uint32_t k = 0; k < block_groups_wt; ++k) { // number of group-tiles
        uint32_t l1_read_addr_offset = q_k_size_per_group;
        for (uint32_t j = 0; j < v_heads_per_group; j++) { // 1 v tile per kv head
            for (uint32_t i = 0; i < block_ht; i++) { // sequence length
                uint64_t v_src_noc_addr = src_noc_addr + l1_read_addr_offset + src_noc_addr_offset_outer;
                noc_async_read(v_src_noc_addr, l1_write_addr_out2, v_out_block_wt_size_bytes);
                l1_write_addr_out2 += v_out_block_wt_size_bytes;
                l1_read_addr_offset += block_wt_size_bytes;
            }
            src_noc_addr_offset_outer += v_out_block_wt_size_bytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_out2, v_num_tiles);
}
