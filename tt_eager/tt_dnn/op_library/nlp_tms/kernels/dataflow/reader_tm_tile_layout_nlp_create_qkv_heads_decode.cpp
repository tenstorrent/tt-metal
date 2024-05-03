// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    uint32_t head_size                     = get_arg_val<uint32_t>(0);
    uint32_t num_q_heads                   = get_arg_val<uint32_t>(1);
    uint32_t num_kv_heads                   = get_arg_val<uint32_t>(2);
    uint32_t head_size_num_tiles           = get_arg_val<uint32_t>(3);
    uint32_t in_tile_offset_by_batch       = get_arg_val<uint32_t>(4);
    uint32_t start_q_x                     = get_arg_val<uint32_t>(5);
    uint32_t start_q_y                     = get_arg_val<uint32_t>(6);
    uint32_t q_start_addr                  = get_arg_val<uint32_t>(7);
    uint32_t k_start_addr                  = get_arg_val<uint32_t>(8);
    uint32_t v_start_addr                  = get_arg_val<uint32_t>(9);

    constexpr uint32_t ELEMENT_SIZE        = get_compile_time_arg_val(0);
    constexpr uint32_t SUBTILE_LINE_BYTES  = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_q_out         = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_k_out         = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_v_out         = get_compile_time_arg_val(4);

    uint32_t num_x                         = get_arg_val<uint32_t>(10);
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(8));
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(8 + num_x));

    // Q
    uint32_t q_x = start_q_x;
    uint32_t q_y = start_q_y;
    uint64_t q_read_addr = get_noc_addr(in0_mcast_noc_x[q_x], in0_mcast_noc_y[q_y], q_start_addr) + in_tile_offset_by_batch;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size/head_size_num_tiles;

    // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
    for (uint32_t q = 0; q < num_q_heads; ++q) {
        uint32_t wptr_offset = q < 16 ? q * SUBTILE_LINE_BYTES : (q - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t q_write_addr = get_write_ptr(cb_id_q_out) + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            noc_async_read(q_read_addr, q_write_addr, SUBTILE_LINE_BYTES);
            noc_async_read_barrier();
            // Read second phase
            noc_async_read(q_read_addr+256*ELEMENT_SIZE, q_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            noc_async_read_barrier();

            q_read_addr += tile_size;
            q_write_addr += tile_size;
        }
    }

    // K
    uint32_t k_x = start_q_x;
    uint32_t k_y = start_q_y;
    uint64_t k_read_addr = get_noc_addr(in0_mcast_noc_x[k_x], in0_mcast_noc_y[k_y], k_start_addr) + in_tile_offset_by_batch;
    uint32_t k_write_addr = 0;

    // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
    for (uint32_t k = 0; k < num_kv_heads; ++k) {
        uint32_t wptr_offset = k < 16 ? k * SUBTILE_LINE_BYTES : (k - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t k_write_addr = get_write_ptr(cb_id_k_out) + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            noc_async_read(k_read_addr, k_write_addr, SUBTILE_LINE_BYTES);
            noc_async_read_barrier();
            // Read second phase
            noc_async_read(k_read_addr+256*ELEMENT_SIZE, k_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            noc_async_read_barrier();

            k_read_addr += tile_size;
            k_write_addr += tile_size;
        }
    }

    // v
    uint32_t v_x = start_q_x;
    uint32_t v_y = start_q_y;
    uint64_t v_read_addr = get_noc_addr(in0_mcast_noc_x[v_x], in0_mcast_noc_y[v_y], v_start_addr) + in_tile_offset_by_batch;
    uint32_t v_write_addr = 0;

    // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
    for (uint32_t v = 0; v < num_kv_heads; ++v) {
        uint32_t wptr_offset = v < 16 ? v * SUBTILE_LINE_BYTES : (v - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t v_write_addr = get_write_ptr(cb_id_v_out) + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            noc_async_read(v_read_addr, v_write_addr, SUBTILE_LINE_BYTES);
            noc_async_read_barrier();
            // Read second phase
            noc_async_read(v_read_addr+256*ELEMENT_SIZE, v_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            noc_async_read_barrier();

            v_read_addr += tile_size;
            v_write_addr += tile_size;
        }
    }
}
