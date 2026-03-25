// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM reader for diagonal helper cores with M_block x N_block streaming.
// Phase 1: Reads N_block odd K-columns from DRAM into c_1 for compute.
//          Loop: for msb: for nsb: for blk: read N_block rows.
// Phase 2: Writes combined output (c_6, after accumulation) to DRAM row by row (per msb).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

inline void fill_tile_zeros(uint32_t write_addr, uint32_t tile_bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t bytes_left = tile_bytes;
    while (bytes_left > 0) {
        uint32_t read_size = (bytes_left > MEM_ZEROS_SIZE) ? MEM_ZEROS_SIZE : bytes_left;
        noc_async_read(zeros_noc_addr, write_addr, read_size);
        write_addr += read_size;
        bytes_left -= read_size;
    }
}

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(5);
    constexpr uint32_t Mpc = get_compile_time_arg_val(6);
    constexpr uint32_t padded_out_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t M_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t M_block = get_compile_time_arg_val(9);
    constexpr uint32_t N_num_subblocks = get_compile_time_arg_val(10);

    constexpr auto input_ta = TensorAccessorArgs<11>();
    constexpr auto output_ta = TensorAccessorArgs<input_ta.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(argidx++);  // row offset (y * Mpc)
    get_arg_val<uint32_t>(argidx++);                               // Mpc (unused, available as compile-time arg)
    const uint32_t K_tiles = get_arg_val<uint32_t>(argidx++);      // padded K for loop structure
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_M_tiles = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_K_tiles = get_arg_val<uint32_t>(argidx++);

    const auto reader = TensorAccessor(input_ta, src_addr, tile_size);

    constexpr uint32_t num_blocks = num_tiles / block_size;
    // block_size = K_block_tiles * M_block (compile-time), so K_block_tiles = block_size / M_block
    constexpr uint32_t K_block_tiles = block_size / M_block;

    for (uint32_t msb = 0; msb < M_num_subblocks; msb++) {
        uint32_t M_start = msb * M_block;
        uint32_t current_M_block = (M_block < Mpc - M_start) ? M_block : (Mpc - M_start);

        // --- Phase 1: read odd K-columns from DRAM, N_block rows per pass ---
        for (uint32_t nsb = 0; nsb < N_num_subblocks; nsb++) {
            uint32_t row_base = nsb * M_block;

            for (uint32_t blk = 0; blk < num_blocks; blk++) {
                uint32_t first_k_col = blk * K_block_tiles * 2 + 1;

                cb_reserve_back(cb_id, block_size);
                uint32_t base_addr = get_write_ptr(cb_id);
                for (uint32_t kb = 0; kb < K_block_tiles; kb++) {
                    uint32_t k_col = first_k_col + kb * 2;
                    for (uint32_t m = 0; m < M_block; m++) {
                        uint32_t cb_offset = (kb * M_block + m) * tile_size;
                        uint32_t global_row = tile_offset + row_base + m;
                        if (global_row < logical_M_tiles && k_col < logical_K_tiles) {
                            uint32_t dram_tile = global_row * logical_K_tiles + k_col;
                            noc_async_read_tile(dram_tile, reader, base_addr + cb_offset);
                        } else {
                            fill_tile_zeros(base_addr + cb_offset, tile_size);
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_id, block_size);
            }
        }

        const auto out_writer = TensorAccessor(output_ta, out_addr, out_tile_size);

#ifdef PER_NSB_REDUCTION
        for (uint32_t m = 0; m < current_M_block; m++) {
            cb_wait_front(cb_out, Mpc);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint32_t row = M_start_tile + M_start + m;
            for (uint32_t n = 0; n < Mpc; n++) {
                uint32_t col = N_start_tile + n;
                if (row < logical_M_tiles && col < logical_M_tiles) {
                    uint32_t tid = row * padded_out_tiles + col;
                    noc_async_write_tile(tid, out_writer, l1_read_addr + n * out_tile_size);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, Mpc);
        }
#else
        for (uint32_t n = 0; n < Mpc; n++) {
            cb_wait_front(cb_out, current_M_block);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint32_t col = N_start_tile + n;
            for (uint32_t m = 0; m < current_M_block; m++) {
                uint32_t row = M_start_tile + M_start + m;
                if (row < logical_M_tiles && col < logical_M_tiles) {
                    uint32_t tid = row * padded_out_tiles + col;
                    noc_async_write_tile(tid, out_writer, l1_read_addr + m * out_tile_size);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, current_M_block);
        }
#endif
    }
}
