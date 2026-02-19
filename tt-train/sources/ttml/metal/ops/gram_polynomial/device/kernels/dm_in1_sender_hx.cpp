// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Phase 3 in1 DM kernel for muon_precondition: X' = H @ X + a*X.
// Adapted from Phase 2's dm_in1_sender_gsq.cpp.
// Key difference: reads X[k,n] instead of G[k,n]. Shape is [M,K] rectangular.
//

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gsq_dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(11);
    uint32_t in1_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));
    uint32_t in1_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(13));
    uint32_t in1_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(15);

    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);

    // Tensor accessor for X [M, K_x] — in1 reads X columns for the H@X matmul
    constexpr auto in1_args = TensorAccessorArgs<16>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr, in1_tile_size);

    // X shape for in1: rows = M (inner dim of H@X), cols = N_tiles (K of X)
    // read_in1_block_sync indexes as [k, n] where k iterates over M_tiles (inner dim)
    // and n iterates over the output columns. For H@X: inner dim = M, output cols = K_x.
    // in1_shape.logical_d0 = M_tiles (inner dim), in1_shape.logical_d1 = N_tiles (output cols)
    const TensorShape2D in1_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    volatile tt_l1_ptr uint32_t* in1_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_valid_semaphore_addr);
    *(in1_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in1_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in1_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_semaphore_addr);
    const uint64_t in1_sender_semaphore_noc_addr =
        get_noc_addr(in1_sender_noc_x, in1_sender_noc_y, in1_sender_semaphore_addr);

    const uint64_t in1_receiver_semaphore_noc_addr =
        get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, in1_receiver_semaphore_addr);

    const uint64_t in1_unicast_data_base_addr = get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, 0);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    bool k_forward = true;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        k_forward = true;

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

                uint32_t in1_start_address = get_write_ptr(cb_id_in1);
                if constexpr (is_injector_core) {
                    read_in1_block_sync<K_block_tiles, N_block_tiles>(
                        in1_reader,
                        in1_shape,
                        in1_start_address,
                        in1_tile_size,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles,
                        n_tile,
                        n_tile_end);
                } else {
                    noc_semaphore_set(in1_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in1_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in1_receiver_semaphore_addr_ptr, VALID);
                }

                cb_push_back(cb_id_in1, in1_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in1_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);

                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        uint64_t in1_unicast_data_addr = in1_unicast_data_base_addr | in1_start_address;
                        noc_async_write(in1_start_address, in1_unicast_data_addr, current_N_tiles_bytes);
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                }
            }

            k_forward = !k_forward;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
