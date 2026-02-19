// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Phase 3 in0 DM kernel for muon_precondition: X' = H @ X + a*X.
// Adapted from Phase 2's dm_in0_sender_gsq.cpp.
// Key differences:
//   - Two TensorAccessors: H (in0 for matmul rows) and X (c_4 for aX epilogue)
//   - Reads X[m,n] output block into c_4 instead of G[i,j]
//   - Output shape is [M,K] rectangular, not [M,M] square
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
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    uint32_t in0_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(13));
    uint32_t in0_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    uint32_t in0_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(16);

    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t x_addr = get_arg_val<uint32_t>(argidx++);

    // Input tensor accessor for H (in0 matmul rows)
    constexpr auto in0_args = TensorAccessorArgs<17>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, in0_tile_size);

    // Output tensor accessor (X')
    constexpr uint32_t out_tensor_args_cta_offset = in0_args.next_compile_time_args_offset();
    constexpr auto out_args = TensorAccessorArgs<out_tensor_args_cta_offset>();
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);

    // X tensor accessor (for aX epilogue reads into c_4)
    constexpr uint32_t x_tensor_args_cta_offset = out_args.next_compile_time_args_offset();
    constexpr auto x_args = TensorAccessorArgs<x_tensor_args_cta_offset>();
    const auto x_reader = TensorAccessor(x_args, x_addr, in0_tile_size);

    // H is square [M,M]: in0 shape uses M_tiles for both dims
    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    // Output X' is [M,N] where N = K_tiles of X (rectangular)
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    // X is [M,N] — same shape as output, used for aX epilogue
    const TensorShape2D x_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_x_input = tt::CBIndex::c_4;

    volatile tt_l1_ptr uint32_t* in0_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_valid_semaphore_addr);
    *(in0_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);

    volatile tt_l1_ptr uint32_t* in0_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_semaphore_addr);
    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_semaphore_addr);

    const uint64_t in0_receiver_semaphore_noc_addr =
        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);

    bool k_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
        uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;

        reuse_block = false;
        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);

            // Issue X[m,n] DRAM reads asynchronously — they overlap with the K-block loop.
            // Same pattern as Phase 2's G[i,j] reads.
            cb_reserve_back(cb_id_x_input, out_block_num_tiles);
            uint32_t x_write_ptr = get_write_ptr(cb_id_x_input);
            read_g_block_async<M_block_tiles, N_block_tiles>(
                x_reader, x_shape, x_write_ptr, in0_tile_size, m_tile, m_tile_end, n_tile, n_tile_end);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    cb_wait_front(cb_id_out, out_block_num_tiles);
                    uint32_t out_read_ptr = get_read_ptr(cb_id_out);

                    write_block_sync<M_block_tiles, N_block_tiles>(
                        out_writer,
                        out_shape,
                        out_read_ptr,
                        out_tile_size,
                        defer_write_m_tile,
                        defer_write_m_tile_end,
                        defer_write_n_tile,
                        defer_write_n_tile_end);
                    cb_pop_front(cb_id_out, out_block_num_tiles);
                }

                if (reuse_block && k_block_iter == 0) {
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);

                uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        in0_start_address,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
                }

                cb_push_back(cb_id_in0, in0_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);

                    uint64_t in0_unicast_data_addr = get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);
                    noc_async_write(in0_start_address, in0_unicast_data_addr, current_block_bytes);

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                }
            }

            // X[m,n] reads were issued before the K-loop. Barrier is a formality.
            noc_async_read_barrier();
            cb_push_back(cb_id_x_input, out_block_num_tiles);

            k_forward = !k_forward;
            reuse_block = true;

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            defer_write = !((m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1)));
            defer_write = defer_write && !is_injector_core;

            if (!defer_write) {
                write_block_sync_granular<M_block_tiles, N_block_tiles>(
                    out_writer, out_shape, cb_id_out, out_tile_size, m_tile, m_tile_end, n_tile, n_tile_end);
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
