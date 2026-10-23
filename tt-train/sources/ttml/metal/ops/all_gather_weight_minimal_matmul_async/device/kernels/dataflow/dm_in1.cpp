// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(8);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(11);
    Semaphore<> in1_sender_semaphore(get_compile_time_arg_val(12));
    Semaphore<> in1_receiver_semaphore(get_compile_time_arg_val(13));
    Semaphore<> in1_valid_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(15);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(17);
    constexpr uint32_t stripe_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t piece_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t hops_before = get_compile_time_arg_val(20);
    constexpr bool transpose_b = get_compile_time_arg_val(21) == 1;

    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_offset = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_width = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    uint32_t stripe_order[ring_size];
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_order[i] = get_arg_val<uint32_t>(argidx++);
    }

    constexpr auto in1_args = TensorAccessorArgs<22>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr);
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);
    constexpr auto pwb_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const uint32_t pwb_addr = get_arg_val<uint32_t>(argidx++);
    const auto pwb_reader = TensorAccessor(pwb_args, pwb_addr);
    uint32_t stripe_sem_addrs[ring_size];
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_sem_addrs[i] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t local_stripe = stripe_order[0];
    uint32_t arrived[ring_size] = {};

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;

    constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;
    CircularBuffer cb_in1(cb_in1_id);

    in1_valid_semaphore.set(VALID);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    uint32_t cols[N_block_tiles];
    uint32_t width = 0;
    DeferredBlockWrite<M_block_tiles, N_block_tiles> deferred;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            const bool any_real = block_columns<N_block_tiles, piece_tiles, ring_size, hops_before>(
                n_block_iter, stripe_order, stripe_tiles, chain_offset, chain_width, cols, width);
            const uint32_t forward_row_bytes = any_real ? width * in1_tile_size : 0;
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));

            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                if constexpr (is_output_writer) {
                    if (deferred.pending && k_block == defer_write_k_block) {
                        deferred.write(out_writer, cb_out_id, M_tiles, N_tiles, out_tile_size);
                    }
                }

                cb_in1.reserve_back(in1_block_num_tiles);

                uint32_t in1_start_address = cb_in1.get_write_ptr();
                if constexpr (is_injector_core) {
                    if (any_real) {
                        wait_for_block_stripes(
                            cols, width, local_stripe, stripe_tiles, stripe_sem_addrs, k_block + 1, arrived);
                        read_in1_block_columns<K_block_tiles, N_block_tiles, transpose_b>(
                            in1_reader,
                            pwb_reader,
                            cb_in1_id,
                            in1_tile_size,
                            K_tiles,
                            N_tiles,
                            k_block * K_block_tiles,
                            cols,
                            width,
                            local_stripe,
                            stripe_tiles);
                    }
                } else {
                    in1_receiver_semaphore.set(INVALID);
                    in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                    in1_receiver_semaphore.wait(VALID);
                }

                cb_in1.push_back(in1_block_num_tiles);

                if (!is_sink_core) {
                    in1_sender_semaphore.wait(1);
                    in1_sender_semaphore.set(0);

                    if (forward_row_bytes > 0) {
                        for (uint32_t i = 0; i < K_block_tiles; i++) {
                            noc.async_write(
                                CoreLocalMem<uint32_t>(in1_start_address),
                                UnicastEndpoint{},
                                forward_row_bytes,
                                {},
                                {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = in1_start_address});
                            in1_start_address += full_N_tiles_bytes;
                        }
                    }

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in1_valid_semaphore.relay_unicast(noc, in1_receiver_semaphore, in1_dest_noc_x, in1_dest_noc_y);
                }
            }

            if constexpr (is_output_writer) {
                if (is_last_block || is_injector_core) {
                    write_block_columns<M_block_tiles, N_block_tiles, true>(
                        out_writer, M_tiles, N_tiles, cb_out_id, out_tile_size, m_tile, m_tile_end, cols, width);
                } else {
                    deferred.defer(m_tile, m_tile_end, cols, width);
                }
            }
        }
    }
    if constexpr (is_injector_core) {
        for (uint32_t s = 0; s < ring_size; s++) {
            if (arrived[s] > 0) {
                noc_semaphore_inc(get_noc_addr(stripe_sem_addrs[s]), -K_num_blocks);
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
