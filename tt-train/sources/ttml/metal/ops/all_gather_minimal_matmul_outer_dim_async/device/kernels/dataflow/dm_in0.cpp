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

// in0 data mover: the injector reads in0 from DRAM, the others get it from upstream, and every core but the last
// forwards it down the chain.
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
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(11);
    Semaphore<> in0_sender_semaphore(get_compile_time_arg_val(12));
    Semaphore<> in0_receiver_semaphore(get_compile_time_arg_val(13));
    Semaphore<> in0_valid_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(15);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(17);
    constexpr uint32_t stripe_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t piece_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t hops_before = get_compile_time_arg_val(20);
    constexpr bool sink_in1 = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t in1_sender_semaphore_id = get_compile_time_arg_val(23);
    constexpr uint32_t in1_receiver_semaphore_id = get_compile_time_arg_val(24);

    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);    // last core of the chain: nothing to forward
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);  // next core in the chain
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);  // previous core, which feeds us
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    // This core's in1 chain and stripe order: the output writer needs them to know which columns it writes.
    const uint32_t chain_offset = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_width = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    uint32_t stripe_order[ring_size];
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_order[i] = get_arg_val<uint32_t>(argidx++);
    }
    // Sink row only: the previous core of the in1 chain.
    const uint32_t in1_upstream_noc_x = sink_in1 ? get_arg_val<uint32_t>(argidx++) : 0;
    const uint32_t in1_upstream_noc_y = sink_in1 ? get_arg_val<uint32_t>(argidx++) : 0;

    constexpr auto in0_args = TensorAccessorArgs<25>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;
    CircularBuffer cb_in0(cb_in0_id);

    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    CircularBuffer cb_in1(cb_in1_id);
    Semaphore<> in1_sender_semaphore(in1_sender_semaphore_id);
    Semaphore<> in1_receiver_semaphore(in1_receiver_semaphore_id);

    in0_valid_semaphore.set(VALID);  // relay_unicast copies this into the next core's receiver semaphore

    uint32_t cols[N_block_tiles];
    uint32_t width = 0;
    DeferredBlockWrite<M_block_tiles, N_block_tiles> deferred;

    for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
        if constexpr (is_output_writer) {
            block_columns<N_block_tiles, piece_tiles, ring_size, hops_before>(
                n_block_iter, stripe_order, stripe_tiles, chain_offset, chain_width, cols, width);
        }

        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
            uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
            uint32_t current_M_block_tiles = m_tile_end - m_tile;
            // A partial M block only forwards its real rows.
            uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;
            bool is_last_block = (m_block_iter == (M_blocks_per_core - 1)) && (n_block_iter == (N_blocks_per_core - 1));

            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                if constexpr (sink_in1) {  // sink row: no in1 kernel here, so take its block too
                    cb_in1.reserve_back(in1_block_num_tiles);
                    in1_receiver_semaphore.set(INVALID);
                    in1_sender_semaphore.up(noc, in1_upstream_noc_x, in1_upstream_noc_y, 1);
                }
                cb_in0.reserve_back(in0_block_num_tiles);

                uint32_t in0_start_address = cb_in0.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        cb_in0_id,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {  // tell upstream we have room, then wait for the block
                    in0_receiver_semaphore.set(INVALID);
                    in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                    in0_receiver_semaphore.wait(VALID);
                }

                cb_in0.push_back(in0_block_num_tiles);
                if constexpr (sink_in1) {
                    in1_receiver_semaphore.wait(VALID);
                    cb_in1.push_back(in1_block_num_tiles);
                }

                if (!is_sink_core) {  // once downstream has room, forward the block and mark it valid
                    in0_sender_semaphore.wait(1);
                    in0_sender_semaphore.set(0);

                    noc.async_write(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        UnicastEndpoint{},
                        current_block_bytes,
                        {},
                        {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();  // the block must leave before its valid flag
#endif

                    in0_valid_semaphore.relay_unicast(noc, in0_receiver_semaphore, in0_dest_noc_x, in0_dest_noc_y);
                }

                if constexpr (is_output_writer) {  // the previous block's output goes out mid-way through this one
                    if (deferred.pending && k_block == defer_write_k_block) {
                        deferred.write(out_writer, cb_out_id, M_tiles, N_tiles, out_tile_size);
                    }
                }
            }

            if constexpr (is_output_writer) {
                if (is_last_block) {
                    write_block_columns<M_block_tiles, N_block_tiles, true>(
                        out_writer, M_tiles, N_tiles, cb_out_id, out_tile_size, m_tile, m_tile_end, cols, width);
                } else {
                    deferred.defer(m_tile, m_tile_end, cols, width);
                }
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
