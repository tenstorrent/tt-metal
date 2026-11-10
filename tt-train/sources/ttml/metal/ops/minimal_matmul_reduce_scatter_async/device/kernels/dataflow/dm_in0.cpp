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
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t m_piece = get_compile_time_arg_val(7);
    constexpr uint32_t M_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t N_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(11);
    Semaphore<> in0_sender_semaphore(get_compile_time_arg_val(12));
    Semaphore<> in0_receiver_semaphore(get_compile_time_arg_val(13));
    Semaphore<> in0_valid_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(15);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(17);
    constexpr uint32_t slice_tiles = get_compile_time_arg_val(18);
    constexpr bool sink_in1 = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t in1_sender_semaphore_id = get_compile_time_arg_val(21);
    constexpr uint32_t in1_receiver_semaphore_id = get_compile_time_arg_val(22);
    constexpr uint32_t go_semaphore_id = get_compile_time_arg_val(23);
    constexpr bool transpose_a = get_compile_time_arg_val(24) == 1;
    constexpr bool forward_transposed = transpose_a && is_injector_core;

    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t buffer_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    CoreShare share;
    share.m_off = get_arg_val<uint32_t>(argidx++);
    share.m_width = get_arg_val<uint32_t>(argidx++);
    share.n_off = get_arg_val<uint32_t>(argidx++);
    share.n_width = get_arg_val<uint32_t>(argidx++);
    RingWriterArgs ring;
    ring.defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    ring.ring_index = get_arg_val<uint32_t>(argidx++);
    ring.landed_sem_addr = get_arg_val<uint32_t>(argidx++);
    ring.sender_noc_x = get_arg_val<uint32_t>(argidx++);
    ring.sender_noc_y = get_arg_val<uint32_t>(argidx++);
    ring.go_sem_addr = get_semaphore(go_semaphore_id);
    const uint32_t in1_upstream_noc_x = sink_in1 ? get_arg_val<uint32_t>(argidx++) : 0;
    const uint32_t in1_upstream_noc_y = sink_in1 ? get_arg_val<uint32_t>(argidx++) : 0;

    constexpr auto in0_args = TensorAccessorArgs<25>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);
    constexpr auto buffer_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto buffer = TensorAccessor(buffer_args, buffer_addr, out_tile_size);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_in0_transposed(tt::CBIndex::c_7);
    CircularBuffer cb_transposed_ready(tt::CBIndex::c_8);

    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    CircularBuffer cb_in1(cb_in1_id);
    Semaphore<> in1_sender_semaphore(in1_sender_semaphore_id);
    Semaphore<> in1_receiver_semaphore(in1_receiver_semaphore_id);

    in0_valid_semaphore.set(VALID);

    RingWriter<M_block_tiles, N_block_tiles, ring_size, slice_tiles, decltype(out_writer), decltype(buffer)> writer(
        out_writer, buffer, out_tile_size, N_tiles, ring);

    for (uint32_t step = 0; step < ring_size; step++) {
        for (uint32_t nb = 0; nb < N_blocks; nb++) {
            const uint32_t slice = mmrs::step_slice(step, ring.ring_index, ring_size);

            for (uint32_t mb = 0; mb < M_blocks; mb++) {
                const BlockRegion region = block_region<slice_tiles, m_piece, M_blocks, N_blocks>(slice, mb, nb, share);
                const uint32_t current_M_block_tiles = mmrs::balanced_span(mb, m_piece, M_blocks).size;
                const uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;
                const bool is_last_block = step == ring_size - 1 && nb == N_blocks - 1 && mb == M_blocks - 1;

                for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                    if constexpr (sink_in1) {
                        cb_in1.reserve_back(in1_block_num_tiles);
                        in1_receiver_semaphore.set(INVALID);
                        in1_sender_semaphore.up(noc, in1_upstream_noc_x, in1_upstream_noc_y, 1);
                    }
                    cb_in0.reserve_back(in0_block_num_tiles);

                    uint32_t in0_start_address = cb_in0.get_write_ptr();
                    if constexpr (is_injector_core) {
                        read_in0_block<K_block_tiles, transpose_a>(
                            in0_reader,
                            cb_in0_id,
                            in0_tile_size,
                            M_tiles,
                            K_tiles,
                            k_block * K_block_tiles,
                            region.row0,
                            region.rows);
                    } else {
                        in0_receiver_semaphore.set(INVALID);
                        in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                        in0_receiver_semaphore.wait(VALID);
                    }

                    cb_in0.push_back(in0_block_num_tiles);
                    if constexpr (sink_in1) {
                        in1_receiver_semaphore.wait(VALID);
                        cb_in1.push_back(in1_block_num_tiles);
                    }

                    if (!is_sink_core) {
                        in0_sender_semaphore.wait(1);
                        in0_sender_semaphore.set(0);

                        uint32_t forward_address = in0_start_address;
                        if constexpr (forward_transposed) {
                            cb_transposed_ready.wait_front(1);
                            forward_address = cb_in0_transposed.get_read_ptr();
                        }
                        noc.async_write(
                            CoreLocalMem<uint32_t>(forward_address),
                            UnicastEndpoint{},
                            current_block_bytes,
                            {},
                            {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});

#ifdef ARCH_BLACKHOLE
                        noc.async_writes_flushed();
#endif

                        in0_valid_semaphore.relay_unicast(noc, in0_receiver_semaphore, in0_dest_noc_x, in0_dest_noc_y);
                        if constexpr (forward_transposed) {
                            noc.async_writes_flushed();
                            cb_transposed_ready.pop_front(1);
                        }
                    }

                    if constexpr (is_output_writer) {
                        writer.at_k_block(k_block);
                        if (k_block == K_num_blocks - 1) {
                            writer.fetch_incoming(step, region);
                        }
                    }
                }

                if constexpr (is_output_writer) {
                    writer.end_block(step, region, is_last_block);
                }
            }
        }
    }
    if constexpr (is_output_writer) {
        writer.finish();
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
