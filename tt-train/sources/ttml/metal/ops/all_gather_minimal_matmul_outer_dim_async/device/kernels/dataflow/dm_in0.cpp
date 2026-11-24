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

// in0 data mover: the injector reads in0 from DRAM - when in0 is the gathered operand, its own stripe from the shard
// and the rest once the senders land them -, the others get it from upstream, and every core but the last forwards it
// down the chain.
void kernel_main() {
    Noc noc;
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr bool gather_in0 = get_compile_time_arg_val(1) == 1;  // in0 is the gathered operand; otherwise in1 is
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t outer_blocks = get_compile_time_arg_val(8);  // blocks per core, gathered operand's outer dim
    constexpr uint32_t inner_blocks = get_compile_time_arg_val(9);  // blocks per core, local operand's outer dim
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
    constexpr bool gathered = gather_in0;  // in0 is the gathered operand

    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);    // last core of the chain: nothing to forward
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);  // next core in the chain
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);  // previous core, which feeds us
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    // This core's range of the local operand's outer dimension: M tiles, or N tiles when in0 is gathered.
    const uint32_t local_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_end_tile = get_arg_val<uint32_t>(argidx++);
    // This core's chain of the gathered operand: its slice of every stripe.
    const uint32_t chain_offset = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_width = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    uint32_t stripe_order[ring_size];  // stripes in the order this device uses them; [0] is our own
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_order[i] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t local_stripe = stripe_order[0];

    constexpr auto in0_args = TensorAccessorArgs<25>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);
    // Only the gathered operand has a persistent buffer and stripe semaphores. A local in0 is one stripe that holds it
    // all, so its buffer accessor is never read and stands in as the input's.
    constexpr auto pgb_args = TensorAccessorArgs<gathered ? out_args.next_compile_time_args_offset() : 25>();
    const uint32_t pgb_addr = gathered ? get_arg_val<uint32_t>(argidx++) : in0_addr;
    const auto pgb_reader = TensorAccessor(pgb_args, pgb_addr);
    uint32_t stripe_sem_addrs[ring_size] = {};
    if constexpr (gathered) {
        for (uint32_t i = 0; i < ring_size; i++) {
            stripe_sem_addrs[i] = get_arg_val<uint32_t>(argidx++);
        }
    }
    const StripeSource in0_source =
        gathered ? StripeSource{M_tiles, local_stripe * stripe_tiles, stripe_tiles} : StripeSource{M_tiles, 0, M_tiles};
    // Sink row only: the previous core of the in1 chain.
    const uint32_t in1_upstream_noc_x = sink_in1 ? get_arg_val<uint32_t>(argidx++) : 0;
    const uint32_t in1_upstream_noc_y = sink_in1 ? get_arg_val<uint32_t>(argidx++) : 0;
    uint32_t arrived[ring_size] = {};  // K blocks already seen per stripe

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;
    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_in1(cb_in1_id);
    Semaphore<> in1_sender_semaphore(in1_sender_semaphore_id);
    Semaphore<> in1_receiver_semaphore(in1_receiver_semaphore_id);

    in0_valid_semaphore.set(VALID);  // relay_unicast copies this into the next core's receiver semaphore

    // The gathered operand's blocks are the outer loop: every block of the local operand runs against one of them
    // before the next one starts, so remote stripes are needed as late as possible.
    constexpr uint32_t local_block_tiles = gather_in0 ? N_block_tiles : M_block_tiles;
    constexpr uint32_t local_tiles = gather_in0 ? N_tiles : M_tiles;
    using Block = BlockTiles<gather_in0, M_block_tiles, N_block_tiles, piece_tiles, ring_size, hops_before>;
    Block block;
    DeferredBlockWrite<Block> deferred;

    for (uint32_t outer = 0; outer < outer_blocks; outer++) {
        block.set_gathered(outer, stripe_order, stripe_tiles, chain_offset, chain_width);

        for (uint32_t inner = 0; inner < inner_blocks; inner++) {
            const uint32_t local_tile = local_start_tile + (inner * local_block_tiles);
            block.set_local(local_tile, std::min(local_tile + local_block_tiles, local_end_tile), local_tiles);
            // All-padding blocks aren't sent on, and a partial block only forwards its rows.
            const uint32_t forward_bytes = block.rows_real ? block.height * K_block_tiles * in0_tile_size : 0;
            const bool is_last_block = (outer == outer_blocks - 1) && (inner == inner_blocks - 1);

            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                if constexpr (sink_in1) {  // sink row: no in1 kernel here, so take its block too
                    cb_in1.reserve_back(in1_block_num_tiles);
                    in1_receiver_semaphore.set(INVALID);
                    in1_sender_semaphore.up(noc, in1_upstream_noc_x, in1_upstream_noc_y, 1);
                }
                cb_in0.reserve_back(in0_block_num_tiles);

                const uint32_t in0_start_address = cb_in0.get_write_ptr();
                if constexpr (is_injector_core) {
                    if (block.rows_real) {
                        if constexpr (gathered) {
                            // tiles of other devices' stripes: wait until this K block is in the persistent buffer
                            wait_for_block_stripes(
                                block.rows,
                                block.height,
                                local_stripe,
                                stripe_tiles,
                                stripe_sem_addrs,
                                k_block + 1,
                                arrived);
                        }
                        read_in0_block_rows<K_block_tiles>(
                            in0_reader,
                            pgb_reader,
                            in0_source,
                            cb_in0_id,
                            in0_tile_size,
                            K_tiles,
                            k_block * K_block_tiles,
                            block.rows,
                            block.height);
                    }
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

                    if (forward_bytes > 0) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(in0_start_address),
                            UnicastEndpoint{},
                            forward_bytes,
                            {},
                            {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});
                    }

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();  // the block must leave before its valid flag
#endif

                    in0_valid_semaphore.relay_unicast(noc, in0_receiver_semaphore, in0_dest_noc_x, in0_dest_noc_y);
                }

                if constexpr (is_output_writer) {  // the previous block's output goes out mid-way through this one
                    if (deferred.pending && k_block == defer_write_k_block) {
                        deferred.write(out_writer, cb_out_id, N_tiles, out_tile_size);
                    }
                }
            }

            if constexpr (is_output_writer) {
                if (is_last_block) {
                    write_block_tiles<true>(out_writer, N_tiles, cb_out_id, out_tile_size, block);
                } else {
                    deferred.defer(block);
                }
            }
        }
    }
    if constexpr (is_injector_core && gathered) {
        release_block_stripes(stripe_sem_addrs, arrived, ring_size, K_num_blocks);
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
