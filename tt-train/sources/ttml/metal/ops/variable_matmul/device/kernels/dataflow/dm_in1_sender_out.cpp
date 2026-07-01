// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "matmul_dataflow_common.hpp"

void kernel_main() {
    // M_tiles, padded_M_tiles, M_blocks_per_core, K_tiles come from runtime args (padded_K_tiles
    // is derived locally below).
    constexpr uint32_t N_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    Semaphore in1_sender_sem(get_compile_time_arg_val(8));
    Semaphore in1_receiver_sem(get_compile_time_arg_val(9));
    Semaphore in1_valid_sem(get_compile_time_arg_val(10));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(11);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(12);
    constexpr bool transpose_b = static_cast<bool>(get_compile_time_arg_val(13));
    constexpr bool use_offset_in1 = static_cast<bool>(get_compile_time_arg_val(14));
    constexpr bool use_out_offset = static_cast<bool>(get_compile_time_arg_val(15));

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    // OFFSET_ROW_MODE overrides these from on-device offsets (per-core M re-derived).
    uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    // Host passes core.y here; we compute defer_write_k_block from runtime K_num_blocks below.
    const uint32_t core_y_index = get_arg_val<uint32_t>(argidx++);

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output address comes next.

    // Tensor accessors (scalar CTAs 0..15; tensor accessors start at 16: in1, output, then offsets).
    constexpr auto in1_args = TensorAccessorArgs<16>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr, in1_tile_size);

    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto out_accessor = TensorAccessor(out_args, get_arg_val<uint32_t>(out_addr_rt_arg_idx), out_tile_size);

    // Variable-M: read actual M values from runtime args (after output address).
    // OFFSET_ROW_MODE overrides M_tiles and M_blocks_per_core from on-device offsets.
    uint32_t M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 1);
    const uint32_t padded_M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 2);
    uint32_t M_blocks_per_core = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 3);
    // Row and K offsets are initialized to 0 here and overwritten below from offsets[start..start+2].
    uint32_t out_row_offset_tiles = 0U;
    uint32_t in1_k_offset_tiles = 0U;
    uint32_t parent_K_tiles_stride_in1 = 0U;
    if constexpr (use_offset_in1) {
        parent_K_tiles_stride_in1 = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 4);
    }
    // OFFSET_K_MODE overrides K_tiles from on-device offsets[start..start+2].
    uint32_t K_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 5);

    Noc noc;

    // Read offsets from a 1-D UINT32 ROW_MAJOR device tensor. One mode per role:
    //   OFFSET_ROW_MODE (InputAndOutputRow): re-derives per-core M_start / M_end / M_blocks_per_core
    //                    locally (matches dm_in0_sender so both kernels agree) and, on the writer
    //                    kernel (transpose_core_grid), sets out_row_offset_tiles from offsets[start].
    //   OFFSET_K_MODE   (InputAndWeightK):    sets in1_k_offset_tiles + K_tiles locally (dm_in0
    //                    owns the K offset's cb_ctrl publish).
    {
        const uint32_t offsets_addr = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 6);
        const uint32_t offsets_start_index = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 7);
        constexpr auto offsets_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
        const auto offsets_acc = TensorAccessor(offsets_args, offsets_addr);

        // Use the in1 CB's L1 write pointer as scratch — unused at kernel startup; the
        // real in1 tile reads only begin inside the K-loop below.
        // Limitation: this reads exactly ONE page (page 0) of the offsets tensor. The
        // (start, end) pair must therefore fall within page 0 — i.e. (E + 1) * sizeof(uint32_t)
        // <= the offsets tensor's page size, where E is num_experts.
        const uint32_t offsets_l1_addr = CircularBuffer(tt::CBIndex::c_1).get_write_ptr();
        noc.async_read(
            offsets_acc,
            CoreLocalMem<uint32_t>(offsets_l1_addr),
            offsets_acc.get_aligned_page_size(),
            {.page_id = 0},
            {});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* offsets_stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offsets_l1_addr);
        const uint32_t row_start = offsets_stage[offsets_start_index];
        const uint32_t row_end = offsets_stage[offsets_start_index + 1U];
        // Contract: offsets must be tile-aligned — the windows below are addressed at tile
        // granularity (offset / 32), so a non-multiple-of-32 offset would alias the wrong tile.
        ASSERT(row_start % 32U == 0U && row_end % 32U == 0U);
#ifdef OFFSET_ROW_MODE
        {
            const uint32_t in0_idx = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 8);
            const uint32_t actual_eff_M = (row_end - row_start) / 32U;
            // Empty-expert (actual=0) → M_blocks_per_core=0 (loop skipped). Still clamp
            // M_tiles to >=1 for shape construction (TensorShape2D asserts d0>0).
            M_tiles = actual_eff_M > 0U ? actual_eff_M : 1U;
            // On the writer kernel (transpose_core_grid) override out_row_offset_tiles.
            // dm_in0_sender publishes M values to cb_ctrl; we re-derive them locally here
            // (both kernels read the same offsets).
            if constexpr (is_output_writer) {
                out_row_offset_tiles = row_start / 32U;
            }
            constexpr uint32_t kAxisCores = IN0_AXIS_CORES;
            const uint32_t per_core = (actual_eff_M + kAxisCores - 1U) / kAxisCores;
            // Uniform M_blocks_per_core across cores — matches dm_in0_sender; bounds checks
            // clip out-of-range reads/writes for the tail cores.
            M_start_tile = per_core * in0_idx;
            M_end_tile = per_core * (in0_idx + 1U);
            M_blocks_per_core = (per_core + M_block_tiles - 1U) / M_block_tiles;
        }
#endif  // OFFSET_ROW_MODE
#ifdef OFFSET_K_MODE
        // in1 owns its K offset; dm_in0 owns the cb_ctrl publish for compute.
        in1_k_offset_tiles = row_start / 32U;
        K_tiles = (row_end - row_start) / 32U;
#endif  // OFFSET_K_MODE
    }
    const uint32_t padded_K_tiles = ((K_tiles + K_block_tiles - 1U) / K_block_tiles) * K_block_tiles;

    // Empty expert (K_tiles==0): K_num_blocks is 0 so the K-loop is skipped and in1_shape is
    // never read; clamp its K extent to 1 only to satisfy TensorShape2D's d>0 assert.
    const uint32_t K_tiles_shape = K_tiles > 0U ? K_tiles : 1U;
    const uint32_t padded_K_tiles_shape = padded_K_tiles > 0U ? padded_K_tiles : K_block_tiles;

    // Storage layout: without transpose_b the weight is stored as [K, N]; with it, as [N, K].
    const TensorShape2D in1_shape = transpose_b
                                        ? TensorShape2D(N_tiles, K_tiles_shape, padded_N_tiles, padded_K_tiles_shape)
                                        : TensorShape2D(K_tiles_shape, N_tiles, padded_K_tiles_shape, padded_N_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    const uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    const uint32_t defer_write_k_block = compute_defer_write_k_block(core_y_index, Y_AXIS_CORES, K_num_blocks);
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
    CircularBuffer cb_in1(cb_id_in1);

    in1_valid_sem.set(VALID);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    bool k_forward = true;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);

        k_forward = true;

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            const uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            const uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            const uint32_t current_N_block_tiles = n_tile_end - n_tile;
            const uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        do_deferred_block_write<M_block_tiles, N_block_tiles, use_out_offset>(
                            out_accessor,
                            out_shape,
                            cb_id_out,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile,
                            defer_write_n_tile_end,
                            out_row_offset_tiles);
                    }
                }

                const uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_in1.reserve_back(in1_block_num_tiles);

                uint32_t in1_start_address = cb_in1.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in1_block_sync<K_block_tiles, N_block_tiles, transpose_b, use_offset_in1>(
                        in1_reader,
                        in1_shape,
                        in1_start_address,
                        cb_id_in1,
                        in1_tile_size,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles,
                        n_tile,
                        n_tile_end,
                        in1_k_offset_tiles,
                        parent_K_tiles_stride_in1);
                } else {
                    in1_receiver_sem.set(INVALID);
                    in1_sender_sem.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                    in1_receiver_sem.wait(VALID);
                }

                // Critical to performance for the sender to push data to compute before forwarding.
                // This frees the sender to start the next read earlier.
                cb_in1.push_back(in1_block_num_tiles);

                if (!is_sink_core) {
                    in1_sender_sem.wait(1);
                    in1_sender_sem.set(0);

                    /**
                     * in1 is K_block_tiles x N_block_tiles. When N block is partial, we don't need to write the
                     * padded tiles. For each tile in the K block, write only the non-padded N tiles. Use
                     * `current_N_tiles_bytes`.
                     */
                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(in1_start_address),
                            UnicastEndpoint{},
                            current_N_tiles_bytes,
                            {},
                            {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = in1_start_address});
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in1_valid_sem.relay_unicast(noc, in1_receiver_sem, in1_dest_noc_x, in1_dest_noc_y);
                }
            }

            k_forward = !k_forward;
            // We have an output block to write out

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !((m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1)));
            defer_write = defer_write && !is_injector_core;
            // K_num_blocks == 0 (empty expert): the in-K-loop deferred-write trigger never
            // fires, so force direct writes to flush compute's zero-init to DRAM.
            if (K_num_blocks == 0U) {
                defer_write = false;
            }

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    write_block_sync_granular<M_block_tiles, N_block_tiles, use_out_offset>(
                        out_accessor,
                        out_shape,
                        cb_id_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile,
                        n_tile_end,
                        out_row_offset_tiles);
                }
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
