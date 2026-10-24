// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qwen_ag_matmul in1 (weight) data movement + output writer: the stock minimal_matmul dm_in1_sender_out.cpp
// (ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels) for the transposed core grid (N over y,
// in1 relayed along each grid row, this kernel writes the output), one N block per core, no bias / ternary /
// SwiGLU, with the M blocks taken from an explicit per-column list (same list as the column's in0 kernel).
//
// IN1_RESIDENT: the in1 CB holds all K blocks of this core's N block. The first block reads (injector) / receives
// them as the stock kernel does; every later block only re-pushes the same CB pages (the data stays in L1), so the
// columns stop relaying in1 and run independently after their first block. Compute sees the same tiles in the same
// order.
//
// CT: 0 K_tiles, 1 padded_K_tiles, 2 N_tiles, 3 M_block_tiles, 4 K_block_tiles, 5 N_block_tiles, 6 in1_tile_size,
//     7 out_tile_size, 8 in1_sender_sem, 9 in1_receiver_sem, 10 in1_valid_sem, 11 is_injector_core, 12 M_tiles,
//     13 GX, 14 GY, 15 N_tiles_per_core, 16 k_blocks_per_core, then accessors: weight (in1), out
// Common RT (no per-core args): 0 in1_addr, 1 out_addr, then GX NOC x of the grid columns, GY NOC y of the grid
//     rows, GX list offsets, then the lists (per grid column: num_blocks, num_blocks x (m_start_tile, rows))

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
#ifdef PROBE_NOMM
    return;
#endif
    Noc noc;
    constexpr uint32_t K_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    Semaphore<> in1_sender_semaphore(get_compile_time_arg_val(8));
    Semaphore<> in1_receiver_semaphore(get_compile_time_arg_val(9));
    Semaphore<> in1_valid_semaphore(get_compile_time_arg_val(10));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(11);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t GX = get_compile_time_arg_val(13);
    constexpr uint32_t GY = get_compile_time_arg_val(14);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(15);
    constexpr uint32_t k_blocks_per_core = get_compile_time_arg_val(16);

    // in1 is relayed along the grid row in the NOC_1 order of build_core_order_for_axis: x = 0, GX-1, ..., 1
    const uint32_t cx = get_absolute_logical_x();
    const uint32_t cy = get_absolute_logical_y();
    const uint32_t in1_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t PX = 2, PY = 2 + GX;
    auto order_x = [](uint32_t i) -> uint32_t { return i == 0 ? 0 : GX - i; };
    const uint32_t i1 = cx == 0 ? 0 : GX - cx;
    const uint32_t is_sink_core = i1 == GX - 1;
    const uint32_t in1_dest_noc_x = get_common_arg_val<uint32_t>(PX + order_x(i1 + 1 < GX ? i1 + 1 : GX - 1));
    const uint32_t in1_dest_noc_y = get_common_arg_val<uint32_t>(PY + cy);
    const uint32_t in1_sender_noc_x = get_common_arg_val<uint32_t>(PX + order_x(i1 > 0 ? i1 - 1 : 0));
    const uint32_t in1_sender_noc_y = in1_dest_noc_y;
    const uint32_t N_start_tile = cy * N_tiles_per_core;
    const uint32_t N_end_tile = N_start_tile + N_tiles_per_core;
    constexpr uint32_t K_num_blocks_ = padded_K_tiles / K_block_tiles;
    const uint32_t defer_write_k_block =
        cy * k_blocks_per_core < K_num_blocks_ - 1 ? cy * k_blocks_per_core : K_num_blocks_ - 1;
    const uint32_t list_at = get_common_arg_val<uint32_t>(PY + GY + cx);
    const uint32_t num_blocks = get_common_arg_val<uint32_t>(list_at);
    const uint32_t blocks_idx = list_at + 1;

    constexpr auto in1_args = TensorAccessorArgs<17>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr);
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;
    CircularBuffer cb_in1(cb_in1_id);
    CircularBuffer cb_out(cb_out_id);

    in1_valid_semaphore.set(VALID);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;
    const uint32_t n_tile = N_start_tile;
    const uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
    const uint32_t current_N_block_tiles = n_tile_end - n_tile;
    const uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;

    // write_block_sync (matmul_dataflow_common.hpp): the whole block from the front of cb_out
    auto write_block = [&](uint32_t m0, uint32_t m1) {
        uint32_t read_ptr = cb_out.get_read_ptr();
        for (uint32_t i = m0; i < m1; i++) {
            if (i >= M_tiles) {
                break;
            }
            for (uint32_t j = n_tile; j < n_tile_end; j++) {
                if (j >= N_tiles) {
                    read_ptr += out_tile_size;
                    continue;
                }
                noc.async_write(
                    CoreLocalMem<uint32_t>(read_ptr), out_writer, out_tile_size, {}, {.page_id = i * N_tiles + j});
                read_ptr += out_tile_size;
            }
            read_ptr += (N_block_tiles - (n_tile_end - n_tile)) * out_tile_size;
        }
        noc.async_writes_flushed();
    };

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    bool defer_write = false;

    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t m_tile = get_common_arg_val<uint32_t>(blocks_idx + 2 * b);
        const uint32_t m_tile_end = m_tile + get_common_arg_val<uint32_t>(blocks_idx + 2 * b + 1);
        const bool is_last_block = b == num_blocks - 1;

        for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
            if (defer_write && k_block == defer_write_k_block) {
                cb_out.wait_front(out_block_num_tiles);
                write_block(defer_write_m_tile, defer_write_m_tile_end);
                cb_out.pop_front(out_block_num_tiles);
            }

            cb_in1.reserve_back(in1_block_num_tiles);
#ifdef IN1_RESIDENT
            if (b > 0) {
                cb_in1.push_back(in1_block_num_tiles);
                continue;
            }
#endif

            uint32_t in1_start_address = cb_in1.get_write_ptr();
            if constexpr (is_injector_core) {
                // read_in1_block_sync on K rows [k_block * K_block_tiles, +K_block_tiles), N [n_tile, n_tile_end)
                uint32_t write_ptr = in1_start_address;
                for (uint32_t i = k_block * K_block_tiles; i < (k_block + 1) * K_block_tiles; i++) {
                    for (uint32_t j = n_tile; j < n_tile_end; j++) {
                        if (j >= N_tiles) {
                            write_ptr += in1_tile_size;
                            continue;
                        }
                        if (i < K_tiles) {
                            noc.async_read(
                                in1_reader,
                                CoreLocalMem<uint32_t>(write_ptr),
                                in1_tile_size,
                                {.page_id = i * N_tiles + j},
                                {});
                        } else {
                            noc.async_write_zeros(
                                cb_in1, in1_tile_size, {.offset_bytes = write_ptr - in1_start_address});
                        }
                        write_ptr += in1_tile_size;
                    }
                    write_ptr += (N_block_tiles - (n_tile_end - n_tile)) * in1_tile_size;
                }
                noc.async_read_barrier();
                noc.write_zeros_l1_barrier();
            } else {
                in1_receiver_semaphore.set(INVALID);
                in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                in1_receiver_semaphore.wait(VALID);
            }

            // Critical to performance for sender to push data to compute before mcasting
            // This frees sender to start next read earlier
            cb_in1.push_back(in1_block_num_tiles);

            if (!is_sink_core) {
                in1_sender_semaphore.wait(1);
                in1_sender_semaphore.set(0);

                for (uint32_t i = 0; i < K_block_tiles; i++) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(in1_start_address),
                        UnicastEndpoint{},
                        current_N_tiles_bytes,
                        {},
                        {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = in1_start_address});
                    in1_start_address += full_N_tiles_bytes;
                }

                in1_valid_semaphore.relay_unicast(noc, in1_receiver_semaphore, in1_dest_noc_x, in1_dest_noc_y);
            }
        }

        defer_write_m_tile = m_tile;
        defer_write_m_tile_end = m_tile_end;
        // If this isn't the last output block, defer writing until the defer_write_k_block iteration of the next one.
        defer_write = !is_last_block && !is_injector_core;

        if (!defer_write) {
            // write_block_sync_granular: row by row as compute packs them
            for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
                cb_out.wait_front(N_block_tiles);
                const uint32_t mt = m_tile + m_id;
                if (mt < m_tile_end && mt < M_tiles) {
                    uint32_t out_read_ptr = cb_out.get_read_ptr();
                    for (uint32_t j = n_tile; j < n_tile_end; j++) {
                        if (j >= N_tiles) {
                            break;
                        }
                        noc.async_write(
                            CoreLocalMem<uint32_t>(out_read_ptr),
                            out_writer,
                            out_tile_size,
                            {},
                            {.page_id = mt * N_tiles + j});
                        out_read_ptr += out_tile_size;
                    }
                }
                noc.async_writes_flushed();
                cb_out.pop_front(N_block_tiles);
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
