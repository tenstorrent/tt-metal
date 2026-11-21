// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qwen_ag_matmul in0 (activation) data movement: the stock minimal_matmul dm_in0_sender.cpp
// (ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels) for the transposed core grid (M over x,
// in0 relayed down each grid column, never the output writer), one N block per core, no bias / ternary / SwiGLU,
// with the M blocks taken from an explicit per-column list.
//
// The injector (grid row 0) reads block b only once the hub has released it (ready semaphore >= b + 1, FUSED only),
// and reads the rows of this device's own slot from x_local instead of the gathered buffer.
//
// CT: 0 K_tiles, 1 padded_K_tiles, 2 M_block_tiles, 3 K_block_tiles, 4 in0_tile_size,
//     5 in0_sender_sem, 6 in0_receiver_sem, 7 in0_valid_sem, 8 is_injector_core, 9 ready_sem, 10 slot_rows,
//     11 my_slot, 12 M_tiles, 13 GX, 14 GY, then accessors: gathered (in0), x_local
// Common RT (no per-core args; the core's place comes from its logical coordinates): 0 in0_addr, 1 x_local_addr,
//     then GX NOC x of the grid columns, GY NOC y of the grid rows, GX list offsets, then the lists (per grid
//     column: num_blocks, num_blocks x (m_start_tile, rows))

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
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(4);
    Semaphore<> in0_sender_semaphore(get_compile_time_arg_val(5));
    Semaphore<> in0_receiver_semaphore(get_compile_time_arg_val(6));
    Semaphore<> in0_valid_semaphore(get_compile_time_arg_val(7));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(8);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t slot_rows = get_compile_time_arg_val(10);
    constexpr uint32_t my_slot = get_compile_time_arg_val(11);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t GX = get_compile_time_arg_val(13);
    constexpr uint32_t GY = get_compile_time_arg_val(14);

    // in0 is relayed down the grid column (NOC_0 order: y increasing)
    const uint32_t cx = get_absolute_logical_x();
    const uint32_t cy = get_absolute_logical_y();
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t xl_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t PX = 2, PY = 2 + GX;
    const uint32_t is_sink_core = cy == GY - 1;
    const uint32_t in0_dest_noc_x = get_common_arg_val<uint32_t>(PX + cx);
    const uint32_t in0_dest_noc_y = get_common_arg_val<uint32_t>(PY + (cy + 1 < GY ? cy + 1 : GY - 1));
    const uint32_t in0_sender_noc_x = in0_dest_noc_x;
    const uint32_t in0_sender_noc_y = get_common_arg_val<uint32_t>(PY + (cy > 0 ? cy - 1 : 0));
    const uint32_t list_at = get_common_arg_val<uint32_t>(PY + GY + cx);
    const uint32_t num_blocks = get_common_arg_val<uint32_t>(list_at);
    const uint32_t blocks_idx = list_at + 1;

    constexpr auto in0_args = TensorAccessorArgs<15>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);
    constexpr auto xl_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto xl_reader = TensorAccessor(xl_args, xl_addr);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    CircularBuffer cb_in0(cb_in0_id);

#ifdef FUSED
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id));
#endif

    in0_valid_semaphore.set(VALID);

    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t m_tile = get_common_arg_val<uint32_t>(blocks_idx + 2 * b);
        const uint32_t rows = get_common_arg_val<uint32_t>(blocks_idx + 2 * b + 1);
        const uint32_t m_tile_end = m_tile + rows;
        const uint32_t current_block_bytes = rows * K_block_tiles * in0_tile_size;
        // rows of this device's own slot come from x_local (FUSED); every other row from the gathered buffer
        bool local = false;
        uint32_t row_base = m_tile;
#ifdef FUSED
        if constexpr (is_injector_core) {
            local = rows > 0 && (m_tile / slot_rows) == my_slot;
            if (local) {
                row_base = m_tile - my_slot * slot_rows;
            } else {
#ifndef PROBE_NOWAIT
                noc_semaphore_wait_min(ready, b + 1);
#endif
            }
        }
#endif

        for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
            cb_in0.reserve_back(in0_block_num_tiles);

            uint32_t in0_start_address = cb_in0.get_write_ptr();
            if constexpr (is_injector_core) {
                // read_in0_block_sync (matmul_dataflow_common.hpp) on rows [m_tile, m_tile_end), K block k_block
                uint32_t write_ptr = in0_start_address;
                const uint32_t d1_start = k_block * K_block_tiles;
                const uint32_t d1_end = d1_start + K_block_tiles;
                for (uint32_t i = 0; i < rows; i++) {
                    if (m_tile + i >= M_tiles) {
                        break;
                    }
                    for (uint32_t j = d1_start; j < d1_end; j++) {
                        if (j < K_tiles) {
                            const uint32_t tile_id = (row_base + i) * K_tiles + j;
                            if (local) {
                                noc.async_read(
                                    xl_reader,
                                    CoreLocalMem<uint32_t>(write_ptr),
                                    in0_tile_size,
                                    {.page_id = tile_id},
                                    {});
                            } else {
                                noc.async_read(
                                    in0_reader,
                                    CoreLocalMem<uint32_t>(write_ptr),
                                    in0_tile_size,
                                    {.page_id = tile_id},
                                    {});
                            }
                        } else {
                            noc.async_write_zeros(
                                cb_in0, in0_tile_size, {.offset_bytes = write_ptr - in0_start_address});
                        }
                        write_ptr += in0_tile_size;
                    }
                }
                noc.async_read_barrier();
                noc.write_zeros_l1_barrier();
            } else {
                // Get from previous device
                in0_receiver_semaphore.set(INVALID);
                in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                in0_receiver_semaphore.wait(VALID);
            }

            // Critical to performance for sender to push data to compute before mcasting
            // This frees sender to start next read earlier
            cb_in0.push_back(in0_block_num_tiles);

            if (!is_sink_core) {
                in0_sender_semaphore.wait(1);
                in0_sender_semaphore.set(0);

                if (current_block_bytes > 0) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        UnicastEndpoint{},
                        current_block_bytes,
                        {},
                        {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});
                }

                in0_valid_semaphore.relay_unicast(noc, in0_receiver_semaphore, in0_dest_noc_x, in0_dest_noc_y);
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
