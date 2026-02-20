// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"

/**
 * @brief Reader kernel: feeds the input matrix tiles and a constant scaler tile to the compute kernel.
 *
 * Tiles are delivered in row-major tile order: all Nt column tiles for row 0,
 * then all Nt column tiles for row 1, and so on through Mt rows.
 * The scaler tile (a 32×32 tile of all 1.0 values) is sent once at the start and
 * is never popped by the compute kernel, so it remains in the scaler CB throughout.
 *
 * Runtime arguments:
 *   0: src_addr   - DRAM address of the input matrix (tilized, Mt×Nt tiles).
 *   1: scaler_addr - DRAM address of the scaler tile (one 32×32 tile of 1.0).
 *   2: Mt          - Number of tile rows in the input matrix.
 *   3: Nt          - Number of tile columns in the input matrix (reduction dimension).
 */
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t scaler_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in = 0;
    constexpr uint32_t cb_id_scaler = 1;

    // Set up tile accessors for both DRAM buffers.
    // TensorAccessorArgs encodes the buffer layout (interleaved, page size, etc.)
    // as compile-time arguments appended to the kernel's compile_args list.
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, get_tile_size(cb_id_in));

    constexpr auto scaler_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto scaler = TensorAccessor(scaler_args, scaler_addr, get_tile_size(cb_id_scaler));

    // Send the scaler tile (all 1.0) once before the main loop.
    // The compute kernel never pops it, so it stays available for all reduce operations.
    {
        cb_reserve_back(cb_id_scaler, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_scaler);
        noc_async_read_tile(0, scaler, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_scaler, 1);
    }

    // Send input tiles in row-major tile order.
    // The input CB is double-buffered (2 tiles), and the compute kernel pops each
    // tile immediately after reducing it, so the reader and compute pipeline naturally.
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            // DPRINT << "Processing tile (mt, nt) = (" << mt << ", " << nt << ")" << ENDL();

            uint32_t tile_index = mt * Nt + nt;  // Row-major tile index in the Mt×Nt grid.
            cb_reserve_back(cb_id_in, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in);
            noc_async_read_tile(tile_index, src, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in, 1);
        }
    }
}
