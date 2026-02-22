// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Reader kernel: feeds input tiles and the scaler tile to the compute kernel.
 *
 * For each tile row mt the input is delivered twice:
 *   Pass 1 (row-wise max):  Nt tiles in column order → cb_in.
 *   Pass 2 (exp(x − max)): the same Nt tiles again → cb_in.
 *
 * The scaler tile (all 1.0) is pushed to cb_scaler once before the first pass
 * and is never popped, so it stays resident for all reduce operations.
 *
 * Circular buffers:
 *   c_0 (cb_in):     Input tiles, consumed one at a time by the compute kernel.
 *   c_1 (cb_scaler): Scaler tile; pushed once, never popped.
 *
 * Runtime arguments:
 *   0: src_addr    - DRAM address of the input matrix (tilized, Mt×Nt tiles).
 *   1: scaler_addr - DRAM address of the scaler tile (one 32×32 tile of 1.0).
 *   2: Mt          - Number of tile rows.
 *   3: Nt          - Number of tile columns.
 */
void kernel_main() {
    // TODO: implement reader kernel

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t scaler_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in = 0;
    constexpr uint32_t cb_id_scaler = 1;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, get_tile_size(cb_id_in));

    constexpr auto scaler_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto scaler = TensorAccessor(scaler_args, scaler_addr, get_tile_size(cb_id_scaler));

    // Push the scaler tile once before the main loop.
    {
        cb_reserve_back(cb_id_scaler, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_scaler);
        noc_async_read_tile(0, scaler, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_scaler, 1);
    }

    for (uint32_t mt = 0; mt < Mt; mt++) {
        // for calculating max
        for (uint32_t nt = 0; nt < Nt; nt++) {
            uint32_t tile_index = mt * Nt + nt;
            cb_reserve_back(cb_id_in, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in);
            noc_async_read_tile(tile_index, src, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in, 1);
        }

        // for calculating exp and sum
        for (uint32_t nt = 0; nt < Nt; nt++) {
            uint32_t tile_index = mt * Nt + nt;
            cb_reserve_back(cb_id_in, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in);
            noc_async_read_tile(tile_index, src, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in, 1);
        }
    }
}