// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Reader kernel: feeds input tiles and the scaler tile to the compute kernel (multi-core).
 *
 * Identical in structure to the single-core version except that each core reads only
 * the tile-row groups assigned to it.  Global tile indices are computed as
 * (mt_start + mt) * Nt + nt.
 *
 * Each row is sent twice: once for pass 1 (row-wise max) and once for pass 2
 * (exp(x − max)).  The scaler tile is pushed once and never popped.
 *
 * Runtime arguments:
 *   0: src_addr    - DRAM address of the input matrix (tilized, Mt×Nt tiles).
 *   1: scaler_addr - DRAM address of the scaler tile (all 1.0).
 *   2: mt_start    - First tile-row group index assigned to this core.
 *   3: mt_count    - Number of tile-row groups assigned to this core.
 *   4: Nt          - Number of tile columns.
 *
 * Compile-time arguments:
 *   [0..N): TensorAccessorArgs for src_dram_buffer.
 *   [N..M): TensorAccessorArgs for scaler_dram_buffer.
 */
void kernel_main() {
    uint32_t src_addr    = get_arg_val<uint32_t>(0);
    uint32_t scaler_addr = get_arg_val<uint32_t>(1);
    uint32_t mt_start    = get_arg_val<uint32_t>(2);
    uint32_t mt_count    = get_arg_val<uint32_t>(3);
    uint32_t Nt          = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in     = 0;
    constexpr uint32_t cb_id_scaler = 1;

    constexpr auto src_args    = TensorAccessorArgs<0>();
    const auto     src         = TensorAccessor(src_args, src_addr, get_tile_size(cb_id_in));

    constexpr auto scaler_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto     scaler      = TensorAccessor(scaler_args, scaler_addr, get_tile_size(cb_id_scaler));

    // Push the scaler tile once; the compute kernel never pops it.
    {
        cb_reserve_back(cb_id_scaler, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_scaler);
        noc_async_read_tile(0, scaler, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_scaler, 1);
    }

    for (uint32_t mt = 0; mt < mt_count; mt++) {
        uint32_t global_mt = mt_start + mt;

        // Pass 1: row-wise max — send Nt tiles for this tile-row.
        for (uint32_t nt = 0; nt < Nt; nt++) {
            uint32_t tile_index = global_mt * Nt + nt;
            cb_reserve_back(cb_id_in, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in);
            noc_async_read_tile(tile_index, src, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in, 1);
        }

        // Pass 2: exp(x − max) — re-send the same Nt tiles.
        for (uint32_t nt = 0; nt < Nt; nt++) {
            uint32_t tile_index = global_mt * Nt + nt;
            cb_reserve_back(cb_id_in, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in);
            noc_async_read_tile(tile_index, src, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in, 1);
        }
    }
}
