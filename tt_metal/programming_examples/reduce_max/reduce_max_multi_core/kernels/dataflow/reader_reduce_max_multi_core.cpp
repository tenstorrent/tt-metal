// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Reader kernel: feeds the input matrix tiles and the scaler tile to the compute kernel (multi-core).
 *
 * Identical in structure to the single-core version, except that only the subset of tile-row groups
 * assigned to this core are sent.  The reader streams tiles for mt_count row groups starting at
 * mt_start, delivering all Nt column tiles for each row group in order.
 *
 * The scaler tile (all 1.0) is sent exactly once at the start.  Because the compute kernel never
 * pops it, it stays resident in the scaler CB for the full lifetime of this core's work.
 *
 * Runtime arguments:
 *   0: src_addr    - DRAM address of the full input matrix (tilized, Mt×Nt tiles).
 *   1: scaler_addr - DRAM address of the scaler tile (one 32×32 tile of 1.0).
 *   2: mt_start    - First tile-row group index assigned to this core.
 *   3: mt_count    - Number of tile-row groups assigned to this core.
 *   4: Nt          - Number of tile columns in the input matrix (reduction dimension).
 *
 * Compile-time arguments:
 *   [0..N): TensorAccessorArgs for src_dram_buffer.
 *   [N..M): TensorAccessorArgs for scaler_dram_buffer.
 */
void kernel_main() {
    // TODO: implement multi-core reader kernel.
    //
    // Steps:
    //   1. Read runtime args: src_addr (0), scaler_addr (1), mt_start (2), mt_count (3), Nt (4).
    //   2. Construct TensorAccessors for src and scaler using TensorAccessorArgs<offset>().
    //      (Same offset pattern as the single-core reader.)
    //   3. Send the scaler tile once:
    //        cb_reserve_back(cb_id_scaler, 1);
    //        noc_async_read_tile(0, scaler, get_write_ptr(cb_id_scaler));
    //        noc_async_read_barrier();
    //        cb_push_back(cb_id_scaler, 1);
    //   4. For each mt in [0, mt_count):
    //        For each nt in [0, Nt):
    //          tile_index = (mt_start + mt) * Nt + nt;
    //          cb_reserve_back(cb_id_in, 1);
    //          noc_async_read_tile(tile_index, src, get_write_ptr(cb_id_in));
    //          noc_async_read_barrier();
    //          cb_push_back(cb_id_in, 1);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t scaler_addr = get_arg_val<uint32_t>(1);
    uint32_t mt_start = get_arg_val<uint32_t>(2);
    uint32_t mt_count = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in = 0;
    constexpr uint32_t cb_id_scaler = 1;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, get_tile_size(cb_id_in));

    constexpr auto scaler_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto scaler = TensorAccessor(scaler_args, scaler_addr, get_tile_size(cb_id_scaler));

    {
        cb_reserve_back(cb_id_scaler, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_scaler);
        noc_async_read_tile(0, scaler, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_scaler, 1);
    }

    for (uint32_t mt = mt_start; mt < mt_start + mt_count; mt++) {
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
