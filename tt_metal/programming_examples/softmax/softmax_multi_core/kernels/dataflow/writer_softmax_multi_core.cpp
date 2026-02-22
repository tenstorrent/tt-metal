// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

/**
 * @brief Writer kernel: writes the final softmax output tiles from L1 to DRAM (multi-core).
 *
 * Identical in structure to the single-core version, but each core writes only
 * the mt_count × Nt tiles assigned to it.  Global tile indices are computed as
 * (mt_start + mt) * Nt + nt so that each core's output lands in the correct
 * position within the shared Mt×Nt output buffer.
 *
 * Runtime arguments:
 *   0: dst_addr  - DRAM address of the output buffer (Mt×Nt tiles total).
 *   1: mt_start  - First tile-row group index assigned to this core.
 *   2: mt_count  - Number of tile-row groups assigned to this core.
 *   3: Nt        - Number of tile columns.
 *
 * Compile-time arguments:
 *   [0..N): TensorAccessorArgs for dst_dram_buffer.
 */
void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t mt_start  = get_arg_val<uint32_t>(1);
    uint32_t mt_count  = get_arg_val<uint32_t>(2);
    uint32_t Nt        = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = 16;
    constexpr auto     s_args    = TensorAccessorArgs<0>();
    const auto         s         = TensorAccessor(s_args, dst_addr, get_tile_size(cb_id_out));

    for (uint32_t mt = 0; mt < mt_count; mt++) {
        uint32_t global_mt = mt_start + mt;
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_id_out, 1);
            uint32_t tile_index   = global_mt * Nt + nt;
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_tile(tile_index, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }
}
