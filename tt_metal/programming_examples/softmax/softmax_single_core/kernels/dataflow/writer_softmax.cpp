// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

/**
 * @brief Writer kernel: writes the final softmax output tiles from L1 to DRAM.
 *
 * Reads Mt×Nt tiles from cb_out (c_16) in row-major tile order as the compute
 * kernel produces them, and writes each tile to the corresponding index in the
 * output DRAM buffer.
 *
 * Circular buffers:
 *   c_16 (cb_out): Softmax output tiles; Mt×Nt tiles total.
 *
 * Runtime arguments:
 *   0: dst_addr - DRAM address of the output buffer (Mt×Nt tiles).
 *   1: Mt        - Number of tile rows.
 *   2: Nt        - Number of tile columns.
 */
void kernel_main() {
    // TODO: implement writer kernel

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_out = 16;
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr, get_tile_size(cb_id_out));

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_id_out, 1);

            uint32_t tile_index = mt * Nt + nt;
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_tile(tile_index, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }
}