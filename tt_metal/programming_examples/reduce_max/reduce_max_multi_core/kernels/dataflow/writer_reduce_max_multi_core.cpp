// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

/**
 * @brief Writer kernel: writes the reduced output tiles from L1 to DRAM (multi-core).
 *
 * Identical in structure to the single-core version, except that only the mt_count output
 * tiles assigned to this core are written, starting at tile offset mt_start in the output buffer.
 *
 * Runtime arguments:
 *   0: dst_addr  - DRAM address of the output buffer (Mt tiles total).
 *   1: mt_start  - First output tile index assigned to this core.
 *   2: mt_count  - Number of output tiles assigned to this core.
 *
 * Compile-time arguments:
 *   [0..N): TensorAccessorArgs for dst_dram_buffer.
 */
void kernel_main() {
    // TODO: implement multi-core writer kernel.
    //
    // Steps:
    //   1. Read runtime args: dst_addr (0), mt_start (1), mt_count (2).
    //   2. Construct a TensorAccessor for the output buffer using TensorAccessorArgs<0>().
    //   3. For each mt in [0, mt_count):
    //        cb_wait_front(cb_id_out, 1);
    //        output_tile_index = mt_start + mt;
    //        noc_async_write_tile(output_tile_index, s, get_read_ptr(cb_id_out));
    //        noc_async_write_barrier();
    //        cb_pop_front(cb_id_out, 1);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t mt_start = get_arg_val<uint32_t>(1);
    uint32_t mt_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_out = 16;

    // Set up the tile accessor for the output DRAM buffer.
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr, get_tile_size(cb_id_out));

    for (uint32_t mt = mt_start; mt < mt_start + mt_count; mt++) {
        cb_wait_front(cb_id_out, 1);

        // DPRINT << "[WRITER] " << TSLICE(cb_out, 0, SliceRange::h0_32_w0(), TSLICE_RD_PTR) << ENDL();
        // DPRINT << "[WRITER] " << TSLICE(cb_out, 0, SliceRange::h0_w0_32(), TSLICE_RD_PTR) << ENDL();
        DPRINT << ENDL() << ENDL();

        uint32_t output_tile_index = mt;
        noc_async_write_tile(output_tile_index, s, get_read_ptr(cb_id_out));
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }
}
