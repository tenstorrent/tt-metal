// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"

/**
 * @brief Writer kernel: writes the reduced output tiles from L1 to DRAM.
 *
 * The compute kernel produces Mt output tiles, one per row group of the input matrix.
 * Each output tile is a 32×32 bfloat16 tile where column 0 holds the 32 row maxima
 * and all other columns are zero.
 *
 * Runtime arguments:
 *   0: dst_addr - DRAM address of the output buffer (Mt tiles of 32×32).
 *   1: Mt        - Number of output tiles (one per tile row of the input).
 */
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_out = 16;

    // Set up the tile accessor for the output DRAM buffer.
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr, get_tile_size(cb_id_out));

    // Write one output tile per row group.  Tile index mt corresponds to the
    // maxima for the 32 matrix rows in the mt-th tile row of the input.
    for (uint32_t mt = 0; mt < Mt; mt++) {
        cb_wait_front(cb_id_out, 1);

        // DPRINT << "[WRITER] " << TSLICE(cb_out, 0, SliceRange::h0_32_w0(), TSLICE_RD_PTR) << ENDL();
        // DPRINT << "[WRITER] " << TSLICE(cb_out, 0, SliceRange::h0_w0_32(), TSLICE_RD_PTR) << ENDL();
        DPRINT << ENDL() << ENDL();

        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(mt, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }
}
