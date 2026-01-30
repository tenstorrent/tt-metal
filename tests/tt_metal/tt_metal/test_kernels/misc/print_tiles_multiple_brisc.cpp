// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "api/debug/ring_buffer.h"

#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_id = tt::CBIndex::c_0;

    uint32_t tile_size_bytes = get_tile_size(cb_id);
    uint32_t log2_tile_size = __builtin_ctz(tile_size_bytes);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t is_tilized = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint64_t noc_addr = get_dram_noc_addr(i, tile_size_bytes, src_addr);

        cb_reserve_back(cb_id, 1);
        noc_async_read(noc_addr, get_write_ptr(cb_id), tile_size_bytes);
        noc_async_read_barrier();

        DPRINT << "Write tile " << i << ":" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_WR_PTR, true, is_tilized) << ENDL();

        cb_push_back(cb_id, 1);
    }

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id, 1);
        DPRINT << "Read tile " << i << ":" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, is_tilized) << ENDL();
        cb_pop_front(cb_id, 1);
    }
}
