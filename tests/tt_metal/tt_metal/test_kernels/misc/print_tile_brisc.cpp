// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint.h"
#include "debug/ring_buffer.h"

#include "dataflow_api.h"
void kernel_main() {
    // Read out the tile we want to print using BRISC, put it in c_in0
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_1;
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t is_tilized = get_arg_val<uint32_t>(2);
    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_addr);
    cb_reserve_back(cb_id, 1);
    noc_async_read(src_noc_addr, get_write_ptr(cb_id), get_tile_size(cb_id));
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);

    cb_wait_front(cb_id, 1);
    // Print the tile from each RISC, one after another
    DPRINT << "Print tile from Data0:" << ENDL();
    DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, is_tilized) << ENDL();
    DPRINT << RAISE{1};
}
