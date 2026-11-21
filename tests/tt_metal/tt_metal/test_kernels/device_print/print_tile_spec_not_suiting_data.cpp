// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // Read the tile into c_in0, as print_tile_brisc.cpp does.
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t is_tilized = get_arg_val<uint32_t>(2);
    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_addr);
    cb_reserve_back(cb_id, 1);
    noc_async_read(src_noc_addr, get_write_ptr(cb_id), get_tile_size(cb_id));
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);

    cb_wait_front(cb_id, 1);
    // {:.3f} suits Float16_b values, so they print with it; {:x} doesn't, so they print as {} would, after a warning
    // line.
    DPRINT(
        "Fixed:\n{:.3f}\n", TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, is_tilized));
    DPRINT("Hex:\n{:x}\n", TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, is_tilized));
}
