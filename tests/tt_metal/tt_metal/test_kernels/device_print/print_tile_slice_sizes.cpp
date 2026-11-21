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
    // TileSlice sizes other than 64 and 128: TileSlice<100> with this slice's 16 Float16_b values (32 bytes), and the
    // largest allowed, TileSlice<252>, with 112 values (224 bytes).
    DPRINT(
        "TileSlice<100>:\n{}\n",
        TileSlice<100>(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, is_tilized));
    DPRINT(
        "TileSlice<252>:\n{}\n",
        TileSlice<252>(
            cb_id,
            0,
            SliceRange{.h0 = 0, .h1 = 28, .hs = 4, .w0 = 0, .w1 = 32, .ws = 2},
            TSLICE_INPUT_CB,
            TSLICE_RD_PTR,
            true,
            is_tilized));
}
