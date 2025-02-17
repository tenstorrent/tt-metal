// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint.h"
#include "debug/ring_buffer.h"
#include "dataflow_api.h"
void dump_tile(uint32_t cb_id) {
    uint32_t cb1_rd_addr = CB_RD_PTR(cb_id);
    volatile tt_l1_ptr uint32_t* p = (volatile tt_l1_ptr uint32_t*)cb1_rd_addr;
    for (int r = 0; r < 17; r++) {
        DPRINT << DEC() << "[" << SETW(3) << (r - 1) * 16 << ":" << SETW(3) << r * 16 - 1 << "]: " << HEX();
        for (int c = 15; c >= 0; c--) {
            DPRINT << "0x" << SETW(8) << p[r * 16 + c] << " ";
        }
        DPRINT << ENDL();
    }
}

void kernel_main() {
    // Read out the tile we want to print using BRISC, put it in c_in0
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_1;
    uint32_t is_tilized = get_arg_val<uint32_t>(0);

    cb_wait_front(cb_id, 1);
    if (is_tilized) {
        cb_wait_front(cb_intermed, 1);
    }
    DPRINT << WAIT{4};
    DPRINT << "Print tile from Data1:" << ENDL();
    // Use NCRISC to test printing untilized
    DPRINT
        << TSLICE(
               is_tilized ? cb_intermed : cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false)
        << ENDL();
}
