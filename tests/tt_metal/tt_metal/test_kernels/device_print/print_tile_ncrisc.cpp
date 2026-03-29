// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "api/debug/ring_buffer.h"

void kernel_main() {
    // Read out the tile we want to print using BRISC, put it in c_in0
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_1;
    uint32_t is_tilized = get_arg_val<uint32_t>(0);

    cb_wait_front(cb_id, 1);
    if (is_tilized) {
        cb_wait_front(cb_intermed, 1);
    }
    DEVICE_PRINT(
        "Print tile from Data1:\n{}\n",
        TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, is_tilized));
}
