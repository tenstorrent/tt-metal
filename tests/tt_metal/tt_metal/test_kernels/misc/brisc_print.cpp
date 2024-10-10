// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_test_common.h"

/*
 * Test printing from a kernel running on BRISC.
*/

void kernel_main() {
    // Write some data to the CB that will be used to test TSLICE.
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    cb_reserve_back(cb_id_in0, 1);
    auto ptr = reinterpret_cast<BF16*>(get_write_ptr(cb_id_in0));
    uint16_t bfloat16_base = 0x3dfb;
    for (uint16_t idx = 0; idx < 32*32; idx++) {
        ptr[idx] = BF16(idx + bfloat16_base);
    }
    cb_push_back(cb_id_in0, 1);

    cb_reserve_back(tt::CB::c_in2, 1);
    auto float_ptr = reinterpret_cast<float *>(get_write_ptr(tt::CB::c_in2));
    float float_base = 3.14159274101257324219;
    for (uint16_t idx = 0; idx < 32*32; idx++) {
        float_ptr[idx] = float_base + (float) idx;
    }
    cb_push_back(tt::CB::c_in2, 1);

    DPRINT_DATA0(
        DPRINT << "Test Debug Print: Data0" << ENDL();
        print_test_data();
        // Let the next core (PACK) know to start printing.
        DPRINT << RAISE{1};
    );
}
