// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dprint.h"
#include "compute_kernel_api.h"

void dprint_tensix_dest_reg() {
    dbg_halt();
    MATH({
        DPRINT_MATH(DPRINT << FIXED() << SETPRECISION(2));
        uint32_t rd_data[8+1]; // data + array_type
        for (int row = 0; row < 64; row++) {
            dbg_read_dest_acc_row(row, rd_data);
            DPRINT_MATH(DPRINT << SETW(6) << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_TensixRegister_FP16_B, rd_data, 8));
            if (row % 2 == 1) DPRINT_MATH(DPRINT << ENDL());
        }
    })
    dbg_unhalt();
}
