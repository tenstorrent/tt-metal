// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "debug/dprint.h"

// A helper function to exercise print features.
inline void print_test_data() {
    uint32_t my_int = 123456;
    float my_float = 3.14159f;
    DPRINT << "Basic Types:\n" << 101 << -1.6180034f << '@' << BF16(0x3dfb) << ENDL();
    DPRINT << "SETP/FIXP:\n";
    DPRINT << SETP(5) << my_float << ENDL();
    DPRINT << SETP(9) << my_float << ENDL();
    DPRINT << FIXP();
    // See issue #3990, disable this line for now until it's fixed.
    // DPRINT << SETP(5) << my_float << ENDL();
    DPRINT << SETP(9) << my_float << ENDL();
    DPRINT << "SETW (sticky):\n" << SETW(10) << my_int << my_int << ENDL();
    DPRINT << "SETW (non-sticky):\n" << SETW(10, false) << my_int << my_int << ENDL();
    DPRINT << "HEX:\n" << HEX() << my_int << ENDL();
    DPRINT << "SLICE:\n";
// See issue #3970, disable TSLICE printing for math for now.
#ifndef UCK_CHLKC_MATH
    DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_8());
    //DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_4());
#endif
}
