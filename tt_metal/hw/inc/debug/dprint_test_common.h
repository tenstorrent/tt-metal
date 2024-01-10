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
    DPRINT << "SETPRECISION/FIXED/DEFAULTFLOAT:\n";
    DPRINT << SETPRECISION(5) << my_float << ENDL();
    DPRINT << SETPRECISION(9) << my_float << ENDL();
    DPRINT << FIXED();
    DPRINT << SETPRECISION(5) << my_float << ENDL();
    DPRINT << SETPRECISION(9) << my_float << ENDL();
    DPRINT << DEFAULTFLOAT();
    DPRINT << "SETW:\n" << SETW(10) << my_int << my_int << SETW(4) << "ab" << ENDL();
    DPRINT << "HEX/OCT/DEC:\n" << HEX() << my_int << OCT() << my_int << DEC() << my_int << ENDL();
#ifndef COMPILE_FOR_ERISC
    // Eth cores don't have CBs, so don't try TSLICE printing.
    DPRINT << "SLICE:\n";
    cb_wait_front(tt::CB::c_in0, 1);
    DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_8());
    DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_4());
#endif
}
