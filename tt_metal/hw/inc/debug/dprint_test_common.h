// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "debug/dprint.h"

// A helper function to exercise print features.
inline void print_test_data() {
    uint8_t  my_uint8  = 101;
    uint16_t my_uint16 = 555;
    uint32_t my_uint32 = 123456;
    uint64_t my_uint64 = 9123456789;
    int8_t  my_int8  = -17;
    int16_t my_int16 = -343;
    int32_t my_int32 = -44444;
    int64_t my_int64 = -5123456789;
    float my_float = 3.14159f;
    DPRINT << "Basic Types:\n" << 101 << -1.6180034f << '@' << BF16(0x3dfb) << ENDL();
    DPRINT << my_uint8 << my_uint16 << my_uint32 << my_uint64 << ENDL();
    DPRINT << my_int8 << my_int16 << my_int32 << my_int64 << ENDL();
    DPRINT << "SETPRECISION/FIXED/DEFAULTFLOAT:\n";
    DPRINT << SETPRECISION(5) << my_float << ENDL();
    DPRINT << SETPRECISION(9) << my_float << ENDL();
    DPRINT << FIXED();
    DPRINT << SETPRECISION(5) << my_float << ENDL();
    DPRINT << SETPRECISION(9) << my_float << ENDL();
    DPRINT << DEFAULTFLOAT();
    DPRINT << "SETW:\n" << SETW(10) << my_uint32 << my_uint32 << SETW(4) << "ab" << ENDL();
    DPRINT << "HEX/OCT/DEC:\n" << HEX() << my_uint32 << OCT() << my_uint32 << DEC() << my_uint32 << ENDL();
#if !defined(COMPILE_FOR_ERISC) && !defined(COMPILE_FOR_IDLE_ERISC)
    // Eth cores don't have CBs, so don't try TSLICE printing.
    DPRINT << "SLICE:\n";
    cb_wait_front(tt::CB::c_in0, 1);
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    // Since brisc is writing to the CB before printing, should look at read pointer
    DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR);
    DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_4(), TSLICE_INPUT_CB, TSLICE_RD_PTR);
    // This one has an unsupported data type, should show a warning instead of data
    DPRINT << TSLICE(tt::CB::c_in1, 0, SliceRange::hw0_32_4(), TSLICE_INPUT_CB, TSLICE_RD_PTR);
#else
    DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_8());
    DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_4());
    DPRINT << TSLICE(tt::CB::c_in1, 0, SliceRange::hw0_32_4());
#endif
#endif
}
