// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
*
* Type ids shared between the debug print server and on-device debug prints.
*
*/

#pragma once

#include <cstddef>

constexpr int DEBUG_PRINT_TYPEID_CSTR           = 0;
constexpr int DEBUG_PRINT_TYPEID_ENDL           = 1; // std::endl
constexpr int DEBUG_PRINT_TYPEID_SETW           = 2; // std::setw
constexpr int DEBUG_PRINT_TYPEID_UINT32         = 3;
constexpr int DEBUG_PRINT_TYPEID_FLOAT32        = 4;
constexpr int DEBUG_PRINT_TYPEID_CHAR           = 5;
constexpr int DEBUG_PRINT_TYPEID_RAISE          = 6;
constexpr int DEBUG_PRINT_TYPEID_WAIT           = 7;
constexpr int DEBUG_PRINT_TYPEID_BFLOAT16       = 8;
constexpr int DEBUG_PRINT_TYPEID_SETPRECISION   = 9; // std::setprecision
constexpr int DEBUG_PRINT_TYPEID_FIXED          = 10; // std::fixed
constexpr int DEBUG_PRINT_TYPEID_DEFAULTFLOAT   = 11; // std::defaultfloat
constexpr int DEBUG_PRINT_TYPEID_HEX            = 12; // std::hex
constexpr int DEBUG_PRINT_TYPEID_OCT            = 13; // std::oct
constexpr int DEBUG_PRINT_TYPEID_DEC            = 14; // std::dec
constexpr int DEBUG_PRINT_TYPEID_INT32          = 15;
constexpr int DEBUG_PRINT_TYPEID_TILESLICE      = 16;
constexpr int DEBUG_PRINT_TYPEID_UINT64         = 17;

// We need to set thw wpos, rpos pointers to 0 in the beginning of the kernel startup
// Because there's no mechanism (known to me) to initialize values at fixed mem locations in kernel code,
// in order to initialize the pointers in the buffers we use a trick with print server writing
// a magic value to a fixed location, we look for it on device, and only if it's present we initialize
// the read and write ptrs to 0 in DebugPrinter() constructor. This check is actually done every time
// a DebugPrinter() object is created (which can be many times), but results in resetting the pointers
// only once.
constexpr int DEBUG_PRINT_SERVER_STARTING_MAGIC = 0x12341234;
constexpr int DEBUG_PRINT_SERVER_DISABLED_MAGIC = 0x23455432;

// In case a single argument to operator << (such as a string) is larger than the buffer size
// (making it impossible to print) we will instead print this message.
constexpr const char* debug_print_overflow_error_message = "*** INTERNAL DEBUG PRINT BUFFER OVERFLOW ***\n\n";

#define ATTR_ALIGN4 __attribute__((aligned(4)))
#define ATTR_ALIGN2 __attribute__((aligned(2)))
#define ATTR_ALIGN1 __attribute__((aligned(1)))
#define ATTR_PACK   __attribute__((packed))

struct DebugPrintMemLayout {
    struct Aux {
        // current writer offset in buffer
        uint32_t wpos ATTR_ALIGN4;
        uint32_t rpos ATTR_ALIGN4;
        uint16_t core_x ATTR_ALIGN2;
        uint16_t core_y ATTR_ALIGN2;
    } aux ATTR_ALIGN4;
    uint8_t data[PRINT_BUFFER_SIZE-sizeof(Aux)];

    static size_t rpos_offs() { return offsetof(DebugPrintMemLayout::Aux, rpos) + offsetof(DebugPrintMemLayout, aux); }

} ATTR_PACK;

template<int MAXCOUNT=0>
struct TileSliceHostDev {
    uint32_t ptr_                ATTR_ALIGN4; // also print the cb fifo pointer for debugging
    uint16_t h0_                 ATTR_ALIGN2;
    uint16_t h1_                 ATTR_ALIGN2;
    uint16_t hs_                 ATTR_ALIGN2;
    uint16_t w0_                 ATTR_ALIGN2;
    uint16_t w1_                 ATTR_ALIGN2;
    uint16_t ws_                 ATTR_ALIGN2;
    uint16_t count_              ATTR_ALIGN2;
    uint16_t endl_rows_          ATTR_ALIGN2;
    uint16_t samples_[MAXCOUNT]  ATTR_ALIGN2;
} ATTR_PACK;

static_assert(sizeof(DebugPrintMemLayout) == PRINT_BUFFER_SIZE);
