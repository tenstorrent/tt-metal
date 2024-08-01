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
#include <dev_msgs.h>

#define DPRINT_TYPES                 \
    DPRINT_PREFIX(CSTR)              \
    DPRINT_PREFIX(ENDL)              \
    DPRINT_PREFIX(SETW)              \
    DPRINT_PREFIX(UINT8)             \
    DPRINT_PREFIX(UINT16)            \
    DPRINT_PREFIX(UINT32)            \
    DPRINT_PREFIX(UINT64)            \
    DPRINT_PREFIX(INT8)              \
    DPRINT_PREFIX(INT16)             \
    DPRINT_PREFIX(INT32)             \
    DPRINT_PREFIX(INT64)             \
    DPRINT_PREFIX(FLOAT32)           \
    DPRINT_PREFIX(CHAR)              \
    DPRINT_PREFIX(RAISE)             \
    DPRINT_PREFIX(WAIT)              \
    DPRINT_PREFIX(BFLOAT16)          \
    DPRINT_PREFIX(SETPRECISION)      \
    DPRINT_PREFIX(NOC_LOG_XFER)      \
    DPRINT_PREFIX(FIXED)             \
    DPRINT_PREFIX(DEFAULTFLOAT)      \
    DPRINT_PREFIX(HEX)               \
    DPRINT_PREFIX(OCT)               \
    DPRINT_PREFIX(DEC)               \
    DPRINT_PREFIX(TILESLICE)         \
    DPRINT_PREFIX(U32_ARRAY)         \
    DPRINT_PREFIX(TYPED_U32_ARRAY)   // Same as U32_ARRAY, but with the last element indicating the type of array elements

enum DPrintTypeID : uint8_t {
// clang-format off
#define DPRINT_PREFIX(a) DPrint ## a,
    DPRINT_TYPES
#undef DPRINT_PREFIX
    DPrintTypeID_Count,
// clang-format on
};
static_assert(DPrintTypeID_Count < 64, "Exceeded number of dprint types");


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
    uint8_t data[DPRINT_BUFFER_SIZE-sizeof(Aux)];

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

enum TypedU32_ARRAY_Format {
    TypedU32_ARRAY_Format_INVALID,

    TypedU32_ARRAY_Format_Raw,                                     // A raw uint32_t array
    TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, // Array of numbers with format specified in subtype

    TypedU32_ARRAY_Format_COUNT,
};

static_assert(sizeof(DebugPrintMemLayout) == DPRINT_BUFFER_SIZE);
