// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 *
 * Type ids shared between the debug print server and on-device debug prints.
 *
 */

#pragma once

// DataFormat comes from tt_backend_api_types.hpp for SW, and tensix_types.h for HW...
// But wait there's more, SW also includes tensix_types.h so there's both tt::DataFormat and DataFormat there. Use a
// different name here so that this header can be included in both.
#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)  // SW
#include "common/tt_backend_api_types.hpp"
typedef tt::DataFormat CommonDataFormat;
#else  // HW already includes tensix_types.h
#include "core_config.h"
typedef DataFormat CommonDataFormat;
#endif

#include <cstddef>

constexpr static std::uint32_t DPRINT_BUFFER_SIZE = 204;  // per thread
// TODO: when device specific headers specify number of processors
// (and hal abstracts them on host), get these from there
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
constexpr static std::uint32_t DPRINT_BUFFERS_COUNT = 1;
#else
constexpr static std::uint32_t DPRINT_BUFFERS_COUNT = 5;
#endif

// Used to index into the DPRINT buffers. Erisc is separate because it only has one buffer.
enum DebugPrintHartIndex : unsigned int {
    DPRINT_RISCV_INDEX_NC = 0,
    DPRINT_RISCV_INDEX_TR0 = 1,
    DPRINT_RISCV_INDEX_TR1 = 2,
    DPRINT_RISCV_INDEX_TR2 = 3,
    DPRINT_RISCV_INDEX_BR = 4,
    DPRINT_RISCV_INDEX_ER = 0,
    DPRINT_RISCV_INDEX_ER1 = 1,
};
#define DPRINT_NRISCVS 5
#ifdef ARCH_BLACKHOLE
#define DPRINT_NRISCVS_ETH 2
#else
#define DPRINT_NRISCVS_ETH 1
#endif

#define DPRINT_TYPES            \
    DPRINT_PREFIX(CSTR)         \
    DPRINT_PREFIX(ENDL)         \
    DPRINT_PREFIX(SETW)         \
    DPRINT_PREFIX(UINT8)        \
    DPRINT_PREFIX(UINT16)       \
    DPRINT_PREFIX(UINT32)       \
    DPRINT_PREFIX(UINT64)       \
    DPRINT_PREFIX(INT8)         \
    DPRINT_PREFIX(INT16)        \
    DPRINT_PREFIX(INT32)        \
    DPRINT_PREFIX(INT64)        \
    DPRINT_PREFIX(FLOAT32)      \
    DPRINT_PREFIX(CHAR)         \
    DPRINT_PREFIX(RAISE)        \
    DPRINT_PREFIX(WAIT)         \
    DPRINT_PREFIX(BFLOAT16)     \
    DPRINT_PREFIX(SETPRECISION) \
    DPRINT_PREFIX(FIXED)        \
    DPRINT_PREFIX(DEFAULTFLOAT) \
    DPRINT_PREFIX(HEX)          \
    DPRINT_PREFIX(OCT)          \
    DPRINT_PREFIX(DEC)          \
    DPRINT_PREFIX(TILESLICE)    \
    DPRINT_PREFIX(U32_ARRAY)    \
    DPRINT_PREFIX(              \
        TYPED_U32_ARRAY)  // Same as U32_ARRAY, but with the last element indicating the type of array elements

enum DPrintTypeID : uint8_t {
// clang-format off
#define DPRINT_PREFIX(a) DPrint ## a,
    DPRINT_TYPES
#undef DPRINT_PREFIX
    DPrintTypeID_Count,
    // clang-format on
};
static_assert(DPrintTypeID_Count < 64, "Exceeded number of dprint types");

// We need to set the wpos, rpos pointers to 0 in the beginning of the kernel startup
// Because there's no mechanism (known to me) to initialize values at fixed mem locations in kernel code,
// in order to initialize the pointers in the buffers we use a trick with print server writing
// a magic value to a fixed location, we look for it on device, and only if it's present we initialize
// the read and write ptrs to 0 in DebugPrinter() constructor. This check is actually done every time
// a DebugPrinter() object is created (which can be many times), but results in resetting the pointers
// only once.
// These magic values must not be equal to any real wpos/rpos values.
constexpr uint32_t DEBUG_PRINT_SERVER_STARTING_MAGIC = 0x98989898;
constexpr uint32_t DEBUG_PRINT_SERVER_DISABLED_MAGIC = 0xf8f8f8f8;

#define ATTR_PACK __attribute__((packed))

struct DebugPrintMemLayout {
    struct Aux {
        // current writer offset in buffer
        uint32_t wpos;
        uint32_t rpos;
        uint16_t core_x;
        uint16_t core_y;
    } aux ATTR_PACK;
    uint8_t data[DPRINT_BUFFER_SIZE - sizeof(Aux)];

    static size_t rpos_offs() { return offsetof(DebugPrintMemLayout::Aux, rpos) + offsetof(DebugPrintMemLayout, aux); }

} ATTR_PACK;

struct SliceRange {
    // A slice object encoding semantics of np.slice(h0:h1:hs, w0:w1:ws)
    // This is only used with DPRINT for TileSlice object
    uint8_t h0, h1, hs, w0, w1, ws;
    // [0:32:16, 0:32:16]
    static inline SliceRange hw0_32_16() {
        return SliceRange{.h0 = 0, .h1 = 32, .hs = 16, .w0 = 0, .w1 = 32, .ws = 16};
    }
    // [0:32:8, 0:32:8]
    static inline SliceRange hw0_32_8() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8}; }
    // [0:32:4, 0:32:4]
    static inline SliceRange hw0_32_4() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 4, .w0 = 0, .w1 = 32, .ws = 4}; }
    // [0, 0:32]
    static inline SliceRange h0_w0_32() { return SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}; }
    // [0:32, 0]
    static inline SliceRange h0_32_w0() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1}; }
    // [0:32:1, 1]
    static inline SliceRange h0_32_w1() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 1, .w1 = 2, .ws = 1}; }
    // [0:4:1, 0:4:1]
    static inline SliceRange hw041() { return SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1}; }
} ATTR_PACK;

template <int MAX_BYTES = 0>
struct TileSliceHostDev {
    uint32_t cb_ptr;
    struct SliceRange slice_range;
    uint8_t cb_id;
    uint8_t data_format;
    uint8_t data_count;
    uint8_t endl_rows;
    uint8_t return_code;
    uint8_t pad;
    uint8_t data[MAX_BYTES];
} ATTR_PACK;

enum dprint_tileslice_return_code_enum {
    DPrintOK = 2,
    DPrintErrorBadTileIdx = 3,
    DPrintErrorBadPointer = 4,
    DPrintErrorUnsupportedFormat = 5,
    DPrintErrorMath = 6,
    DPrintErrorEthernet = 7,
};
enum TypedU32_ARRAY_Format {
    TypedU32_ARRAY_Format_INVALID,

    TypedU32_ARRAY_Format_Raw,                                      // A raw uint32_t array
    TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type,  // Array of numbers with format specified in subtype

    TypedU32_ARRAY_Format_COUNT,
};

static_assert(sizeof(DebugPrintMemLayout) == DPRINT_BUFFER_SIZE);
// We use DebugPrintMemLayout to hold noc xfer data, 32 buckets (one for each bit in noc xfer length field).
static_assert(sizeof(DebugPrintMemLayout().data) >= sizeof(uint32_t) * 8 * sizeof(uint32_t));

// Size of datum in bytes, dprint-specific to support device-side and bfp* DataFormats
static inline constexpr uint32_t dprint_datum_size(const CommonDataFormat& format) {
    switch (format) {
        case CommonDataFormat::Bfp2:
        case CommonDataFormat::Bfp2_b:
        case CommonDataFormat::Bfp4:
        case CommonDataFormat::Bfp4_b:
        case CommonDataFormat::Bfp8:
        case CommonDataFormat::Bfp8_b: return 1;  // Round up to 1 byte
        case CommonDataFormat::Float16:
        case CommonDataFormat::Float16_b: return 2;
        case CommonDataFormat::Float32: return 4;
        case CommonDataFormat::Int8: return 1;
        case CommonDataFormat::Lf8: return 1;
        case CommonDataFormat::UInt8: return 1;
        case CommonDataFormat::UInt16: return 2;
        case CommonDataFormat::UInt32: return 4;
        case CommonDataFormat::Int32: return 4;
        case CommonDataFormat::Invalid: return 0;  // Invalid
        default: return 0;                         // Unknown
    }
}

static inline constexpr bool is_bfp(const CommonDataFormat& format) {
    switch (format) {
        case CommonDataFormat::Bfp2:
        case CommonDataFormat::Bfp2_b:
        case CommonDataFormat::Bfp4:
        case CommonDataFormat::Bfp4_b:
        case CommonDataFormat::Bfp8:
        case CommonDataFormat::Bfp8_b: return true;
        case CommonDataFormat::Float16:
        case CommonDataFormat::Float16_b:
        case CommonDataFormat::Float32:
        case CommonDataFormat::Int8:
        case CommonDataFormat::Lf8:
        case CommonDataFormat::UInt8:
        case CommonDataFormat::UInt16:
        case CommonDataFormat::UInt32:
        case CommonDataFormat::Int32:
        case CommonDataFormat::Invalid:
        default: return false;
    }
}

static inline constexpr bool is_supported_format(const CommonDataFormat& format) {
    switch (format) {
        case CommonDataFormat::Bfp2:
        case CommonDataFormat::Bfp2_b:
        case CommonDataFormat::Bfp4: return false;
        case CommonDataFormat::Bfp4_b: return true;
        case CommonDataFormat::Bfp8: return false;
        case CommonDataFormat::Bfp8_b: return true;
        case CommonDataFormat::Float16: return false;
        case CommonDataFormat::Float16_b: return true;
        case CommonDataFormat::Float32: return true;
        case CommonDataFormat::Int8:
        case CommonDataFormat::UInt8:
        case CommonDataFormat::UInt16:
        case CommonDataFormat::UInt32:
        case CommonDataFormat::Int32: return true;
        case CommonDataFormat::Lf8:
        case CommonDataFormat::Invalid:
        default: return false;
    }
}
