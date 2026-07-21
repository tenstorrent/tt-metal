// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
#if !defined(KERNEL_BUILD) && !defined(FW_BUILD) && !defined(ENV_LLK_INFRA)  // SW
#include <tt-metalium/tt_backend_api_types.hpp>
using CommonDataFormat = tt::DataFormat;
#else
#include "core_config.h"
#include "tensix_types.h"
using CommonDataFormat = DataFormat;
#endif

#include <cstddef>

#if !defined(ENV_LLK_INFRA)
// Deprecated buffer size from old DPRINT implementation. Used only to verify that buffers are still the same size.
constexpr static std::uint32_t DPRINT_BUFFER_SIZE = 204;  // per thread
#endif

// Magic values the server writes into the device print buffer's wpos slot to signal init state to
// the kernel-side writer. STARTING means "server is alive, kernel may reset pointers and write";
// DISABLED means "server is not draining, kernel should skip prints." Must not collide with any
// legitimate wpos/rpos value.
constexpr uint32_t DEBUG_PRINT_SERVER_STARTING_MAGIC = 0x98989898;
constexpr uint32_t DEBUG_PRINT_SERVER_DISABLED_MAGIC = 0xf8f8f8f8;
constexpr uint32_t DEVICE_PRINT_RESET_BUFFER_MAGIC = 0xF0E1D2C3;
constexpr uint32_t DEVICE_PRINT_WRITE_STALL_FLAG = 1u << 31;

#define ATTR_PACK __attribute__((packed))

struct SliceRange {
    // A slice object encoding semantics of np.slice(h0:h1:hs, w0:w1:ws)
    // This is only used with DPRINT for TileSlice object
    uint8_t h0, h1, hs, w0, w1, ws;
    // [0:32:16, 0:32:16]
    static SliceRange hw0_32_16() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 16, .w0 = 0, .w1 = 32, .ws = 16}; }
    // [0:32:8, 0:32:8]
    static SliceRange hw0_32_8() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8}; }
    // [0:32:4, 0:32:4]
    static SliceRange hw0_32_4() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 4, .w0 = 0, .w1 = 32, .ws = 4}; }
    // [0, 0:32]
    static SliceRange h0_w0_32() { return SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}; }
    // [0:32, 0]
    static SliceRange h0_32_w0() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1}; }
    // [0:32:1, 1]
    static SliceRange h0_32_w1() { return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 1, .w1 = 2, .ws = 1}; }
    // [0:4:1, 0:4:1]
    static SliceRange hw041() { return SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1}; }
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
