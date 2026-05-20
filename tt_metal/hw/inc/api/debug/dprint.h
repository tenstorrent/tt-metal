// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_common.h"

/*
 * Device-side debug print API for device kernels.
 * Works on either one of NC/BR/TR threads.
 * On device the use is as follows:
 *
 * DPRINT << SETW(2) << 0 << 0.1f << "string" << ENDL();
 *
 * This DebugPrinter object can be created multiple times.
 *
 * On the host it's required to start the print server first, otherwise the behavior will be incorrect.
 * This is because the host print server writes a special value that is used in DebugPrinter() constructor
 * to initialize the read/write pointers to 0 only once.
 * It is also needed to empty the print buffer, otherwise device code will stall waiting on the host to flush it.
 *
 * Use impl/debug/dprint_server.h APIs to start the host-side print server.
 *
 */

#include <cstdint>
#include <type_traits>
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
// TODO(AP): this ifdef doesn't seem to make sense given we include risc_common.h
// The issue is some files included inside risc_common.h only apply to NC/BRISCS
// But moving this ifdef inside of the header breaks other code
// So there are some not fully decoupled dependencies in this header.
#include "risc_common.h"
#endif
#include "hostdevcommon/dprint_common.h"

#include "internal/debug/dprint_buffer.h"
#include "waypoint.h"

namespace internal_ {
void risc_context_switch(bool);
}

// Deprecated structures that should be removed once all users are migrated to the new API.
struct [[deprecated("Use DEVICE_PRINT's bf16_t instead")]] BF16 {
    BF16(uint16_t) {}
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] F32 {
    F32(float) {}
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] U32 {
    U32(uint32_t) {}
} ATTR_PACK;

struct [[deprecated("Use DEVICE_PRINT instead")]] ENDL {
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] SETPRECISION {
    SETPRECISION(char) {}
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] FIXED {
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] DEFAULTFLOAT {
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] HEX {
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] OCT {
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] DEC {
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT instead")]] SETW {
    SETW(char) {}
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT's dp_typed_array_t<> instead")]] U32_ARRAY {
    U32_ARRAY(uint32_t*, uint32_t) {}
} ATTR_PACK;
struct [[deprecated("Use DEVICE_PRINT's dp_typed_array_t<> instead")]] TYPED_U32_ARRAY {
    TYPED_U32_ARRAY(uint16_t, uint16_t, uint32_t*, uint32_t) {}
} ATTR_PACK;

// Fake printer object used only to report nice errors for deprecated API usage.
class OldStyleDevicePrint {
} DPRINT;

template <typename Type>
__attribute__((__noinline__)) OldStyleDevicePrint operator<<(OldStyleDevicePrint dp, Type) {
    static_assert(sizeof(Type) == 0, "Old style DPRINT is deprecated. Use DEVICE_PRINT instead.");
    return dp;
}

// Tile printing only supported in kernels
#if defined(KERNEL_BUILD)
#include "dprint_tile.h"
#endif

// Support for DEVICE_PRINT.
#include "device_print.h"

// Forward declaration of DEVICE_PRINT's internal implementation, which we will call from the public API.
#define DPRINT(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
