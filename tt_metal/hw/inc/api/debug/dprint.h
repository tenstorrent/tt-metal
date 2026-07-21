// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_common.h"

/*
 * Device-side debug print API for device kernels.
 * Works on any hardware thread (BR/NC/TR0-2/ER).
 * On device the use is as follows:
 *
 *   DPRINT("formatted value = {:.3f}, hex = {:#x}\n", x, y);
 *
 * DPRINT is a thin alias for DEVICE_PRINT (see device_print.h), accepting an `fmt`-style format
 * string with compile-time format/argument checking. Format strings are stored host-side in
 * dedicated ELF sections, so they do not consume device L1.
 *
 * The legacy stream-style API (`DPRINT << ... << ENDL();`) has been removed. Existing call sites
 * trip a static_assert pointing at the deprecation. See
 * tech_reports/Debugging/DEVICE_PRINT_replaces_DPRINT.md for the migration guide.
 *
 * On the host it's required to start the print server first, otherwise the behavior will be
 * incorrect. The host server writes the starting magic into the device print buffer so the
 * kernel-side writer knows it can proceed; it also drains the buffer, otherwise device code
 * stalls waiting on the host to flush it.
 *
 * Use impl/debug/dprint_server.hpp APIs to start the host-side print server.
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
    static_assert(sizeof(Type) == 0, "Old style DPRINT is deprecated. Use DPRINT(format, ...) instead.");
    return dp;
}

// Tile printing only supported in kernels
#if defined(KERNEL_BUILD)
#include "dprint_tile.h"
#endif

// DPRINT is the user-facing printf-style debug print API. The macros below forward to the
// internal DEVICE_PRINT implementation in device_print.h.
#include "device_print.h"

#define DPRINT(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
#define DPRINT_UNPACK(format, ...) DEVICE_PRINT_UNPACK(format, ##__VA_ARGS__)
#define DPRINT_MATH(format, ...) DEVICE_PRINT_MATH(format, ##__VA_ARGS__)
#define DPRINT_PACK(format, ...) DEVICE_PRINT_PACK(format, ##__VA_ARGS__)
#define DPRINT_DATA0(format, ...) DEVICE_PRINT_DATA0(format, ##__VA_ARGS__)
#define DPRINT_DATA1(format, ...) DEVICE_PRINT_DATA1(format, ##__VA_ARGS__)
