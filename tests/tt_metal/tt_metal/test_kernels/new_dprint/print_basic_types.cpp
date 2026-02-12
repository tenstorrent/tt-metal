// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/new_dprint.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    int8_t i8 = -8;
    NEW_DPRINT("int8_t: {}", i8);
    uint8_t u8 = 8;
    NEW_DPRINT("uint8_t: {}", u8);
    int16_t i16 = -16;
    NEW_DPRINT("int16_t: {}", i16);
    uint16_t u16 = 16;
    NEW_DPRINT("uint16_t: {}", u16);
    int32_t i32 = -32;
    NEW_DPRINT("int32_t: {}", i32);
    uint32_t u32 = 32;
    NEW_DPRINT("uint32_t: {}", u32);
    int64_t i64 = -64;
    NEW_DPRINT("int64_t: {}", i64);
    uint64_t u64 = 64;
    NEW_DPRINT("uint64_t: {}", u64);
    float f32 = 3.14f;
    NEW_DPRINT("float: {}", f32);
    double f64 = 6.28;
    NEW_DPRINT("double: {}", f64);
    bool b = true;
    NEW_DPRINT("bool: {}", b);

    NEW_DPRINT("Reordered args: {} {} {} {}", b, i16, i32, i64);
    NEW_DPRINT("Reordered args: {0} {1} {2} {3}", b, i16, i32, i64);
}
