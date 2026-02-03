// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/new_dprint.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    int8_t i8 = -8;
    NEW_DPRINT("int8_t: {: >-10}", i8);
    uint8_t u8 = 8;
    NEW_DPRINT("uint8_t: {:#B}", u8);
    int16_t i16 = -16;
    NEW_DPRINT("int16_t: {: <-10}", i16);
    uint16_t u16 = 16;
    NEW_DPRINT("uint16_t: {:#X}", u16);
    int32_t i32 = -32;
    NEW_DPRINT("int32_t: {: ^-10}", i32);
    uint32_t u32 = 32;
    NEW_DPRINT("uint32_t: {:#x}", u32);
    int64_t i64 = -64;
    NEW_DPRINT("int64_t: {: }", i64);
    uint64_t u64 = 64;
    NEW_DPRINT("uint64_t: {:#08X}", u64);
    float f32 = 3.14f;
    NEW_DPRINT("float: {:3.3g}", f32);
    double f64 = 6.28;
    NEW_DPRINT("double: {:.5f}", f64);
    bool b = true;
    NEW_DPRINT("bool: {}", b);
}
