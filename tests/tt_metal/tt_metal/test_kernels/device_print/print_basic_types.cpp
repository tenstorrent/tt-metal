// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    int8_t i8 = -8;
    DEVICE_PRINT("int8_t: {}", i8);
    uint8_t u8 = 8;
    DEVICE_PRINT("uint8_t: {}", u8);
    int16_t i16 = -16;
    DEVICE_PRINT("int16_t: {}", i16);
    uint16_t u16 = 16;
    DEVICE_PRINT("uint16_t: {}", u16);
    int32_t i32 = -32;
    DEVICE_PRINT("int32_t: {}", i32);
    uint32_t u32 = 32;
    DEVICE_PRINT("uint32_t: {}", u32);
    int64_t i64 = -64;
    DEVICE_PRINT("int64_t: {}", i64);
    uint64_t u64 = 64;
    DEVICE_PRINT("uint64_t: {}", u64);
    float f32 = 3.14f;
    DEVICE_PRINT("float: {}", f32);
    double f64 = 6.28;
    DEVICE_PRINT("double: {}", f64);
    bool b = true;
    DEVICE_PRINT("bool: {}", b);

    DEVICE_PRINT("Reordered args: {} {} {} {}", b, i16, i32, i64);
    DEVICE_PRINT("Reordered args: {0} {1} {2} {3}", b, i16, i32, i64);
}
