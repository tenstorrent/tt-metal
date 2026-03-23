// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    int8_t i8 = -8;
    DEVICE_PRINT("int8_t: {}\n", i8);
    uint8_t u8 = 8;
    DEVICE_PRINT("uint8_t: {}\n", u8);
    int16_t i16 = -16;
    DEVICE_PRINT("int16_t: {}\n", i16);
    uint16_t u16 = 16;
    DEVICE_PRINT("uint16_t: {}\n", u16);
    int32_t i32 = -32;
    DEVICE_PRINT("int32_t: {}\n", i32);
    uint32_t u32 = 32;
    DEVICE_PRINT("uint32_t: {}\n", u32);
    int64_t i64 = -64;
    DEVICE_PRINT("int64_t: {}\n", i64);
    uint64_t u64 = 64;
    DEVICE_PRINT("uint64_t: {}\n", u64);
    float f32 = 3.14f;
    DEVICE_PRINT("float: {}\n", f32);
    double f64 = 6.28;
    DEVICE_PRINT("double: {}\n", f64);
    bool b = true;
    DEVICE_PRINT("bool: {}\n", b);
    auto bf4 = bf4_t(128, 1);
    DEVICE_PRINT("bf4_t: {}\n", bf4);
    auto bf8 = bf8_t(130, 3);
    DEVICE_PRINT("bf8_t: {}\n", bf8);
    auto bf16 = bf16_t(0x3dfb);
    DEVICE_PRINT("bf16_t: {}\n", bf16);

    // Here we are testing how compile-time code generation in kernels handles argument reordering.
    // Arguments are reordered by size when sent to the host to guarantee proper alignment during serialization.
    // This means the format string in the ELF will be stored as "Reordered args: {3,?} {2,h} {1,i} {0,q}\n" instead of
    // "Reordered args: {0,?} {1,h} {2,i} {3,q}\n". In other words, the compiler generates code as if the user wrote:
    // DEVICE_PRINT("Reordered args: {3} {2} {1} {0}\n", i64, i32, i16, b);
    DEVICE_PRINT("Reordered args: {} {} {} {}\n", b, i16, i32, i64);
    DEVICE_PRINT("Reordered args: {0} {1} {2} {3}\n", b, i16, i32, i64);
}
