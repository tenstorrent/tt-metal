// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    uint32_t index = 42;
    std::uintptr_t addr = static_cast<std::uintptr_t>(0x12345678);
    void* ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xABCDEF00));
    const char* str = "runtime string";
    char mut_str[] = "mutable string";
    float f32 = 3.14159f;
    uint64_t u64 = 0x123456789ABCDEF0ull;
    int64_t i64 = -0x123456789LL;
    double dbl = 6.25;

    DEVICE_PRINT("idx={} uintptr={}\n", index, addr);
    DEVICE_PRINT("uintptr={} idx={}\n", addr, index);
    DEVICE_PRINT("uintptr={}\n", addr);
    DEVICE_PRINT("idx={} void_ptr={}\n", index, ptr);
    DEVICE_PRINT("idx={} const_char_ptr={}\n", index, str);
    DEVICE_PRINT("idx={} char_ptr={}\n", index, mut_str);
    DEVICE_PRINT("idx={} float={:.2f}\n", index, f32);
    DEVICE_PRINT("idx={} ctstr={}\n", index, CTSTR("compile time string"));
    DEVICE_PRINT("idx={} u64={}\n", index, u64);
    DEVICE_PRINT("idx={} i64={}\n", index, i64);
    DEVICE_PRINT("idx={} double={:.2f}\n", index, dbl);
}
