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

    DPRINT("uintptr={}\n", addr);
    DPRINT("idx={} uintptr={}\n", index, addr);
    DPRINT("uintptr={} idx={}\n", addr, index);
    DPRINT("uintptr={}\n", addr);
    DPRINT("idx={} void_ptr={}\n", index, ptr);
    DPRINT("idx={} const_char_ptr={}\n", index, str);
    DPRINT("idx={} char_ptr={}\n", index, mut_str);
    DPRINT("idx={} float={:.2f}\n", index, f32);
    DPRINT("idx={} ctstr={}\n", index, CTSTR("compile time string"));
    DPRINT("idx={} u64={}\n", index, u64);
    DPRINT("idx={} i64={}\n", index, i64);
    DPRINT("idx={} double={:.2f}\n", index, dbl);
}
