// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "dprint.h"

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT_INITIALIZE_LOCK();
    DEVICE_PRINT("unpack: i8={} u8={} i16={} u16={}\n", (std::int8_t)-1, (std::uint8_t)255, (std::int16_t)-100, (std::uint16_t)65535);
    DEVICE_PRINT("unpack: str={}\n", CTSTR("_unpack"));
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT_INITIALIZE_LOCK();

    // Print a value from each type category.
    DEVICE_PRINT("math: i32={} u32={}\n", (std::int32_t)-1, (std::uint32_t)65536);
    DEVICE_PRINT("math: float={}\n", 1.0f);
    DEVICE_PRINT("math: bool={} {}\n", true, false);
    DEVICE_PRINT("math: ptr={}\n", reinterpret_cast<std::uint32_t*>(0xDEADBEEF));
    DEVICE_PRINT("math: str={}\n", CTSTR("_math"));
    DEVICE_PRINT("math: hex={:08x}\n", (std::uint32_t)0xABC);
    DEVICE_PRINT("math: pad={:>8}\n", CTSTR("test"));

    // Flood the buffer to force a drain.
    for (std::uint32_t i = 0; i < 160u; ++i)
    {
        DEVICE_PRINT("w={}\n", i);
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT_INITIALIZE_LOCK();
    DEVICE_PRINT("pack: i64={}\n", (std::int64_t)-1000000LL);
    DEVICE_PRINT("pack: str={}\n", CTSTR("_pack"));
}

#endif
