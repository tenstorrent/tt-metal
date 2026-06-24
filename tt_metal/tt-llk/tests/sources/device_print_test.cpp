// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "dprint.h"
#include "tensix_types.h"

enum class Color : std::uint8_t
{
    Red   = 0,
    Green = 1,
    Blue  = 2
};

enum class Perm : std::uint32_t
{
    R = 1,
    W = 2,
    X = 4
};

constexpr Perm operator|(Perm a, Perm b)
{
    return static_cast<Perm>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

// Declaration order != ascending value order. Used to verify the parser
// iterates enumerators in DWARF source order (matching dprint_parser.cpp).
enum class Rev : std::uint32_t
{
    Z = 4,
    Y = 2,
    X = 1
};

constexpr Rev operator|(Rev a, Rev b)
{
    return static_cast<Rev>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT("unpack: i8={} u8={} i16={} u16={}", (std::int8_t)-1, (std::uint8_t)255, (std::int16_t)-100, (std::uint16_t)65535);
    DEVICE_PRINT("unpack: str={}", CTSTR("_unpack"));
    DEVICE_PRINT("unpack: enum={}", Color::Green);
    DEVICE_PRINT("unpack: flag={}", Perm::R | Perm::X);
    DEVICE_PRINT("unpack: flag_full={:#}", Perm::R | Perm::W);
    DEVICE_PRINT("unpack: flag_unk={}", static_cast<Perm>(0x18));
    DEVICE_PRINT("unpack: flag_rev={}", Rev::X | Rev::Y | Rev::Z);

    // dp_typed_array_t wire-format smoke (bit patterns for 1.0, 2.0, 3.0, 4.0).
    std::uint32_t arr[4] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000};
    DEVICE_PRINT("unpack: array={}", dp_typed_array_t<4>(static_cast<std::uint16_t>(DataFormat::Float32), arr));
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS)
{
    // Print a value from each type category.
    DEVICE_PRINT("math: i32={} u32={}", (std::int32_t)-1, (std::uint32_t)65536);
    DEVICE_PRINT("math: float={}", 1.0f);
    DEVICE_PRINT("math: bool={} {}", true, false);
    DEVICE_PRINT("math: ptr={}", reinterpret_cast<std::uint32_t*>(0xDEADBEEF));
    DEVICE_PRINT("math: str={}", CTSTR("_math"));
    DEVICE_PRINT("math: hex={:08x}", (std::uint32_t)0xABC);
    DEVICE_PRINT("math: pad={:>8}", CTSTR("test"));

    // Flood the buffer to force a drain.
    for (std::uint32_t i = 0; i < 2048; ++i)
    {
        DEVICE_PRINT("w={}", i);
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT("pack: i64={}", (std::int64_t)-1000000LL);
    DEVICE_PRINT("pack: str={}", CTSTR("_pack"));
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

void run_kernel(RUNTIME_PARAMETERS)
{
    DEVICE_PRINT("sfpu: u8={} i8={}", (std::uint8_t)3, (std::int8_t)-1);
    DEVICE_PRINT("sfpu: str={}", CTSTR("_sfpu"));
}

#endif
