// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The parts of the report vocabulary that are the same wherever the sanitizer is built. Reached
// through deps/device.h or deps/host.h, which bind the rest to what their platform supplies.

#pragma once

#include <cstdint>

#if !defined(SAN_DEPS_CARRIERS_FROM_METAL) || !defined(SAN_DEPS_PRINT_LIVE)
#error "llk::san | fault   | deps/common.h is reached through deps/device.h or deps/host.h, which set the selectors first"
#endif

// sizeof is unevaluated: the arguments are type-checked and count as used, but nothing is emitted.
#define SAN_DISCARD(...) ((void)sizeof(__VA_ARGS__))

#if !defined(FULL_KERNEL_NAME)
#define FULL_KERNEL_NAME "<unknown>"
#endif

#if !SAN_DEPS_PRINT_LIVE
#define SAN_PRINT(...) SAN_DISCARD(__VA_ARGS__)
#endif

#if !SAN_DEPS_CARRIERS_FROM_METAL

namespace llk::san
{

struct string
{
    const char* ptr;
};

// The device twin subtracts the kernel offset and rewinds ra; neither has meaning off-device.
struct callstack
{
    std::uintptr_t pc;
    std::uintptr_t ra;
    std::uintptr_t skip_frames;

    callstack(std::uintptr_t pc, std::uintptr_t ra, std::uintptr_t skip_frames) : pc(pc), ra(ra), skip_frames(skip_frames)
    {
    }
};

// Tag type. The device serializer parks T's name in .device_print_strings, so nothing is carried here.
template <typename T>
struct type_name
{
};

} // namespace llk::san

#define SAN_STRING(literal) (llk::san::string {literal})

#endif // !SAN_DEPS_CARRIERS_FROM_METAL
