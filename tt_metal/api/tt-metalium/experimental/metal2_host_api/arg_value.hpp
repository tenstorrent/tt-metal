// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <type_traits>

#include <fmt/base.h>

namespace tt::tt_metal::experimental {

// ============================================================================
//  ArgValue
// ============================================================================
//
// The value of a kernel argument.
//
// Kernel arguments are currently all uint32_t on the device, but other argument
// types will be supported in the future.
//
// NOTES:
//  - An ArgValue is implicitly constructible from uint32_t or any value that converts
//    implicitly to uint32_t, with exactly that conversion's result (-1 becomes 0xFFFFFFFF).
//  - To read an ArgValue back as a uint32_t, use to_uint32(). ArgValue deliberately has
//    no implicit conversion to uint32_t: once other argument types exist, it would
//    silently convert them.
//
// ============================================================================

class ArgValue {
public:
    ArgValue() = default;

    // Implicit construction from uint32_t (or any implicitly convertible type)
    // (Taken by const reference and converted implicitly, so it accepts exactly what assigning
    //  to a uint32_t accepts, with the same result: non-copyable sources like std::atomic and
    //  bit-fields work, and an explicit conversion operator is never chosen.)
    template <typename T>
        requires std::is_convertible_v<const T&, std::uint32_t>
    ArgValue(const T& value) : u32_(implicitly_converted(value)) {}

    // Get the arg value as a uint32_t (static_cast<uint32_t>(original value))
    std::uint32_t to_uint32() const { return u32_; }

    // Equality comparison (there is deliberately no ordering)
    friend bool operator==(const ArgValue&, const ArgValue&) = default;

    friend std::ostream& operator<<(std::ostream& os, const ArgValue& value) { return os << value.to_uint32(); }

private:
    // MAINTAINER: Keep the representation private. Typed (and possibly 64-bit) arguments will
    // change what is stored, not how values are written or read:
    //  - Every construction that compiles today must keep compiling and, for a uint32_t argument,
    //    keep producing the same argument word. So keep the single constructor template; an added
    //    overload (e.g. for float) would silently change what existing calls store.
    //  - to_uint32() is a conversion, not a view of the representation. It must keep meaning
    //    static_cast<uint32_t>(original value), even where that isn't the word sent to the device.
    std::uint32_t u32_ = 0;

    // Passing a value as this parameter applies only implicit conversions, as assignment does.
    // (Initializing u32_ directly, or a static_cast, could choose an explicit conversion operator.)
    static std::uint32_t implicitly_converted(std::uint32_t value) { return value; }
};

}  // namespace tt::tt_metal::experimental

template <>
struct std::hash<tt::tt_metal::experimental::ArgValue> {
    std::size_t operator()(const tt::tt_metal::experimental::ArgValue& value) const noexcept {
        return std::hash<std::uint32_t>{}(value.to_uint32());
    }
};

template <>
struct fmt::formatter<tt::tt_metal::experimental::ArgValue> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::experimental::ArgValue& value, format_context& ctx) const
        -> format_context::iterator {
        return fmt::format_to(ctx.out(), "{}", value.to_uint32());
    }
};
