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
// An ArgValue is implicitly constructible from any value that converts implicitly to
// uint32_t, with exactly the result of that conversion (integers wrap, floating point
// values truncate, bools and unscoped enums convert):
//   KernelSpec{.compile_time_args = {{"Ht", Ht}}}
//   run_args.runtime_arg_values["num_tiles"][node] = n;
//   run_args.common_runtime_arg_values["sentinel"] = -1;  // 0xFFFFFFFF, as for a uint32_t
//
// Kernel arguments are currently all uint32_t on the device.
//
// An ArgValue is written, not read back: it has no conversion to an integer.
// ArgValues compare equal (==) when they hold the same value.
//
// ============================================================================

class ArgValue {
public:
    ArgValue() = default;

    // Implicit, so that values can be used wherever an ArgValue is expected.
    template <typename T>
        requires std::is_convertible_v<T, std::uint32_t>
    ArgValue(T value) : u32_(static_cast<std::uint32_t>(value)) {}

    friend bool operator==(const ArgValue&, const ArgValue&) = default;

    friend std::ostream& operator<<(std::ostream& os, const ArgValue& value) { return os << value.u32_; }

private:
    // MAINTAINER: Keep the representation private, and don't add a public way to read it as
    // a uint32_t. The stored value will change when typed (and possibly 64-bit) arguments
    // are supported; the constructor's overload set must stay as it is.
    friend struct ArgValueAccess;  // Implementation-only reader (defined in tt_metal/impl)
    friend struct std::hash<ArgValue>;
    friend struct fmt::formatter<ArgValue>;

    std::uint32_t u32_ = 0;
};

}  // namespace tt::tt_metal::experimental

template <>
struct std::hash<tt::tt_metal::experimental::ArgValue> {
    std::size_t operator()(const tt::tt_metal::experimental::ArgValue& value) const noexcept {
        return std::hash<std::uint32_t>{}(value.u32_);
    }
};

template <>
struct fmt::formatter<tt::tt_metal::experimental::ArgValue> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::experimental::ArgValue& value, format_context& ctx) const
        -> format_context::iterator {
        return fmt::format_to(ctx.out(), "{}", value.u32_);
    }
};
