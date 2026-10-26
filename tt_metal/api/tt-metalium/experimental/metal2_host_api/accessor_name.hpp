// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <compare>
#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace tt::tt_metal::experimental {

// ============================================================================
//  AccessorName
// ============================================================================
//
// The name kernel source code uses to reach something: a resource binding's accessor
// (dfb::in, tensor::a, ...) or an argument's accessor (args::num_tiles).
//
// An AccessorName is implicitly constructible from a string literal, std::string or
// std::string_view, so a name can be written as a plain string wherever one is expected:
//   KernelSpec{.runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}}}
//   run_args.runtime_arg_values["num_tiles"][node] = n;
//
// It compares equal to a string with the same characters (name == "num_tiles").
// To read the characters back, use view() or str(). (There is deliberately no implicit conversion
// back to a string: it would make comparisons such as name == str ambiguous.)
//
// CONSTRAINTS (validated with the ProgramSpec, not on construction):
//   - A name must be a valid C++ identifier. (It is used verbatim in generated kernel code.)
//
// ============================================================================

class AccessorName {
public:
    AccessorName() = default;

    // Implicit, so that strings can be used wherever a name is expected.
    AccessorName(const char* name) : name_(name) {}
    AccessorName(std::string name) : name_(std::move(name)) {}
    AccessorName(std::string_view name) : name_(name) {}

    std::string_view view() const noexcept { return name_; }
    const std::string& str() const noexcept { return name_; }

    friend bool operator==(const AccessorName&, const AccessorName&) = default;
    friend auto operator<=>(const AccessorName&, const AccessorName&) = default;

    friend std::string_view format_as(const AccessorName& name) { return name.view(); }
    friend std::ostream& operator<<(std::ostream& os, const AccessorName& name) { return os << name.view(); }

private:
    // MAINTAINER: Keep the representation private. It may change later (e.g. to carry a scope, or
    // to inline storage, in which case str() would have to return by value); view() and str()
    // must keep meaning the name as written.
    std::string name_;
};

// The name of a kernel argument (args::<name>). Same type; reads better at argument sites.
using ArgName = AccessorName;

}  // namespace tt::tt_metal::experimental

template <>
struct std::hash<tt::tt_metal::experimental::AccessorName> {
    std::size_t operator()(const tt::tt_metal::experimental::AccessorName& name) const noexcept {
        return std::hash<std::string_view>{}(name.view());
    }
};
