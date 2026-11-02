// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace tt::tt_metal {

// A kernel-visible name (an argument name, a binding's accessor_name, or a tensor binding
// sequence_name) is either unscoped ("name") or scoped ("scope.name"). Kernel code reaches an
// unscoped name in its kind's namespace (args::name, dfb::name, ...), and a scoped name as a
// static member of the generated struct for its scope (scope::name).
// See "NAMES AND SCOPES" in kernel_spec.hpp.
inline constexpr char SCOPE_SEPARATOR = '.';

struct ScopedName {
    std::string_view scope;  // empty for an unscoped name
    std::string_view name;   // the name within the scope
};

// Splits a name at its scope separator. Validation allows at most one separator, with a
// C++ identifier on each side (see IsValidKernelVisibleName).
inline ScopedName split_scoped_name(std::string_view qualified_name) {
    const std::size_t separator = qualified_name.find(SCOPE_SEPARATOR);
    if (separator == std::string_view::npos) {
        return {.scope = {}, .name = qualified_name};
    }
    return {.scope = qualified_name.substr(0, separator), .name = qualified_name.substr(separator + 1)};
}

inline bool is_scoped_name(std::string_view qualified_name) {
    return qualified_name.find(SCOPE_SEPARATOR) != std::string_view::npos;
}

// The namespaces the generated kernel headers declare (args, ... sem_internal) or refer to (the rest).
// A scope's struct is declared at global namespace scope, so a scope can't take one of these names.
// (A clash with any other global name in device code is a kernel compile error.)
inline constexpr std::array<std::string_view, 12> RESERVED_SCOPE_NAMES = {
    "args",
    "dfb",
    "pipe",
    "sem",
    "scratch",
    "tensor",
    "sem_internal",
    "binding_details",
    "experimental",
    "internal",
    "std",
    "tensor_accessor"};

// Every scope's struct also gets the by-name binding lookups: the scoped counterparts of
// dfb::get_token_if_present, tensor::get_token_if_present and scratch::get_token_if_present.
// They need one name per binding kind, since a scope holds every kind together.
inline constexpr std::string_view SCOPED_DFB_LOOKUP_NAME = "get_dfb_token_if_present";
inline constexpr std::string_view SCOPED_TENSOR_LOOKUP_NAME = "get_tensor_token_if_present";
inline constexpr std::string_view SCOPED_SCRATCH_LOOKUP_NAME = "get_scratch_token_if_present";

}  // namespace tt::tt_metal
