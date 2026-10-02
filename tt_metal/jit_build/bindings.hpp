// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <set>

namespace tt::tt_metal {

// Represent a static object instantiation of a given binding entry.
//
// Models:
// constexpr BindingType<template_args...> <name>(<args...>);
//
// Note that this allows "novel" binding entries:
// e.g. constexpr auto <name>(std::make_tuple(....));. // args = {"std::make_tuple(....)"}
struct BindingEntry {
    std::string name;
    std::vector<std::string> args;
    std::vector<std::string> template_args;
};

// Config for generating programmatic binding token getter for the binding.
//
// Programmatic binding token getter is a token getter structured like this:
// auto get_token_if_present() {
//     if constexpr (name == "entry_name") {
//         return &entry_name;
//     }
//     ...
//     else {
//         return nullptr;
//     }
// }
//
// This is emitted regardless of there is any entries in the binding.
struct ProgrammaticBindingTokenGetterConfig {
    // The pointed-to type to return when there's no entries within the binding.
    // If this is absent, it will use the binding type associated with the binding instead.
    //
    // auto get_token_if_present() { // No binding entries.
    //   using null_token_ptr_t = const <null_binding_type>*;
    //   return null_token_ptr_t{nullptr};
    // }
    std::optional<std::string> null_binding_type;
};

// Represent a class of Binding (e.g. Scratchpad binding)
struct Binding {
    // Object invariant:
    // - If is_binding_type_templated is false,
    //   then for all entries, BindingEntry::template_args must be empty.

    std::string name, emission_namespace, binding_type;
    bool is_binding_type_templated = false;
    std::set<std::string> includes;
    std::vector<BindingEntry> entries;

    // Generates programmatic binding token getter for the binding.
    std::optional<ProgrammaticBindingTokenGetterConfig> programmatic_getter_config;
};

}  // namespace tt::tt_metal
