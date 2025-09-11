// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace nanobind {
class module_;
}  // namespace nanobind

namespace nb = nanobind;
// TODO: put the follwing in its own file
#include <nanobind/nanobind.h>

#include <enchantum/enchantum.hpp>

namespace nanobind {
template <typename E, typename... Extra>
inline nb::enum_<E> export_enum(const nb::handle& scope, const std::string& name = "", Extra&&... extra) {
    nb::enum_<E> enum_type(
        scope, name.empty() ? enchantum::type_name<E>.data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto [value, name] : enchantum::entries_generator<E>) {
        enum_type.value(name.data(), value);
    }
    enum_type.export_values();

    return enum_type;
}
}  // namespace nanobind
