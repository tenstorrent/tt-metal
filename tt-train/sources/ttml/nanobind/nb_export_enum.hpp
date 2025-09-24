
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include <enchantum/enchantum.hpp>

namespace nanobind {
template <typename E, typename... Extra>
inline nanobind::enum_<E> export_enum(const nanobind::handle& scope, const std::string& name = "", Extra&&... extra) {
    nanobind::enum_<E> enum_type(
        scope, name.empty() ? enchantum::type_name<E>.data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto [value, name] : enchantum::entries_generator<E>) {
        enum_type.value(name.data(), value);
    }

    return enum_type;
}
}  // namespace nanobind
