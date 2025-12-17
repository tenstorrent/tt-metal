// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include <enchantum/enchantum.hpp>

#include "nb_fwd.hpp"

namespace ttml::nanobind::util {

template <typename E, typename... Extra>
inline nb::enum_<E> export_enum(const nb::handle& scope, const std::string& name = "", Extra&&... extra) {
    nb::enum_<E> enum_type(
        scope, name.empty() ? enchantum::type_name<E>.data() : name.c_str(), std::forward<Extra>(extra)...);
    for (const auto [value, name] : enchantum::entries_generator<E>) {
        enum_type.value(name.data(), value);
    }

    return enum_type;
}

}  // namespace ttml::nanobind::util
