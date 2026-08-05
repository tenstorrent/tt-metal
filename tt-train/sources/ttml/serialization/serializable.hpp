// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>
#include <variant>

#include "autograd/tensor.hpp"
#include "flatbuffer_file.hpp"

namespace ttml::serialization {
using NamedParameters = std::unordered_map<std::string, ttml::autograd::TensorPtr>;
using SerializableType = std::variant<ValueType, ttnn::Tensor, ttml::autograd::TensorPtr, NamedParameters>;
using StateDict = std::unordered_map<std::string, SerializableType>;

template <typename T>
concept IsValueType = requires {
    { std::get<T>(std::declval<ValueType>()) };
};

// Compatibility groups for integer widening in get_value_type.
//
// We allow widening between integer alternatives of the same "kind" but
// not across kinds. Three disjoint groups:
//   * {bool}                          -- only matches bool
//   * {char}                          -- only matches char
//   * {int, uint32_t, size_t, ...}    -- any "regular" integer matches any other
//
template <typename T>
constexpr bool is_regular_integer_v = std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, char>;

template <typename T, typename Held>
constexpr bool integer_compatible_v =
    (std::is_same_v<T, bool> && std::is_same_v<Held, bool>) ||
    (std::is_same_v<T, char> && std::is_same_v<Held, char>) || (is_regular_integer_v<T> && is_regular_integer_v<Held>);

template <IsValueType T>
T get_value_type(const StateDict& dict, const std::string& key) {
    const auto& val_type = std::get<ValueType>(dict.at(key));
    if constexpr (std::is_integral_v<T>) {
        return std::visit(
            [](const auto& held) -> T {
                using Held = std::decay_t<decltype(held)>;
                if constexpr (integer_compatible_v<T, Held>) {
                    // Reject signed-negative -> unsigned conversions, which would
                    // wrap to a huge positive value via two's-complement.
                    if constexpr (std::is_unsigned_v<T> && std::is_signed_v<Held>) {
                        if (held < 0) {
                            throw std::bad_variant_access{};
                        }
                    }
                    return static_cast<T>(held);
                } else {
                    throw std::bad_variant_access{};
                }
            },
            val_type);
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::visit(
            [](const auto& held) -> T {
                using Held = std::decay_t<decltype(held)>;
                if constexpr (std::is_floating_point_v<Held>) {
                    return static_cast<T>(held);
                } else {
                    throw std::bad_variant_access{};
                }
            },
            val_type);
    } else {
        return std::get<T>(val_type);
    }
}

}  // namespace ttml::serialization
