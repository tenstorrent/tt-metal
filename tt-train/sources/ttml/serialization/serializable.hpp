// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>
#include <variant>

#include "autograd/tensor.hpp"
#include "msgpack_file.hpp"

namespace ttml::serialization {
using NamedParameters = std::unordered_map<std::string, ttml::autograd::TensorPtr>;
using SerializableType = std::variant<ValueType, ttnn::Tensor, ttml::autograd::TensorPtr, NamedParameters>;
using StateDict = std::unordered_map<std::string, SerializableType>;

template <typename T>
concept IsValueType = requires {
    { std::get<T>(std::declval<ValueType>()) };
};

template <IsValueType T>
const T& get_value_type(const StateDict& dict, const std::string& key) {
    const auto& val_type = std::get<ValueType>(dict.at(key));
    return std::get<T>(val_type);
}

}  // namespace ttml::serialization
