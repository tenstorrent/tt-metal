// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ttnn/tensor/tensor.hpp>
#include <variant>

#include "autograd/tensor.hpp"
#include "msgpack_file.hpp"

namespace ttml::serialization {
using NamedParameters = std::unordered_map<std::string, ttml::autograd::TensorPtr>;
using SerializableType = std::variant<ValueType, ttnn::Tensor, ttml::autograd::TensorPtr, NamedParameters>;
using StateDict = std::unordered_map<std::string, SerializableType>;

/*/
TODO: add template check in this functions that T belongs to the ValueType.
*/
template <class T>
const T& get_value_type(const StateDict& dict, const std::string& key) {
    const auto& val_type = std::get<ValueType>(dict.at(key));
    return std::get<T>(val_type);
}

}  // namespace ttml::serialization
