// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tensor/owned_buffer.hpp"

#include <vector>

namespace tt {

namespace tt_metal {

namespace owned_buffer {

template<typename T>
Buffer<T> create(std::vector<T>&& storage) {
    return Buffer<T>{std::make_shared<std::vector<T>>(std::forward<std::vector<T>>(storage))};
}

template<typename T>
Buffer<T> create(std::size_t size) {
    return create(std::vector<T>(size, 0));
}

template<typename T>
void validate_datatype(const Tensor& tensor) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT32 or tensor.get_dtype() == DataType::BFLOAT8_B or tensor.get_dtype() == DataType::BFLOAT4_B);
    } else if constexpr (std::is_same_v<T, float>) {
        TT_FATAL(tensor.get_dtype() == DataType::FLOAT32);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT16);
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported DataType");
    }
}

template<typename T>
Buffer<T> get_as(OwnedBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template<typename T>
const Buffer<T> get_as(const OwnedBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template<typename T>
Buffer<T> get_as(Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [](auto&& storage) -> Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have OwnedStorage");
            }
        },
        tensor.get_storage());
}

template<typename T>
const Buffer<T> get_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [](auto&& storage) -> Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have OwnedStorage");
            }
        },
        tensor.get_storage());
}

}  // namespace owned_buffer

}  // namespace tt_metal

}  // namespace tt
