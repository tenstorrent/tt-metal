// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tensor/borrowed_buffer.hpp"

#include <vector>

namespace tt {

namespace tt_metal {

namespace borrowed_buffer {

template<typename T>
void validate_datatype(const Tensor& tensor) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT32);
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
Buffer<T> get_as(BorrowedBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template<typename T>
const Buffer<T> get_as(const BorrowedBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template<typename T>
Buffer<T> get_as(Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [](auto&& storage) -> Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have BorrowedStorage");
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
            if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have BorrowedStorage");
            }
        },
        tensor.get_storage());
}

}  // namespace borrowed_buffer

}  // namespace tt_metal

}  // namespace tt
