#pragma once

#include "tensor/tensor.hpp"
#include "tensor/owned_buffer.hpp"

#include <vector>

namespace tt {

namespace tt_metal {

namespace owned_buffer {

template<typename T>
Buffer<T> create(std::vector<T>&& storage) {
    return Buffer<T>{std::make_shared<std::vector<T>>(std::move(storage))};
}

template<typename T>
Buffer<T> create(std::size_t size) {
    return create(std::vector<T>(size, 0));
}

template<typename T>
void validate_datatype(const Tensor& tensor) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        TT_ASSERT(tensor.dtype() == DataType::UINT32);
    } else if constexpr (std::is_same_v<T, float>) {
        TT_ASSERT(tensor.dtype() == DataType::FLOAT32 or tensor.dtype() == DataType::BFLOAT8_B);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        TT_ASSERT(tensor.dtype() == DataType::BFLOAT16);
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
    auto& buffer = tensor.owned_storage().value().buffer;
    return get_as<T>(buffer);
}

template<typename T>
const Buffer<T> get_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    const auto& buffer = tensor.owned_storage().value().buffer;
    return get_as<T>(buffer);
}

}  // namespace owned_buffer

}  // namespace tt_metal

}  // namespace tt
