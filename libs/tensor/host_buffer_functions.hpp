#pragma once

#include "tensor/tensor.hpp"
#include "tensor/host_buffer.hpp"

#include <memory>
#include <vector>

namespace tt {

namespace tt_metal {

namespace host_buffer {

template<typename T>
HostBufferForDataType<T> create(std::vector<T>&& storage) {
    return HostBufferForDataType<T>{std::make_shared<std::vector<T>>(std::move(storage))};
}

template<typename T>
HostBufferForDataType<T> create(std::size_t size) {
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
HostBufferForDataType<T> get_as(HostBuffer& host_buffer) {
    return std::get<HostBufferForDataType<T>>(host_buffer);
}

template<typename T>
const HostBufferForDataType<T> get_as(const HostBuffer& host_buffer) {
    return std::get<HostBufferForDataType<T>>(host_buffer);
}

template<typename T>
HostBufferForDataType<T> get_as(Tensor& tensor) {
    validate_datatype<T>(tensor);
    auto& buffer = tensor.host_storage().value().buffer;
    return host_buffer::get_as<T>(buffer);
}

template<typename T>
const HostBufferForDataType<T> get_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    const auto& buffer = tensor.host_storage().value().buffer;
    return host_buffer::get_as<T>(buffer);
}

}  // namespace host_buffer

}  // namespace tt_metal

}  // namespace tt
