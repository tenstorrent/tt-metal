#pragma once

#include "tensor/tensor.hpp"
#include "tensor/span.hpp"

namespace tt {

namespace tt_metal {

namespace host_buffer {

template<typename T>
HostBuffer create(std::size_t size) {
    auto host_buffer = std::make_shared<HostBufferContainer>(sizeof(T) * size, 0);
    return host_buffer;
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
span_t<T> view_as(HostBuffer& host_buffer) {
    auto address = reinterpret_cast<T*>(host_buffer->data());
    auto size = host_buffer->size() / sizeof(T);
    return span_t(address, size);
}

template<typename T>
const span_t<T> view_as(const HostBuffer& host_buffer) {
    auto address = reinterpret_cast<T*>(host_buffer->data());
    auto size = host_buffer->size() / sizeof(T);
    return span_t(address, size);
}

template<typename T>
const span_t<T> view_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    auto buffer = tensor.host_storage().value().buffer;
    return host_buffer::view_as<T>(buffer);
}

template<typename T>
const span_t<T> view_as(Tensor&& tensor) = delete;

}  // namespace host_buffer

}  // namespace tt_metal

}  // namespace tt
