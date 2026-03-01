// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <enchantum/enchantum.hpp>

namespace tt::tt_metal {

// Returns true if the tensor data is stored in row-major layout and the logical shape matches the physical shape.
// When true, no encoding/decoding is needed to convert between logical and physical representations.
bool logical_matches_physical(const TensorSpec& tensor_spec);

namespace host_buffer {

template <typename T>
void validate_datatype(const HostTensor& tensor) {
    using BaseType = std::remove_cvref_t<T>;
    auto dtype_str = enchantum::to_string(tensor.dtype());
    if constexpr (std::is_same_v<BaseType, uint32_t>) {
        TT_FATAL(
            tensor.dtype() == DataType::UINT32 or tensor.dtype() == DataType::BFLOAT8_B or
                tensor.dtype() == DataType::BFLOAT4_B,
            "Incorrect data type {}",
            dtype_str);
    } else if constexpr (std::is_same_v<BaseType, int32_t>) {
        TT_FATAL(tensor.dtype() == DataType::INT32, "Incorrect data type {}", dtype_str);
    } else if constexpr (std::is_same_v<BaseType, float>) {
        TT_FATAL(tensor.dtype() == DataType::FLOAT32, "Incorrect data type {}", dtype_str);
    } else if constexpr (std::is_same_v<BaseType, bfloat16>) {
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Incorrect data type {}", dtype_str);
    } else if constexpr (std::is_same_v<BaseType, uint16_t>) {
        TT_FATAL(tensor.dtype() == DataType::UINT16, "Incorrect data type {}", dtype_str);
    } else if constexpr (std::is_same_v<BaseType, uint8_t>) {
        TT_FATAL(tensor.dtype() == DataType::UINT8, "Incorrect data type {}", dtype_str);
    } else {
        static_assert(sizeof(BaseType) == 0, "Unsupported DataType");
    }
}

HostBuffer get_host_buffer(const HostTensor& tensor);

template <typename T>
tt::stl::Span<const T> get_as(const HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
tt::stl::Span<T> get_as(HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
tt::stl::Span<const T> get_as(const HostTensor& tensor) {
    validate_datatype<T>(tensor);
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
tt::stl::Span<T> get_as(HostTensor& tensor) {
    validate_datatype<T>(tensor);
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

}  // namespace host_buffer

}  // namespace tt::tt_metal
