// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>
#include <type_traits>

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal::host_buffer {

// Tensor here can be Runtime host/device tensor or ttnn::Tensor,
// this is rather ugly and insane.
// TODO: we should yoink this out entirely or provide a ttnn overload for ttnn::Tensor.
template <typename Tensor, typename T>
void validate_datatype(const Tensor& tensor) {
    using BaseType = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<BaseType, uint32_t>) {
        TT_FATAL(
            tensor.dtype() == DataType::UINT32 or tensor.dtype() == DataType::BFLOAT8_B or
                tensor.dtype() == DataType::BFLOAT4_B,
            "Incorrect data type {}",
            tensor.dtype());
    } else if constexpr (std::is_same_v<BaseType, int32_t>) {
        TT_FATAL(tensor.dtype() == DataType::INT32, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<BaseType, float>) {
        TT_FATAL(tensor.dtype() == DataType::FLOAT32, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<BaseType, bfloat16>) {
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<BaseType, uint16_t>) {
        TT_FATAL(tensor.dtype() == DataType::UINT16, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<BaseType, uint8_t>) {
        TT_FATAL(tensor.dtype() == DataType::UINT8, "Incorrect data type {}", tensor.dtype());
    } else {
        static_assert(tt::stl::concepts::always_false_v<BaseType>, "Unsupported DataType");
    }
}

template <typename T>
tt::stl::Span<const T> get_as(const HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
tt::stl::Span<T> get_as(HostBuffer& buffer) {
    return buffer.view_as<T>();
}

}  // namespace tt::tt_metal::host_buffer
