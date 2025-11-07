// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/tensor_impl.hpp>
#include <utility>

namespace tt::tt_metal::tensor_impl {

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(DataType dtype, Func&& func, Args&&... args) {
    switch (dtype) {
        case DataType::BFLOAT16:
            return (std::forward<Func>(func)).template operator()<bfloat16>(std::forward<Args>(args)...);
        case DataType::FLOAT32:
            return (std::forward<Func>(func)).template operator()<float>(std::forward<Args>(args)...);
        case DataType::INT32:
            return (std::forward<Func>(func)).template operator()<int32_t>(std::forward<Args>(args)...);
        case DataType::UINT32:
            return (std::forward<Func>(func)).template operator()<uint32_t>(std::forward<Args>(args)...);
        case DataType::UINT16:
            return (std::forward<Func>(func)).template operator()<uint16_t>(std::forward<Args>(args)...);
        case DataType::UINT8:
            return (std::forward<Func>(func)).template operator()<uint8_t>(std::forward<Args>(args)...);
        case DataType::BFLOAT8_B:
            return (std::forward<Func>(func)).template operator()<bfloat8_b>(std::forward<Args>(args)...);
        case DataType::BFLOAT4_B:
            return (std::forward<Func>(func)).template operator()<bfloat4_b>(std::forward<Args>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

// NOLINTBEGIN(bugprone-macro-parentheses)
#define AS_LAMBDA(func) []<typename T>(auto&&... args) { return func<T>(std::forward<decltype(args)>(args)...); }

#define WRAP_FUNCTION(func)                                                                                     \
    template <typename... Args>                                                                                 \
    auto func##_wrapper(Args&&... args) {                                                                       \
        return dispatch(                                                                                        \
            std::get<0>(std::forward_as_tuple(args...)).dtype(), AS_LAMBDA(func), std::forward<Args>(args)...); \
    }
// NOLINTEND(bugprone-macro-parentheses)

WRAP_FUNCTION(to_device)
WRAP_FUNCTION(to_host)
WRAP_FUNCTION(copy_to_device)
WRAP_FUNCTION(copy_to_host)
WRAP_FUNCTION(extract_shard)
WRAP_FUNCTION(to_layout)
WRAP_FUNCTION(pad)
WRAP_FUNCTION(unpad)
WRAP_FUNCTION(to_string)

#undef WRAP_FUNCTION
#undef AS_LAMBDA

Tensor to_dtype(const Tensor& input_tensor, DataType dtype);

}  // namespace tt::tt_metal::tensor_impl
