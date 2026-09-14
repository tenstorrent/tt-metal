// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <initializer_list>
#include <string_view>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Checks that `tensor` is an allocated DEVICE tensor with the given layout, memory layout and one of the
// allowed dtypes. `op` and `name` only feed the error message. Meant for validate_on_program_cache_miss;
// call it before touching shapes so a host or unallocated tensor fails here and not in the kernel.
inline void check_device_tensor(
    const ttnn::Tensor& tensor,
    std::string_view op,
    std::string_view name,
    std::initializer_list<tt::tt_metal::DataType> allowed_dtypes = {tt::tt_metal::DataType::BFLOAT16},
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE,
    tt::tt_metal::TensorMemoryLayout memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "{}: {} must be on Device. Storage type: {}",
        op,
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "{}: {} buffer is null", op, name);
    TT_FATAL(
        tensor.layout() == layout,
        "{}: {} requires {} layout. Got: {}",
        op,
        name,
        enchantum::to_string(layout),
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        std::find(allowed_dtypes.begin(), allowed_dtypes.end(), tensor.dtype()) != allowed_dtypes.end(),
        "{}: {} has unsupported dtype {}",
        op,
        name,
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == memory_layout,
        "{}: {} requires {} memory layout. Got: {}",
        op,
        name,
        enchantum::to_string(memory_layout),
        enchantum::to_string(tensor.memory_config().memory_layout()));
}

// Checks that `tensor` lives on the same device as `reference`. Kernels are handed raw buffer addresses,
// so a buffer on another device would be read as if it were local. Call after check_device_tensor on both.
inline void check_same_device(
    const ttnn::Tensor& tensor,
    const ttnn::Tensor& reference,
    std::string_view op,
    std::string_view name,
    std::string_view reference_name) {
    TT_FATAL(
        tensor.device() == reference.device(), "{}: {} must be on the same device as {}", op, name, reference_name);
}

}  // namespace ttml::metal
