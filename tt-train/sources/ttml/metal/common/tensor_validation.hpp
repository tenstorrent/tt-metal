// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// What a device op requires of one of its tensors. The defaults are the tt-train norm: bf16, TILE with 32x32
// tiles, interleaved, any buffer type. Override single fields with designated initializers, e.g.
// `{.dtypes = {DataType::UINT32}, .layout = Layout::ROW_MAJOR}` for an index tensor,
// `{.buffer_type = BufferType::DRAM}` for a kernel that addresses DRAM directly, or
// `{.memory_layout = std::nullopt}` for an op whose kernels go through TensorAccessor and take any memory layout.
struct DeviceTensorRequirements {
    std::vector<tt::tt_metal::DataType> dtypes = {tt::tt_metal::DataType::BFLOAT16};
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
    std::optional<tt::tt_metal::TensorMemoryLayout> memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    std::optional<tt::tt_metal::BufferType> buffer_type = std::nullopt;
    // {height, width} of a TILE layout tensor's tiles. Kernels that index tiles and faces assume 32x32.
    std::optional<std::array<uint32_t, 2>> tile_shape =
        std::array<uint32_t, 2>{tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
};

// Checks that `tensor` is an allocated DEVICE tensor meeting `req`. `op` and `name` only feed the error
// message. Meant for validate_on_program_cache_miss; call it before touching shapes so a host or
// unallocated tensor fails here and not in the kernel.
inline void check_device_tensor(
    const ttnn::Tensor& tensor, std::string_view op, std::string_view name, const DeviceTensorRequirements& req = {}) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "{}: {} must be on Device. Storage type: {}",
        op,
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "{}: {} buffer is null", op, name);
    if (req.buffer_type.has_value()) {
        TT_FATAL(
            tensor.buffer()->buffer_type() == *req.buffer_type,
            "{}: {} must be in {}. Got: {}",
            op,
            name,
            enchantum::to_string(*req.buffer_type),
            enchantum::to_string(tensor.buffer()->buffer_type()));
    }
    TT_FATAL(
        tensor.layout() == req.layout,
        "{}: {} requires {} layout. Got: {}",
        op,
        name,
        enchantum::to_string(req.layout),
        enchantum::to_string(tensor.layout()));
    if (req.layout == tt::tt_metal::Layout::TILE && req.tile_shape.has_value()) {
        const auto& tile = tensor.tensor_spec().tile();
        const auto& [height, width] = *req.tile_shape;
        TT_FATAL(
            tile.get_height() == height && tile.get_width() == width,
            "{}: {} requires {}x{} tiles. Got: {}x{}",
            op,
            name,
            height,
            width,
            tile.get_height(),
            tile.get_width());
    }
    TT_FATAL(
        std::find(req.dtypes.begin(), req.dtypes.end(), tensor.dtype()) != req.dtypes.end(),
        "{}: {} has unsupported dtype {}",
        op,
        name,
        enchantum::to_string(tensor.dtype()));
    if (req.memory_layout.has_value()) {
        TT_FATAL(
            tensor.memory_config().memory_layout() == *req.memory_layout,
            "{}: {} requires {} memory layout. Got: {}",
            op,
            name,
            enchantum::to_string(*req.memory_layout),
            enchantum::to_string(tensor.memory_config().memory_layout()));
    }
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
