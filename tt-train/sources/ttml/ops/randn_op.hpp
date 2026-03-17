// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <optional>

namespace ttml::ops {

tt::tt_metal::Tensor randn(
    const ttnn::Shape& shape,
    MeshDevice& device,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const MemoryConfig& memory_config = ttnn::types::DRAM_MEMORY_CONFIG,
    float mean = 0.0f,
    float stddev = 1.0f,
    std::optional<uint32_t> seed = std::nullopt);

}  // namespace ttml::ops
