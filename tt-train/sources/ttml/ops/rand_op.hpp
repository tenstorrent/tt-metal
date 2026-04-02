// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr rand(
    const ttnn::Shape& shape,
    float a = 0.0f,
    float b = 1.0f,
    std::optional<uint32_t> seed = std::nullopt,
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::BFLOAT16,
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE);

}  // namespace ttml::ops
