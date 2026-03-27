// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

namespace detail {
// ttnn::uniform treats seed==0 as "no seed" (uses random entropy).
// Shift by +1 to keep all seeds deterministic. UINT32_MAX maps to 1.
inline uint32_t avoid_zero_seed(uint32_t seed) {
    return seed == std::numeric_limits<uint32_t>::max() ? 1 : seed + 1;
}
}  // namespace detail

autograd::TensorPtr rand(
    const ttnn::Shape& shape,
    float a = 0.0f,
    float b = 1.0f,
    std::optional<uint32_t> seed = std::nullopt,
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::BFLOAT16,
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE);

void rand_(
    const autograd::TensorPtr& tensor, float a = 0.0f, float b = 1.0f, std::optional<uint32_t> seed = std::nullopt);

}  // namespace ttml::ops
