// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/shape.hpp>

#include <vector>

namespace tt::tt_metal::detail {

UncompressedBufferPageMapping compute_page_mapping(
    const Shape& tensor_shape,
    const Shape& shard_shape,
    const std::vector<CoreCoord>& cores,
    ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D);

}  // namespace tt::tt_metal::detail
