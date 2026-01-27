// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This file is a copy of TTNN's ttnn/api/ttnn/tensor/types.hpp
// at commit 9f3856801448f589170defe41b23c8b9b43e33a2, with modifications to
// use experimental tensor types.

#pragma once

#include <iostream>  // For std::ostream
#include <utility>   // For std::move

// For ShardSpec
#include <tt-metalium/buffer.hpp>
// For Shape, CoreRangeSet, etc.
#include <tt-metalium/experimental/tensor/spec/shape/shape.hpp>

namespace tt::tt_metal /*::tensor*/ {

struct NdShardSpec {
    Shape shard_shape;
    CoreRangeSet grid;
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D;

    NdShardSpec with_shard_shape(Shape new_shard_shape) const {
        return NdShardSpec{std::move(new_shard_shape), grid, orientation, shard_distribution_strategy};
    }

    bool operator==(const NdShardSpec& other) const = default;
    bool operator!=(const NdShardSpec& other) const = default;
};

std::ostream& operator<<(std::ostream& os, const NdShardSpec& spec);

}  // namespace tt::tt_metal
