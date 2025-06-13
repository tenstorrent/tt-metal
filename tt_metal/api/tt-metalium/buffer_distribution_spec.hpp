// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>

namespace tt::tt_metal {

class BufferDistributionSpec {
public:
    static BufferDistributionSpec from_shard_spec(
        tt::tt_metal::Shape tensor_shape,
        tt::tt_metal::Shape shard_shape,
        tt::tt_metal::Shape2D page_shape,
        CoreRangeSet core_range_set,
        ShardOrientation shard_orientation);

    BufferDistributionSpec(
        tt::tt_metal::Shape tensor_shape_in_pages,
        tt::tt_metal::Shape shard_shape_in_pages,
        CoreRangeSet core_range_set,
        ShardOrientation shard_orientation);

    tt::tt_metal::Shape get_tensor_shape_in_pages() const { return tensor_shape_in_pages_; }
    tt::tt_metal::Shape get_shard_shape_in_pages() const { return shard_shape_in_pages_; }

    size_t num_shards() const;
    size_t num_shards_per_core() const;
    size_t num_dev_pages_per_core() const;
    size_t num_cores() const { return cores_.size(); }
    const std::vector<CoreCoord>& get_cores() const { return cores_; }

    BufferPageMapping compute_page_mapping() const;

private:
    tt::tt_metal::Shape tensor_shape_in_pages_;
    tt::tt_metal::Shape shard_shape_in_pages_;
    ShardOrientation shard_orientation_ = ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> cores_;
};

}  // namespace tt::tt_metal
