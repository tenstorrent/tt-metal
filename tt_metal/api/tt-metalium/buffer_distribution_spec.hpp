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

namespace detail {
UncompressedBufferPageMapping compute_page_mapping(
    const Shape& tensor_shape, const Shape& shard_shape, const std::vector<CoreCoord>& cores);
}

class BufferDistributionSpec {
public:
    static BufferDistributionSpec from_shard_spec(
        tt::tt_metal::Shape tensor_shape,
        tt::tt_metal::Shape shard_shape,
        tt::tt_metal::Shape2D page_shape,
        CoreRangeSet core_range_set,
        ShardOrientation shard_orientation,
        ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D);

    BufferDistributionSpec(
        tt::tt_metal::Shape tensor_shape_in_pages,
        tt::tt_metal::Shape shard_shape_in_pages,
        CoreRangeSet core_range_set,
        ShardOrientation shard_orientation,
        ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D);

    tt::tt_metal::Shape get_tensor_shape_in_pages() const { return tensor_shape_in_pages_; }
    tt::tt_metal::Shape get_shard_shape_in_pages() const { return shard_shape_in_pages_; }

    size_t num_shards() const;
    size_t max_num_shards_per_core() const;
    size_t max_num_dev_pages_per_core() const;
    size_t num_cores() const { return cores_.size(); }
    size_t num_cores_with_data() const;
    const std::vector<CoreCoord>& get_cores() const { return cores_; }
    std::vector<CoreCoord> get_cores_with_data() const;

    size_t num_shards_per_core(size_t core_idx) const;
    size_t num_dev_pages_per_core(size_t core_idx) const;

    struct CoreGroup {
        size_t num_shards = 0;
        std::vector<CoreCoord> cores;
    };
    std::pair<CoreGroup, CoreGroup> get_core_groups_by_num_shards() const;

    UncompressedBufferPageMapping compute_page_mapping() const {
        return detail::compute_page_mapping(tensor_shape_in_pages_, shard_shape_in_pages_, cores_);
    }

private:
    std::vector<CoreCoord> compute_core_list(
        const CoreRangeSet& core_range_set, ShardDistributionStrategy shard_distribution_strategy);

    tt::tt_metal::Shape tensor_shape_in_pages_;
    tt::tt_metal::Shape shard_shape_in_pages_;
    ShardOrientation shard_orientation_ = ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> cores_;
};

}  // namespace tt::tt_metal
