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
        Shape tensor_shape,
        Shape shard_shape,
        Shape2D page_shape,
        CoreRangeSet core_range_set,
        ShardOrientation shard_orientation,
        ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D);

    BufferDistributionSpec(
        Shape tensor_shape_in_pages,
        Shape shard_shape_in_pages,
        CoreRangeSet core_range_set,
        ShardOrientation shard_orientation,
        ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D);

    const Shape& tensor_shape_in_pages() const { return tensor_shape_in_pages_; }
    const Shape& shard_shape_in_pages() const { return shard_shape_in_pages_; }

    size_t num_shards() const;
    size_t max_num_shards_per_core() const;
    size_t max_num_dev_pages_per_core() const;
    size_t num_cores() const { return cores_.size(); }
    size_t num_cores_with_data() const;
    const std::vector<CoreCoord>& cores() const { return cores_; }
    const std::vector<CoreCoord>& cores_with_data() const { return cores_with_data_; }

    size_t num_shards_per_core(size_t core_idx) const;
    size_t num_dev_pages_per_core(size_t core_idx) const;

    struct CoreGroups {
        CoreRangeSet cores_with_data;
        CoreRangeSet cores_in_group_1;
        CoreRangeSet cores_in_group_2;
        size_t num_shards_per_core_in_group_1 = 0;
        size_t num_shards_per_core_in_group_2 = 0;
    };
    const CoreGroups& core_groups() const { return core_groups_; }
    // CoreGroups represented as a tuple compatible with split_work_to_cores function.
    // Returns: number of cores with data, cores with data, cores in group 1, cores in group 2,
    // number of shards per core in group 1, number of shards per core in group 2.
    std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> core_groups_tuple() const;

    UncompressedBufferPageMapping compute_page_mapping() const {
        return detail::compute_page_mapping(tensor_shape_in_pages_, shard_shape_in_pages_, cores_);
    }

private:
    static std::vector<CoreCoord> compute_core_list(
        const Shape& tensor_shape_in_pages,
        const Shape& shard_shape_in_pages,
        const CoreRangeSet& core_range_set,
        ShardOrientation shard_orientation,
        ShardDistributionStrategy shard_distribution_strategy);
    void init_precomputed_data();

    Shape tensor_shape_in_pages_;
    Shape shard_shape_in_pages_;
    ShardOrientation shard_orientation_ = ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> cores_;

    // Precomputed data
    CoreGroups core_groups_;
    std::vector<CoreCoord> cores_with_data_;
    size_t shard_volume_ = 0;
    size_t num_shards_ = 0;
};

}  // namespace tt::tt_metal
