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

class BufferDistributionSpecImpl;
class BufferDistributionSpec {
public:
    static BufferDistributionSpec from_shard_spec(
        Shape tensor_shape,
        Shape shard_shape,
        Shape2D page_shape,
        const CoreRangeSet& core_range_set,
        ShardOrientation shard_orientation,
        ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D);

    BufferDistributionSpec(
        const Shape& tensor_shape_in_pages,
        const Shape& shard_shape_in_pages,
        const CoreRangeSet& core_range_set,
        ShardOrientation shard_orientation,
        ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D);

    BufferDistributionSpec(
        const Shape& tensor_shape_in_pages, const Shape& shard_shape_in_pages, std::vector<CoreCoord> cores);

    ~BufferDistributionSpec();
    BufferDistributionSpec(const BufferDistributionSpec& other);
    BufferDistributionSpec& operator=(const BufferDistributionSpec& other);
    BufferDistributionSpec(BufferDistributionSpec&& other) noexcept;
    BufferDistributionSpec& operator=(BufferDistributionSpec&& other) noexcept;

    const Shape& tensor_shape_in_pages() const;
    const Shape& shard_shape_in_pages() const;

    size_t num_shards() const;
    size_t max_num_dev_pages_per_core() const;
    size_t num_cores_with_data() const;
    const std::vector<CoreCoord>& cores_with_data() const;

    // CoreGroups represented as a tuple compatible with split_work_to_cores function.
    // Returns: number of cores with data, cores with data, cores in group 1, cores in group 2,
    // number of shards per core in group 1, number of shards per core in group 2.
    std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> core_groups_tuple() const;

    BufferDistributionSpecImpl* impl();
    const BufferDistributionSpecImpl* impl() const;

private:
    std::unique_ptr<BufferDistributionSpecImpl> impl_;
};

}  // namespace tt::tt_metal
