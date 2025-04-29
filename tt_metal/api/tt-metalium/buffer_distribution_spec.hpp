// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distribution_spec.hpp>
#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal {

class BufferDistributionSpec {
public:
    static BufferDistributionSpec from_shard_spec(
        const tt::tt_metal::Shape& tensor_shape,
        const tt::tt_metal::Shape& physical_shard_shape,
        const Shape2D& page_shape,
        const CoreRangeSet& corerangeset,
        const ShardOrientation shard_orientation);

    size_t num_dev_pages_per_core() const {
        return page_distribution_spec_.get_shard_shape().volume() *
               page_distribution_spec_.get_max_num_shards_per_target();
    }
    size_t num_cores() const { return cores_.size(); }
    std::vector<CoreCoord> get_cores() const { return cores_; }

private:
    BufferDistributionSpec(const DistributionSpec& page_distribution_spec, const std::vector<CoreCoord>& cores);

    DistributionSpec page_distribution_spec_;
    std::vector<CoreCoord> cores_;
};

}  // namespace tt::tt_metal
