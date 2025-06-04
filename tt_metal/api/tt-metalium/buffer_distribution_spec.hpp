// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

    const std::vector<std::vector<std::optional<uint32_t>>>& get_page_mapping() const { return page_mapping_; }

private:
    void compute_page_mapping();

    tt::tt_metal::Shape tensor_shape_in_pages_;
    tt::tt_metal::Shape shard_shape_in_pages_;
    ShardOrientation shard_orientation_;

    std::vector<std::vector<std::optional<uint32_t>>> page_mapping_;
    std::vector<CoreCoord> cores_;
};

}  // namespace tt::tt_metal
