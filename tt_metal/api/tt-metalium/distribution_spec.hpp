// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <variant>

#include <tt-metalium/shape.hpp>

namespace tt::tt_metal {

class DistributionSpec {
public:
    static DistributionSpec from_shard_shape(
        const tt::tt_metal::Shape& tensor_shape, const tt::tt_metal::Shape& shard_shape, size_t num_targets);

    tt::tt_metal::Shape get_shard_shape() const { return shard_shape_; }
    size_t get_num_targets() const { return num_targets_; }

    // ChunkMapping defined as:
    // - src: Element offset from tensor shape
    // - dst: Element offset on target
    // - size: Number of contiguous elements from src/dst
    struct ChunkMapping {
        size_t src = 0;
        size_t dst = 0;
        size_t size = 0;

        bool operator==(const ChunkMapping& other) const {
            return src == other.src && dst == other.dst && size == other.size;
        }
    };

    // TargetData decribes all the data intended for one target in terms of ChunkMapping
    // When constructing the vector, it accounts for partial shards and multiple shards per core
    // When consuming TargetData, user can be oblivious to such details and just iterate through the mappings
    using TargetData = std::vector<ChunkMapping>;

    // When computing the metadata for all targets, we can either:
    // - MappingMode::NONCOALESCED:
    //  * Always return mapping element by element (ie. ChunkMapping.size will always be 1)
    //  * Useful for single device where we need to copy page by page if page is non-aligned
    // - MappingMode::COALESCED:
    //  * Coalesce as much contiguous data as possible
    //    ** Example: ChunkMapping{0, 0, 1}, ChunkMapping{1, 1, 1} is equivalent to ChunkMapping{0, 0, 2}
    //  * Useful for working with larger amounts of data together whenever possible
    enum MappingMode { NONCOALESCED = 0, COALESCED = 1 };

    // Metadata for all targets stores one TargetData per num_targets
    // The mapping to target is implied in the position of the TargetData
    // When consuming the metadata, it is user's responsibility to map TargetData to the correct device/core
    std::vector<TargetData> compute_metadata_for_targets(const MappingMode mapping_mode) const;

private:
    struct ShardSize {
        size_t size = 0;
    };

    struct ReplicateCount {
        size_t count = 0;
    };
    using DistributionType = std::variant<std::monostate, ShardSize, ReplicateCount>;

    DistributionSpec(
        const tt::tt_metal::Shape& tensor_shape,
        const tt::stl::SmallVector<DistributionType>& spec,
        size_t num_targets);

    tt::tt_metal::Shape tensor_shape_;
    tt::stl::SmallVector<DistributionType> spec_;
    tt::tt_metal::Shape shard_shape_;  // Determined based on spec_
    size_t num_targets_ = 0;
};

}  // namespace tt::tt_metal
