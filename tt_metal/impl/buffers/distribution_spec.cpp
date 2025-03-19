// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distribution_spec.hpp"
#include "math.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

std::pair<tt::tt_metal::Shape, tt::tt_metal::Shape> maybe_convert_shapes_for_mapping_mode(
    const auto& original_tensor_shape, const auto& original_shard_shape, const auto mapping_mode) {
    switch (mapping_mode) {
        case DistributionSpec::MappingMode::COALESCED: {
            auto new_rank = original_tensor_shape.rank();

            if (original_tensor_shape[-1] != original_shard_shape[-1]) {
                // Return the original tensor_shape and shard_shape as a tuple
                return {original_tensor_shape, original_shard_shape};
            }

            size_t tensor_accum = 1;
            size_t shard_accum = 1;
            while (new_rank > 1 and original_tensor_shape[new_rank - 1] == original_shard_shape[new_rank - 1]) {
                tensor_accum *= original_tensor_shape[new_rank - 1];
                shard_accum *= original_shard_shape[new_rank - 1];
                new_rank--;
            }

            auto copy_up_to_new_rank_and_multiply_last_dim =
                [](const auto& original_shape, const size_t new_rank, const size_t accum) {
                    auto new_shape =
                        tt::stl::SmallVector<uint32_t>(original_shape.cbegin(), original_shape.cbegin() + new_rank);
                    new_shape[new_rank - 1] *= accum;
                    return tt::tt_metal::Shape(new_shape);
                };

            // Return the modified tensor_shape and shard_shape as a tuple
            return {
                copy_up_to_new_rank_and_multiply_last_dim(original_tensor_shape, new_rank, tensor_accum),
                copy_up_to_new_rank_and_multiply_last_dim(original_shard_shape, new_rank, shard_accum)};
        }

        case DistributionSpec::MappingMode::NONCOALESCED: {
            auto copy_and_insert_one_to_end_of_shape = [](const auto& original_shape) {
                auto new_shape = tt::stl::SmallVector<uint32_t>(original_shape.cbegin(), original_shape.cend());
                new_shape.push_back(1);
                return tt::tt_metal::Shape(new_shape);
            };

            // Return the modified tensor_shape and shard_shape as a tuple
            return {
                copy_and_insert_one_to_end_of_shape(original_tensor_shape),
                copy_and_insert_one_to_end_of_shape(original_shard_shape)};
        }
    }
    TT_THROW("MappingMode {} is unsupported in maybe_convert_shapes_for_mapping_mode!", mapping_mode);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

DistributionSpec::DistributionSpec(
    const tt::tt_metal::Shape& tensor_shape, const tt::stl::SmallVector<DistributionType>& spec, size_t num_targets) :
    tensor_shape_(tensor_shape), spec_(spec) {
    auto rank = tensor_shape.rank();
    TT_FATAL(spec.size() == rank, "Spec rank ({}) must be same as tensor shape rank ({})!", spec.size(), rank);

    tt::stl::SmallVector<uint32_t> shard_shape(rank);
    size_t num_shards = 1;
    for (size_t i = 0; i < rank; i++) {
        const auto [num_shards_along_dim, shard_size] = std::visit(
            [tensor_size = tensor_shape[i]](auto&& distribution_type) -> std::pair<size_t, size_t> {
                using SpecType = std::decay_t<decltype(distribution_type)>;
                if constexpr (std::is_same_v<SpecType, ShardSize>) {
                    auto shard_size = distribution_type.size;
                    return {tt::div_up(tensor_size, shard_size), shard_size};
                } else {
                    TT_THROW("Replication is not supported in DistributionSpec yet!");
                }
            },
            spec[i]);
        num_shards *= num_shards_along_dim;
        shard_shape[i] = shard_size;
    }

    shard_shape_ = tt::tt_metal::Shape(shard_shape);

    // Always set num_targets to num_targets used if num_targets provided > num_shards
    num_targets_ = std::min(num_targets, num_shards);
}

DistributionSpec DistributionSpec::from_shard_shape(
    const tt::tt_metal::Shape& tensor_shape, const tt::tt_metal::Shape& shard_shape, size_t num_targets) {
    auto rank = tensor_shape.rank();
    TT_FATAL(
        shard_shape.rank() == rank,
        "Shard shape rank ({}) must be same as tensor shape rank ({})!",
        shard_shape.rank(),
        rank);
    // Create spec with ShardSize only
    tt::stl::SmallVector<DistributionType> spec(rank);
    for (size_t i = 0; i < rank; i++) {
        spec[i] = ShardSize{shard_shape[i]};
    }

    return DistributionSpec(tensor_shape, spec, num_targets);
}

std::vector<DistributionSpec::TargetData> DistributionSpec::compute_metadata_for_targets(
    const MappingMode mapping_mode) const {
    // Compute mapping algorithm treats the last dim of the shard as contiguous
    // To handle the two cases of MappingMode:
    // - MappingMode::NONCOALESCED:
    //  * Insert an extra dim of 1 at the back of tensor_shape and shard_shape
    //  * This way, we treat all original dims as non-contiguous
    // - MappingMode::COALESCED:
    //  * Iteratively stack tensor_shape and shard_shape from the back if they are equal
    //  * Example: [2, 3, 4] cut by [1, 2, 4] is equivalent to [2, 12] cut by [1, 8]

    // TODO: Two more ways to coalesce
    // 1. For single-device, we have an optimization specifically for this case:
    //    * If HEIGHT sharded and pages are aligned, then copy entire tensor as is
    //      ** In ND sharding, HEIGHT sharded means we only cut along second last dim
    //    * Distribute the entire shards across targets at a lower level through a separate path
    // 2. If num_targets = 1, then every shard is contiguous within one target
    //    * In best case, everything is contiguous if all shards are full shards
    //    * Otherwise, can coalesce neighbouring pieces of shards together
    //      ** Example, [2, 3, 2] cut by [1, 2, 2] should give {(0, 0, 6), (6, 8, 6)} for target 0
    //      ** Probably best to do this by doing a separate pass through the final mapping
    // NOTES: 1. is important but should revisit later; 2. is low priority
    const auto [tensor_shape, shard_shape] =
        CMAKE_UNIQUE_NAMESPACE::maybe_convert_shapes_for_mapping_mode(tensor_shape_, shard_shape_, mapping_mode);

    const auto rank = tensor_shape.rank();
    const auto tensor_strides = tt::tt_metal::compute_strides(tensor_shape);
    const auto shard_strides = tt::tt_metal::compute_strides(shard_shape);
    const auto shard_volume = shard_shape.volume();

    // Compute shard_grid to iterate over; rounds up to account for partial shards
    tt::stl::SmallVector<uint32_t> shard_grid(rank);
    for (size_t dim = 0; dim < rank; dim++) {
        shard_grid[dim] = tt::div_up(tensor_shape[dim], shard_shape[dim]);
    }

    using IterateWithinShardFunc =
        std::function<void(TargetData&, const tt::tt_metal::Shape&, size_t&, size_t, size_t, size_t)>;
    IterateWithinShardFunc iterate_within_shard = [&tensor_strides, &shard_strides, &iterate_within_shard, &rank](
                                                      TargetData& target_data,
                                                      const tt::tt_metal::Shape& actual_shard_shape,
                                                      size_t& chunk_id,
                                                      size_t src,
                                                      size_t dst,
                                                      size_t dim) {
        // Base case: if we have processed all dims except last
        if (dim == rank - 1) {
            // Last dim of shard is treated as contiguous
            target_data[chunk_id] = ChunkMapping{src, dst, actual_shard_shape[rank - 1]};
            chunk_id++;
            return;
        }

        // Iterate over the current dimension
        for (size_t i = 0; i < actual_shard_shape[dim]; ++i) {
            // Recursively iterate on the next dimension
            iterate_within_shard(target_data, actual_shard_shape, chunk_id, src, dst, dim + 1);
            src += tensor_strides[dim];
            dst += shard_strides[dim];
        }
    };

    using IterateOverShardsFunc =
        std::function<void(std::vector<TargetData>&, tt::tt_metal::Shape&, size_t&, size_t, size_t)>;
    IterateOverShardsFunc iterate_over_shards = [&iterate_over_shards,
                                                 &iterate_within_shard,
                                                 &rank,
                                                 &tensor_shape,
                                                 &shard_shape,
                                                 &tensor_strides,
                                                 &shard_grid,
                                                 &shard_volume,
                                                 &num_targets = num_targets_](
                                                    std::vector<TargetData>& metadata_for_targets,
                                                    tt::tt_metal::Shape& actual_shard_shape,
                                                    size_t& shard_id,
                                                    size_t src_offset,
                                                    size_t dim) {
        // Base case: if we have processed all dims
        if (dim == rank) {
            // Round-robin distribute shards over num_targets
            const auto target_id = shard_id % num_targets;
            // dst is from the perspective of the target and is based off of full shard volume
            const auto dst_offset = (shard_id / num_targets) * shard_volume;

            // New entries for TargetData is number of elements in shard except last dim
            const auto num_chunks = actual_shard_shape.volume() / actual_shard_shape[-1];

            // Resize TargetData and populate it with iterate_within_shard
            // chunk_id is used to iterate through TargetData
            // src and dst values are computed based off of offsets in iterate_within_shard
            size_t chunk_id = metadata_for_targets[target_id].size();
            metadata_for_targets[target_id].resize(chunk_id + num_chunks);
            iterate_within_shard(
                metadata_for_targets[target_id], actual_shard_shape, chunk_id, src_offset, dst_offset, 0);

            shard_id++;
            return;
        }

        // Iterate over the current dimension
        for (size_t i = 0; i < shard_grid[dim]; ++i) {
            // Compute actual shard size along current dim
            const auto shard_size = shard_shape[dim];
            // If last shard, set it to partial shard size if we have partial shards
            if (i == shard_grid[dim] - 1) {
                const auto partial_shard_size = tensor_shape[dim] % shard_size;
                actual_shard_shape[dim] = partial_shard_size == 0 ? shard_size : partial_shard_size;
            } else {
                actual_shard_shape[dim] = shard_size;
            }

            // Recursively iterate on the next dimension
            iterate_over_shards(
                metadata_for_targets,
                actual_shard_shape,
                shard_id,
                src_offset + i * shard_shape[dim] * tensor_strides[dim],
                dim + 1);
        }
    };

    // Set up metadata_for_targets and populate it with iterate_over_shards
    // actual_shard_shape is either full or partial shard shape and is computed as we iterate over shards
    // shard_id is used to keep track of how many shards we have iterated over
    std::vector<TargetData> metadata_for_targets(num_targets_);
    tt::tt_metal::Shape actual_shard_shape(shard_shape);
    size_t shard_id = 0;
    iterate_over_shards(metadata_for_targets, actual_shard_shape, shard_id, 0, 0);

    return metadata_for_targets;
}

}  // namespace tt::tt_metal
