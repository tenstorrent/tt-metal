// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>

namespace addr_gen_utils {

template <size_t... Dims>
struct ShapeWrapper {
    static constexpr size_t rank = sizeof...(Dims);
    static_assert(rank > 0, "Shape rank must be greater than 0!");
    static constexpr std::array<uint32_t, rank> shape = {Dims...};

    // Compute shape properities at compile time
    static constexpr std::pair<size_t, std::array<uint32_t, rank>> compute_volume_and_strides(
        const std::array<uint32_t, rank>& shape) {
        std::array<uint32_t, rank> strides;
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return {strides[0] * shape[0], strides};
    }

    // Compiler should optimize out the second call
    // static constexpr size_t volume = compute_volume_and_strides(shape).first;
    // static constexpr std::array<uint32_t, rank> strides = compute_volume_and_strides(shape).second;
    static constexpr auto volume = compute_volume_and_strides(shape).first;
    static constexpr auto strides = compute_volume_and_strides(shape).second;
};

template <typename F, size_t... Is>
constexpr auto make_shape_wrapper(F, std::index_sequence<Is...>) -> ShapeWrapper<F{}()[Is]...>;

}  // namespace addr_gen_utils

#define USING_SHAPE_WRAPPER(name, arr)                      \
    struct name##_fn {                                      \
        constexpr auto operator()() const { return (arr); } \
    };                                                      \
    using name = decltype(::addr_gen_utils::make_shape_wrapper(name##_fn{}, std::make_index_sequence<(arr).size()>{}))

template <typename TensorShape, typename ShardShape, size_t NumBanks>
struct KernelDistributionSpec {
    static_assert(TensorShape::rank == ShardShape::rank, "Tensor and shard shapes must have the same rank!");
    static constexpr auto rank = TensorShape::rank;
    static_assert(rank > 0, "Tensor and shard shape ranks must be greater than 0!");
    static constexpr auto num_banks = NumBanks;

    static constexpr auto tensor_shape = TensorShape::shape;
    static constexpr auto tensor_strides = TensorShape::strides;
    static constexpr auto tensor_volume = TensorShape::volume;

    static constexpr auto shard_shape = ShardShape::shape;
    static constexpr auto shard_strides = ShardShape::strides;
    static constexpr auto shard_volume = ShardShape::volume;

    // Compute shard grid and shard grid strides at compile time
    static constexpr std::pair<std::array<uint32_t, rank>, std::array<uint32_t, rank>> compute_shard_grid_and_strides(
        const std::array<uint32_t, rank>& tensor_shape, const std::array<uint32_t, rank>& shard_shape) {
        std::array<uint32_t, rank> shard_grid;
        std::array<uint32_t, rank> shard_grid_strides;
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            shard_grid[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
            shard_grid_strides[i] = stride;
            stride *= shard_grid[i];
        }
        return {shard_grid, shard_grid_strides};
    }

    // Compiler should optimize out the second call
    // NOTE: shard_grid is not really needed, but probably more idiomatic to keep it with shard_grid_strides
    static constexpr auto shard_grid = compute_shard_grid_and_strides(tensor_shape, shard_shape).first;
    static constexpr auto shard_grid_strides = compute_shard_grid_and_strides(tensor_shape, shard_shape).second;

    // Check that the number of shards is greater than or equal to the number of banks
    // Here, shard_grid_strides[0] * shard_grid[0] is the total number of shards
    static_assert(
        shard_grid_strides[0] * shard_grid[0] >= num_banks,
        "Number of shards must be greater than or equal to number of banks!");
};

template <typename DSpec>
struct ShardedAccessor {
    static constexpr DSpec DSPEC_CONSTANTS{};

    std::pair<size_t, size_t> get_bank_and_offset(uint32_t page_id) const {
        // Check if page_id is within bounds of total tensor volume
        TT_FATAL(
            page_id <= DSPEC_CONSTANTS.tensor_volume,
            "Page id {} must be less than tensor volume {}!",
            page_id,
            DSPEC_CONSTANTS.tensor_volume);

        std::cout << "page_coord: ";
        std::cout << "page_id: " << page_id << std::endl;
        std::array<uint32_t, DSPEC_CONSTANTS.rank> page_coord;
        for (int i = DSPEC_CONSTANTS.rank - 1; i >= 0; --i) {
            page_coord[i] = page_id % DSPEC_CONSTANTS.tensor_shape[i];
            page_id /= DSPEC_CONSTANTS.tensor_shape[i];
        }
        return get_bank_and_offset(page_coord);
    }

    std::pair<size_t, size_t> get_bank_and_offset(const std::array<uint32_t, DSPEC_CONSTANTS.rank> page_coord) const {
        // Check if page_coord is within bounds of tensor shape at each dimension
        for (size_t i = 0; i < DSPEC_CONSTANTS.rank; ++i) {
            TT_FATAL(
                page_coord[i] <= DSPEC_CONSTANTS.tensor_shape[i],
                "Page coord {} must be less than tensor shape {} at rank {}!",
                page_coord[i],
                DSPEC_CONSTANTS.tensor_shape[i],
                i);
        }

        std::cout << "page_coord: ";
        for (size_t i = 0; i < DSPEC_CONSTANTS.rank; ++i) {
            std::cout << page_coord[i] << " ";
        }
        std::cout << std::endl;

        std::array<uint32_t, DSPEC_CONSTANTS.rank> shard_coord;
        std::array<uint32_t, DSPEC_CONSTANTS.rank> local_coord;
        for (size_t i = 0; i < DSPEC_CONSTANTS.rank; ++i) {
            shard_coord[i] = page_coord[i] / DSPEC_CONSTANTS.shard_shape[i];
            local_coord[i] = page_coord[i] % DSPEC_CONSTANTS.shard_shape[i];
        }

        // Compute flattened shard id
        size_t shard_id = 0;
        for (size_t i = 0; i < DSPEC_CONSTANTS.rank; ++i) {
            shard_id += shard_coord[i] * DSPEC_CONSTANTS.shard_grid_strides[i];
        }

        // Bank id is round-robin assigned
        size_t bank_id = shard_id % DSPEC_CONSTANTS.num_banks;
        size_t bank_shard_index = shard_id / DSPEC_CONSTANTS.num_banks;

        // Compute offset within shard
        size_t offset_within_shard = 0;
        for (size_t i = 0; i < DSPEC_CONSTANTS.rank; ++i) {
            offset_within_shard += local_coord[i] * DSPEC_CONSTANTS.shard_strides[i];
        }

        // Total offset in bank = bank_shard_index * shard_volume + offset_within_shard
        size_t offset_in_bank = bank_shard_index * DSPEC_CONSTANTS.shard_volume + offset_within_shard;

        return {bank_id, offset_in_bank};
    }
};
