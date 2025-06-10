// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace detail {

template <size_t... Dims>
struct ShapeWrapper {
    // Check that rank is > 0
    static constexpr size_t rank = sizeof...(Dims);
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    // Check that all Dims are > 0
    static_assert(((Dims > 0) && ...), "Shape dims must be greater than 0!");

    static constexpr std::array<uint32_t, rank> shape = {Dims...};

    // Compute shape properities at compile time
    static constexpr std::pair<size_t, std::array<uint32_t, rank>> compute_volume_and_strides(
        const std::array<uint32_t, rank>& shape) {
        std::array<uint32_t, rank> strides = {};
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return {strides[0] * shape[0], strides};
    }

    // Compiler should optimize out the second call
    static constexpr auto volume = compute_volume_and_strides(shape).first;
    static constexpr auto strides = compute_volume_and_strides(shape).second;
};

template <size_t... PackedCoords>
struct BankCoordWrapper {
    static constexpr size_t num_banks = sizeof...(PackedCoords);
    // TODO: Each bank coord is packed as one uint32_t (ie. (16 bits) <x> | (16 bits) <y>)
    // This can be optimized to be 8 bits per coord, so we pack two bank coords in one uint32_t compile time arg
    static constexpr std::array<uint32_t, num_banks> packed_xy_coords = {PackedCoords...};
};

//
template <template <size_t...> class Wrapper, size_t BASE_IDX, size_t... Is>
constexpr auto make_struct_from_sequence_wrapper(std::index_sequence<Is...>)
    -> Wrapper<get_compile_time_arg_val(BASE_IDX + Is)...>;

template <template <size_t...> class Wrapper, size_t base, size_t rank>
using struct_sequence_wrapper_t =
    decltype(make_struct_from_sequence_wrapper<Wrapper, base>(std::make_index_sequence<rank>{}));

template <typename TensorShape, typename ShardShape, typename BankCoords>
struct DistributionSpec {
    static_assert(TensorShape::rank == ShardShape::rank, "Tensor and shard shapes must have the same rank!");
    static constexpr auto rank = TensorShape::rank;
    static_assert(rank > 0, "Tensor and shard shape ranks must be greater than 0!");

    static constexpr auto tensor_shape = TensorShape::shape;
    static constexpr auto tensor_strides = TensorShape::strides;
    static constexpr auto tensor_volume = TensorShape::volume;

    static constexpr auto shard_shape = ShardShape::shape;
    static constexpr auto shard_strides = ShardShape::strides;
    static constexpr auto shard_volume = ShardShape::volume;

    // Compute shard grid and shard grid strides at compile time
    static constexpr std::pair<std::array<uint32_t, rank>, std::array<uint32_t, rank>> compute_shard_grid_and_strides(
        const std::array<uint32_t, rank>& tensor_shape, const std::array<uint32_t, rank>& shard_shape) {
        std::array<uint32_t, rank> shard_grid = {};
        std::array<uint32_t, rank> shard_grid_strides = {};
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

    static constexpr auto num_banks = BankCoords::num_banks;
    static constexpr auto packed_xy_coords = BankCoords::packed_xy_coords;
    // Check that the number of shards is greater than or equal to the number of banks
    // Here, shard_grid_strides[0] * shard_grid[0] is the total number of shards
    static_assert(
        shard_grid_strides[0] * shard_grid[0] >= num_banks,
        "Number of shards must be greater than or equal to number of banks!");
};

template <size_t BASE, size_t RANK, size_t NUM_BANKS>
struct DistributionSpecWrapper {
    using dspec = DistributionSpec<
        struct_sequence_wrapper_t<ShapeWrapper, BASE, RANK>,
        struct_sequence_wrapper_t<ShapeWrapper, BASE + RANK, RANK>,
        struct_sequence_wrapper_t<BankCoordWrapper, BASE + 2 * RANK, NUM_BANKS>>;
};

}  // namespace detail
