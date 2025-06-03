// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <tuple>

namespace detail {

template <size_t... Dims>
struct ShapeWrapper {
    static constexpr size_t rank = sizeof...(Dims);
    static constexpr bool is_static = true;
    using ShapeBase = std::array<uint32_t, rank>;
    static constexpr ShapeBase shape = {Dims...};

    // Check that rank is > 0
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    // Check that all Dims are > 0
    static_assert(((Dims > 0) && ...), "Shape dims must be greater than 0!");

    // Compute shape properities at compile time
    static constexpr std::pair<size_t, ShapeBase> compute_volume_and_strides(const ShapeBase& shape) {
        ShapeBase strides = {};
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

    constexpr explicit ShapeWrapper() = default;
    constexpr explicit ShapeWrapper(const ShapeBase&) {}
    constexpr explicit ShapeWrapper(ShapeBase&&) {}
};

template <size_t Rank>
struct ShapeWrapperDynamic {
    static constexpr size_t rank = Rank;
    static constexpr bool is_static = false;
    using ShapeBase = std::array<uint32_t, rank>;
    ShapeBase shape;    // runtime shape
    ShapeBase strides;  // runtime strides
    size_t volume;      // runtime volume

    // Check that rank is > 0
    static_assert(rank > 0, "Shape rank must be greater than 0!");

    template <class... Ts, std::enable_if_t<sizeof...(Ts) == Rank, int> = 0>
    constexpr explicit ShapeWrapperDynamic(Ts... exts) : shape{static_cast<uint32_t>(exts)...} {
        compute_volume_and_strides(shape);
    }

    constexpr explicit ShapeWrapperDynamic() = default;

    constexpr explicit ShapeWrapperDynamic(const ShapeBase& shape) : shape{shape} { compute_volume_and_strides(shape); }

    constexpr explicit ShapeWrapperDynamic(ShapeBase&& shape) : shape{std::move(shape)} {
        compute_volume_and_strides(shape);
    }

    inline void compute_volume_and_strides(const ShapeBase& shape) {
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        volume = strides[0] * shape[0];
    }
};

template <size_t... PackedCoords>
struct BankCoordWrapper {
    static constexpr bool is_static = true;
    static constexpr size_t num_banks = sizeof...(PackedCoords);
    // TODO: Each bank coord is packed as one uint32_t (ie. (16 bits) <x> | (16 bits) <y>)
    // This can be optimized to be 8 bits per coord, so we pack two bank coords in one uint32_t compile time arg
    using PackedCoordsArray = std::array<uint32_t, num_banks>;
    static constexpr PackedCoordsArray packed_xy_coords = {PackedCoords...};
    constexpr explicit BankCoordWrapper() = default;
    constexpr explicit BankCoordWrapper(const PackedCoordsArray&) {}
    constexpr explicit BankCoordWrapper(PackedCoordsArray&&) {}
};

template <size_t NumBanks>
struct BankCoordWrapperDynamic {
    static constexpr bool is_static = false;
    static constexpr size_t num_banks = NumBanks;
    using PackedCoordsArray = std::array<uint32_t, num_banks>;
    PackedCoordsArray packed_xy_coords;
    constexpr explicit BankCoordWrapperDynamic() = default;
    constexpr explicit BankCoordWrapperDynamic(const PackedCoordsArray& banks_coords) :
        packed_xy_coords(banks_coords) {}
    constexpr explicit BankCoordWrapperDynamic(PackedCoordsArray&& banks_coords) :
        packed_xy_coords(std::move(banks_coords)) {}
};

//
template <template <size_t...> class Wrapper, size_t BASE_IDX, size_t... Is>
constexpr auto make_struct_from_sequence_wrapper(std::index_sequence<Is...>)
    -> Wrapper<get_compile_time_arg_val(BASE_IDX + Is)...>;

template <template <size_t...> class Wrapper, size_t base, size_t rank>
using struct_sequence_wrapper_t =
    decltype(make_struct_from_sequence_wrapper<Wrapper, base>(std::make_index_sequence<rank>{}));

// Helper to generate array using index sequence
template <std::size_t Base, std::size_t... Is>
constexpr std::array<uint32_t, sizeof...(Is)> make_runtime_array_from_sequence(std::index_sequence<Is...>) {
    return {get_arg_val<uint32_t>(Base + Is)...};
}

// Public interface
template <std::size_t Base, std::size_t Size>
constexpr auto runtime_array_sequence_wrapper() {
    return make_runtime_array_from_sequence<Base>(std::make_index_sequence<Size>{});
}

template <typename TensorShape, typename ShardShape, typename BankCoords>
struct DistributionSpec {
    static_assert(TensorShape::rank == ShardShape::rank, "Tensor and shard shapes must have the same rank!");
    static_assert(
        std::is_same_v<typename TensorShape::ShapeBase, typename ShardShape::ShapeBase>,
        "Tensor and shard shapes bases must be the same");
    static_assert(TensorShape::rank > 0, "Tensor and shard shape ranks must be greater than 0!");
    static constexpr auto rank = TensorShape::rank;
    using TensorShapeT = TensorShape;
    using ShardShapeT = ShardShape;
    using BankCoordsT = BankCoords;
    using ShapeBase = typename TensorShape::ShapeBase;
    using PackedCoordsArray = typename BankCoords::PackedCoordsArray;

    static constexpr bool all_shapes_static = TensorShape::is_static && ShardShape::is_static;
    static constexpr bool bank_coords_static = BankCoords::is_static;
    static constexpr bool is_static = all_shapes_static && bank_coords_static;

    static constexpr size_t num_banks = BankCoords::num_banks;

    constexpr DistributionSpec() {
        static_assert(is_static, "Cannot use default constructor for non-static DistributionSpec!");
        static_assert(
            shard_grid_ct_[0] * shard_grid_strides_ct_[0] >= num_banks,
            "Number of shards must be greater than or equal to number of banks!");
    };

    template <
        typename TensorShapeArr = ShapeBase,
        typename ShardShapeArr = ShapeBase,
        typename BankCoordsArr = PackedCoordsArray>
    constexpr DistributionSpec(
        TensorShapeArr&& tensor_shape_arr, ShardShapeArr&& shard_shape_arr = {}, BankCoordsArr&& bank_coords_arr = {}) :
        tensor_shape_(std::forward<TensorShapeArr>(tensor_shape_arr)),
        shard_shape_(std::forward<ShardShapeArr>(shard_shape_arr)),
        bank_coords_(std::forward<BankCoordsArr>(bank_coords_arr)) {
        static_assert(!(is_static), "Everything is static, this constructor is obsolete!");
        if constexpr (!all_shapes_static) {
            // Tensor shape has bad rank!
            ASSERT(get_tensor_shape().size() == rank);
            // Shard shape has bad rank!
            ASSERT(get_shard_shape().size() == rank);
            compute_shard_grid_and_strides_rt(get_tensor_shape(), get_shard_shape());
        }
        if constexpr (!bank_coords_static) {
            // Number of bank coordinates must match the number of banks!"
            ASSERT(bank_coords_arr.size() == num_banks);
        }
    }

    constexpr const ShapeBase& get_shard_grid() const {
        if constexpr (all_shapes_static) {
            return shard_grid_ct_;
        } else {
            return shard_grid_rt;
        }
    }

    constexpr const ShapeBase& get_shard_grid_strides() const {
        if constexpr (all_shapes_static) {
            return shard_grid_strides_ct_;
        } else {
            return shard_grid_strides_rt;
        }
    }

    constexpr const ShapeBase& get_tensor_shape() const {
        if constexpr (TensorShape::is_static) {
            return TensorShape::shape;
        } else {
            return tensor_shape_.shape;
        }
    }

    constexpr const ShapeBase& get_tensor_strides() const {
        if constexpr (TensorShape::is_static) {
            return TensorShape::strides;
        } else {
            return tensor_shape_.strides;
        }
    }

    constexpr size_t get_tensor_volume() const {
        if constexpr (TensorShape::is_static) {
            return TensorShape::volume;
        } else {
            return tensor_shape_.volume;
        }
    }

    constexpr const ShapeBase& get_shard_shape() const {
        if constexpr (ShardShape::is_static) {
            return ShardShape::shape;
        } else {
            return shard_shape_.shape;
        }
    }

    constexpr const ShapeBase& get_shard_strides() const {
        if constexpr (ShardShape::is_static) {
            return ShardShape::strides;
        } else {
            return shard_shape_.strides;
        }
    }

    constexpr size_t get_shard_volume() const {
        if constexpr (ShardShape::is_static) {
            return ShardShape::volume;
        } else {
            return shard_shape_.volume;
        }
    }

    constexpr const PackedCoordsArray& get_packed_xy_coords() const {
        if constexpr (BankCoords::is_static) {
            return BankCoords::packed_xy_coords;
        } else {
            return bank_coords_.packed_xy_coords;
        }
    }

    // Compute shard grid and shard grid strides at compile time
    static constexpr ShapeBase shard_grid_ct(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        // If shapes are dynamic, we cannot compute shard grid at compile time
        if (!all_shapes_static) {
            return {};
        }
        ShapeBase shard_grid = {};
        for (int i = rank - 1; i >= 0; --i) {
            shard_grid[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid;
    }

    static constexpr ShapeBase shard_grid_strides_ct(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        // If shapes are dynamic, we cannot compute strides at compile time
        if (!all_shapes_static) {
            return {};
        }
        ShapeBase shard_grid_strides = {};
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            shard_grid_strides[i] = stride;
            stride *= (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid_strides;
    }

    void compute_shard_grid_and_strides_rt(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        uint32_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            shard_grid_rt[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
            shard_grid_strides_rt[i] = stride;
            stride *= shard_grid_rt[i];
        }
        // Check that the number of shards is greater than or equal to the number of banks
        ASSERT(shard_grid_rt[0] * shard_grid_strides_rt[0] >= num_banks);
    }

    std::conditional_t<all_shapes_static, std::monostate, ShapeBase> shard_grid_rt{};
    std::conditional_t<all_shapes_static, std::monostate, ShapeBase> shard_grid_strides_rt{};

    static constexpr ShapeBase shard_grid_ct_ = shard_grid_ct(TensorShape::shape, ShardShape::shape);
    static constexpr ShapeBase shard_grid_strides_ct_ = shard_grid_strides_ct(TensorShape::shape, ShardShape::shape);

    TensorShape tensor_shape_;  // Tensor shape
    ShardShape shard_shape_;    // Shard shape
    BankCoords bank_coords_;    // Bank coordinates

    // static constexpr auto packed_xy_coords = BankCoords::packed_xy_coords;
};

template <
    size_t CTA_BASE,
    size_t RANK,
    size_t NUM_BANKS,
    bool TensorShapeDynamic = false,
    bool ShardShapeDynamic = false,
    bool BankCoordsDynamic = false>
struct DistributionSpecWrapper {
    using dspec = DistributionSpec<
        std::conditional_t<
            TensorShapeDynamic,
            ShapeWrapperDynamic<RANK>,
            struct_sequence_wrapper_t<ShapeWrapper, CTA_BASE, RANK>>,
        std::conditional_t<
            ShardShapeDynamic,
            ShapeWrapperDynamic<RANK>,
            struct_sequence_wrapper_t<ShapeWrapper, CTA_BASE + RANK * !TensorShapeDynamic, RANK>>,
        std::conditional_t<
            BankCoordsDynamic,
            BankCoordWrapperDynamic<RANK>,
            struct_sequence_wrapper_t<
                BankCoordWrapper,
                CTA_BASE + RANK * !TensorShapeDynamic + RANK * !ShardShapeDynamic,
                NUM_BANKS>>>;
};

template <size_t RTA_BASE, typename DSpec>
auto build_dspec_from_runtime_args() {
    static constexpr bool TensorShapeDynamic = !DSpec::TensorShapeT::is_static;
    static constexpr bool ShardShapeDynamic = !DSpec::ShardShapeT::is_static;
    static constexpr bool BankCoordsDynamic = !DSpec::BankCoordsT::is_static;
    static constexpr size_t RANK = DSpec::rank;
    static constexpr size_t NUM_BANKS = DSpec::num_banks;

    std::array<uint32_t, RANK> tensor_shape_array;
    std::array<uint32_t, RANK> shard_shape_array;
    std::array<uint32_t, NUM_BANKS> bank_coord_array;
    if constexpr (TensorShapeDynamic) {
        tensor_shape_array = runtime_array_sequence_wrapper<RTA_BASE, RANK>();
    }
    if constexpr (ShardShapeDynamic) {
        shard_shape_array = runtime_array_sequence_wrapper<RTA_BASE + RANK * TensorShapeDynamic, RANK>();
    }
    if constexpr (BankCoordsDynamic) {
        bank_coord_array = runtime_array_sequence_wrapper<
            RTA_BASE + RANK * TensorShapeDynamic + RANK * ShardShapeDynamic,
            NUM_BANKS>();
    }

    return DSpec(std::move(tensor_shape_array), std::move(shard_shape_array), std::move(bank_coord_array));
}

}  // namespace detail
