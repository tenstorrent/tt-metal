// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <tuple>
#include "helpers.hpp"
#include "shape_wrapper.hpp"
#include "bank_coords.hpp"
#include <hostdevcommon/flags.hpp>

namespace nd_sharding {
namespace detail {

namespace {
// TODO: This exact enum is defined on host. Maybe somehow reuse it?
enum class ArgConfig : uint8_t {
    CTA = 0,
    RuntimeTensorShape = 1 << 0,
    RuntimeShardShape = 1 << 1,
    RuntimeBankCoords = 1 << 2,
    CRTA = RuntimeTensorShape | RuntimeShardShape | RuntimeBankCoords
};

using ArgsConfig = Flags<ArgConfig>;
constexpr ArgsConfig operator|(ArgConfig a, ArgConfig b) noexcept { return ArgsConfig(a) | b; }
constexpr ArgsConfig operator|(ArgConfig a, ArgsConfig b) noexcept { return ArgsConfig(a) | b; }
constexpr ArgsConfig operator|(ArgsConfig a, ArgConfig b) noexcept { return a | ArgsConfig(b); }
}  // namespace

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

// Helper template for selecting the appropriate type (static or dynamic)
template <
    bool IsDynamic,
    template <size_t...> class StaticWrapper,
    template <size_t> class DynamicWrapper,
    size_t BASE,
    size_t SIZE>
struct TypeSelector;

// Specialization for dynamic types
template <template <size_t...> class StaticWrapper, template <size_t> class DynamicWrapper, size_t BASE, size_t SIZE>
struct TypeSelector<true, StaticWrapper, DynamicWrapper, BASE, SIZE> {
    using type = DynamicWrapper<SIZE>;
};

// Specialization for static types
template <template <size_t...> class StaticWrapper, template <size_t> class DynamicWrapper, size_t BASE, size_t SIZE>
struct TypeSelector<false, StaticWrapper, DynamicWrapper, BASE, SIZE> {
    using type = struct_cta_sequence_wrapper_t<StaticWrapper, BASE, SIZE>;
};

template <size_t CTA_BASE, size_t RANK, size_t NUM_BANKS>
struct DistributionSpecWrapper {
    constexpr static auto args_config =
        ArgsConfig(static_cast<ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_BASE)));
    constexpr static bool TensorShapeDynamic = args_config.test(ArgConfig::RuntimeTensorShape);
    constexpr static bool ShardShapeDynamic = args_config.test(ArgConfig::RuntimeShardShape);
    constexpr static bool BankCoordsDynamic = args_config.test(ArgConfig::RuntimeBankCoords);

    // Calculate offsets based on which previous shapes are dynamic
    constexpr static size_t TensorShapeBase = CTA_BASE + 1;
    static constexpr size_t ShardShapeBase = TensorShapeBase + (TensorShapeDynamic ? 0 : RANK);
    static constexpr size_t BankCoordsBase = ShardShapeBase + (ShardShapeDynamic ? 0 : RANK);

    using TensorShapeType = typename TypeSelector<
        TensorShapeDynamic,
        ShapeWrapperStaticDimsStaticRank,
        ShapeWrapperDynamicDimsStaticRank,
        TensorShapeBase,
        RANK>::type;
    using ShardShapeType = typename TypeSelector<
        ShardShapeDynamic,
        ShapeWrapperStaticDimsStaticRank,
        ShapeWrapperDynamicDimsStaticRank,
        ShardShapeBase,
        RANK>::type;
    using BankCoordsType = typename TypeSelector<
        BankCoordsDynamic,
        BankCoordWrapperStaticNBanksStaticCoords,
        BankCoordWrapperDynamicStaticNBanksDynamicCoords,
        BankCoordsBase,
        NUM_BANKS>::type;
    using dspec = DistributionSpec<TensorShapeType, ShardShapeType, BankCoordsType>;
};

template <size_t CRTA_BASE, typename DSpec>
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
        tensor_shape_array = array_crta_sequence_wrapper<CRTA_BASE, RANK>();
    }
    if constexpr (ShardShapeDynamic) {
        shard_shape_array = array_crta_sequence_wrapper<CRTA_BASE + RANK * TensorShapeDynamic, RANK>();
    }
    if constexpr (BankCoordsDynamic) {
        bank_coord_array =
            array_crta_sequence_wrapper<CRTA_BASE + RANK * TensorShapeDynamic + RANK * ShardShapeDynamic, NUM_BANKS>();
    }

    return DSpec(std::move(tensor_shape_array), std::move(shard_shape_array), std::move(bank_coord_array));
}

}  // namespace detail
}  // namespace nd_sharding
