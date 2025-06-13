// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include "helpers.hpp"
#include "const.hpp"
#include "array_wrapper.hpp"
#include "args_location.hpp"

namespace nd_sharding {
namespace detail {

/**
 * @brief Holds all the distribution specification information for a tensor: rank, number of banks, tensor shape, shard
 * shape, bank coordinates. Each of these can be static or dynamic. ArgsLoc_ is used to determine the compile-time and
 * runtime arguments for the distribution specification.
 *
 * @tparam TensorShape_
 * @tparam ShardShape_
 * @tparam BankCoords_
 * @tparam ArgsLoc_
 */
template <
    uint32_t RankCT = 0,
    uint32_t NumBanksCT = 0,
    typename TensorShapeWrapper_ = ArrayDynamicWrapper,
    typename ShardShapeWrapper_ = ArrayDynamicWrapper,
    typename BankCoordsWrapper_ = ArrayDynamicWrapper>
struct DistributionSpec {
    using TensorShapeWrapper = TensorShapeWrapper_;
    using ShardShapeWrapper = ShardShapeWrapper_;
    using BankCoordsWrapper = BankCoordsWrapper_;

    static constexpr bool has_static_rank = RankCT != 0;
    static constexpr bool has_static_num_banks = NumBanksCT != 0;
    static constexpr bool tensor_shape_static = has_static_rank && TensorShapeWrapper::is_static;
    static constexpr bool shard_shape_static = has_static_rank && ShardShapeWrapper::is_static;
    static constexpr bool bank_coords_static = has_static_num_banks && BankCoordsWrapper::is_static;
    static constexpr bool shapes_static = has_static_rank && tensor_shape_static && shard_shape_static;
    static constexpr bool is_static = shapes_static && bank_coords_static;

    static constexpr auto rank_ct = RankCT;
    static constexpr uint32_t num_banks_ct = NumBanksCT;

    using Shape = Span<uint32_t>;
    using ShapeStatic = std::array<uint32_t, rank_ct>;
    using BankCoordsStatic = std::array<uint32_t, num_banks_ct>;

    // This constructor is only used for completely static DistributionSpec
    template <typename T = void, typename = std::enable_if_t<is_static, T>>
    constexpr DistributionSpec() {
        static_assert(
            shard_grid_ct[0] * shard_grid_strides_ct[0] >= num_banks_ct,
            "Number of shards must be greater than or equal to number of banks!");
    };

    template <typename TensorShape = Shape, typename ShardShape = Shape, typename BankCoords = Shape>
    constexpr DistributionSpec(
        TensorShape&& tensor_shape_arr, ShardShape&& shard_shape_arr = {}, BankCoords&& bank_coords_arr = {}) :
        tensor_shape_rt(std::forward<TensorShape>(tensor_shape_arr)),
        shard_shape_rt(std::forward<ShardShape>(shard_shape_arr)),
        bank_coords_rt(std::forward<BankCoords>(bank_coords_arr)) {
        if constexpr (!has_static_rank) {
            // Rank is not known at compile time, use runtime rank
            rank_rt = tensor_shape_rt.size();
        }
        if constexpr (!has_static_num_banks) {
            // Number of banks is not known at compile time, use runtime number of banks
            num_banks_rt = bank_coords_rt.size();
        }
        if constexpr (!shapes_static) {
            shard_grid_rt = Shape(shard_grid_rt_buf.value, rank_rt);
            shard_grid_strides_rt = Shape(shard_grid_strides_rt_buf.value, rank_rt);

            tensor_strides_rt = Shape(tensor_strides_rt_buf.value, rank_rt);
            shard_strides_rt = Shape(shard_strides_rt_buf.value, rank_rt);
        }
        if constexpr (!tensor_shape_static) {
            // If tensor shape is not static, we need to compute strides and volume at runtime
            compute_strides_volume_rt(get_tensor_shape(), tensor_strides_rt, tensor_volume_rt);
        }
        if constexpr (!shard_shape_static) {
            // If shard shape is not static, we need to compute strides and volume at runtime
            compute_strides_volume_rt(get_shard_shape(), shard_strides_rt, shard_volume_rt);
        }
        if constexpr (!shapes_static) {
            compute_shard_grid_and_strides_rt(get_tensor_shape(), get_shard_shape());
            ASSERT(shard_grid_rt[0] * shard_grid_strides_rt[0] >= get_num_banks());
        }
    }

// Helper macro to avoid code duplication in getters
#define getter_helper(is_static, val_ct, val_rt) \
    if constexpr (is_static) {                   \
        return val_ct;                           \
    } else {                                     \
        return val_rt;                           \
    }
    // Getters
    FORCE_INLINE constexpr uint32_t get_rank() const { getter_helper(has_static_rank, rank_ct, rank_rt); }

    FORCE_INLINE constexpr uint32_t get_num_banks() const {
        getter_helper(has_static_num_banks, num_banks_ct, num_banks_rt)}

    FORCE_INLINE constexpr const
        auto& get_shard_grid() const {getter_helper(shapes_static, shard_grid_ct, shard_grid_rt)}

    FORCE_INLINE constexpr const auto& get_shard_grid_strides() const {
        getter_helper(shapes_static, shard_grid_strides_ct, shard_grid_strides_rt)}

    FORCE_INLINE constexpr const auto& get_tensor_shape() const {
        getter_helper(tensor_shape_static, TensorShapeWrapper::elements, tensor_shape_rt)}

    FORCE_INLINE constexpr const
        auto& get_tensor_strides() const {getter_helper(tensor_shape_static, tensor_strides_ct, tensor_strides_rt)}

    FORCE_INLINE constexpr size_t
        get_tensor_volume() const {getter_helper(tensor_shape_static, tensor_volume_ct, tensor_volume_rt)}

    FORCE_INLINE constexpr const
        auto& get_shard_shape() const {getter_helper(shard_shape_static, ShardShapeWrapper::elements, shard_shape_rt)}

    FORCE_INLINE constexpr const
        auto& get_shard_strides() const {getter_helper(shard_shape_static, shard_strides_ct, shard_strides_rt)}

    FORCE_INLINE constexpr size_t
        get_shard_volume() const {getter_helper(shard_shape_static, shard_volume_ct, shard_volume_rt)}

    FORCE_INLINE constexpr const auto& get_packed_xy_coords() const {
        getter_helper(bank_coords_static, BankCoordsWrapper::elements, bank_coords_rt)
    }

#undef getter_helper

private:
    static constexpr ShapeStatic precompute_shard_grid_ct(
        const ShapeStatic& tensor_shape, const ShapeStatic& shard_shape) {
        // If shapes are dynamic, we cannot compute shard grid at compile time
        if (!shapes_static) {
            return {};
        }
        ShapeStatic shard_grid = {};
        for (int i = rank_ct - 1; i >= 0; --i) {
            shard_grid[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid;
    }

    static constexpr ShapeStatic precompute_shard_grid_strides_ct(
        const ShapeStatic& tensor_shape, const ShapeStatic& shard_shape) {
        ShapeStatic shard_grid_strides = {};
        uint32_t stride = 1;
        for (int i = rank_ct - 1; i >= 0; --i) {
            shard_grid_strides[i] = stride;
            stride *= (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid_strides;
    }

    static constexpr size_t precompute_volume_ct(const ShapeStatic& shape) {
        size_t volume = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            volume *= shape[i];
        }
        return volume;
    }

    static constexpr ShapeStatic precompute_strides_ct(const ShapeStatic& shape) {
        ShapeStatic strides = {};
        uint32_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    template <typename TensorShape, typename ShardShape>
    void compute_strides_volume_rt(const TensorShape& shape, ShardShape& strides, size_t& volume) const {
        uint32_t stride = 1;
        volume = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
            volume *= shape[i];
        }
    }

    template <typename TensorShape, typename ShardShape>
    void compute_shard_grid_and_strides_rt(const TensorShape& tensor_shape, const ShardShape& shard_shape) {
        uint32_t stride = 1;
        for (int i = get_rank() - 1; i >= 0; --i) {
            shard_grid_rt[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
            shard_grid_strides_rt[i] = stride;
            stride *= shard_grid_rt[i];
        }
        // Check that the number of shards is greater than or equal to the number of banks
        ASSERT(shard_grid_rt[0] * shard_grid_strides_rt[0] >= get_num_banks());
    }

    uint32_t rank_rt = 0;
    uint32_t num_banks_rt = 0;

    Shape tensor_shape_rt;
    Shape shard_shape_rt;
    Shape bank_coords_rt;

    std::conditional_t<shapes_static, std::monostate, Shape> shard_grid_rt{};
    std::conditional_t<shapes_static, std::monostate, Shape> shard_grid_strides_rt{};

    // Buffers to wrap around span in case of dynamic rank
    mutable detail::ConditionalField<!shapes_static, uint32_t[MAX_RANK]> shard_grid_rt_buf;
    mutable detail::ConditionalField<!shapes_static, uint32_t[MAX_RANK]> shard_grid_strides_rt_buf;

    static constexpr ShapeStatic shard_grid_ct =
        precompute_shard_grid_ct(TensorShapeWrapper::elements, ShardShapeWrapper::elements);
    static constexpr ShapeStatic shard_grid_strides_ct =
        precompute_shard_grid_strides_ct(TensorShapeWrapper::elements, ShardShapeWrapper::elements);

    mutable detail::ConditionalField<!shapes_static, uint32_t[MAX_RANK]> tensor_strides_rt_buf;
    mutable detail::ConditionalField<!shapes_static, uint32_t[MAX_RANK]> shard_strides_rt_buf;
    Shape tensor_strides_rt = {};
    Shape shard_strides_rt = {};
    static constexpr ShapeStatic tensor_strides_ct = precompute_strides_ct(TensorShapeWrapper::elements);
    static constexpr ShapeStatic shard_strides_ct = precompute_strides_ct(ShardShapeWrapper::elements);

    size_t tensor_volume_rt = 0;
    size_t shard_volume_rt = 0;
    static constexpr size_t tensor_volume_ct = precompute_volume_ct(TensorShapeWrapper::elements);
    static constexpr size_t shard_volume_ct = precompute_volume_ct(ShardShapeWrapper::elements);
};

/**
 * @brief Helper function to build a DistributionSpec from commom runtime arguments. Parses tensor shape, shard shape,
 * bank coordinates if needed, and passes to DSpec constructor.
 *
 * @tparam DSpec DistributionSpec type
 * @return auto DistributionSpec instance built from common runtime arguments.
 */
template <typename ArgsOffsets>
auto build_dspec_from_args(const ArgsOffsets& args_offsets) {
    using Loc = typename ArgsOffsets::ArgsLoc;

    // Dispatch to the appropriate ShapeWrapper and BankCoordsWrapper types based on the "staticness"
    using TensorShapeType = typename ArrayWrapperTypeSelector<
        Loc::TensorShapeStatic,
        ArgsOffsets::TensorShapeCTAOffset,
        ArgsOffsets::RankCT>::type;
    using ShardShapeType = typename ArrayWrapperTypeSelector<
        Loc::ShardShapeStatic,
        ArgsOffsets::ShardShapeCTAOffset,
        ArgsOffsets::RankCT>::type;
    using BankCoordsType = typename ArrayWrapperTypeSelector<
        Loc::BankCoordsStatic,
        ArgsOffsets::BankCoordsCTAOffset,
        ArgsOffsets::NumBanksCT>::type;

    using DSpec =
        DistributionSpec<ArgsOffsets::RankCT, ArgsOffsets::NumBanksCT, TensorShapeType, ShardShapeType, BankCoordsType>;

    static constexpr bool TensorShapeCRTA = Loc::TensorShapeCRTA;
    static constexpr bool ShardShapeCRTA = Loc::ShardShapeCRTA;
    static constexpr bool BankCoordsCRTA = Loc::BankCoordsCRTA;

    auto rank = args_offsets.fetch_rank();
    auto num_banks = args_offsets.fetch_num_banks();

    ASSERT(rank > 0);
    ASSERT(num_banks > 0);

    typename DSpec::Shape tensor_shape_array;
    typename DSpec::Shape shard_shape_array;
    typename DSpec::Shape bank_coord_array;

    static_assert(
        Loc::RankStatic or !Loc::TensorShapeStatic, "Tensor shape must be CRTA if rank is not known at compile time!");
    static_assert(
        Loc::RankStatic or !Loc::ShardShapeStatic, "Shard shape must be CRTA if rank is not known at compile time!");
    static_assert(
        Loc::NumBanksStatic or !Loc::BankCoordsStatic,
        "Bank coords must be CRTA if num_banks is not known at compile time!");

    if constexpr (TensorShapeCRTA) {
        auto* tensor_shape_ptr = (uint32_t*)(get_common_arg_addr(args_offsets.tensor_shape_crta_offset()));
        tensor_shape_array = typename DSpec::Shape(tensor_shape_ptr, rank);
    }
    if constexpr (ShardShapeCRTA) {
        auto* shard_shape_ptr = (uint32_t*)(get_common_arg_addr(args_offsets.shard_shape_crta_offset()));
        shard_shape_array = typename DSpec::Shape(shard_shape_ptr, rank);
    }
    if constexpr (BankCoordsCRTA) {
        auto* bank_coords_ptr = (uint32_t*)(get_common_arg_addr(args_offsets.bank_coords_crta_offset()));
        bank_coord_array = typename DSpec::Shape(bank_coords_ptr, num_banks);
    }

    auto dspec = DSpec(std::move(tensor_shape_array), std::move(shard_shape_array), std::move(bank_coord_array));
    return dspec;
}

}  // namespace detail
}  // namespace nd_sharding
