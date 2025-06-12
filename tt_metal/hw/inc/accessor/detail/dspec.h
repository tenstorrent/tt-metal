// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include "helpers.hpp"
#include "const.hpp"
#include "shape_wrapper.hpp"
#include "bank_coords_wrapper.hpp"
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
template <typename TensorShapeWrapper_, typename ShardShapeWrapper_, typename BankCoordsWrapper_, typename ArgsLoc_>
struct DistributionSpec {
    using ArgsLoc = ArgsLoc_;
    using TensorShapeWrapper = TensorShapeWrapper_;
    using ShardShapeWrapper = ShardShapeWrapper_;
    using BankCoordsWrapper = BankCoordsWrapper_;

    // std::array if rank/num_banks are static, Span otherwise
    using ShapeBase = typename TensorShapeWrapper::ShapeBase;
    using PackedCoordsBase = typename BankCoordsWrapper::PackedCoordsBase;

    static constexpr bool has_static_rank = ArgsLoc::RankStatic;
    static constexpr bool has_static_num_banks = ArgsLoc::NumBanksStatic;

    static constexpr bool tensor_shape_static = has_static_rank && ArgsLoc::TensorShapeStatic;
    static constexpr bool shard_shape_static = has_static_rank && ArgsLoc::ShardShapeStatic;
    static constexpr bool bank_coords_static = has_static_num_banks && ArgsLoc::BankCoordsStatic;

    static constexpr bool shapes_static = has_static_rank && tensor_shape_static && shard_shape_static;
    static constexpr bool is_static = shapes_static && bank_coords_static;

    static constexpr auto rank_ct = has_static_rank ? TensorShapeWrapper::rank : detail::UNKNOWN;
    static constexpr uint32_t num_banks_ct = has_static_num_banks ? BankCoordsWrapper::num_banks : detail::UNKNOWN;

    // This constructor is only used for completely static DistributionSpec
    template <typename T = void, typename = std::enable_if_t<is_static, T>>
    constexpr DistributionSpec() {
        static_assert(
            shard_grid_ct[0] * shard_grid_strides_ct[0] >= num_banks_ct,
            "Number of shards must be greater than or equal to number of banks!");
    };

    template <
        typename TensorShapeArr = ShapeBase,
        typename ShardShapeArr = ShapeBase,
        typename BankCoordsArr = PackedCoordsBase>
    // typename T = void,
    // typename = std::enable_if_t<!is_static, T>>
    constexpr DistributionSpec(
        TensorShapeArr&& tensor_shape_arr, ShardShapeArr&& shard_shape_arr = {}, BankCoordsArr&& bank_coords_arr = {}) :
        tensor_shape_rt(std::forward<TensorShapeArr>(tensor_shape_arr)),
        shard_shape_rt(std::forward<ShardShapeArr>(shard_shape_arr)),
        bank_coords_rt(std::forward<BankCoordsArr>(bank_coords_arr)) {
        if constexpr (!has_static_rank) {
            // Rank is not known at compile time, use runtime rank
            rank_rt = tensor_shape_rt.shape.size();
            // !has_static_rank means ShapeBase is span<uint32_t>
            shard_grid_rt = ShapeBase(shard_grid_rt_buf.value, rank_rt);
            shard_grid_strides_rt = ShapeBase(shard_grid_strides_rt_buf.value, rank_rt);

            tensor_strides_rt = ShapeBase(tensor_strides_rt_buf.value, rank_rt);
            shard_strides_rt = ShapeBase(shard_strides_rt_buf.value, rank_rt);
        }
        if constexpr (!has_static_num_banks) {
            // Number of banks is not known at compile time, use runtime number of banks
            num_banks_rt = bank_coords_rt.packed_xy_coords.size();
        }
        if constexpr (!tensor_shape_static) {
            // If tensor shape is not static, we need to compute strides and volume at runtime
            compute_strides_volume_rt(tensor_shape_rt.shape, tensor_strides_rt, tensor_volume_rt);
        }
        if constexpr (!shard_shape_static) {
            // If shard shape is not static, we need to compute strides and volume at runtime
            compute_strides_volume_rt(shard_shape_rt.shape, shard_strides_rt, shard_volume_rt);
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
    constexpr const uint32_t get_rank() const { getter_helper(has_static_rank, rank_ct, rank_rt); }

    constexpr const uint32_t get_num_banks() const { getter_helper(has_static_num_banks, num_banks_ct, num_banks_rt) }

    constexpr const ShapeBase& get_shard_grid() const { getter_helper(shapes_static, shard_grid_ct, shard_grid_rt) }

    constexpr const ShapeBase& get_shard_grid_strides() const {
        getter_helper(shapes_static, shard_grid_strides_ct, shard_grid_strides_rt)
    }

    constexpr const ShapeBase& get_tensor_shape() const {
        getter_helper(tensor_shape_static, TensorShapeWrapper::shape, tensor_shape_rt.shape)
    }

    constexpr const ShapeBase& get_tensor_strides() const {
        getter_helper(tensor_shape_static, tensor_strides_ct, tensor_strides_rt)
    }

    constexpr size_t get_tensor_volume() const {
        getter_helper(tensor_shape_static, tensor_volume_ct, tensor_volume_rt)
    }

    constexpr const ShapeBase& get_shard_shape() const {
        getter_helper(shard_shape_static, ShardShapeWrapper::shape, shard_shape_rt.shape)
    }

    constexpr const ShapeBase& get_shard_strides() const {
        getter_helper(shard_shape_static, shard_strides_ct, shard_strides_rt)
    }

    constexpr size_t get_shard_volume() const { getter_helper(shard_shape_static, shard_volume_ct, shard_volume_rt) }

    constexpr const PackedCoordsBase& get_packed_xy_coords() const {
        getter_helper(bank_coords_static, BankCoordsWrapper::packed_xy_coords, bank_coords_rt.packed_xy_coords)
    }

#undef getter_helper

private:
    static constexpr ShapeBase precompute_shard_grid_ct(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        // If shapes are dynamic, we cannot compute shard grid at compile time
        if (!shapes_static) {
            return {};
        }
        ShapeBase shard_grid = {};
        for (int i = rank_ct - 1; i >= 0; --i) {
            shard_grid[i] = (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid;
    }

    static constexpr ShapeBase precompute_shard_grid_strides_ct(
        const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        ShapeBase shard_grid_strides = {};
        uint32_t stride = 1;
        for (int i = rank_ct - 1; i >= 0; --i) {
            shard_grid_strides[i] = stride;
            stride *= (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid_strides;
    }

    static constexpr size_t precompute_volume_ct(const ShapeBase& shape) {
        size_t volume = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            volume *= shape[i];
        }
        return volume;
    }

    static constexpr ShapeBase precompute_strides_ct(const ShapeBase& shape) {
        ShapeBase strides = {};
        uint32_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    void compute_strides_volume_rt(const ShapeBase& shape, ShapeBase& strides, size_t& volume) const {
        uint32_t stride = 1;
        volume = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
            volume *= shape[i];
        }
    }

    void compute_shard_grid_and_strides_rt(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
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

    TensorShapeWrapper tensor_shape_rt;
    ShardShapeWrapper shard_shape_rt;
    BankCoordsWrapper bank_coords_rt;

    std::conditional_t<shapes_static, std::monostate, ShapeBase> shard_grid_rt{};
    std::conditional_t<shapes_static, std::monostate, ShapeBase> shard_grid_strides_rt{};

    // Buffers to wrap around span in case of dynamic rank
    mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]> shard_grid_rt_buf;
    mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]> shard_grid_strides_rt_buf;

    static constexpr ShapeBase shard_grid_ct =
        precompute_shard_grid_ct(TensorShapeWrapper::shape, ShardShapeWrapper::shape);
    static constexpr ShapeBase shard_grid_strides_ct =
        precompute_shard_grid_strides_ct(TensorShapeWrapper::shape, ShardShapeWrapper::shape);

    mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]> tensor_strides_rt_buf;
    mutable detail::ConditionalField<!has_static_rank, uint32_t[MAX_RANK]> shard_strides_rt_buf;
    ShapeBase tensor_strides_rt = {};
    ShapeBase shard_strides_rt = {};
    static constexpr ShapeBase tensor_strides_ct = precompute_strides_ct(TensorShapeWrapper::shape);
    static constexpr ShapeBase shard_strides_ct = precompute_strides_ct(ShardShapeWrapper::shape);

    size_t tensor_volume_rt = 0;
    size_t shard_volume_rt = 0;
    static constexpr size_t tensor_volume_ct = precompute_volume_ct(TensorShapeWrapper::shape);
    static constexpr size_t shard_volume_ct = precompute_volume_ct(ShardShapeWrapper::shape);
};

/**
 * @brief Helper function to build a DistributionSpec from commom runtime arguments. Parses tensor shape, shard shape,
 * bank coordinates if needed, and passes to DSpec constructor.
 *
 * @tparam DSpec DistributionSpec type
 * @return auto DistributionSpec instance built from common runtime arguments.
 */
template <typename ArgsOffsets>
auto build_dspec_from_args_proxy(const ArgsOffsets& args_offsets) {
    using Loc = typename ArgsOffsets::ArgsLoc;

    // Dispatch to the appropriate ShapeWrapper and BankCoordsWrapper types based on the "staticness"
    using TensorShapeType = typename ShapeWrapperTypeSelector<
        Loc::RankStatic,
        Loc::TensorShapeStatic,
        ArgsOffsets::TensorShapeCTAOffset,
        ArgsOffsets::RankCT>::type;
    using ShardShapeType = typename ShapeWrapperTypeSelector<
        Loc::RankStatic,
        Loc::ShardShapeStatic,
        ArgsOffsets::ShardShapeCTAOffset,
        ArgsOffsets::RankCT>::type;
    using BankCoordsType = typename BankCoordsWrapperTypeSelector<
        Loc::NumBanksStatic,
        Loc::BankCoordsStatic,
        ArgsOffsets::BankCoordsCTAOffset,
        ArgsOffsets::NumBanksCT>::type;

    using DSpec = DistributionSpec<TensorShapeType, ShardShapeType, BankCoordsType, Loc>;

    static constexpr bool TensorShapeCRTA = Loc::TensorShapeCRTA;
    static constexpr bool ShardShapeCRTA = Loc::ShardShapeCRTA;
    static constexpr bool BankCoordsCRTA = Loc::BankCoordsCRTA;

    auto rank = args_offsets.fetch_rank();
    auto num_banks = args_offsets.fetch_num_banks();

    ASSERT(rank > 0);
    ASSERT(num_banks > 0);

    // DSpec::ShapeBase == std::array<uint32_t, RANK> if RankStatic is true, otherwise it is Span<uint32_t>
    typename DSpec::ShapeBase tensor_shape_array;
    typename DSpec::ShapeBase shard_shape_array;
    typename DSpec::PackedCoordsBase bank_coord_array;

    // Construct tensor and shard shapes
    if constexpr (Loc::RankStatic) {
        // In such case shape base is std::array<uint32_t, RANK>
        if constexpr (TensorShapeCRTA) {
            auto tensor_shape_crta_offset = args_offsets.tensor_shape_crta_offset();
            for (size_t i = 0; i < rank; ++i) {
                tensor_shape_array[i] = get_common_arg_val<uint32_t>(tensor_shape_crta_offset + i);
                ASSERT(tensor_shape_array[i] > 0);
            }
        }
        if constexpr (ShardShapeCRTA) {
            auto shard_shape_crta_offset = args_offsets.shard_shape_crta_offset();
            for (size_t i = 0; i < rank; ++i) {
                shard_shape_array[i] = get_common_arg_val<uint32_t>(shard_shape_crta_offset + i);
                ASSERT(shard_shape_array[i] > 0);
            }
        }
    } else {
        // In such case shape base is Span<uint32_t>
        static_assert(TensorShapeCRTA, "Tensor shape must be CRTA if rank is not known at compile time!");
        static_assert(ShardShapeCRTA, "Shard shape must be CRTA if rank is not known at compile time!");

        // (C)RTA are contiguous in memory, so we can do 0-copy construction from pointer to first value
        if constexpr (TensorShapeCRTA) {
            auto* tensor_shape_ptr = (uint32_t*)(get_common_arg_addr(args_offsets.tensor_shape_crta_offset()));
            tensor_shape_array = typename DSpec::ShapeBase(tensor_shape_ptr, rank);
        }
        if constexpr (ShardShapeCRTA) {
            auto* shard_shape_ptr = (uint32_t*)(get_common_arg_addr(args_offsets.shard_shape_crta_offset()));
            shard_shape_array = typename DSpec::ShapeBase(shard_shape_ptr, rank);
        }
    }

    // Construct bank coordinates
    if constexpr (Loc::NumBanksStatic) {
        // In such case packed coords base is std::array<uint32_t, NUM_BANKS>
        if constexpr (BankCoordsCRTA) {
            for (size_t i = 0; i < num_banks; ++i) {
                bank_coord_array[i] = get_common_arg_val<uint32_t>(
                    args_offsets.bank_coords_crta_offset() + i);  // Get packed coords from CRTA
            }
        }
    } else {
        // In such case packed coords base is Span<uint32_t>
        // TODO: figure out how to handle case of CRTA num_banks and CTA bank coords
        static_assert(BankCoordsCRTA, "Bank coords must be RTA if rank is not known at compile time!");
        if constexpr (BankCoordsCRTA) {
            auto* bank_coords_ptr = (uint32_t*)(get_common_arg_addr(args_offsets.bank_coords_crta_offset()));
            bank_coord_array = typename DSpec::PackedCoordsBase(bank_coords_ptr, num_banks);
        }
    }

    auto dspec = DSpec(std::move(tensor_shape_array), std::move(shard_shape_array), std::move(bank_coord_array));
    return dspec;
}

}  // namespace detail
}  // namespace nd_sharding
