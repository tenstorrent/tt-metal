// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include "helpers.hpp"
#include "shape_wrapper.hpp"
#include "bank_coords_wrapper.hpp"
#include "args_location.hpp"

namespace nd_sharding {
namespace detail {

constexpr size_t UNKNOWN = static_cast<size_t>(-1);

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
template <typename TensorShape_, typename ShardShape_, typename BankCoords_, typename ArgsLoc_>
struct DistributionSpec {
    using ArgsLoc = ArgsLoc_;
    using TensorShape = TensorShape_;
    using ShardShape = ShardShape_;
    using BankCoords = BankCoords_;

    // std::array if rank/num_banks are static, Span otherwise
    using ShapeBase = typename TensorShape::ShapeBase;
    using PackedCoordsBase = typename BankCoords::PackedCoordsBase;

    static constexpr bool has_static_rank = ArgsLoc::RankStatic;
    static constexpr bool has_static_num_banks = ArgsLoc::NumBanksStatic;

    static constexpr bool tensor_shape_static = has_static_rank && ArgsLoc::TensorShapeStatic;
    static constexpr bool shard_shape_static = has_static_rank && ArgsLoc::ShardShapeStatic;
    static constexpr bool bank_coords_static = has_static_num_banks && ArgsLoc::BankCoordsStatic;

    static constexpr bool shapes_static = has_static_rank && tensor_shape_static && shard_shape_static;
    static constexpr bool is_static = shapes_static && bank_coords_static;

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
        typename BankCoordsArr = PackedCoordsBase,
        typename T = void,
        typename = std::enable_if_t<!is_static, T>>
    constexpr DistributionSpec(
        TensorShapeArr&& tensor_shape_arr, ShardShapeArr&& shard_shape_arr = {}, BankCoordsArr&& bank_coords_arr = {}) :
        tensor_shape_rt(std::forward<TensorShapeArr>(tensor_shape_arr)),
        shard_shape_rt(std::forward<ShardShapeArr>(shard_shape_arr)),
        bank_coords_rt(std::forward<BankCoordsArr>(bank_coords_arr)) {
        if constexpr (!has_static_rank) {
            // Rank is not known at compile time, use runtime rank
            rank_rt = tensor_shape_rt.rank_rt;
            // !has_static_rank means ShapeBase is span<uint32_t>
            shard_grid_rt = ShapeBase(shard_grid_rt_buf.value, rank_rt);
            shard_grid_strides_rt = ShapeBase(shard_grid_strides_rt_buf.value, rank_rt);
        }
        if constexpr (!has_static_num_banks) {
            // Number of banks is not known at compile time, use runtime number of banks
            num_banks_rt = bank_coords_rt.num_banks_rt;
        }
        if constexpr (!shapes_static) {
            compute_shard_grid_and_strides_rt(get_tensor_shape(), get_shard_shape());
        }
    }

    // Getters
    constexpr const uint32_t get_rank() const {
        if constexpr (has_static_rank) {
            return rank_ct;
        } else {
            return rank_rt;
        }
    }

    constexpr const uint32_t get_num_banks() const {
        if constexpr (has_static_num_banks) {
            return num_banks_ct;
        } else {
            return num_banks_rt;
        }
    }

    constexpr const ShapeBase& get_shard_grid() const {
        if constexpr (shapes_static) {
            return shard_grid_ct;
        } else {
            return shard_grid_rt;
        }
    }

    constexpr const ShapeBase& get_shard_grid_strides() const {
        if constexpr (shapes_static) {
            return shard_grid_strides_ct;
        } else {
            return shard_grid_strides_rt;
        }
    }

    constexpr const ShapeBase& get_tensor_shape() const {
        if constexpr (tensor_shape_static) {
            return TensorShape::shape;
        } else {
            return tensor_shape_rt.shape;
        }
    }

    constexpr const ShapeBase& get_tensor_strides() const {
        if constexpr (tensor_shape_static) {
            return TensorShape::strides;
        } else {
            return tensor_shape_rt.strides;
        }
    }

    constexpr size_t get_tensor_volume() const {
        if constexpr (tensor_shape_static) {
            return TensorShape::volume;
        } else {
            return tensor_shape_rt.volume;
        }
    }

    constexpr const ShapeBase& get_shard_shape() const {
        if constexpr (shard_shape_static) {
            return ShardShape::shape;
        } else {
            return shard_shape_rt.shape;
        }
    }

    constexpr const ShapeBase& get_shard_strides() const {
        if constexpr (shard_shape_static) {
            return ShardShape::strides;
        } else {
            return shard_shape_rt.strides;
        }
    }

    constexpr size_t get_shard_volume() const {
        if constexpr (shard_shape_static) {
            return ShardShape::volume;
        } else {
            return shard_shape_rt.volume;
        }
    }

    constexpr const PackedCoordsBase& get_packed_xy_coords() const {
        if constexpr (bank_coords_static) {
            return BankCoords::packed_xy_coords;
        } else {
            return bank_coords_rt.packed_xy_coords;
        }
    }

    // Compute shard grid and shard grid strides at compile time
    static constexpr ShapeBase calc_shard_grid_ct(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
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

    static constexpr ShapeBase calc_shard_grid_strides_ct(const ShapeBase& tensor_shape, const ShapeBase& shard_shape) {
        ShapeBase shard_grid_strides = {};
        uint32_t stride = 1;
        for (int i = rank_ct - 1; i >= 0; --i) {
            shard_grid_strides[i] = stride;
            stride *= (tensor_shape[i] - 1) / shard_shape[i] + 1;  // div_up
        }
        return shard_grid_strides;
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

    static constexpr auto rank_ct = ArgsLoc::RankCT;
    static constexpr uint32_t num_banks_ct = ArgsLoc::NumBanksCT;
    uint32_t rank_rt = 0;
    uint32_t num_banks_rt = 0;

    TensorShape tensor_shape_rt;
    ShardShape shard_shape_rt;
    BankCoords bank_coords_rt;

    std::conditional_t<shapes_static, std::monostate, ShapeBase> shard_grid_rt{};
    std::conditional_t<shapes_static, std::monostate, ShapeBase> shard_grid_strides_rt{};

    // Buffers to wrap around span in case of dynamic rank
    mutable detail::ConditionalBuffer<!has_static_rank, uint32_t, MAX_RANK> shard_grid_rt_buf;
    mutable detail::ConditionalBuffer<!has_static_rank, uint32_t, MAX_RANK> shard_grid_strides_rt_buf;

    static constexpr ShapeBase shard_grid_ct = calc_shard_grid_ct(TensorShape::shape, ShardShape::shape);
    static constexpr ShapeBase shard_grid_strides_ct =
        calc_shard_grid_strides_ct(TensorShape::shape, ShardShape::shape);
};

// Helper to properly build a DistributionSpec type from compile-time arguments
template <size_t CTA_OFFSET, size_t CRTA_OFFSET>
struct BuildDistributionSpec {
    // Fetch configuration of arguments, i.e. which are CTA which are CRTA
    using ArgsLoc = ArgsLocation<CTA_OFFSET, CRTA_OFFSET>;

    // Dispatch to the appropriate ShapeWrapper and BankCoordsWrapper types based on the "staticness"
    using TensorShapeType = typename ShapeWrapperTypeSelector<
        ArgsLoc::RankStatic,
        ArgsLoc::TensorShapeStatic,
        ArgsLoc::TensorShapeStaticOffset,
        ArgsLoc::RankCT>::type;
    using ShardShapeType = typename ShapeWrapperTypeSelector<
        ArgsLoc::RankStatic,
        ArgsLoc::ShardShapeStatic,
        ArgsLoc::ShardShapeStaticOffset,
        ArgsLoc::RankCT>::type;
    using BankCoordsType = typename BankCoordsWrapperTypeSelector<
        ArgsLoc::NumBanksStatic,
        ArgsLoc::BankCoordsStatic,
        ArgsLoc::BankCoordsStaticOffset,
        ArgsLoc::NumBanksCT>::type;

    using dspec = DistributionSpec<TensorShapeType, ShardShapeType, BankCoordsType, ArgsLoc>;
};

template <typename DSpec>
auto build_dspec_from_args() {
    using Loc = typename DSpec::ArgsLoc;
    static constexpr bool TensorShapeCRTA = Loc::TensorShapeCRTA;
    static constexpr bool ShardShapeCRTA = Loc::ShardShapeCRTA;
    static constexpr bool BankCoordsCRTA = Loc::BankCoordsCRTA;

    auto rank = Loc::fetch_rank();
    auto num_banks = Loc::fetch_num_banks();

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
            auto tensor_shape_crta_offset = Loc::tensor_shape_crta_offset();
            for (size_t i = 0; i < rank; ++i) {
                tensor_shape_array[i] = get_common_arg_val<uint32_t>(tensor_shape_crta_offset + i);
            }
        }
        if constexpr (ShardShapeCRTA) {
            auto shard_shape_crta_offset = Loc::shard_shape_crta_offset();
            for (size_t i = 0; i < rank; ++i) {
                shard_shape_array[i] = get_common_arg_val<uint32_t>(shard_shape_crta_offset + i);
            }
        }
    } else {
        // In such case shape base is Span<uint32_t>
        static_assert(TensorShapeCRTA, "Tensor shape must be CRTA if rank is not known at compile time!");
        static_assert(ShardShapeCRTA, "Shard shape must be CRTA if rank is not known at compile time!");

        // (C)RTA are contiguous in memory, so we can do 0-copy construction
        if constexpr (TensorShapeCRTA) {
            auto* tensor_shape_ptr = (uint32_t*)(get_common_arg_addr(Loc::tensor_shape_crta_offset()));
            tensor_shape_array = typename DSpec::ShapeBase(tensor_shape_ptr, rank);
        }
        if constexpr (ShardShapeCRTA) {
            auto* shard_shape_ptr = (uint32_t*)(get_common_arg_addr(Loc::shard_shape_crta_offset()));
            shard_shape_array = typename DSpec::ShapeBase(shard_shape_ptr, rank);
        }
    }

    // Construct bank coordinates
    if constexpr (Loc::NumBanksStatic) {
        // In such case packed coords base is std::array<uint32_t, NUM_BANKS>
        if constexpr (BankCoordsCRTA) {
            for (size_t i = 0; i < num_banks; ++i) {
                bank_coord_array[i] =
                    get_common_arg_val<uint32_t>(Loc::bank_coords_crta_offset() + i);  // Get packed coords from CRTA
            }
        }
    } else {
        // In such case packed coords base is Span<uint32_t>
        // TODO: figure out how to handle case of CRTA num_banks and CTA bank coords
        static_assert(BankCoordsCRTA, "Bank coords must be RTA if rank is not known at compile time!");
        if constexpr (BankCoordsCRTA) {
            auto* bank_coords_ptr = (uint32_t*)(get_common_arg_addr(Loc::bank_coords_crta_offset()));
            bank_coord_array = typename DSpec::PackedCoordsBase(bank_coords_ptr, num_banks);
        }
    }

    auto dspec = DSpec(std::move(tensor_shape_array), std::move(shard_shape_array), std::move(bank_coord_array));
    return dspec;
}

}  // namespace detail
}  // namespace nd_sharding
