// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include "helpers.hpp"
#include "shape_wrapper.hpp"
#include "bank_coords.hpp"
#include <hostdevcommon/flags.hpp>

namespace nd_sharding {
namespace detail {

constexpr size_t UNKNOWN = static_cast<size_t>(-1);

template <typename TensorShape, typename ShardShape, typename BankCoords, typename ArgsLoc>
struct DistributionSpec {
    using ArgumentsLocation = ArgsLoc;
    using TensorShapeT = TensorShape;
    using ShardShapeT = ShardShape;
    using BankCoordsT = BankCoords;

    // compile time shape/coords arrays with rank known at compile time
    using ShapeBase = typename TensorShape::ShapeBase;
    using PackedCoordsBase = typename BankCoords::PackedCoordsBase;

    static constexpr bool all_shapes_static = TensorShape::is_static && ShardShape::is_static;
    static constexpr bool bank_coords_static = BankCoords::is_static;
    static constexpr bool is_static = all_shapes_static && bank_coords_static;
    static constexpr bool has_static_rank = TensorShape::has_static_rank || ShardShape::has_static_rank;
    static constexpr bool has_static_num_banks = BankCoords::has_static_num_banks;

    // This constructor is only used for completely static DistributionSpec
    constexpr DistributionSpec() {
        static_assert(is_static, "Cannot use default constructor for non-static DistributionSpec!");
        static_assert(
            shard_grid_ct_[0] * shard_grid_strides_ct_[0] >= num_banks_ct,
            "Number of shards must be greater than or equal to number of banks!");
    };

    template <
        typename TensorShapeArr = ShapeBase,
        typename ShardShapeArr = ShapeBase,
        typename BankCoordsArr = PackedCoordsBase>
    constexpr DistributionSpec(
        TensorShapeArr&& tensor_shape_arr, ShardShapeArr&& shard_shape_arr = {}, BankCoordsArr&& bank_coords_arr = {}) :
        tensor_shape_(std::forward<TensorShapeArr>(tensor_shape_arr)),
        shard_shape_(std::forward<ShardShapeArr>(shard_shape_arr)),
        bank_coords_(std::forward<BankCoordsArr>(bank_coords_arr)) {
        if constexpr (!has_static_rank) {
            // Rank is not known at compile time, use runtime rank
            rank_rt = tensor_shape_.rank_rt;
            // !has_static_rank means ShapeBase is span<uint32_t>
            shard_grid_rt = ShapeBase(shard_grid_rt_buf.value, rank_rt);
            shard_grid_strides_rt = ShapeBase(shard_grid_strides_rt_buf.value, rank_rt);
        }
        if constexpr (!has_static_num_banks) {
            // Number of banks is not known at compile time, use runtime number of banks
            num_banks_rt = bank_coords_.num_banks_rt;
        }
        static_assert(!(is_static), "Everything is static, this constructor is obsolete!");
        if constexpr (TensorShape::has_static_rank and ShardShape::has_static_rank) {
            // If both tensor and shard ranks are static, they should be the same
            static_assert(TensorShape::rank == ShardShape::rank, "Tensor and shard shapes must have the same rank!");
            static_assert(
                std::is_same_v<typename TensorShape::ShapeBase, typename ShardShape::ShapeBase>,
                "Tensor and shard shapes bases must be the same");
        }
        if constexpr (!all_shapes_static) {
            // Tensor shape has bad rank!
            ASSERT(get_tensor_shape().size() == get_rank());
            // Shard shape has bad rank!
            ASSERT(get_shard_shape().size() == get_rank());
            compute_shard_grid_and_strides_rt(get_tensor_shape(), get_shard_shape());
        }
        if constexpr (!bank_coords_static) {
            // Number of bank coordinates must match the number of banks!"
            ASSERT(bank_coords_arr.size() == get_num_banks());
        }
    }

    constexpr const size_t get_rank() const {
        if constexpr (has_static_rank) {
            return rank_ct;
        } else {
            return rank_rt;
        }
    }

    constexpr const size_t get_num_banks() const {
        if constexpr (has_static_num_banks) {
            return num_banks_ct;
        } else {
            return num_banks_rt;
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

    constexpr const PackedCoordsBase& get_packed_xy_coords() const {
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
        for (int i = rank_ct - 1; i >= 0; --i) {
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

    static constexpr auto rank_ct = TensorShape::has_static_rank  ? TensorShape::rank
                                    : ShardShape::has_static_rank ? ShardShape::rank
                                                                  : UNKNOWN;
    static constexpr size_t num_banks_ct = BankCoords::has_static_num_banks ? BankCoords::num_banks : UNKNOWN;
    size_t rank_rt = 0;
    size_t num_banks_rt = 0;

    std::conditional_t<all_shapes_static, std::monostate, ShapeBase> shard_grid_rt{};
    std::conditional_t<all_shapes_static, std::monostate, ShapeBase> shard_grid_strides_rt{};
    mutable detail::ConditionalBuffer<!has_static_rank, uint32_t, MAX_RANK> shard_grid_rt_buf;
    mutable detail::ConditionalBuffer<!has_static_rank, uint32_t, MAX_RANK> shard_grid_strides_rt_buf;

    static constexpr ShapeBase shard_grid_ct_ = shard_grid_ct(TensorShape::shape, ShardShape::shape);
    static constexpr ShapeBase shard_grid_strides_ct_ = shard_grid_strides_ct(TensorShape::shape, ShardShape::shape);

    TensorShape tensor_shape_;  // Tensor shape
    ShardShape shard_shape_;    // Shard shape
    BankCoords bank_coords_;    // Bank coordinates

    static constexpr size_t fetch_rank() {
        if constexpr (has_static_rank) {
            return rank_ct;
        } else {
            uint32_t rank_crta_offset = ArgumentsLocation::CRTA_BASE;
            return get_common_arg_val<uint32_t>(rank_crta_offset);
        }
    }

    static constexpr size_t fetch_num_banks() {
        if constexpr (has_static_num_banks) {
            return num_banks_ct;
        } else {
            uint32_t num_banks_crta_offset = ArgumentsLocation::CRTA_BASE + ArgumentsLocation::RankCRTA;
            return get_common_arg_val<uint32_t>(num_banks_crta_offset);
        }
    }
};

// TODO: Should we merge ShapeWrapperTypeSelector and BankCoordsWrapperTypeSelector into one?
template <bool RankCTA, bool ShapeCTA, size_t CTA_BASE, size_t Rank>
struct ShapeWrapperTypeSelector;

template <size_t CTA_BASE, size_t Rank>
struct ShapeWrapperTypeSelector<true, true, CTA_BASE, Rank> {
    // Both rank and dims are known at compile time -- we can construct a static wrapper
    using type = struct_cta_sequence_wrapper_t<ShapeWrapperStaticDimsStaticRank, CTA_BASE, Rank>;
};

template <size_t CTA_BASE, size_t Rank>
struct ShapeWrapperTypeSelector<true, false, CTA_BASE, Rank> {
    // Rank is known at compile time, but dims are not
    using type = ShapeWrapperDynamicDimsStaticRank<Rank>;
};

template <bool ShapeCTA, size_t CTA_BASE, size_t Rank>
struct ShapeWrapperTypeSelector<false, ShapeCTA, CTA_BASE, Rank> {
    // Rank is not known at compile time, doesn't matter if dims are known or not, use poorly dynamic wrapper
    using type = ShapeWrapperDynamicRank;
};

template <bool NumBanksCTA, bool BankCoordsCTA, size_t CTA_BASE, size_t NumBanks>
struct BankCoordsWrapperTypeSelector;

template <size_t CTA_BASE, size_t NumBanks>
struct BankCoordsWrapperTypeSelector<true, true, CTA_BASE, NumBanks> {
    // Both num_banks and coords are known at compile time -- we can construct a static wrapper
    using type = struct_cta_sequence_wrapper_t<BankCoordWrapperStaticNBanksStaticCoords, CTA_BASE, NumBanks>;
};

template <size_t CTA_BASE, size_t NumBanks>
struct BankCoordsWrapperTypeSelector<true, false, CTA_BASE, NumBanks> {
    // Num_banks is known at compile time, but coords are not
    using type = BankCoordWrapperDynamicStaticNBanksDynamicCoords<NumBanks>;
};

template <bool BankCoordsCTA, size_t CTA_BASE, size_t NumBanks>
struct BankCoordsWrapperTypeSelector<false, BankCoordsCTA, CTA_BASE, NumBanks> {
    // Num_banks is not known at compile time, doesn't matter if coords are known or not, use poorly dynamic wrapper
    using type = BankCoordWrapperDynamicsNBanks;
};

namespace {
// TODO: This exact enum is defined on host. Maybe somehow reuse it?
enum class ArgConfig : uint8_t {
    CTA = 0,
    RankCRTA = 1 << 0,
    NumBanksCRTA = 1 << 1,
    TensorShapeCRTA = 1 << 2,
    ShardShapeCRTA = 1 << 3,
    BankCoordsCRTA = 1 << 4,
    CRTA = RankCRTA | NumBanksCRTA | TensorShapeCRTA | ShardShapeCRTA | BankCoordsCRTA
};

using ArgsConfig = Flags<ArgConfig>;
constexpr ArgsConfig operator|(ArgConfig a, ArgConfig b) noexcept { return ArgsConfig(a) | b; }
constexpr ArgsConfig operator|(ArgConfig a, ArgsConfig b) noexcept { return ArgsConfig(a) | b; }
}  // namespace

// compile-time helper to figure out if argument is compile-time or common runtime
template <size_t CTA_BASE_, size_t CRTA_BASE_ = static_cast<size_t>(-1)>
struct ArgsLocation {
    static constexpr size_t CTA_BASE = CTA_BASE_;
    static constexpr size_t CRTA_BASE = CRTA_BASE_;

    static constexpr auto args_config =
        ArgsConfig(static_cast<ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_BASE)));

    static constexpr bool RankCRTA = args_config.test(ArgConfig::RankCRTA);
    static constexpr bool NumBanksCRTA = args_config.test(ArgConfig::NumBanksCRTA);
    static constexpr bool TensorShapeCRTA = args_config.test(ArgConfig::TensorShapeCRTA);
    static constexpr bool ShardShapeCRTA = args_config.test(ArgConfig::ShardShapeCRTA);
    static constexpr bool BankCoordsCRTA = args_config.test(ArgConfig::BankCoordsCRTA);
};

// Helper to properly build a DistributionSpec type from compile-time arguments
template <size_t CTA_BASE, size_t CRTA_BASE>
struct DistributionSpecWrapper {
    // Fetch configuration of arguments, i.e. which are CTA which are CRTA
    using ArgsLoc = ArgsLocation<CTA_BASE, CRTA_BASE>;

    constexpr static bool RankStatic = !ArgsLoc::RankCRTA;
    constexpr static bool NumBanksStatic = !ArgsLoc::NumBanksCRTA;
    // For shapes and bank coords to be static, both rank and dims/coords must be static
    constexpr static bool TensorShapeStatic = !ArgsLoc::TensorShapeCRTA and RankStatic;
    constexpr static bool ShardShapeStatic = !ArgsLoc::ShardShapeCRTA and RankStatic;
    constexpr static bool BankCoordsStatic = !ArgsLoc::BankCoordsCRTA and NumBanksStatic;

    static_assert(
        RankStatic or (!RankStatic and !TensorShapeStatic and !ShardShapeStatic),
        "If rank is not static, tensor and shard shapes must be dynamic!");
    static_assert(
        NumBanksStatic or (!NumBanksStatic and !BankCoordsStatic),
        "If number of banks is not static, bank coordinates must be dynamic!");

    // Figure out locations (offsets) of compile-time arguments (if they are compile time)
    constexpr static size_t NEW_CTA_BASE = CTA_BASE + 1;  // +1 for args_config
    constexpr static size_t RankBase = NEW_CTA_BASE;
    constexpr static size_t NumBanksBase = RankBase + (RankStatic ? 1 : 0);

    constexpr static size_t RANK = RankStatic ? get_compile_time_arg_val(RankBase) : 0;
    constexpr static size_t NUM_BANKS = NumBanksStatic ? get_compile_time_arg_val(NumBanksBase) : 0;

    constexpr static size_t TensorShapeBase = NumBanksBase + (NumBanksStatic ? 1 : 0);
    static constexpr size_t ShardShapeBase = TensorShapeBase + (TensorShapeStatic ? RANK : 0);
    static constexpr size_t BankCoordsBase = ShardShapeBase + (ShardShapeStatic ? RANK : 0);

    using TensorShapeType =
        typename ShapeWrapperTypeSelector<RankStatic, TensorShapeStatic, TensorShapeBase, RANK>::type;
    using ShardShapeType = typename ShapeWrapperTypeSelector<RankStatic, ShardShapeStatic, ShardShapeBase, RANK>::type;
    using BankCoordsType =
        typename BankCoordsWrapperTypeSelector<NumBanksStatic, BankCoordsStatic, BankCoordsBase, NUM_BANKS>::type;

    using dspec = DistributionSpec<TensorShapeType, ShardShapeType, BankCoordsType, ArgsLoc>;
};

template <typename DSpec>
auto build_dspec_from_args() {
    static constexpr bool TensorShapeCRTA = !DSpec::TensorShapeT::is_static;
    static constexpr bool ShardShapeCRTA = !DSpec::ShardShapeT::is_static;
    static constexpr bool BankCoordsCRTA = !DSpec::BankCoordsT::is_static;
    static constexpr size_t CRTA_BASE = DSpec::ArgumentsLocation::CRTA_BASE;
    static constexpr bool RankStatic = DSpec::has_static_rank;
    static constexpr bool NumBanksStatic = DSpec::has_static_num_banks;

    // Rank known at compile time, but shapes and bank coords possibly not
    typename DSpec::ShapeBase tensor_shape_array;
    typename DSpec::ShapeBase shard_shape_array;
    typename DSpec::PackedCoordsBase bank_coord_array;

    // Calculate CRTA offsets that can be calculated at compile time
    static constexpr size_t rank_crta_offset = CRTA_BASE;
    static constexpr size_t num_banks_crta_offset = CRTA_BASE + (!RankStatic ? 1 : 0);
    static constexpr size_t tensor_shape_crta_offset = num_banks_crta_offset + (!NumBanksStatic ? 1 : 0);

    size_t rank = DSpec::fetch_rank();
    size_t num_banks = DSpec::fetch_num_banks();

    if constexpr (RankStatic) {
        // In such case shape base is std::array<uint32_t, RANK>
        if constexpr (TensorShapeCRTA) {
            tensor_shape_array = array_crta_sequence_wrapper<tensor_shape_crta_offset, DSpec::rank_ct>();
        }
        if constexpr (ShardShapeCRTA) {
            shard_shape_array = array_crta_sequence_wrapper<
                tensor_shape_crta_offset + DSpec::rank_ct * TensorShapeCRTA,
                DSpec::rank_ct>();
        }
    } else {
        // In such case shape base is Span<uint32_t>
        static_assert(TensorShapeCRTA, "Tensor shape must be CRTA if rank is not known at compile time!");
        static_assert(ShardShapeCRTA, "Shard shape must be CRTA if rank is not known at compile time!");

        size_t shard_shape_crta_offset = tensor_shape_crta_offset + (TensorShapeCRTA ? rank : 0);

        if constexpr (TensorShapeCRTA) {
            // TODO: (C)RTA are contiguous in memory? verify that
            auto* tensor_shape_ptr = (uint32_t*)(get_common_arg_addr(tensor_shape_crta_offset));
            tensor_shape_array = typename DSpec::ShapeBase(tensor_shape_ptr, rank);
        }
        if constexpr (ShardShapeCRTA) {
            auto* shard_shape_ptr = (uint32_t*)(get_common_arg_addr(shard_shape_crta_offset));
            shard_shape_array = typename DSpec::ShapeBase(shard_shape_ptr, rank);
        }
    }

    size_t bank_coords_crta_offset =
        tensor_shape_crta_offset + (TensorShapeCRTA ? rank : 0) + (ShardShapeCRTA ? rank : 0);
    if constexpr (NumBanksStatic) {
        // In such case packed coords base is std::array<uint32_t, NUM_BANKS>
        if constexpr (BankCoordsCRTA) {
            for (size_t i = 0; i < num_banks; ++i) {
                bank_coord_array[i] =
                    get_common_arg_val<uint32_t>(bank_coords_crta_offset + i);  // Get packed coords from CRTA
            }
        }
    } else {
        // In such case packed coords base is Span<uint32_t>
        // TODO: figure out how to handle case of CRTA num_banks and CTA bank coords
        static_assert(BankCoordsCRTA, "Bank coords must be RTA if rank is not known at compile time!");
        if constexpr (BankCoordsCRTA) {
            auto* bank_coords_ptr = (uint32_t*)(get_common_arg_addr(bank_coords_crta_offset));
            bank_coord_array = typename DSpec::PackedCoordsBase(bank_coords_ptr, num_banks);
        }
    }
    auto dspec = DSpec(std::move(tensor_shape_array), std::move(shard_shape_array), std::move(bank_coord_array));
    return dspec;
}

}  // namespace detail
}  // namespace nd_sharding
