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
#include <hostdevcommon/flags.hpp>

namespace nd_sharding {
namespace detail {

constexpr size_t UNKNOWN = static_cast<size_t>(-1);

template <typename TensorShape_, typename ShardShape_, typename BankCoords_, typename ArgsLoc_>
struct DistributionSpec {
    using ArgsLoc = ArgsLoc_;
    using TensorShape = TensorShape_;
    using ShardShape = ShardShape_;
    using BankCoords = BankCoords_;

    // std::array if rank/num_banks are static, Span otherwise
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
        static_assert(!(is_static), "Everything is static, this constructor is obsolete!");
        if constexpr (TensorShape::has_static_rank and ShardShape::has_static_rank) {
            // If both tensor and shard ranks are static, they should be the same
            static_assert(TensorShape::rank == ShardShape::rank, "Tensor and shard shapes must have the same rank!");
            static_assert(
                std::is_same_v<typename TensorShape::ShapeBase, typename ShardShape::ShapeBase>,
                "Tensor and shard shapes bases must be the same");
        }

        if constexpr (!has_static_rank) {
            // Rank is not known at compile time, use runtime rank
            ASSERT(
                tensor_shape_.rank_rt == shard_shape_.rank_rt,
                "Tensor and shard shapes must have the same rank at runtime!");
            rank_rt = tensor_shape_.rank_rt;
            // !has_static_rank means ShapeBase is span<uint32_t>
            shard_grid_rt = ShapeBase(shard_grid_rt_buf.value, rank_rt);
            shard_grid_strides_rt = ShapeBase(shard_grid_strides_rt_buf.value, rank_rt);
        }
        if constexpr (!has_static_num_banks) {
            // Number of banks is not known at compile time, use runtime number of banks
            num_banks_rt = bank_coords_.num_banks_rt;
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
    static constexpr uint32_t num_banks_ct = BankCoords::has_static_num_banks ? BankCoords::num_banks : UNKNOWN;
    uint32_t rank_rt = 0;
    uint32_t num_banks_rt = 0;

    std::conditional_t<all_shapes_static, std::monostate, ShapeBase> shard_grid_rt{};
    std::conditional_t<all_shapes_static, std::monostate, ShapeBase> shard_grid_strides_rt{};
    mutable detail::ConditionalBuffer<!has_static_rank, uint32_t, MAX_RANK> shard_grid_rt_buf;
    mutable detail::ConditionalBuffer<!has_static_rank, uint32_t, MAX_RANK> shard_grid_strides_rt_buf;

    static constexpr ShapeBase shard_grid_ct_ = shard_grid_ct(TensorShape::shape, ShardShape::shape);
    static constexpr ShapeBase shard_grid_strides_ct_ = shard_grid_strides_ct(TensorShape::shape, ShardShape::shape);

    TensorShape tensor_shape_;  // Tensor shape
    ShardShape shard_shape_;    // Shard shape
    BankCoords bank_coords_;    // Bank coordinates
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
template <size_t CTA_OFFSET_, size_t CRTA_OFFSET_ = static_cast<size_t>(-1)>
struct ArgsLocation {
    static constexpr size_t CTA_OFFSET = CTA_OFFSET_;
    static constexpr size_t CRTA_OFFSET = CRTA_OFFSET_;

    static constexpr auto args_config =
        ArgsConfig(static_cast<ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_OFFSET)));

    // Fetch locations of the arguments
    static constexpr bool RankCRTA = args_config.test(ArgConfig::RankCRTA);
    static constexpr bool NumBanksCRTA = args_config.test(ArgConfig::NumBanksCRTA);
    static constexpr bool TensorShapeCRTA = args_config.test(ArgConfig::TensorShapeCRTA);
    static constexpr bool ShardShapeCRTA = args_config.test(ArgConfig::ShardShapeCRTA);
    static constexpr bool BankCoordsCRTA = args_config.test(ArgConfig::BankCoordsCRTA);

    static constexpr bool RankStatic = !RankCRTA;
    static constexpr bool NumBanksStatic = !NumBanksCRTA;
    static constexpr bool TensorShapeStatic = !TensorShapeCRTA;
    static constexpr bool ShardShapeStatic = !ShardShapeCRTA;
    static constexpr bool BankCoordsStatic = !BankCoordsCRTA;

    // Impossible to have runtime rank without runtime tensor and shard shapes since then impossible to calculate CTA
    // offsets in compile time
    static_assert(
        !RankCRTA or (RankCRTA and TensorShapeCRTA and ShardShapeCRTA),
        "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    static_assert(
        !NumBanksCRTA or (NumBanksCRTA and BankCoordsCRTA),
        "If num_banks is runtime, bank_coords must also be runtime");

    // Calculate offsets for compile-time arguments
    static constexpr uint32_t ArgsConfigCTAOFfset = CTA_OFFSET;
    static constexpr uint32_t RankCTAOffset = ArgsConfigCTAOFfset + 1;
    static constexpr uint32_t NumBanksStaticOffset = RankCTAOffset + (RankCRTA ? 0 : 1);

    static constexpr uint32_t RankCT = RankCRTA ? 0 : get_compile_time_arg_val(RankCRTA ? CTA_OFFSET : RankCTAOffset);
    static constexpr uint32_t NumBanksCT =
        NumBanksCRTA ? 0 : get_compile_time_arg_val(NumBanksCRTA ? CTA_OFFSET : NumBanksStaticOffset);

    static constexpr uint32_t TensorShapeStaticOffset = NumBanksStaticOffset + (NumBanksCRTA ? 0 : 1);
    static constexpr uint32_t ShardShapeStaticOffset = TensorShapeStaticOffset + (TensorShapeCRTA ? 0 : RankCT);
    static constexpr uint32_t BankCoordsStaticOffset = ShardShapeStaticOffset + (ShardShapeCRTA ? 0 : RankCT);

    static constexpr uint32_t NumArgsCT =
        BankCoordsStaticOffset + (BankCoordsCRTA ? 0 : NumBanksCT) - CTA_OFFSET;  // Number of compile-time arguments

    // Functions to calculate offsets for common runtime arguments
    static constexpr uint32_t rank_crta_offset() { return CRTA_OFFSET; }
    static constexpr uint32_t num_banks_crta_offset() { return CRTA_OFFSET + RankCRTA; }

    static constexpr uint32_t fetch_rank() {
        if constexpr (RankStatic) {
            return RankCT;
        } else {
            return get_common_arg_val<uint32_t>(rank_crta_offset());
        }
    }

    static constexpr uint32_t fetch_num_banks() {
        if constexpr (NumBanksStatic) {
            return NumBanksCT;
        } else {
            return get_common_arg_val<uint32_t>(num_banks_crta_offset());
        }
    }

    static constexpr uint32_t tensor_shape_crta_offset() { return num_banks_crta_offset() + NumBanksCRTA; }

    static constexpr uint32_t shard_shape_crta_offset() {
        return tensor_shape_crta_offset() + (TensorShapeCRTA ? fetch_rank() : 0);
    }

    static constexpr uint32_t bank_coords_crta_offset() {
        return shard_shape_crta_offset() + (ShardShapeCRTA ? fetch_rank() : 0);
    }

    static constexpr uint32_t num_args_crta() {
        return bank_coords_crta_offset() + (BankCoordsCRTA ? fetch_num_banks() : 0) - CRTA_OFFSET;
    }
};

// Helper to properly build a DistributionSpec type from compile-time arguments
template <size_t CTA_OFFSET, size_t CRTA_OFFSET>
struct DistributionSpecWrapper {
    // Fetch configuration of arguments, i.e. which are CTA which are CRTA
    using ArgsLoc = ArgsLocation<CTA_OFFSET, CRTA_OFFSET>;

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
    static constexpr size_t CRTA_OFFSET = Loc::CRTA_OFFSET;
    static constexpr bool RankStatic = Loc::RankStatic;
    static constexpr bool NumBanksStatic = Loc::NumBanksStatic;

    auto rank = Loc::fetch_rank();
    auto num_banks = Loc::fetch_num_banks();

    // DSpec::ShapeBase == std::array<uint32_t, RANK> if RankStatic is true, otherwise it is Span<uint32_t>
    typename DSpec::ShapeBase tensor_shape_array;
    typename DSpec::ShapeBase shard_shape_array;
    typename DSpec::PackedCoordsBase bank_coord_array;

    if constexpr (RankStatic) {
        // In such case shape base is std::array<uint32_t, RANK>
        if constexpr (TensorShapeCRTA) {
            tensor_shape_array = array_crta_sequence_wrapper<Loc::tensor_shape_crta_offset(), DSpec::rank_ct>();
        }
        if constexpr (ShardShapeCRTA) {
            shard_shape_array = array_crta_sequence_wrapper<
                Loc::tensor_shape_crta_offset() + DSpec::rank_ct * TensorShapeCRTA,
                DSpec::rank_ct>();
        }
    } else {
        // In such case shape base is Span<uint32_t>
        static_assert(TensorShapeCRTA, "Tensor shape must be CRTA if rank is not known at compile time!");
        static_assert(ShardShapeCRTA, "Shard shape must be CRTA if rank is not known at compile time!");

        if constexpr (TensorShapeCRTA) {
            // TODO: (C)RTA are contiguous in memory? verify that
            auto* tensor_shape_ptr = (uint32_t*)(get_common_arg_addr(Loc::tensor_shape_crta_offset()));
            tensor_shape_array = typename DSpec::ShapeBase(tensor_shape_ptr, rank);
        }
        if constexpr (ShardShapeCRTA) {
            auto* shard_shape_ptr = (uint32_t*)(get_common_arg_addr(Loc::shard_shape_crta_offset()));
            shard_shape_array = typename DSpec::ShapeBase(shard_shape_ptr, rank);
        }
    }

    if constexpr (NumBanksStatic) {
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
