// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <hostdevcommon/flags.hpp>
#include "const.hpp"

namespace {
/**
 * @brief Encodes which arguments are compile-time and which are common runtime.
 */
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

namespace nd_sharding {
namespace detail {
using std::size_t;

/**
 * @brief Keeps which DSpec arguments are compile-time and which are common runtime.
 *
 */
template <
    bool RankCRTA_ = false,
    bool NumBanksCRTA_ = false,
    bool TensorShapeCRTA_ = false,
    bool ShardShapeCRTA_ = false,
    bool BankCoordsCRTA_ = false>
struct ArgsLocation {
    // Fetch locations of the arguments
    static constexpr bool RankCRTA = RankCRTA_;
    static constexpr bool NumBanksCRTA = NumBanksCRTA_;
    static constexpr bool TensorShapeCRTA = TensorShapeCRTA_;
    static constexpr bool ShardShapeCRTA = ShardShapeCRTA_;
    static constexpr bool BankCoordsCRTA = BankCoordsCRTA_;

    static constexpr bool RankStatic = !RankCRTA;
    static constexpr bool NumBanksStatic = !NumBanksCRTA;
    static constexpr bool TensorShapeStatic = !TensorShapeCRTA;
    static constexpr bool ShardShapeStatic = !ShardShapeCRTA;
    static constexpr bool BankCoordsStatic = !BankCoordsCRTA;
};

/**
 * @brief Holds offsets for compile-time and common runtime arguments used for creation of DistributionSpec.
 * The order of arguments in the args array is: [rank, num_banks, tensor_shape, shard_shape, bank_coords].
 *
 * @tparam CTA_OFFSET_  base index of compile-time arguments in the args array
 * @tparam CRTA_OFFSET_ base index of common runtime arguments in the args array, if set to UNKNOWN, it will be set in
 * constructor
 */
template <size_t CTA_OFFSET_, size_t CRTA_OFFSET_ = UNKNOWN>
struct ArgsOffsets {
    static constexpr size_t CTA_OFFSET = CTA_OFFSET_;
    static constexpr size_t CRTA_OFFSET = CRTA_OFFSET_;

    static constexpr auto args_config =
        ArgsConfig(static_cast<ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_OFFSET)));

    using ArgsLoc = ArgsLocation<
        args_config.test(ArgConfig::RankCRTA),
        args_config.test(ArgConfig::NumBanksCRTA),
        args_config.test(ArgConfig::TensorShapeCRTA),
        args_config.test(ArgConfig::ShardShapeCRTA),
        args_config.test(ArgConfig::BankCoordsCRTA)>;

    // Impossible to have runtime rank without runtime tensor and shard shapes since then impossible to calculate CTA
    // offsets in compile time
    static_assert(
        !ArgsLoc::RankCRTA or (ArgsLoc::RankCRTA and ArgsLoc::TensorShapeCRTA and ArgsLoc::ShardShapeCRTA),
        "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    static_assert(
        !ArgsLoc::NumBanksCRTA or (ArgsLoc::NumBanksCRTA and ArgsLoc::BankCoordsCRTA),
        "If num_banks is runtime, bank_coords must also be runtime");

    // Calculate offsets for compile-time arguments
    static constexpr uint32_t ArgsConfigCTAOFfset = CTA_OFFSET;
    static constexpr uint32_t RankCTAOffset = ArgsConfigCTAOFfset + 1;
    static constexpr uint32_t NumBanksCTAOffset = RankCTAOffset + (ArgsLoc::RankCRTA ? 0 : 1);

    static constexpr uint32_t RankCT =
        ArgsLoc::RankCRTA ? 0 : get_compile_time_arg_val(ArgsLoc::RankCRTA ? CTA_OFFSET : RankCTAOffset);
    static constexpr uint32_t NumBanksCT =
        ArgsLoc::NumBanksCRTA ? 0 : get_compile_time_arg_val(ArgsLoc::NumBanksCRTA ? CTA_OFFSET : NumBanksCTAOffset);
    static constexpr uint32_t PhysicalNumBanksCT = (NumBanksCT - 1) / 2 + 1;  // Size of bank copordinates array (2
                                                                              // coordinates packed in one uint32_t)

    static_assert(!ArgsLoc::RankStatic or RankCT > 0, "Rank must be greater than 0!");
    static_assert(
        !ArgsLoc::NumBanksStatic or NumBanksCT > 0,
        "Number of banks must be greater than 0!");  // Number of banks must be > 0

    static constexpr uint32_t TensorShapeCTAOffset = NumBanksCTAOffset + (ArgsLoc::NumBanksCRTA ? 0 : 1);
    static constexpr uint32_t ShardShapeCTAOffset = TensorShapeCTAOffset + (ArgsLoc::TensorShapeCRTA ? 0 : RankCT);
    static constexpr uint32_t BankCoordsCTAOffset = ShardShapeCTAOffset + (ArgsLoc::ShardShapeCRTA ? 0 : RankCT);

    static constexpr uint32_t NumArgsCT = BankCoordsCTAOffset + (ArgsLoc::BankCoordsCRTA ? 0 : PhysicalNumBanksCT) -
                                          CTA_OFFSET;  // Number of compile-time arguments

    uint32_t crta_offset_rt = CRTA_OFFSET;  // Default CRTA offset

    template <typename ArgsOffsets_ = ArgsOffsets, std::enable_if_t<ArgsOffsets_::CRTA_OFFSET == UNKNOWN, int> = 0>
    ArgsOffsets(size_t crta_offset = CRTA_OFFSET) : crta_offset_rt(crta_offset) {}  // Constructor to set CRTA offset

    template <typename ArgsOffsets_ = ArgsOffsets, std::enable_if_t<ArgsOffsets_::CRTA_OFFSET != UNKNOWN, int> = 0>
    ArgsOffsets() {}

    // Functions to calculate offsets for common runtime arguments
    constexpr uint32_t crta_offset() const {
        if constexpr (CRTA_OFFSET != UNKNOWN) {
            return CRTA_OFFSET;
        } else {
            return crta_offset_rt;
        }
    }

    constexpr uint32_t rank_crta_offset() const { return crta_offset(); }
    constexpr uint32_t num_banks_crta_offset() const { return crta_offset() + ArgsLoc::RankCRTA; }

    constexpr uint32_t get_rank() const {
        if constexpr (ArgsLoc::RankStatic) {
            return RankCT;
        } else {
            return get_common_arg_val<uint32_t>(rank_crta_offset());
        }
    }

    constexpr uint32_t get_num_banks() const {
        if constexpr (ArgsLoc::NumBanksStatic) {
            return NumBanksCT;
        } else {
            return get_common_arg_val<uint32_t>(num_banks_crta_offset());
        }
    }

    constexpr uint32_t get_physical_num_banks() const {
        // 2 coordinates are packed in one uint32_t
        return (get_num_banks() - 1) / 2 + 1;
    }

    constexpr uint32_t tensor_shape_crta_offset() const { return num_banks_crta_offset() + ArgsLoc::NumBanksCRTA; }

    constexpr uint32_t shard_shape_crta_offset() const {
        return tensor_shape_crta_offset() + (ArgsLoc::TensorShapeCRTA ? get_rank() : 0);
    }

    constexpr uint32_t bank_coords_crta_offset() const {
        return shard_shape_crta_offset() + (ArgsLoc::ShardShapeCRTA ? get_rank() : 0);
    }

    /**
     * @brief Calculates the number of compile-time arguments used when building a DistributionSpec. Note that
     * compile_time_args_skip is required to be constexpr since cta argument index must be constexpr
     *
     * @return constexpr size_t     Number of compile-time arguments used by the DistributionSpec.
     */
    static constexpr uint32_t compile_time_args_skip() { return NumArgsCT; }

    /**
     * @brief Calculates the number of common runtime arguments used when building a DistributionSpec.
     *
     * @return constexpr size_t     Number of common runtime arguments used by the DistributionSpec.
     */
    constexpr uint32_t runtime_args_skip() const {
        return bank_coords_crta_offset() + (ArgsLoc::BankCoordsCRTA ? get_physical_num_banks() : 0) - crta_offset();
    }
};

}  // namespace detail
}  // namespace nd_sharding
