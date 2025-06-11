// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/flags.hpp>

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

using std::size_t;

/**
 * @brief Encapsulates the locations of arguments in compile-time and common runtime and calculates offsets
 * for CTA in compile time, and CRTA in compile time if possible.
 *
 * @tparam CTA_OFFSET_  Starting offset for compile-time arguments.
 * @tparam CRTA_OFFSET_ Starting offset for common runtime arguments.
 * runtime arguments.
 */
template <size_t CTA_OFFSET_, size_t CRTA_OFFSET_>
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

    static_assert(!RankStatic or RankCT > 0, "Rank must be greater than 0!");
    static_assert(
        !NumBanksStatic or NumBanksCT > 0, "Number of banks must be greater than 0!");  // Number of banks must be > 0

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
