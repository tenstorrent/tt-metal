// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <hostdevcommon/sharded_accessor/arg_config.hpp>
#include "const.hpp"

namespace nd_sharding {
namespace detail {
using std::size_t;

template <size_t CTA_OFFSET, size_t CRTA_OFFSET = UNKNOWN>
struct ArgsOffsets {
    static constexpr auto args_config =
        ArgsConfig(static_cast<ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_OFFSET)));

    static constexpr bool rank_is_crta = args_config.test(ArgConfig::RankCRTA);
    static constexpr bool num_banks_is_crta = args_config.test(ArgConfig::NumBanksCRTA);
    static constexpr bool tensor_shape_is_crta = args_config.test(ArgConfig::TensorShapeCRTA);
    static constexpr bool shard_shape_is_crta = args_config.test(ArgConfig::ShardShapeCRTA);
    static constexpr bool bank_coords_is_crta = args_config.test(ArgConfig::BankCoordsCRTA);

    // Impossible to have runtime rank without runtime tensor and shard shapes since then impossible to calculate CTA
    // offsets in compile time
    static_assert(
        !rank_is_crta || (rank_is_crta and tensor_shape_is_crta and shard_shape_is_crta),
        "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    static_assert(
        !num_banks_is_crta || (num_banks_is_crta and bank_coords_is_crta),
        "If num_banks is runtime, bank_coords must also be runtime");

    // Calculate offsets for compile-time arguments
    static constexpr uint32_t ArgsConfigCTAOFfset = CTA_OFFSET;
    static constexpr uint32_t RankCTAOffset = ArgsConfigCTAOFfset + 1;
    static constexpr uint32_t NumBanksCTAOffset = RankCTAOffset + (rank_is_crta ? 0 : 1);

    static constexpr uint32_t RankCT =
        rank_is_crta ? 0 : get_compile_time_arg_val(rank_is_crta ? CTA_OFFSET : RankCTAOffset);
    static constexpr uint32_t NumBanksCT =
        num_banks_is_crta ? 0 : get_compile_time_arg_val(num_banks_is_crta ? CTA_OFFSET : NumBanksCTAOffset);
    static constexpr uint32_t PhysicalNumBanksCT = (NumBanksCT - 1) / 2 + 1;

    static_assert(rank_is_crta || RankCT > 0, "Rank must be greater than 0!");
    static_assert(num_banks_is_crta || NumBanksCT > 0, "Number of banks must be greater than 0!");

    static constexpr uint32_t TensorShapeCTAOffset = NumBanksCTAOffset + (num_banks_is_crta ? 0 : 1);
    static constexpr uint32_t ShardShapeCTAOffset = TensorShapeCTAOffset + (tensor_shape_is_crta ? 0 : RankCT);
    static constexpr uint32_t BankCoordsCTAOffset = ShardShapeCTAOffset + (shard_shape_is_crta ? 0 : RankCT);

    static constexpr uint32_t NumArgsCT =
        BankCoordsCTAOffset + (bank_coords_is_crta ? 0 : PhysicalNumBanksCT) - CTA_OFFSET;

    uint32_t crta_offset_rt = CRTA_OFFSET;  // Default CRTA offset

    template <size_t C = CRTA_OFFSET, std::enable_if_t<C == UNKNOWN, int> = 0>
    ArgsOffsets(size_t crta_offset = CRTA_OFFSET) : crta_offset_rt(crta_offset) {}

    template <size_t C = CRTA_OFFSET, std::enable_if_t<C != UNKNOWN, int> = 0>
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
    constexpr uint32_t num_banks_crta_offset() const { return crta_offset() + rank_is_crta; }

    constexpr uint32_t get_rank() const {
        if constexpr (!rank_is_crta) {
            return RankCT;
        } else {
            return get_common_arg_val<uint32_t>(rank_crta_offset());
        }
    }

    constexpr uint32_t get_num_banks() const {
        if constexpr (!num_banks_is_crta) {
            return NumBanksCT;
        } else {
            return get_common_arg_val<uint32_t>(num_banks_crta_offset());
        }
    }

    constexpr uint32_t get_physical_num_banks() const {
        // 2 coordinates are packed in one uint32_t
        return (get_num_banks() - 1) / 2 + 1;
    }

    constexpr uint32_t tensor_shape_crta_offset() const { return num_banks_crta_offset() + num_banks_is_crta; }

    constexpr uint32_t shard_shape_crta_offset() const {
        return tensor_shape_crta_offset() + (tensor_shape_is_crta ? get_rank() : 0);
    }

    constexpr uint32_t bank_coords_crta_offset() const {
        return shard_shape_crta_offset() + (shard_shape_is_crta ? get_rank() : 0);
    }

    /**
     * @brief Calculates the number of compile-time arguments used when building a DistributionSpec. Note that
     * compile_time_args_skip is required to be constexpr since cta argument index must be constexpr
     *
     * @return constexpr uint32_t Number of compile-time arguments used by the DistributionSpec.
     */
    static constexpr uint32_t compile_time_args_skip() { return NumArgsCT; }

    /**
     * @brief Calculates the number of common runtime arguments used when building a DistributionSpec.
     * Evaluated at compile time if rank and num_banks are compile-time.
     *
     * @return constexpr uint32_t Number of common runtime arguments used by the DistributionSpec.
     */
    constexpr uint32_t runtime_args_skip() const {
        return bank_coords_crta_offset() + (bank_coords_is_crta ? get_physical_num_banks() : 0) - crta_offset();
    }
};

}  // namespace detail
}  // namespace nd_sharding
