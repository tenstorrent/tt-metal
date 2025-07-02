// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include "const.h"

// Forward declared from dataflow_api.h
template <typename T>
T get_common_arg_val(int arg_idx);

namespace tensor_accessor {

template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET = UNKNOWN>
struct ArgsOffsets {
    static constexpr auto args_config =
        ArgsConfig(static_cast<ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_OFFSET)));

    static constexpr bool is_sharded = args_config.test(ArgConfig::Sharded);
    static constexpr bool is_dram = args_config.test(ArgConfig::IsDram);
    static constexpr bool rank_is_crta = args_config.test(ArgConfig::RuntimeRank);
    static constexpr bool num_banks_is_crta = args_config.test(ArgConfig::RuntimeNumBanks);
    static constexpr bool tensor_shape_is_crta = args_config.test(ArgConfig::RuntimeTensorShape);
    static constexpr bool shard_shape_is_crta = args_config.test(ArgConfig::RuntimeShardShape);
    static constexpr bool bank_coords_is_crta = args_config.test(ArgConfig::RuntimeBankCoords);

    // Impossible to have runtime rank without runtime tensor and shard shapes since then impossible to calculate CTA
    // offsets in compile time
    static_assert(
        !is_sharded || !rank_is_crta || (rank_is_crta && tensor_shape_is_crta && shard_shape_is_crta),
        "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    static_assert(
        !is_sharded || !num_banks_is_crta || (num_banks_is_crta && bank_coords_is_crta),
        "If num_banks is runtime, bank_coords must also be runtime");

    // Calculate offsets for compile-time arguments
    static constexpr uint32_t RankCTAOffset = CTA_OFFSET + 1;
    static constexpr uint32_t NumBanksCTAOffset = RankCTAOffset + (rank_is_crta ? 0 : 1);

    static constexpr uint32_t RankCT = [] {
        if constexpr (!is_sharded || rank_is_crta) {
            return 0;
        } else {
            return get_compile_time_arg_val(RankCTAOffset);
        }
    }();

    static constexpr uint32_t NumBanksCT = [] {
        if constexpr (!is_sharded || num_banks_is_crta) {
            return 0;
        } else {
            return get_compile_time_arg_val(NumBanksCTAOffset);
        }
    }();

    static constexpr uint32_t PhysicalNumBanksCT = (NumBanksCT + 1) / 2;

    static_assert(!is_sharded || rank_is_crta || RankCT > 0, "Rank must be greater than 0!");
    static_assert(!is_sharded || num_banks_is_crta || NumBanksCT > 0, "Number of banks must be greater than 0!");

    static constexpr uint32_t TensorShapeCTAOffset = NumBanksCTAOffset + (num_banks_is_crta ? 0 : 1);
    static constexpr uint32_t ShardShapeCTAOffset = TensorShapeCTAOffset + (tensor_shape_is_crta ? 0 : RankCT);
    static constexpr uint32_t BankCoordsCTAOffset = ShardShapeCTAOffset + (shard_shape_is_crta ? 0 : RankCT);

    static constexpr uint32_t NumArgsCT =
        is_sharded ? (BankCoordsCTAOffset + (bank_coords_is_crta ? 0 : PhysicalNumBanksCT) - CTA_OFFSET) : 1;

private:
    uint32_t crta_offset_rt_;

public:
    constexpr ArgsOffsets() : crta_offset_rt_(0) {}
    constexpr explicit ArgsOffsets(uint32_t crta_offset) : crta_offset_rt_(crta_offset) {
        static_assert(CRTA_OFFSET == UNKNOWN, "Do not pass crta_offset when CRTA_OFFSET is known");
    }
    constexpr uint32_t crta_offset() const {
        if constexpr (CRTA_OFFSET != UNKNOWN) {
            return CRTA_OFFSET;
        } else {
            return crta_offset_rt_;
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
        return (get_num_banks() + 1) / 2;
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
        if constexpr (!is_sharded) {
            return 0;
        }
        return bank_coords_crta_offset() + (bank_coords_is_crta ? get_physical_num_banks() : 0) - crta_offset();
    }
};

}  // namespace tensor_accessor
