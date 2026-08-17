// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <tuple>

#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include "internal/tensor/const.h"
#include "api/compile_time_args.h"

// Forward declared from dataflow_api.h
template <typename T>
T get_common_arg_val(int arg_idx);

template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET = tensor_accessor::UNKNOWN>
struct TensorAccessorArgs {
    static constexpr auto args_config = tensor_accessor::ArgsConfig(
        static_cast<tensor_accessor::ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_OFFSET)));

    static constexpr bool is_sharded = args_config.test(tensor_accessor::ArgConfig::Sharded);
    static constexpr bool is_dram = args_config.test(tensor_accessor::ArgConfig::IsDram);
    static constexpr bool rank_is_crta = args_config.test(tensor_accessor::ArgConfig::RuntimeRank);
    static constexpr bool num_banks_is_crta = args_config.test(tensor_accessor::ArgConfig::RuntimeNumBanks);
    static constexpr bool tensor_shape_is_crta = args_config.test(tensor_accessor::ArgConfig::RuntimeTensorShape);
    static constexpr bool shard_shape_is_crta = args_config.test(tensor_accessor::ArgConfig::RuntimeShardShape);
    static constexpr bool bank_coords_is_crta = args_config.test(tensor_accessor::ArgConfig::RuntimeBankCoords);
    static constexpr bool page_size_is_crta = args_config.test(tensor_accessor::ArgConfig::RuntimePageSize);

    // Impossible to have runtime rank without runtime tensor and shard shapes since then impossible to calculate CTA
    // offsets in compile time
    static_assert(
        !is_sharded || !rank_is_crta || (rank_is_crta && tensor_shape_is_crta && shard_shape_is_crta),
        "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    static_assert(
        !is_sharded || !num_banks_is_crta || (num_banks_is_crta && bank_coords_is_crta),
        "If num_banks is runtime, bank_coords must also be runtime");
    // A runtime (CRTA) page size is required for an interleaved-row-major tensor with dynamic shape.
    // (Because in that configuration, page size = last_dim_width * element_size)
    // For a sharded tensor, or interleaved-tiled tensor, page size is static.
    static_assert(
        !is_sharded || !page_size_is_crta,
        "RuntimePageSize (the runtime page-size relaxation) is not supported on sharded tensors");

    // aligned_page_size comes from one of two places depending on the configuration:
    //  - static case:  AlignedPageSize = CTA_OFFSET + 1
    //  - dynamic case: AlignedPageSize is IGNORED (handled by the getter; see below)
    // (The conditional lambda keeps the CTA read from being instantiated when the slot doesn't exist.)
    static constexpr uint32_t AlignedPageSizeCTAOffset = CTA_OFFSET + 1;
    static constexpr uint32_t AlignedPageSize = [] {
        if constexpr (page_size_is_crta) {
            return 0;  // ignored, getter uses the CRTA value instead
        } else {
            return get_compile_time_arg_val(AlignedPageSizeCTAOffset);
        }
    }();

    // Calculate offsets for compile-time arguments.
    // When the page size is a runtime field, its CTA slot is omitted, so the words after the config
    // word shift down by one.
    // (RuntimePageSize + sharded tensor is a forbidden combo, so page_size_is_crta is always false
    // for a sharded accessor. Then this stays CTA_OFFSET + 2.)
    static constexpr uint32_t RankCTAOffset = CTA_OFFSET + 2 - (page_size_is_crta ? 1 : 0);
    static constexpr uint32_t NumBanksCTAOffset = RankCTAOffset + (rank_is_crta ? 0 : 1);

    static constexpr uint32_t RankCT = [] {
        if constexpr (!is_sharded || rank_is_crta) {
            return 0;
        } else {
            return get_compile_time_arg_val(RankCTAOffset);
        }
    }();

    // Raw compile-time num_banks word: the bank count in the low bits and the shard-contiguous flag in the top bit
    // (see arg_config.hpp). NumBanksCT / is_shard_contiguous are the unpacked views used everywhere else.
    static constexpr uint32_t NumBanksRawCT = [] {
        if constexpr (!is_sharded || num_banks_is_crta) {
            return 0;
        } else {
            return get_compile_time_arg_val(NumBanksCTAOffset);
        }
    }();

    static constexpr uint32_t NumBanksCT = tensor_accessor::unpack_num_banks(NumBanksRawCT);

    // Shard-contiguous distribution (CONTIGUOUS_1D) vs. round-robin. When num_banks is compile-time this is read from
    // the top bit of the compile-time num_banks word; when num_banks is runtime, the live value is read from the
    // runtime num_banks word instead (see resolve_shard_contiguous()). Carries no extra args/slots.
    static constexpr bool is_shard_contiguous = tensor_accessor::unpack_is_shard_contiguous(NumBanksRawCT);

    static constexpr uint32_t PhysicalNumBanksCT = (NumBanksCT + 1) / 2;

    static_assert(!is_sharded || rank_is_crta || RankCT > 0, "Rank must be greater than 0!");
    static_assert(!is_sharded || num_banks_is_crta || NumBanksCT > 0, "Number of banks must be greater than 0!");

    static constexpr uint32_t TensorShapeCTAOffset = NumBanksCTAOffset + (num_banks_is_crta ? 0 : 1);
    static constexpr uint32_t ShardShapeCTAOffset = TensorShapeCTAOffset + (tensor_shape_is_crta ? 0 : RankCT);
    static constexpr uint32_t BankCoordsCTAOffset = ShardShapeCTAOffset + (shard_shape_is_crta ? 0 : RankCT);

    // Non-sharded payload is the config word + the aligned_page_size word, or just the config word
    // when the page size is a runtime field. Sharded derives from the offset chain above.
    static constexpr uint32_t NumArgsCT =
        is_sharded ? (BankCoordsCTAOffset + (bank_coords_is_crta ? 0 : PhysicalNumBanksCT) - CTA_OFFSET)
                   : (page_size_is_crta ? 1 : 2);

private:
    uint32_t crta_offset_rt_;

public:
    constexpr TensorAccessorArgs() : crta_offset_rt_(0) {}
    constexpr explicit TensorAccessorArgs(uint32_t crta_offset) : crta_offset_rt_(crta_offset) {
        static_assert(CRTA_OFFSET == tensor_accessor::UNKNOWN, "Do not pass crta_offset when CRTA_OFFSET is known");
    }
    constexpr uint32_t crta_offset() const {
        if constexpr (CRTA_OFFSET != tensor_accessor::UNKNOWN) {
            return CRTA_OFFSET;
        } else {
            return crta_offset_rt_;
        }
    }

    // Getter for aligned page size
    // The aligned page size can come from one of two places, depending on the configuration:
    // - static case:  AlignedPageSize (previously retrieved CTA value)
    // - dynamic case: Read the TensorAccessorArgs' first CRTA word (at runtime)
    constexpr uint32_t get_aligned_page_size() const {
        if constexpr (!page_size_is_crta) {
            return AlignedPageSize;
        } else {
            // In the dynamic case, the aligned page size lives in the first args CRTA slot
            return get_common_arg_val<uint32_t>(crta_offset());
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
            // The runtime num_banks word packs the shard-contiguous flag in its top bit; strip it off.
            return tensor_accessor::unpack_num_banks(get_common_arg_val<uint32_t>(num_banks_crta_offset()));
        }
    }

    // Whether shards are distributed contiguously (shard-contiguous, CONTIGUOUS_1D) vs. round-robin. When num_banks is
    // compile-time this is the constexpr is_shard_contiguous (top bit of the compile-time num_banks word); when
    // num_banks is runtime it is read from the top bit of the runtime num_banks word, so it can vary per dispatch
    // without recompiling.
    constexpr bool resolve_shard_contiguous() const {
        if constexpr (!num_banks_is_crta) {
            return is_shard_contiguous;
        } else {
            return tensor_accessor::unpack_is_shard_contiguous(get_common_arg_val<uint32_t>(num_banks_crta_offset()));
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
     * num_compile_time_args is required to be constexpr since cta argument index must be constexpr
     *
     * @return constexpr uint32_t Number of compile-time arguments used by the DistributionSpec.
     */
    static constexpr uint32_t num_compile_time_args() { return NumArgsCT; }

    static constexpr uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }

    /**
     * @brief Calculates the number of common runtime arguments used when building a DistributionSpec.
     * Evaluated at compile time if rank and num_banks are compile-time.
     *
     * @return constexpr uint32_t Number of common runtime arguments used by the DistributionSpec.
     */
    constexpr uint32_t num_common_runtime_args() const {
        if constexpr (!is_sharded) {
            // Interleaved carries one common runtime arg only under the RuntimePageSize relaxation
            // (the page-size word); otherwise none.
            return page_size_is_crta ? 1 : 0;
        }
        return bank_coords_crta_offset() + (bank_coords_is_crta ? get_physical_num_banks() : 0) - crta_offset();
    }

    constexpr uint32_t next_common_runtime_args_offset() const { return crta_offset() + num_common_runtime_args(); }
};

namespace tensor_accessor::detail {
template <uint32_t TENSOR_IDX, uint32_t CTA_OFFSET>
constexpr uint32_t get_tensor_accessor_args_cta_offset() {
    if constexpr (TENSOR_IDX == 0) {
        return CTA_OFFSET;
    } else {
        constexpr auto prev_offset = get_tensor_accessor_args_cta_offset<TENSOR_IDX - 1, CTA_OFFSET>();
        constexpr auto accessor_args = TensorAccessorArgs<prev_offset>();
        return accessor_args.next_compile_time_args_offset();
    }
}

template <uint32_t CTA_OFFSET, uint32_t... INDEXES>
constexpr auto get_tensor_accessor_args_cta_offsets(std::integer_sequence<uint32_t, INDEXES...>) {
    return std::integer_sequence<uint32_t, get_tensor_accessor_args_cta_offset<INDEXES, CTA_OFFSET>()...>();
}

template <uint32_t... CTA_OFFSETS>
constexpr auto make_tensor_accessor_args_tuple_from_cta_offsets(std::integer_sequence<uint32_t, CTA_OFFSETS...>) {
    return std::make_tuple(TensorAccessorArgs<CTA_OFFSETS>()...);
}
}  // namespace tensor_accessor::detail

template <uint32_t NUM_TENSORS, uint32_t CTA_OFFSET>
constexpr auto make_tensor_accessor_args_tuple() {
    constexpr auto cta_offsets = tensor_accessor::detail::get_tensor_accessor_args_cta_offsets<CTA_OFFSET>(
        std::make_integer_sequence<uint32_t, NUM_TENSORS>());
    return tensor_accessor::detail::make_tensor_accessor_args_tuple_from_cta_offsets(cta_offsets);
}
