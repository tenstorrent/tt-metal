// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "accessor/detail/array_wrapper.hpp"
#include "detail/dspec.h"
#include "detail/helpers.hpp"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "dataflow_api.h"
#endif

namespace nd_sharding {
using detail::ArrayDynamicWrapper;
using detail::ArrayStaticWrapperU16;
using detail::ArrayStaticWrapperU32;
template <size_t StartIdx, uint32_t Size>
using array_u32_cta_sequence_wrapper_t = detail::struct_cta_sequence_wrapper_t<ArrayStaticWrapperU32, StartIdx, Size>;
template <size_t StartIdx, uint32_t Size>
using array_packed_u16_cta_sequence_wrapper_t =
    detail::struct_cta_sequence_wrapper_packed_u16_from_u32_t<StartIdx, Size>;

/**
 * @brief Accessor that encapsulates the logic for accessing sharded tensors pages.
 *
 * The ShardedAccessor provides efficient access to pages in a sharded tensor by:
 * 1. Computing which bank contains a given page
 * 2. Calculating the offset within that bank
 * 3. Providing NOC address computation and async operations
 *
 * @tparam DSpec        DistributionSpec type.
 */
template <typename DSpec>
struct ShardedAccessor {
private:
    // DSpec can be static or dynamic, so we use a conditional instance
    using StaticDspec = detail::ConditionalStaticInstance<DSpec, DSpec::is_static>;
    [[no_unique_address]] detail::ConditionalField<!DSpec::is_static, DSpec> dspec_instance;

    [[no_unique_address]] mutable detail::ConditionalField<!DSpec::has_static_rank, uint32_t[detail::MAX_RANK]>
        _page_coord;
    const size_t bank_base_address;

    // Page size is either compile-time constant or runtime value
    const uint32_t page_size;

public:
    template <typename DSpec_ = DSpec, std::enable_if_t<std::is_same_v<std::decay_t<DSpec_>, DSpec>, int> = 0>
    constexpr explicit ShardedAccessor(
        DSpec_&& dspec, const size_t bank_base_address_in, const uint32_t page_size_in = 0) :
        dspec_instance(std::forward<DSpec_>(dspec)), bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    template <typename DSpec_ = DSpec, std::enable_if_t<DSpec_::is_static, int> = 0>
    ShardedAccessor(const size_t bank_base_address_in = 0, uint32_t page_size_in = 0) :
        bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    constexpr auto& dspec() const {
        if constexpr (DSpec::is_static) {
            return StaticDspec::instance;
        } else {
            return dspec_instance.value;
        }
    }

    // NOC APIs
    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t page_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        const auto [bank_id, bank_offset] = this->get_bank_and_offset(page_id);
        const auto& packed_xy_coords = dspec().packed_xy_coords();
        return NOC_XY_ADDR(
            DYNAMIC_NOC_X(noc, (packed_xy_coords[bank_id] >> 8) & 0xFF),
            DYNAMIC_NOC_Y(noc, packed_xy_coords[bank_id] & 0xFF),
            bank_base_address + bank_offset * page_size + offset);
    }

    FORCE_INLINE
    void noc_async_read_page(
        const uint32_t page_id, const uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_read(get_noc_addr(page_id, offset, noc), dest_addr, page_size, noc);
    }

    FORCE_INLINE
    void noc_async_write_page(
        const uint32_t page_id, const uint32_t src_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_write(src_addr, get_noc_addr(page_id, offset, noc), page_size, noc);
    }

    // Helpers
    struct PageMapping {
        size_t bank_id;
        size_t bank_page_offset;
    };

    PageMapping get_bank_and_offset(uint32_t page_id) const {
        // Check that page_id is within bounds
        ASSERT(page_id < dspec().tensor_volume());
        // TODO: Should be possible to directly implement bank_and_offset logic with page_id and skip computing the
        // page_coord
        // std::array<uint32_t, detail::MAX_RANK> page_coord;
        typename DSpec::Shape page_coord;
        if constexpr (!DSpec::has_static_rank) {
            // If rank is not known at compile time, we need to use the _page_coord buffer for span
            page_coord = typename DSpec::Shape(_page_coord.value, dspec().rank());
        }
        for (int i = dspec().rank() - 1; i >= 0; --i) {
            page_coord[i] = page_id % dspec().tensor_shape()[i];
            page_id /= dspec().tensor_shape()[i];
        }
        return get_bank_and_offset(page_coord);
    }

    template <typename ArrType, std::enable_if_t<detail::has_subscript_operator_v<ArrType>, int> = 0>
    PageMapping get_bank_and_offset(const ArrType page_coord) const {
        // Flattened shard id is used to compute the bank id and shard id within a bank
        // - First, get the shard coordinate with page_coord[i] / dspec.shard_shape[i]
        // - Then, multiply by the shard grid strides and accumulate
        // - Repeat for all dims
        // Page offset within shard refers to the offset within the shard the page belongs to
        // - First, get the page coordinate within the shard with page_coord[i] % dspec.shard_shape[i]
        // - Then, multiple by the shard strides and accumulate
        // - Repeat for all dims
        // Final page offset within the bank is simply: bank_shard_id * shard_volume + page_offset_within_shard

        size_t flattened_shard_id = 0;
        size_t page_offset_within_shard = 0;
        for (size_t i = 0; i < dspec().rank(); ++i) {
            // Check that page_coord is within bounds
            ASSERT(page_coord[i] < dspec().tensor_shape()[i]);
            flattened_shard_id += (page_coord[i] / dspec().shard_shape()[i]) * dspec().shard_grid_strides()[i];
            page_offset_within_shard += (page_coord[i] % dspec().shard_shape()[i]) * dspec().shard_strides()[i];
        }

        // NOTE: This assumes shards are round-robin assigned across banks
        size_t bank_id = flattened_shard_id % dspec().num_banks();
        size_t bank_shard_id = flattened_shard_id / dspec().num_banks();

        size_t bank_page_offset = bank_shard_id * dspec().shard_volume() + page_offset_within_shard;

        return {bank_id, bank_page_offset};
    }
};

// Factory functions to create ShardedAccessor instance
template <size_t CTA_BASE, size_t CRTA_BASE>
FORCE_INLINE auto make_args() {
    return detail::ArgsOffsets<CTA_BASE, CRTA_BASE>();
}

template <size_t CTA_BASE>
FORCE_INLINE auto make_args(const size_t crta_base) {
    return detail::ArgsOffsets<CTA_BASE>(crta_base);
}

template <typename ArgsOffsetsT>
FORCE_INLINE auto make_sharded_accessor_from_args(
    const ArgsOffsetsT& args, const size_t bank_base_address_in, const uint32_t page_size_in) {
    auto dspec = detail::make_dspec_from_args(args);
    return ShardedAccessor<decltype(dspec)>(std::move(dspec), bank_base_address_in, page_size_in);
}

template <
    uint32_t RankCT = 0,
    uint32_t NumBanksCT = 0,
    typename TensorShapeWrapper = ArrayDynamicWrapper,
    typename ShardShapeWrapper = ArrayDynamicWrapper,
    typename BankCoordsWrapper = ArrayDynamicWrapper>
FORCE_INLINE auto make_dspec(
    uint32_t rank_rt = 0,
    uint32_t num_banks_rt = 0,
    uint32_t* tensor_shape_ptr = nullptr,
    uint32_t* shard_shape_ptr = nullptr,
    uint16_t* bank_coords_ptr = nullptr) {
    return detail::make_dspec<RankCT, NumBanksCT, TensorShapeWrapper, ShardShapeWrapper, BankCoordsWrapper>(
        rank_rt, num_banks_rt, tensor_shape_ptr, shard_shape_ptr, bank_coords_ptr);
}

template <typename DSpec>
FORCE_INLINE auto make_sharded_accessor_from_dspec(
    DSpec&& dspec, const size_t bank_base_address_in, const uint32_t page_size_in) {
    return ShardedAccessor<std::decay_t<decltype(dspec)>>(
        std::forward<DSpec>(dspec), bank_base_address_in, page_size_in);
}

}  // namespace nd_sharding
