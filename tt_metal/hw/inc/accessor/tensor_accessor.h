// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "tensor_accessor_args.h"
#include "array_wrapper.h"
#include "dspec.h"
#include "helpers.h"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "dataflow_api_addrgen.h"
#endif

namespace tensor_accessor {
// This helper gets proper additional offset from interleaved_addr_gen::get_bank_offset +
//      Adds proper xy coordinates for NOC address
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
uint64_t get_dram_bank_base_offset(uint32_t bank_id, uint8_t noc) {
    // TODO: Should interleaved_addr_gen:: functions moved into common helper?
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<true>(bank_id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<true>(bank_id, bank_offset_index);
    uint32_t bank_offset = interleaved_addr_gen::get_bank_offset<true>(bank_index);
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank_index, noc);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, bank_offset);
    return noc_addr;
}
#endif
}  // namespace tensor_accessor

// Forward declaration for ShardPagesAddressIterator
template <typename Accessor>
class ShardPagesAddressIterator;

/**
 * @brief Accessor that encapsulates the logic for accessing tensors pages.
 *
 * The TensorAccessor provides efficient access to pages in a tensor by:
 * 1. Computing which bank contains a given page
 * 2. Calculating the offset within that bank
 * 3. Providing NOC address computation and async operations
 *
 * @tparam DSpec        DistributionSpec type.
 */
template <typename DSpec__>
struct TensorAccessor {
    using DSpec = DSpec__;

private:
    // DSpec can be static or dynamic, so we use a conditional instance
    using StaticDspec = tensor_accessor::detail::ConditionalStaticInstance<DSpec, DSpec::is_static>;
    [[no_unique_address]] tensor_accessor::detail::ConditionalField<!DSpec::is_static, DSpec> dspec_instance;

    [[no_unique_address]] mutable tensor_accessor::detail::
        ConditionalField<!DSpec::has_static_rank, uint32_t[tensor_accessor::MAX_RANK]> _page_coord;

public:
    template <typename DSpec_ = DSpec, std::enable_if_t<std::is_same_v<std::decay_t<DSpec_>, DSpec>, int> = 0>
    constexpr explicit TensorAccessor(
        DSpec_&& dspec, const size_t bank_base_address_in, const uint32_t page_size_in = 0) :
        dspec_instance(std::forward<DSpec_>(dspec)), bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    template <typename DSpec_ = DSpec, std::enable_if_t<DSpec_::is_static, int> = 0>
    TensorAccessor(const size_t bank_base_address_in = 0, uint32_t page_size_in = 0) :
        bank_base_address(bank_base_address_in), page_size(page_size_in) {}

    template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
    TensorAccessor(
        const TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args,
        const size_t bank_base_address_in,
        const uint32_t page_size_in = 0) :
        TensorAccessor(tensor_accessor::make_dspec_from_args(args), bank_base_address_in, page_size_in) {}

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
        return get_noc_addr(get_bank_and_offset(page_id), offset, noc);
    }

    template <typename ArrType, std::enable_if_t<tensor_accessor::detail::has_subscript_operator_v<ArrType>, int> = 0>
    FORCE_INLINE std::uint64_t get_noc_addr(
        const ArrType page_coord, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return get_noc_addr(get_bank_and_offset(page_coord), offset, noc);
    }

    // Shard NOC APIs
    FORCE_INLINE
    std::uint64_t get_shard_noc_addr(
        const uint32_t shard_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        PageMapping page_mapping{
            .bank_id = shard_id % dspec().num_banks(),
            .bank_page_offset = shard_id / dspec().num_banks() * dspec().shard_volume(),
        };
        return get_noc_addr(page_mapping, offset, noc);
    }

    template <typename ArrType, std::enable_if_t<tensor_accessor::detail::has_subscript_operator_v<ArrType>, int> = 0>
    FORCE_INLINE std::uint64_t get_shard_noc_addr(
        const ArrType shard_coord, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t shard_id = 0;
        for (uint32_t i = 0; i < dspec().rank(); ++i) {
            // Check that shard_coord is within bounds
            ASSERT(shard_coord[i] < dspec().shard_shape()[i]);
            shard_id *= dspec().shard_grid_strides()[i];
        }
        return get_shard_noc_addr(shard_id, offset, noc);
    }

    // Helpers
    struct PageMapping {
        size_t bank_id;
        size_t bank_page_offset;
    };

    PageMapping get_bank_and_offset(uint32_t page_id) const {
        // Check that page_id is within bounds
        ASSERT(page_id < dspec().tensor_volume());
        if (dspec().rank() >= 4) {
            return get_bank_and_offset_from_page_id(page_id);
        }

        // Calculate the page coordinate in the tensor
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

    template <typename ArrType, std::enable_if_t<tensor_accessor::detail::has_subscript_operator_v<ArrType>, int> = 0>
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
        uint32_t bank_id = flattened_shard_id % dspec().num_banks();
        uint32_t bank_shard_id = flattened_shard_id / dspec().num_banks();

        uint32_t bank_page_offset = bank_shard_id * dspec().shard_volume() + page_offset_within_shard;

        return {bank_id, bank_page_offset};
    }

    // Locality APIs
    FORCE_INLINE
    bool is_local_bank(uint32_t virtual_x, uint32_t virtual_y, uint8_t noc = noc_index) const {
        return virtual_x == my_x[noc] && virtual_y == my_y[noc];
    }

    FORCE_INLINE
    bool is_local_addr(const uint64_t noc_addr, uint8_t noc = noc_index) const {
        uint32_t x = NOC_UNICAST_ADDR_X(noc_addr);
        uint32_t y = NOC_UNICAST_ADDR_Y(noc_addr);
        return is_local_bank(x, y, noc);
    }

    FORCE_INLINE
    bool is_local_page(const uint32_t page_id, uint8_t noc = noc_index) const {
        auto page_mapping = get_bank_and_offset(page_id);
        const auto& packed_xy_coords = dspec().packed_xy_coords();
        auto bank_x = get_bank_x(packed_xy_coords[page_mapping.bank_id]);
        auto bank_y = get_bank_y(packed_xy_coords[page_mapping.bank_id]);
        return is_local_bank(bank_x, bank_y, noc);
    }

    FORCE_INLINE
    bool is_local_shard(const uint32_t shard_id, uint8_t noc = noc_index) const {
        uint32_t bank_id = shard_id % dspec().num_banks();

        const auto& packed_xy_coords = dspec().packed_xy_coords();
        auto bank_x = get_bank_x(packed_xy_coords[bank_id]);
        auto bank_y = get_bank_y(packed_xy_coords[bank_id]);
        return is_local_bank(bank_x, bank_y, noc);
    }

    // Shard iterator
    const ShardPagesAddressIterator<TensorAccessor> shard_pages_address_iterator(
        uint32_t shard_id, uint32_t start_page_offset = 0, uint8_t noc = noc_index) const {
        return ShardPagesAddressIterator<TensorAccessor>(*this, shard_id, start_page_offset, noc);
    }

    const ShardPagesAddressIterator<TensorAccessor> shard_pages_begin(
        uint32_t shard_id, uint8_t noc = noc_index) const {
        return shard_pages_address_iterator(shard_id, 0, noc);
    }

    const ShardPagesAddressIterator<TensorAccessor> shard_pages_end(uint32_t shard_id, uint8_t noc = noc_index) const {
        return shard_pages_address_iterator(shard_id, dspec().shard_volume(), noc);
    }

private:
    // NOC APIs
    FORCE_INLINE
    std::uint64_t get_noc_addr(
        const PageMapping page_mapping, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        const auto& packed_xy_coords = dspec().packed_xy_coords();
        auto bank_x = get_bank_x(packed_xy_coords[page_mapping.bank_id]);
        auto bank_y = get_bank_y(packed_xy_coords[page_mapping.bank_id]);
        auto bank_start = DSpec::is_dram ? tensor_accessor::get_dram_bank_base_offset(bank_x, noc)
                                         : NOC_XY_ADDR(DYNAMIC_NOC_X(noc, bank_x), DYNAMIC_NOC_Y(noc, bank_y), 0);
        return bank_start + bank_base_address + page_mapping.bank_page_offset * page_size + offset;
    }

    PageMapping get_bank_and_offset_from_page_id(uint32_t page_id) const {
        size_t flattened_shard_id = 0;
        size_t page_offset_within_shard = 0;
        for (int i = dspec().rank() - 1; i >= 0; --i) {
            // Check that page_coord is within bounds
            uint32_t page_coord = page_id % dspec().tensor_shape()[i];
            ASSERT(page_coord < dspec().tensor_shape()[i]);
            page_id /= dspec().tensor_shape()[i];
            flattened_shard_id += (page_coord / dspec().shard_shape()[i]) * dspec().shard_grid_strides()[i];
            page_offset_within_shard += (page_coord % dspec().shard_shape()[i]) * dspec().shard_strides()[i];
        }

        // NOTE: This assumes shards are round-robin assigned across banks
        size_t bank_id = flattened_shard_id % dspec().num_banks();
        size_t bank_shard_id = flattened_shard_id / dspec().num_banks();

        size_t bank_page_offset = bank_shard_id * dspec().shard_volume() + page_offset_within_shard;

        return {bank_id, bank_page_offset};
    }

    FORCE_INLINE
    uint16_t get_bank_x(uint16_t packed_xy_coord) const { return (packed_xy_coord >> 8) & 0xFF; }

    FORCE_INLINE
    uint16_t get_bank_y(uint16_t packed_xy_coord) const { return packed_xy_coord & 0xFF; }

public:
    const size_t bank_base_address = 0;
    const uint32_t page_size = 0;

    friend class ShardPagesAddressIterator<TensorAccessor>;
};

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
template <
    uint32_t RankCT,
    uint32_t NumBanksCT,
    typename TensorShapeWrapper,
    typename ShardShapeWrapper,
    typename BankCoordsWrapper,
    bool IsDram>
struct TensorAccessor<tensor_accessor::DistributionSpec<
    RankCT,
    NumBanksCT,
    TensorShapeWrapper,
    ShardShapeWrapper,
    BankCoordsWrapper,
    /* IsInterleaved */ true,
    IsDram>> : public InterleavedAddrGen<IsDram> {
    template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
    TensorAccessor(
        const TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args,
        const size_t bank_base_address_in,
        const uint32_t page_size_in = 0) :
        InterleavedAddrGen<IsDram>({.bank_base_address = bank_base_address_in, .page_size = page_size_in}) {}
};
#endif

template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
TensorAccessor(const TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args, size_t, uint32_t)
    -> TensorAccessor<decltype(tensor_accessor::make_dspec_from_args(args))>;

template <typename Accessor>
class ShardPagesAddressIterator {
public:
    using ArrayU32 = std::array<uint32_t, Accessor::DSpec::rank_ct>;
    using PageMapping = typename Accessor::PageMapping;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = const uint64_t;
    using difference_type = std::ptrdiff_t;
    using reference = const uint64_t&;
    using pointer = const uint64_t*;

    // Constructor that initializes the iterator at a starting position
    ShardPagesAddressIterator(
        const Accessor& accessor, uint32_t shard_id = 0, uint32_t start_page_offset = 0, uint8_t noc = noc_index) :
        accessor(accessor), current_page_id_in_shard(start_page_offset), current_shard_id(shard_id), noc(noc) {
        PageMapping current_page_mapping{
            .bank_id = shard_id % accessor.dspec().num_banks(),
            .bank_page_offset =
                shard_id / accessor.dspec().num_banks() * accessor.dspec().shard_volume() + start_page_offset};
        calculate_current_location(shard_id, start_page_offset);
        current_noc_addr = accessor.get_noc_addr(current_page_mapping, 0, noc);
        ASSERT(current_page_id_in_shard <= accessor.dspec().shard_volume());
    }

    // Get NOC address for current position with optional offset
    uint32_t get_page_id() const {
        uint32_t page_id = 0;
        for (uint32_t i = 0; i < accessor.dspec().rank(); ++i) {
            page_id += global_page_coord[i] * accessor.dspec().tensor_strides()[i];
        }
        return page_id;
    }
    uint64_t get_noc_addr(const uint32_t offset = 0) const { return current_noc_addr + offset; }

    reference operator*() const { return current_noc_addr; }

    // Operator ++/--
    ShardPagesAddressIterator& operator++() {
        if (current_page_id_in_shard >= accessor.dspec().shard_volume()) {
            return *this;  // End iterator
        }

        do {
            current_noc_addr += accessor.page_size;
            current_page_id_in_shard++;
            if (current_page_id_in_shard >= accessor.dspec().shard_volume()) {
                break;
            }
        } while (!update_local_global_page_coord());
        ASSERT(current_page_id_in_shard <= accessor.dspec().shard_volume());
        return *this;
    }

    ShardPagesAddressIterator operator++(int) {
        ShardPagesAddressIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    ShardPagesAddressIterator& operator--() { return *this += -1; }

    ShardPagesAddressIterator operator--(int) {
        ShardPagesAddressIterator tmp = *this;
        --(*this);
        return tmp;
    }

    ShardPagesAddressIterator& operator+=(difference_type steps) {
        if (current_page_id_in_shard >= accessor.dspec().shard_volume()) {
            return *this;  // End iterator
        }

        do {
            current_noc_addr += steps * accessor.page_size;
            current_page_id_in_shard += steps;
            if (current_page_id_in_shard >= accessor.dspec().shard_volume()) {
                break;
            }
        } while (!update_local_global_page_coord(steps));
        ASSERT(current_page_id_in_shard <= accessor.dspec().shard_volume());
        return *this;
    }

    ShardPagesAddressIterator operator+(difference_type steps) const {
        ShardPagesAddressIterator tmp = *this;
        tmp += steps;
        return tmp;
    }

    ShardPagesAddressIterator& operator-=(difference_type steps) { return *this += -steps; }

    ShardPagesAddressIterator operator-(difference_type steps) const { return *this + (-steps); }

    difference_type operator-(const ShardPagesAddressIterator& other) const {
        return current_page_id_in_shard - other.current_page_id_in_shard;
    }

    const uint64_t& operator[](difference_type n) const {
        auto temp = *this;
        temp += n;
        return *temp;
    }

    // Equality comparison
    bool operator==(const ShardPagesAddressIterator& other) const { return current_noc_addr == other.current_noc_addr; }

    // Inequality comparison
    bool operator!=(const ShardPagesAddressIterator& other) const { return !(*this == other); }

    bool operator<(const ShardPagesAddressIterator& other) const { return get_page_id() < other.get_page_id(); }

    bool operator>(const ShardPagesAddressIterator& other) const { return other < *this; }

    bool operator<=(const ShardPagesAddressIterator& other) const { return *this < other || *this == other; }

    bool operator>=(const ShardPagesAddressIterator& other) const { return !(*this < other); }

private:
    const Accessor& accessor;
    uint32_t current_page_id_in_shard = 0;
    uint32_t current_shard_id = 0;
    uint64_t current_noc_addr = 0;
    uint8_t noc = noc_index;

    ArrayU32 local_page_coord = {};
    ArrayU32 global_page_coord = {};
    ArrayU32 shard_coord = {};

    // Calculates global page coordinate and checks if it's within logical tensor bounds
    bool update_global_page_coord() {
        const auto& dspec = accessor.dspec();
        for (uint32_t i = 0; i < dspec.rank(); ++i) {
            global_page_coord[i] = shard_coord[i] * dspec.shard_shape()[i] + local_page_coord[i];
            // Check bounds - some shards at edges might have fewer pages (in case of padding)
            if (global_page_coord[i] >= dspec.tensor_shape()[i]) {
                return false;  // Page is outside logical tensor bounds
            }
        }
        return true;
    }

    // Calculates shard coordinate, page coordinate within shard, and global page coordinate
    void calculate_current_location(uint32_t shard_id, uint32_t page_id_in_shard) {
        const auto& dspec = accessor.dspec();

        // Calculate shard coordinates once and store them
        uint32_t remaining_shard_id = shard_id;
        for (int i = dspec.rank() - 1; i >= 0; --i) {
            shard_coord[i] = remaining_shard_id % dspec.shard_grid()[i];
            remaining_shard_id /= dspec.shard_grid()[i];
        }

        // Calculate initial page coordinates within shard
        uint32_t temp_page_id = page_id_in_shard;
        for (int i = dspec.rank() - 1; i >= 0; --i) {
            local_page_coord[i] = temp_page_id % dspec.shard_shape()[i];
            temp_page_id /= dspec.shard_shape()[i];
        }

        // Calculate initial global coordinates
        update_global_page_coord();
    }

    // Updates local_page_coord and global_page_coord by incrementing the rightmost dimension
    // Returns false if page is outside logical tensor bounds (padded tile)
    bool update_local_global_page_coord() {
        const auto& dspec = accessor.dspec();

        // Incrementally update local_page_coord like a multi-dimensional counter
        // Start from the rightmost (last) dimension and carry over when needed
        for (int i = dspec.rank() - 1; i >= 0; --i) {
            local_page_coord[i]++;
            // No overflow
            if (local_page_coord[i] < dspec.shard_shape()[i]) {
                break;
            }
            // Overflow - reset this dimension and continue to next dimension
            local_page_coord[i] = 0;
        }

        return update_global_page_coord();
    }

    // Generic version of update_local_global_page_coord for arbitrary steps
    bool update_local_global_page_coord(uint32_t step) {
        const auto& dspec = accessor.dspec();

        uint32_t carry = step;
        for (int i = dspec.rank() - 1; i >= 0 && carry > 0; --i) {
            uint32_t new_coord = local_page_coord[i] + carry;
            local_page_coord[i] = new_coord % dspec.shard_shape()[i];
            carry = new_coord / dspec.shard_shape()[i];
        }

        return update_global_page_coord();
    }
};
