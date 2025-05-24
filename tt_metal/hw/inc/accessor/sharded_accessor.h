// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "detail.h"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "dataflow_api.h"
#endif

template <size_t BASE, size_t RANK, size_t NUM_BANKS>
using distribution_spec_t = typename detail::DistributionSpecWrapper<BASE, RANK, NUM_BANKS>::dspec;

template <typename DSpec>
constexpr auto compile_time_args_skip = DSpec::rank * 2 + DSpec::num_banks;

template <typename DSpec, size_t PageSize>
struct ShardedAccessor {
    static constexpr DSpec dspec{};
    static constexpr auto page_size = PageSize;

    // Runtime args
    const size_t bank_base_address;

    // Helpers
    struct PageMapping {
        size_t bank_id;
        size_t bank_page_offset;
    };

    PageMapping get_bank_and_offset(uint32_t page_id) const {
        // Check that page_id is within bounds
        ASSERT(page_id < dspec.tensor_volume);
        // TODO: Should be possible to directly implement get_bank_and_offset logic with page_id and skip computing the
        // page_coord
        std::array<uint32_t, dspec.rank> page_coord;
        for (int i = dspec.rank - 1; i >= 0; --i) {
            page_coord[i] = page_id % dspec.tensor_shape[i];
            page_id /= dspec.tensor_shape[i];
        }
        return get_bank_and_offset(page_coord);
    }

    PageMapping get_bank_and_offset(const std::array<uint32_t, dspec.rank> page_coord) const {
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
        for (size_t i = 0; i < dspec.rank; ++i) {
            // Check that page_coord is within bounds
            ASSERT(page_coord[i] < dspec.tensor_shape[i]);
            flattened_shard_id += (page_coord[i] / dspec.shard_shape[i]) * dspec.shard_grid_strides[i];
            page_offset_within_shard += (page_coord[i] % dspec.shard_shape[i]) * dspec.shard_strides[i];
        }

        // NOTE: This assumes shards are round-robin assigned across banks
        size_t bank_id = flattened_shard_id % dspec.num_banks;
        size_t bank_shard_id = flattened_shard_id / dspec.num_banks;

        size_t bank_page_offset = bank_shard_id * dspec.shard_volume + page_offset_within_shard;

        return {bank_id, bank_page_offset};
    }

    // NOC APIs
    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, uint8_t noc = noc_index) const {
        const auto [bank_id, bank_offset] = this->get_bank_and_offset(id);
        return NOC_XY_ADDR(
            DYNAMIC_NOC_X(noc, (dspec.packed_xy_coords[bank_id] >> 16) & 0xFFFF),
            DYNAMIC_NOC_Y(noc, dspec.packed_xy_coords[bank_id] & 0xFFFF),
            bank_base_address + bank_offset * page_size);
    }

    FORCE_INLINE
    void noc_async_read_page(const uint32_t id, const uint32_t dest_addr, uint8_t noc = noc_index) const {
        noc_async_read(get_noc_addr(id, noc), dest_addr, page_size, noc);
    }

    FORCE_INLINE
    void noc_async_write_page(const uint32_t id, const uint32_t src_addr, uint8_t noc = noc_index) const {
        noc_async_write(src_addr, get_noc_addr(id, noc), page_size, noc);
    }
};
