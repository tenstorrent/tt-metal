// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/dataflow/dataflow_api.h"
#endif

/**
 * @brief N-dimensional iteration arguments parsed from kernel runtime args
 *
 * Encapsulates the common pattern of ND iteration over tensor pages:
 * - rank: number of dimensions
 * - For each dimension: (num_pages, offset, stride)
 *
 * @tparam MaxRank Maximum supported rank (default 8)
 */
template <uint32_t MaxRank = 8>
struct NDIterArgs {
    static constexpr uint32_t max_rank = MaxRank;

    uint32_t rank;
    std::array<uint32_t, MaxRank> dim_pages = {0};
    std::array<uint32_t, MaxRank> dim_offsets = {0};
    std::array<uint32_t, MaxRank> dim_strides = {0};

    NDIterArgs() : rank(0) {}

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
    /**
     * @brief Parse ND iteration args from kernel runtime arguments
     *
     * Expected format:
     *   - arg[start_idx]: rank
     *   - For each dimension d in [0, rank):
     *     - arg[start_idx + 1 + d*3 + 0]: dim_pages[d]
     *     - arg[start_idx + 1 + d*3 + 1]: dim_offsets[d]
     *     - arg[start_idx + 1 + d*3 + 2]: dim_strides[d]
     *
     * @param start_idx Starting index in runtime args (default 0)
     * @return Next argument index after parsing
     */
    uint32_t parse(uint32_t start_idx = 0) {
        rank = get_arg_val<uint32_t>(start_idx);
        uint32_t arg_idx = start_idx + 1;

        for (uint32_t d = 0; d < rank; ++d) {
            dim_pages[d] = get_arg_val<uint32_t>(arg_idx++);
            dim_offsets[d] = get_arg_val<uint32_t>(arg_idx++);
            dim_strides[d] = get_arg_val<uint32_t>(arg_idx++);
        }

        return arg_idx;
    }
#endif

    /**
     * @brief Compute total number of pages (product of all dim_pages)
     */
    uint32_t total_pages() const {
        uint32_t total = 1;
        for (uint32_t d = 0; d < rank; ++d) {
            total *= dim_pages[d];
        }
        return total;
    }

    /**
     * @brief Compute initial page_id from offsets and strides
     */
    uint32_t initial_page_id() const {
        uint32_t page_id = 0;
        for (uint32_t d = 0; d < rank; ++d) {
            page_id += dim_offsets[d] * dim_strides[d];
        }
        return page_id;
    }

    /**
     * @brief Compute total pages for all dimensions except the last (for reduction)
     */
    uint32_t total_rows() const {
        uint32_t rows = 1;
        for (uint32_t d = 0; d < rank - 1; ++d) {
            rows *= dim_pages[d];
        }
        return rows;
    }

    /**
     * @brief Get the last dimension's page count (width for reduction)
     */
    uint32_t last_dim_pages() const { return dim_pages[rank - 1]; }
};
