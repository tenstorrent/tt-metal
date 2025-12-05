// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_accessor.h"

/**
 * @brief Accessor for UDM (Unified Device Memory) block tensors
 *
 * BlockTensorAccessor wraps a regular TensorAccessor and provides conversion
 * from global block page IDs to local grid page IDs.
 *
 * For a width-sharded tensor on 1×4 mesh:
 *   Block page layout:
 *     Row 0: [0-3: dev0] [4-7: dev1] [8-11: dev2] [12-15: dev3]
 *     Row 1: [16-19: dev0] [20-23: dev1] [24-27: dev2] [28-31: dev3]
 *
 *   Local page layout (per grid):
 *     [0-3: row0] [4-7: row1] [8-11: row2] [12-15: row3]
 *
 * Usage in kernel:
 *   BlockTensorAccessor accessor(tensor_accessor, device_id, tiles_per_device_per_row, global_tiles_per_row);
 *   uint64_t noc_addr = accessor.get_noc_addr(block_page_id);
 *
 * @tparam DSpecT Distribution spec type for the underlying TensorAccessor
 */
template <typename DSpecT>
struct BlockTensorAccessor {
    using DSpec = DSpecT;

private:
    TensorAccessor<DSpec> tensor_accessor_;
    uint32_t device_id_;
    uint32_t tiles_per_device_per_row_;  // How many tiles this grid owns per row
    uint32_t global_tiles_per_row_;      // Total tiles per row across all grids

public:
    /**
     * @brief Construct a BlockTensorAccessor
     *
     * @param tensor_accessor The underlying local tensor accessor
     * @param device_id Which grid this kernel is running on
     * @param tiles_per_device_per_row How many tiles per row this grid owns
     * @param global_tiles_per_row Total tiles per row globally
     */
    constexpr BlockTensorAccessor(
        const TensorAccessor<DSpec>& tensor_accessor,
        uint32_t device_id,
        uint32_t tiles_per_device_per_row,
        uint32_t global_tiles_per_row) :
        tensor_accessor_(tensor_accessor),
        device_id_(device_id),
        tiles_per_device_per_row_(tiles_per_device_per_row),
        global_tiles_per_row_(global_tiles_per_row) {}

    /**
     * @brief Convert block page ID to local page ID
     *
     * @param block_page_id Global page ID in block space
     * @return Local page ID on this grid
     */
    FORCE_INLINE
    uint32_t block_page_to_local_page(uint32_t block_page_id) const {
        // Which global row?
        uint32_t global_row = block_page_id / global_tiles_per_row_;

        // Which tile within that row?
        uint32_t tile_in_row = block_page_id % global_tiles_per_row_;

        // Local tile index within the grid's portion of this row
        uint32_t local_tile_in_row = tile_in_row % tiles_per_device_per_row_;

        // Local page ID = row * tiles_per_row + local_tile_in_row
        uint32_t local_page_id = global_row * tiles_per_device_per_row_ + local_tile_in_row;

        return local_page_id;
    }

    /**
     * @brief Get NOC address for a block page ID
     *
     * @param block_page_id Global page ID in block space
     * @param offset Offset within the page
     * @param noc NOC to use
     * @return NOC address for the page
     */
    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t block_page_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t local_page_id = block_page_to_local_page(block_page_id);
        return tensor_accessor_.get_noc_addr(local_page_id, offset, noc);
    }

    /**
     * @brief Get bank and offset for a block page ID
     *
     * @param block_page_id Global page ID in block space
     * @return Bank and offset struct
     */
    FORCE_INLINE
    auto get_bank_and_offset(const uint32_t block_page_id) const {
        uint32_t local_page_id = block_page_to_local_page(block_page_id);
        return tensor_accessor_.get_bank_and_offset(local_page_id);
    }

    /**
     * @brief Async read tile from block page ID
     *
     * @param block_page_id Global page ID in block space
     * @param cb_id Circular buffer ID
     * @param noc NOC to use
     */
    FORCE_INLINE
    void noc_async_read_tile(const uint32_t block_page_id, const uint32_t cb_id, uint8_t noc = noc_index) const {
        uint32_t local_page_id = block_page_to_local_page(block_page_id);
        tensor_accessor_.noc_async_read_tile(local_page_id, cb_id, noc);
    }

    /**
     * @brief Async write tile to block page ID
     *
     * @param block_page_id Global page ID in block space
     * @param cb_id Circular buffer ID
     * @param noc NOC to use
     */
    FORCE_INLINE
    void noc_async_write_tile(const uint32_t block_page_id, const uint32_t cb_id, uint8_t noc = noc_index) const {
        uint32_t local_page_id = block_page_to_local_page(block_page_id);
        tensor_accessor_.noc_async_write_tile(local_page_id, cb_id, noc);
    }

    /**
     * @brief Get the underlying tensor accessor
     */
    const TensorAccessor<DSpec>& tensor_accessor() const { return tensor_accessor_; }
};
