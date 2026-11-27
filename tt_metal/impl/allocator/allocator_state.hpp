// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

// Forward declarations
class Allocator;
class BankManager;
enum class BufferType;

/**
 * @brief Represents the complete allocation state across all buffer types
 *
 * This class captures allocation states for all buffer types (DRAM, L1, L1_SMALL, TRACE).
 * It can be extracted from one or more allocators, merged with other states, and applied
 * to target allocators.
 */
class AllocatorState {
public:
    struct BufferTypeState;

    AllocatorState() = default;
    explicit AllocatorState(std::unordered_map<BufferType, BufferTypeState> states_per_buffer_type) :
        states_per_buffer_type_(std::move(states_per_buffer_type)) {}

    /**
     * @brief Merge another allocator state into this one
     *
     * Combines allocation information from both states. For each buffer type:
     * - If only one state has it, that state is kept
     * - If both have it, regions are merged and compatibility is checked
     *
     * @param other The state to merge into this one
     * @throws TT_FATAL if states are incompatible for any overlapping buffer type
     */
    void merge(const AllocatorState& other);

    /**
     * @brief Get total allocated size across all buffer types
     *
     * @return Total number of bytes allocated
     */
    DeviceAddr total_allocated_size() const;

    /**
     * @brief Get allocated regions for a specific buffer type
     *
     * @param buffer_type The buffer type to query
     * @return Vector of (start_addr, end_addr) pairs, empty if buffer type not present
     */
    const std::vector<std::pair<DeviceAddr, DeviceAddr>>& get_allocated_regions(BufferType buffer_type) const;

    /**
     * @brief Get states per buffer type
     *
     * @return Map from buffer type to buffer type state
     */
    const std::unordered_map<BufferType, BufferTypeState>& get_states_per_buffer_type() const {
        return states_per_buffer_type_;
    }

    /**
     * @brief Per-buffer-type allocation state
     *
     * Captures the complete allocation state for a single buffer type.
     */
    struct BufferTypeState {
        // Metadata - must match across allocators being merged
        BufferType buffer_type = BufferType::DRAM;
        uint32_t num_banks = 0;
        DeviceAddr bank_size = 0;
        DeviceAddr interleaved_address_limit = 0;
        uint32_t alignment_bytes = 0;
        std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset;

        /**
         * @brief Allocated address ranges as (start_addr, end_addr) pairs
         *
         * Represents all allocated memory regions. Ranges are normalized:
         * - Sorted by start address
         * - Non-overlapping and non-adjacent (coalesced)
         * - Half-open intervals: [start_addr, end_addr)
         */
        std::vector<std::pair<DeviceAddr, DeviceAddr>> allocated_regions;

        /**
         * @brief Merge another buffer type state into this one
         *
         * @param other The state to merge
         * @throws TT_FATAL if states are incompatible
         */
        void merge(const BufferTypeState& other);

        /**
         * @brief Sort and coalesce overlapping/adjacent regions
         */
        void normalize();

        /**
         * @brief Get total allocated size
         *
         * @return Total bytes allocated
         */
        DeviceAddr total_allocated_size() const;

        /**
         * @brief Check compatibility with another state
         *
         * @param other State to check against
         * @return true if compatible
         */
        bool is_compatible_with(const BufferTypeState& other) const;
    };

private:
    // Map from buffer type to its allocation state
    std::unordered_map<BufferType, BufferTypeState> states_per_buffer_type_;
};

}  // namespace tt::tt_metal
