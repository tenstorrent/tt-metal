// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/allocator_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt {

namespace tt_metal {

// Forward declarations
class Allocator;
class BankManager;
class Buffer;
enum class BufferType;

/**
 * @brief Represents the unified allocation state for a single buffer type across multiple allocators
 *
 * This structure captures the complete allocation state (addresses and sizes) for a specific
 * buffer type. It can be extracted from one or more allocators, merged with other states,
 * and applied to a target allocator.
 */
struct UnifiedAllocatorState {
    // Metadata - must match across allocators being merged
    BufferType buffer_type;
    uint32_t num_banks;
    DeviceAddr bank_size;
    DeviceAddr interleaved_address_limit;
    uint32_t alignment_bytes;
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
     * @brief Optional: tracks which source allocator each region came from
     *
     * Useful for debugging and auditing. Same length as allocated_regions.
     * If empty, source tracking is disabled.
     */
    std::vector<uint32_t> region_source_allocator_ids;

    UnifiedAllocatorState() = default;

    /**
     * @brief Merge another state into this one
     *
     * Combines allocated regions from both states and normalizes the result.
     * States must be compatible (same buffer type, bank configuration, etc.)
     *
     * @param other The state to merge into this one
     * @throws TT_FATAL if states are incompatible
     */
    void merge(const UnifiedAllocatorState& other);

    /**
     * @brief Sort and coalesce overlapping/adjacent regions
     *
     * After merging or construction, call normalize() to ensure:
     * - Regions are sorted by start address
     * - Overlapping regions are merged
     * - Adjacent regions are coalesced
     */
    void normalize();

    /**
     * @brief Check if this state is compatible with another for merging
     *
     * Compatibility requires:
     * - Same buffer type
     * - Same number of banks
     * - Same bank size
     * - Same alignment
     * - Same bank offsets
     *
     * @param other The state to check compatibility with
     * @return true if compatible, false otherwise
     */
    bool is_compatible_with(const UnifiedAllocatorState& other) const;

    /**
     * @brief Get total allocated size across all regions
     *
     * @return Total number of bytes allocated
     */
    DeviceAddr total_allocated_size() const;

    /**
     * @brief Check if a given address range conflicts with allocated regions
     *
     * @param start_addr Start address of range to check
     * @param end_addr End address of range to check (exclusive)
     * @return true if range overlaps with any allocated region
     */
    bool has_conflict(DeviceAddr start_addr, DeviceAddr end_addr) const;
};

/**
 * @brief Represents the complete unified state for all buffer types
 *
 * This structure contains unified states for all buffer types (DRAM, L1, L1_SMALL, TRACE)
 * and tracks all buffer pointers across merged allocators.
 */
struct CompleteUnifiedAllocatorState {
    /**
     * @brief Unified states per buffer type
     *
     * Key: BufferType (DRAM, L1, L1_SMALL, TRACE)
     * Value: UnifiedAllocatorState for that buffer type
     */
    std::unordered_map<BufferType, UnifiedAllocatorState> states_per_buffer_type;

    /**
     * @brief All buffer pointers from merged allocators
     *
     * Contains union of all allocated_buffers_ from source allocators.
     * Note: These pointers may become invalid if buffers are deallocated.
     */
    std::vector<Buffer*> all_allocated_buffers;

    CompleteUnifiedAllocatorState() = default;

    /**
     * @brief Merge another complete state into this one
     *
     * @param other The complete state to merge
     */
    void merge(const CompleteUnifiedAllocatorState& other);

    /**
     * @brief Get total allocated size across all buffer types
     *
     * @return Total number of bytes allocated
     */
    DeviceAddr total_allocated_size() const;

    /**
     * @brief Check if state exists for a given buffer type
     *
     * @param buffer_type The buffer type to check
     * @return true if state exists for this buffer type
     */
    bool has_buffer_type(BufferType buffer_type) const;
};

/**
 * @brief Compute unified state from N allocators
 *
 * Extracts and merges allocation states from all provided allocators.
 *
 * @param allocators Vector of allocator pointers to merge (nullptr entries are skipped)
 * @return Complete unified state across all allocators
 */
CompleteUnifiedAllocatorState compute_unified_state(const std::vector<Allocator*>& allocators);

/**
 * @brief Compute unified state for specific buffer type from N allocators
 *
 * @param allocators Vector of allocator pointers
 * @param buffer_type The specific buffer type to extract and merge
 * @return Unified state for the specified buffer type
 */
UnifiedAllocatorState compute_unified_state(const std::vector<Allocator*>& allocators, const BufferType& buffer_type);

/**
 * @brief Compute unified state for specific buffer type from N allocators (const version)
 *
 * @param allocators Vector of const allocator pointers
 * @param buffer_type The specific buffer type to extract and merge
 * @return Unified state for the specified buffer type
 */
UnifiedAllocatorState compute_unified_state(
    const std::vector<const Allocator*>& allocators, const BufferType& buffer_type);

}  // namespace tt_metal

}  // namespace tt
