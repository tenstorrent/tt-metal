// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "debug/assert.h"
#include "debug/dprint.h"
#include "noc_parameters.h"

namespace tt::tt_fabric::udm {

// Simple memory pool manager for UDM read responses
// This class manages allocation of variable-sized memory chunks from a fixed pool
// in L1 memory for storing read response data temporarily before sending back ACKs.
class UDMMemoryPool {
private:
    inline static uint32_t pool_base_ = 0;
    inline static uint32_t pool_size_ = 0;
    inline static uint32_t current_offset_ = 0;
    inline static uint32_t pool_end_ = 0;
    inline static uint32_t last_alloc_total_ =
        0;  // Track total size (including padding) of last allocation for LIFO deallocation

    // Helper function to align an address to DRAM_ALIGNMENT
    FORCE_INLINE static uint32_t align_address(uint32_t addr) {
        // use DRAM alignment for now as it's the max alignment
        // TODO: have extra field in the packet header for differentiating l1/dram if needed.
        constexpr uint32_t alignment = DRAM_ALIGNMENT;
        return (addr + alignment - 1) & ~(alignment - 1);
    }

public:
    // Initialize the memory pool with base address and size
    // Must be called before any allocate/deallocate operations
    FORCE_INLINE static void init(uint32_t base_address, uint32_t size_bytes) {
        pool_base_ = base_address;
        pool_size_ = size_bytes;
        current_offset_ = 0;
        pool_end_ = base_address + size_bytes;
    }

    // Allocate a chunk of memory of the specified size
    // Returns the L1 address of the allocated memory (aligned to DRAM_ALIGNMENT)
    // Will hang if pool is exhausted
    FORCE_INLINE static uint32_t allocate_memory(uint32_t size_bytes) {
        // Align the current address to DRAM_ALIGNMENT
        uint32_t unaligned_addr = pool_base_ + current_offset_;
        uint32_t allocated_addr = align_address(unaligned_addr);
        uint32_t alignment_padding = allocated_addr - unaligned_addr;

        // Also align the size to ensure next allocation starts aligned
        uint32_t aligned_size = align_address(size_bytes);

        // Total allocation includes padding and aligned size
        uint32_t total_allocation = alignment_padding + aligned_size;

        ASSERT(allocated_addr + aligned_size <= pool_end_);
        if (allocated_addr + aligned_size > pool_end_) {
            DPRINT << "=== UDM MEMORY POOL EXHAUSTION ERROR ==="
                   << "CRITICAL: Insufficient space in UDM memory pool"
                   << "  - Requested Size: " << size_bytes << " bytes"
                   << "  - Aligned Size: " << aligned_size << " bytes"
                   << "  - Alignment Padding: " << alignment_padding << " bytes"
                   << "  - Total Allocation: " << total_allocation << " bytes"
                   << "  - Current Offset: " << current_offset_ << " bytes"
                   << "  - Pool Size: " << pool_size_ << " bytes"
                   << "  - Pool Base: 0x" << pool_base_ << "  - Pool End: 0x" << pool_end_
                   << "  - Allocated Address: 0x" << allocated_addr
                   << "Action: Entering infinite loop to prevent undefined behavior"
                   << "Solution: Increase UDM memory pool size or reduce memory usage"
                   << "=================================================\n";
            while (1) {
            }  // hang intentionally
        }

        // Store the total allocation for LIFO deallocation
        last_alloc_total_ = total_allocation;

        // Update offset to account for alignment padding and the allocated size
        current_offset_ += total_allocation;

        return allocated_addr;
    }

    // Deallocate a chunk of memory of the specified size
    // This implementation uses a simple bump allocator, so deallocation
    // just moves the offset back. Only works correctly if deallocations
    // happen in reverse order of allocations (LIFO).
    // NOTE: size_bytes should match the size passed to allocate_memory
    FORCE_INLINE static void deallocate_memory(uint32_t size_bytes) {
        // Use the stored total allocation (includes padding + aligned size)
        // This ensures we properly reverse the allocation including any alignment padding
        uint32_t total_dealloc = last_alloc_total_;

        ASSERT(current_offset_ >= total_dealloc);
        if (current_offset_ < total_dealloc) {
            DPRINT << "=== UDM MEMORY POOL DEALLOCATION ERROR ==="
                   << "CRITICAL: Attempting to deallocate more than allocated"
                   << "  - Requested Deallocation: " << size_bytes << " bytes"
                   << "  - Total Deallocation: " << total_dealloc << " bytes"
                   << "  - Current Offset: " << current_offset_ << " bytes"
                   << "Action: Entering infinite loop to prevent undefined behavior"
                   << "=================================================\n";
            while (1) {
            }  // hang intentionally
        }

        current_offset_ -= total_dealloc;
    }

    // Get the current pool usage in bytes
    FORCE_INLINE static uint32_t get_usage() { return current_offset_; }

    // Get the available space in bytes
    FORCE_INLINE static uint32_t get_available() { return pool_size_ - current_offset_; }

    // Reset the pool (only use during initialization or teardown)
    FORCE_INLINE static void reset() { current_offset_ = 0; }
};

}  // namespace tt::tt_fabric::udm
