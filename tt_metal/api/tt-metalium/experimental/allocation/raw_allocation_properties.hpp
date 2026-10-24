// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental {

/**
 * Describes an allocation with the same starting address and size across all SRAM banks on each device.
 */
struct LockstepAllocationProperties {
    using size_type = uint32_t;

    // Number of bytes to reserve in each participating bank.
    //
    // Invariant: must be > 0.
    size_type per_bank_size;  // in bytes

    // Currently RawAllocation is SRAM only,
    // but if we want to generalize it:
    //
    // MemoryResource resource = SRAM;
};

/**
 * Describes an allocation with the same starting address and size across the specified SRAM banks on each device.
 */
struct RangeLockstepAllocationProperties {
    using size_type = uint32_t;

    // Nodes whose SRAM banks participate in the allocation.
    //
    // Invariant: must be non-empty.
    NodeRangeSet banks_to_be_allocated_on;

    // Number of bytes to reserve in each participating bank.
    //
    // Invariant: must be > 0.
    size_type per_bank_size;  // in bytes

    // MemoryResource resource = SRAM;
};

using RawAllocationProperties = std::variant<LockstepAllocationProperties, RangeLockstepAllocationProperties>;

}  // namespace tt::tt_metal::experimental
