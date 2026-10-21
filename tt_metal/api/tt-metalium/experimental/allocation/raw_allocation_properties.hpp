// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental {

// Describes lockstep allocation across all banks of a common type.
// The resulting allocation will have a common starting address and extent across all banks within a device.
struct LockstepAllocationProperties {
    using size_type = uint32_t;

    // The size of allocation of each bank.
    //
    // Invariant: must be > 0.
    size_type per_bank_size;

    // Currently the entire RawAllocation is L1 only,
    // but if we want to generalize it:
    //
    // MemoryResource resource = SRAM;
};

struct RangeLockstepAllocationProperties {
    using size_type = uint32_t;

    // Allocation will only be present on these nodes.
    //
    // Invariant: non-empty
    NodeRangeSet banks_to_be_allocated_on;

    // Invariant: must be > 0.
    size_type per_bank_size;

    // MemoryResource resource = SRAM;
};

using RawAllocationProperties = std::variant<LockstepAllocationProperties, RangeLockstepAllocationProperties>;

}  // namespace tt::tt_metal::experimental
