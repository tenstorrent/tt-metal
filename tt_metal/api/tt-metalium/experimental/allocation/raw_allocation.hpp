// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/experimental/allocation/raw_allocation_properties.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental {

struct RawAllocationImpl;

// Represent a unique handle to a piece of memory allocation.
// **Currently L1 only.**
//
// An allocation could be lockstep, per-node or node range allocated.
// Current interface is intentionally minimal as this is currently a shim to permit arena allocation for metal 2.0.
//
// In long term, we should surface dedicated structs to represent each kind of allocation and use RawAllocation class
// to represent "any" allocation.
// Currently, details of RawAllocation is only visible to the implementation details of tt_metal runtime.
struct RawAllocation {
    // No default constructor for now to reduce states & ease implementation.
    // This could be added to express an allocation that owns no resource
    RawAllocation() = delete;
    // Internal constructor
    explicit RawAllocation(RawAllocationImpl&& impl);
    ~RawAllocation();

    // General RAII resource semantics

    // Disable copy
    RawAllocation(const RawAllocation&) = delete;
    RawAllocation& operator=(const RawAllocation&) = delete;

    // Allow move
    RawAllocation(RawAllocation&&) noexcept;
    RawAllocation& operator=(RawAllocation&&) noexcept;

    RawAllocationImpl& impl();
    const RawAllocationImpl& impl() const;

private:
    std::unique_ptr<RawAllocationImpl> pimpl_;
};

/**
 * Reserve L1 memory at a common address across the participating banks on each device in the mesh.
 *
 * Precondition:
 * - properties satisfies its documented invariants.
 *
 * Postconditions:
 * - Returns a valid RawAllocation that owns the memory reservation described by properties.
 * - No other allocation can occupy the reserved memory region until the returned allocation is destroyed.
 */
RawAllocation allocate_l1(distributed::MeshDevice& device, LockstepAllocationProperties properties);

/**
 * Reserve L1 memory at a common address across the banks specified by properties.banks_to_be_allocated_on
 * on each device in the mesh.
 *
 * Precondition:
 * - properties satisfies its documented invariants.
 *
 * Postconditions:
 * - Returns a valid RawAllocation that owns the memory reservation described by properties.
 * - No other allocation can occupy the reserved memory region on the specified banks until the returned
 *   allocation is destroyed.
 */
RawAllocation allocate_l1(distributed::MeshDevice& device, RangeLockstepAllocationProperties properties);

}  // namespace tt::tt_metal::experimental
