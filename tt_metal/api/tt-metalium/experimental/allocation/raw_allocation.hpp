// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/experimental/allocation/raw_allocation_properties.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental {

struct RawAllocationImpl;

/**
 * An opaque handle to a device-side memory reservation with a user-managed lifetime.
 *
 * A RawAllocation represents either a lockstep or range-lockstep allocation.
 * It provides a minimal interface for passing memory reservations between
 * the user and the runtime. It is the sole owner of the memory reservation it
 * represents.
 *
 * **Currently SRAM only**.
 */
struct RawAllocation {
    // No default constructor to be consistent with MeshTensor & MeshBuffer.
    RawAllocation() = delete;
    // Releases the associated reservation
    ~RawAllocation();

    // General move-only RAII resource semantics
    // Disable copy
    RawAllocation(const RawAllocation&) = delete;
    RawAllocation& operator=(const RawAllocation&) = delete;
    // Allow move
    RawAllocation(RawAllocation&&) noexcept;
    RawAllocation& operator=(RawAllocation&&) noexcept;

    // Opaque handle.

    // Implementation details
    explicit RawAllocation(RawAllocationImpl&& impl);
    RawAllocationImpl& impl();
    const RawAllocationImpl& impl() const;

private:
    std::unique_ptr<RawAllocationImpl> pimpl_;
};

/**
 * Reserve SRAM memory at a common address across the entire MeshDevice consistent with given properties (e.g. size).
 *
 * See: LockstepAllocationProperties.
 *
 * Precondition:
 * - properties satisfies its documented invariants.
 *
 * Postconditions:
 * - Returns a valid RawAllocation that owns the memory reservation described by properties.
 * - No other allocation can occupy the reserved memory region until the returned allocation is destroyed.
 */
RawAllocation allocate_on_sram(distributed::MeshDevice& device, LockstepAllocationProperties properties);

/**
 * Reserve SRAM memory at a common address across the banks specified by the properties
 * across the entire MeshDevice.
 *
 * Precondition:
 * - properties satisfies its documented invariants.
 *
 * Postconditions:
 * - Returns a valid RawAllocation that owns the memory reservation described by properties.
 * - No other allocation can occupy the reserved memory region on the specified banks until the returned
 *   allocation is destroyed.
 */
RawAllocation allocate_on_sram(distributed::MeshDevice& device, RangeLockstepAllocationProperties properties);

}  // namespace tt::tt_metal::experimental
