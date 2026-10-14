// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include <tt-metalium/experimental/allocation/device_span.hpp>

#include <memory>
#include <unordered_map>

namespace tt::tt_metal::experimental {

struct AllocationImpl;

// Represent a unique handle to a piece of memory reservation
struct Allocation {
    // An allocation that owns no resource
    Allocation();
    // Internal constructor (internal use only)
    explicit Allocation(AllocationImpl&& impl);
    ~Allocation();

    // Take ownership of the tensor's allocation without moving its device data.
    // The tensor must own an allocated MeshBuffer, rather than a view of an external address.
    // Per-core allocations require a single-device mesh because reservations are keyed only by core.
    static Allocation adopt_from_meshtensor(MeshTensor tensor);

    // Consume the allocation and transfer its tensor owner out.
    // Throws if the allocation is empty or has been moved from.
    static MeshTensor release_as_meshtensor(Allocation allocation);

    // Disable copy
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;

    // Allow move
    Allocation(Allocation&&) noexcept;
    Allocation& operator=(Allocation&&) noexcept;

    AllocationImpl& impl();
    const AllocationImpl& impl() const;

private:
    std::unique_ptr<AllocationImpl> pimpl_;
};

// Per-node allocation related functions for Allocation.
bool is_per_node_allocation(const Allocation& alloc);

// Get the memory reservation at core coord.
DeviceSpan per_node_reservation_at(const Allocation& alloc, const CoreCoord& coord);

std::unordered_map<CoreCoord, DeviceSpan> per_node_reservations_of(const Allocation&);

bool is_lockstep_allocation(const Allocation&);

DeviceSpan lockstep_reservatoins_of(const Allocation&);

}  // namespace tt::tt_metal::experimental
