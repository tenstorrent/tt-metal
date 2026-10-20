// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/mesh_tensor.hpp>

#include <memory>

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
    // An allocation that owns no resource
    RawAllocation();
    // Internal constructor
    explicit RawAllocation(RawAllocationImpl&& impl);
    ~RawAllocation();

    // Take ownership of the tensor's allocation.
    // This is currently the only means of constructing an RawAllocation.
    //
    // Tensor must be backed by an owning and allocated MeshBuffer in L1.
    // The backed buffer can be allocated lockstep, per-node or node-range.
    static RawAllocation adopt_from_meshtensor(MeshTensor tensor);

    // Release the MeshTensor that was used to create the RawAllocation class
    // The MeshTensor returned will be the same as the one supplied in
    // `adopt_from_meshtensor` memory footprint wise,
    // but it's data maybe overwritten.
    //
    // Precondition: allocation is not default initialized
    // Postcondition: MeshTensor is associated with a MeshBuffer that is owning and allocated on L1.
    static MeshTensor release_as_meshtensor(RawAllocation allocation);

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

}  // namespace tt::tt_metal::experimental
