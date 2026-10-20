// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

namespace tt::tt_metal::experimental {

// For an RawAllocation that do not own any resource
struct NoAlloc {};

// Utiltiy struct for holding resource token assocaited with current RawAllocation.
// The depdency here should be inverted once we formalizes RawAllocation,
// RawAllocation should be part of MeshTensor, not the other way around.
struct TensorBackedResource {
    explicit TensorBackedResource(MeshTensor tensor) : resource(std::move(tensor)) {}

    MeshTensor resource;
};

// Represent lockstep allocation across all banks of a common type,
// with a common starting address and uniform extent.
struct LockstepAllocation : TensorBackedResource {
    LockstepAllocation(DeviceAddr addr, uint32_t size, MeshTensor tensor) :
        TensorBackedResource(std::move(tensor)), addr(addr), size(size) {}

    DeviceAddr addr;
    uint32_t size;
};

// Represent per-node allocation on a subset of available banks,
// with potentially different starting addresses and a common extent.
struct PerNodeAllocation : TensorBackedResource {
    PerNodeAllocation(std::unordered_map<CoreCoord, DeviceAddr> addrs, uint32_t size, MeshTensor tensor) :
        TensorBackedResource(std::move(tensor)), addr_per_node(std::move(addrs)), size(size) {}

    std::unordered_map<CoreCoord, DeviceAddr> addr_per_node;
    uint32_t size;
};

// Represent range lockstep allocation on a subset of available banks,
// with a common starting address and uniform extent.
struct RangeLockstepAllocation : TensorBackedResource {
    RangeLockstepAllocation(DeviceAddr addr, uint32_t size, CoreRangeSet nodes, MeshTensor tensor) :
        TensorBackedResource(std::move(tensor)), addr(addr), size(size), allocated_nodes(std::move(nodes)) {}

    DeviceAddr addr;
    uint32_t size;
    CoreRangeSet allocated_nodes;
};

struct RawAllocationImpl {
    // invariant:
    // - _detail, unless in NoAlloc, is always backed by a tensor in L1.

    using Detail = std::variant<NoAlloc, LockstepAllocation, PerNodeAllocation, RangeLockstepAllocation>;

    RawAllocationImpl() = default;
    explicit RawAllocationImpl(Detail detail) : _detail(std::move(detail)) {}

    Detail _detail;
};

}  // namespace tt::tt_metal::experimental
