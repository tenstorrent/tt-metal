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

// Retained for tensor-backed ownership; not used by the allocation representations below.
struct TensorBackedResource {
    explicit TensorBackedResource(MeshTensor tensor) : resource(std::move(tensor)) {}

    MeshTensor resource;
};

// Placeholder for allocation ownership; no resource is held yet.
struct BackedResource {};

// Represent lockstep allocation across all banks of a common type,
// with a common starting address and uniform extent.
struct LockstepAllocation {
    LockstepAllocation(DeviceAddr addr, uint32_t size) : addr(addr), size(size) {}

    DeviceAddr addr;
    uint32_t size;
};

// Represent per-node allocation on a subset of available banks,
// with potentially different starting addresses and a common extent.
struct PerNodeAllocation {
    PerNodeAllocation(std::unordered_map<CoreCoord, DeviceAddr> addrs, uint32_t size) :
        addr_per_node(std::move(addrs)), size(size) {}

    std::unordered_map<CoreCoord, DeviceAddr> addr_per_node;
    uint32_t size;
};

// Represent range lockstep allocation on a subset of available banks,
// with a common starting address and uniform extent.
struct RangeLockstepAllocation {
    RangeLockstepAllocation(DeviceAddr addr, uint32_t size, CoreRangeSet nodes) :
        addr(addr), size(size), allocated_nodes(std::move(nodes)) {}

    DeviceAddr addr;
    uint32_t size;
    CoreRangeSet allocated_nodes;
};

struct RawAllocationImpl {
    using AllocationLayout = std::variant<LockstepAllocation, PerNodeAllocation, RangeLockstepAllocation>;

    RawAllocationImpl() = delete;
    explicit RawAllocationImpl(AllocationLayout detail, BackedResource backing = {}) :
        layout(std::move(detail)), backing(std::move(backing)) {}

    AllocationLayout layout;
    BackedResource backing;
};

}  // namespace tt::tt_metal::experimental
