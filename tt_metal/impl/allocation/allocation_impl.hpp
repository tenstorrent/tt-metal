// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <utility>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/allocation/device_span.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

namespace tt::tt_metal::experimental {

struct NoAlloc {};

struct TensorBackedResource {
    explicit TensorBackedResource(MeshTensor tensor) : resource(std::move(tensor)) {}

    MeshTensor resource;
};

struct LockstepAllocation : TensorBackedResource {
    LockstepAllocation(DeviceSpan span, MeshTensor tensor) :
        TensorBackedResource(std::move(tensor)), address_space(span) {}

    DeviceSpan address_space;
};

struct PerNodeAllocation : TensorBackedResource {
    PerNodeAllocation(std::unordered_map<CoreCoord, DeviceSpan> spans, MeshTensor tensor) :
        TensorBackedResource(std::move(tensor)), address_space_per_node(std::move(spans)) {}

    std::unordered_map<CoreCoord, DeviceSpan> address_space_per_node;
};

struct RangeLockstepAllocation : TensorBackedResource {
    RangeLockstepAllocation(DeviceSpan span, CoreRangeSet nodes, MeshTensor tensor) :
        TensorBackedResource(std::move(tensor)), address_space(span), allocated_nodes(std::move(nodes)) {}

    DeviceSpan address_space;
    CoreRangeSet allocated_nodes;
};

struct AllocationImpl {
    using Detail = std::variant<NoAlloc, LockstepAllocation, PerNodeAllocation, RangeLockstepAllocation>;

    AllocationImpl() = default;
    explicit AllocationImpl(Detail detail) : _detail(std::move(detail)) {}

    Detail _detail;
};

}  // namespace tt::tt_metal::experimental
