// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/allocation/raw_allocation.hpp>

#include <utility>

#include "experimental/allocation/raw_allocation_properties.hpp"
#include "impl/allocation/raw_allocation_impl.hpp"

namespace tt::tt_metal::experimental {

RawAllocation::RawAllocation(RawAllocationImpl&& impl) : pimpl_(std::make_unique<RawAllocationImpl>(std::move(impl))) {}

RawAllocation::~RawAllocation() = default;

RawAllocation::RawAllocation(RawAllocation&&) noexcept = default;

RawAllocation& RawAllocation::operator=(RawAllocation&&) noexcept = default;

RawAllocationImpl& RawAllocation::impl() { return *pimpl_; }

const RawAllocationImpl& RawAllocation::impl() const { return *pimpl_; }

RawAllocation allocate_on_sram(distributed::MeshDevice& device, LockstepAllocationProperties properties) {
    return RawAllocation(RawAllocationImpl::allocate_on_sram(device, properties));
}

RawAllocation allocate_on_sram(distributed::MeshDevice& device, RangeLockstepAllocationProperties properties) {
    return RawAllocation(RawAllocationImpl::allocate_on_sram(device, std::move(properties)));
}

}  // namespace tt::tt_metal::experimental
