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

bool RawAllocationImpl::matches_properties(const RawAllocationProperties& properties) const {
    if (const auto* lockstep = std::get_if<LockstepAllocation>(&layout)) {
        const auto* expected = std::get_if<LockstepAllocationProperties>(&properties);
        return expected != nullptr && lockstep->config.per_bank_size == expected->per_bank_size;
    }
    const auto& range = std::get<RangeLockstepAllocation>(layout);
    const auto* expected = std::get_if<RangeLockstepAllocationProperties>(&properties);
    return expected != nullptr && range.config.per_bank_size == expected->per_bank_size &&
           range.config.banks_to_be_allocated_on == expected->banks_to_be_allocated_on;
}

RawAllocation allocate_on_sram(distributed::MeshDevice& device, LockstepAllocationProperties properties) {
    return RawAllocation(RawAllocationImpl::allocate_on_sram(device, properties));
}

RawAllocation allocate_on_sram(distributed::MeshDevice& device, RangeLockstepAllocationProperties properties) {
    return RawAllocation(RawAllocationImpl::allocate_on_sram(device, std::move(properties)));
}

}  // namespace tt::tt_metal::experimental
