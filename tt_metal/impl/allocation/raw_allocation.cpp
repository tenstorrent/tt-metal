// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/allocation/raw_allocation.hpp>

#include <utility>

#include "impl/allocation/raw_allocation_impl.hpp"

namespace tt::tt_metal::experimental {

RawAllocation::RawAllocation() : pimpl_(std::make_unique<RawAllocationImpl>()) {}

RawAllocation::RawAllocation(RawAllocationImpl&& impl) : pimpl_(std::make_unique<RawAllocationImpl>(std::move(impl))) {}

RawAllocation::~RawAllocation() = default;

RawAllocation::RawAllocation(RawAllocation&&) noexcept = default;

RawAllocation& RawAllocation::operator=(RawAllocation&&) noexcept = default;

RawAllocationImpl& RawAllocation::impl() { return *pimpl_; }

const RawAllocationImpl& RawAllocation::impl() const { return *pimpl_; }

}  // namespace tt::tt_metal::experimental
