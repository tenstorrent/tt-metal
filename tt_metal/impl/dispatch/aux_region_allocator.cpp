// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "aux_region_allocator.hpp"

#include <cstring>
#include <utility>

#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "dispatch_settings.hpp"

namespace tt::tt_metal {

uint32_t AuxRegionAllocator::reserved_bytes_per_cq(uint8_t num_hw_cqs, bool with_d2h_fallback_reserve) {
    TT_ASSERT(num_hw_cqs > 0, "Aux region sizing needs at least one hardware CQ");
    static constexpr uint32_t kAuxPagesPerCq = 2;
    uint32_t bytes = kAuxPagesPerCq * DispatchSettings::TRANSFER_PAGE_SIZE;
    if (with_d2h_fallback_reserve) {
        bytes += tt::align(
            (DispatchSettings::HUGEPAGE_D2H_FALLBACK_RESERVE_BYTES + num_hw_cqs - 1) / num_hw_cqs,
            DispatchSettings::TRANSFER_PAGE_SIZE);
    }
    return bytes;
}

AuxRegionAllocator::AuxRegionAllocator(
    char* host_base, uint32_t device_base, uint32_t size, bool has_d2h_fallback_reserve) :
    host_base_(host_base),
    device_base_(device_base),
    size_(size),
    has_d2h_fallback_reserve_(has_d2h_fallback_reserve) {}

AuxRegionAllocator::AuxRegionAllocator(AuxRegionAllocator&& other) noexcept :
    host_base_(std::exchange(other.host_base_, nullptr)),
    device_base_(std::exchange(other.device_base_, 0)),
    size_(std::exchange(other.size_, 0)),
    used_(std::exchange(other.used_, 0)),
    has_d2h_fallback_reserve_(std::exchange(other.has_d2h_fallback_reserve_, false)) {}

AuxRegionAllocator& AuxRegionAllocator::operator=(AuxRegionAllocator&& other) noexcept {
    if (this != &other) {
        host_base_ = std::exchange(other.host_base_, nullptr);
        device_base_ = std::exchange(other.device_base_, 0);
        size_ = std::exchange(other.size_, 0);
        used_ = std::exchange(other.used_, 0);
        has_d2h_fallback_reserve_ = std::exchange(other.has_d2h_fallback_reserve_, false);
    }
    return *this;
}

std::pair<void*, uint32_t> AuxRegionAllocator::allocate(uint32_t size) {
    if (host_base_ == nullptr || size_ == 0) {
        return {nullptr, 0};
    }

    const uint32_t aligned_size = tt::align(size, kMinAlignment);
    TT_FATAL(
        aligned_size <= size_ - used_,
        "Hugepage auxiliary region exhausted: requested {} bytes ({} aligned), {} of {} used",
        size,
        aligned_size,
        used_,
        size_);

    void* host_ptr = host_base_ + used_;
    const uint32_t device_addr = device_base_ + used_;
    used_ += aligned_size;

    std::memset(host_ptr, 0, aligned_size);
    return {host_ptr, device_addr};
}

}  // namespace tt::tt_metal
