// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/drisc_l1_arena.hpp"

#include <algorithm>

#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

DriscL1Arena::DriscL1Arena(ContextId context_id) {
    const auto& hal = MetalContext::instance(context_id).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DriscL1Arena requires programmable DRAM cores, which auto-enable on Blackhole with firmware "
        ">= 19.12.0.0");

    unreserved_base_ = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    drisc_unreserved_size_ = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    TT_FATAL(
        kGcbZoneSize < drisc_unreserved_size_,
        "DRISC L1 GCB zone ({} B) must leave room for the above-zone kernel working region "
        "(unreserved size: {} B)",
        kGcbZoneSize,
        drisc_unreserved_size_);
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate(uint32_t size, uint32_t alignment) {
    TT_FATAL(size > 0, "DriscL1Arena::allocate requires size > 0");
    TT_FATAL(alignment > 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of two");

    const uint32_t aligned_size = tt::align(size, alignment);
    const DeviceAddr zone_begin = unreserved_base_;
    const DeviceAddr zone_end = unreserved_base_ + kGcbZoneSize;

    std::lock_guard<std::mutex> lock(mutex_);

    // First-fit search across the single uniform-offset pool.
    DeviceAddr candidate = tt::align(zone_begin, alignment);
    while (candidate + aligned_size <= zone_end) {
        // Find first live range whose end is > candidate (the earliest range that could overlap).
        auto it = std::lower_bound(
            live_.begin(), live_.end(), candidate, [](const std::pair<DeviceAddr, uint32_t>& range, DeviceAddr val) {
                return range.first + range.second <= val;
            });
        if (it == live_.end() || it->first >= candidate + aligned_size) {
            // No overlap — insert and return.
            auto insert_it = std::lower_bound(
                live_.begin(),
                live_.end(),
                candidate,
                [](const std::pair<DeviceAddr, uint32_t>& range, DeviceAddr val) { return range.first < val; });
            live_.insert(insert_it, {candidate, aligned_size});
            return std::shared_ptr<DriscL1Allocation>(new DriscL1Allocation(weak_from_this(), candidate, aligned_size));
        }
        // Overlap with `*it` — advance past its end (aligned) and retry.
        candidate = tt::align(it->first + it->second, alignment);
    }

    TT_THROW(
        "DRISC L1 GCB zone full: requested {} B (aligned {} B); zone is {} B starting at 0x{:x}",
        size,
        aligned_size,
        kGcbZoneSize,
        zone_begin);
}

void DriscL1Arena::release(DeviceAddr base, uint32_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::lower_bound(
        live_.begin(), live_.end(), base, [](const std::pair<DeviceAddr, uint32_t>& range, DeviceAddr val) {
            return range.first < val;
        });
    if (it != live_.end() && it->first == base && it->second == size) {
        live_.erase(it);
    }
}

DriscL1Allocation::~DriscL1Allocation() {
    // If the owning MeshDeviceImpl has already dropped the arena (close_impl
    // ran before the user destroyed their GCBs), lock() returns null and the
    // destructor becomes a no-op — no UAF on the arena pointer. Same shape as
    // MeshBuffer::deallocate() locking its weak_ptr<MeshDevice>.
    if (auto arena = arena_.lock()) {
        arena->release(base_, size_);
    }
}

}  // namespace tt::tt_metal
