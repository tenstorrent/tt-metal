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
        kSenderStateZoneSize < drisc_unreserved_size_,
        "DRISC L1 sender-state zone ({} B) must leave room for the above-zone kernel working region "
        "(unreserved size: {} B)",
        kSenderStateZoneSize,
        drisc_unreserved_size_);
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate(uint32_t size, uint32_t alignment) {
    return allocate_impl(/*core=*/std::nullopt, size, alignment);
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate_on(
    const CoreCoord& dram_sender_logical, uint32_t size, uint32_t alignment) {
    return allocate_impl(dram_sender_logical, size, alignment);
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate_impl(
    std::optional<CoreCoord> core, uint32_t size, uint32_t alignment) {
    TT_FATAL(size > 0, "DriscL1Arena allocation requires size > 0");
    TT_FATAL(alignment > 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of two");

    const uint32_t aligned_size = tt::align(size, alignment);
    const DeviceAddr zone_begin = unreserved_base_;
    const DeviceAddr zone_end = unreserved_base_ + kSenderStateZoneSize;

    std::lock_guard<std::mutex> lock(mutex_);

    // Two ranges can share addresses as long as no DRAM bank sees both: a range on one
    // core only blocks a uniform range or another range on that same core.
    auto shares_a_bank_with_request = [&core](const LiveRange& range) {
        return !range.core.has_value() || !core.has_value() || *range.core == *core;
    };

    // First-fit over the blocking ranges. live_ is sorted ascending by base, so a single
    // forward pass that steps past every overlap lands on the lowest free candidate.
    DeviceAddr candidate = tt::align(zone_begin, alignment);
    for (const LiveRange& range : live_) {
        if (!shares_a_bank_with_request(range)) {
            continue;
        }
        if (range.base >= candidate + aligned_size) {
            break;  // sorted: this and every later range start above the candidate window
        }
        if (candidate < range.base + range.size) {
            candidate = tt::align(range.base + range.size, alignment);
        }
    }
    if (candidate + aligned_size <= zone_end) {
        auto insert_it =
            std::lower_bound(live_.begin(), live_.end(), candidate, [](const LiveRange& range, DeviceAddr val) {
                return range.base < val;
            });
        live_.insert(insert_it, LiveRange{candidate, aligned_size, core});
        return std::shared_ptr<DriscL1Allocation>(
            new DriscL1Allocation(weak_from_this(), candidate, aligned_size, core));
    }

    TT_THROW(
        "DRISC L1 sender zone full: requested {} B (aligned {} B) {}; zone is {} B starting at 0x{:x}",
        size,
        aligned_size,
        core.has_value() ? fmt::format("on DRAM sender core ({}, {})", core->x, core->y) : "on every bank",
        kSenderStateZoneSize,
        zone_begin);
}

void DriscL1Arena::release(DeviceAddr base, uint32_t size, const std::optional<CoreCoord>& core) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::lower_bound(
        live_.begin(), live_.end(), base, [](const LiveRange& range, DeviceAddr val) { return range.base < val; });
    // Ranges on different cores can share a base, so match the core too.
    for (; it != live_.end() && it->base == base; ++it) {
        if (it->size == size && it->core == core) {
            live_.erase(it);
            return;
        }
    }
}

DriscL1Allocation::~DriscL1Allocation() {
    // If the owning MeshDeviceImpl has already dropped the arena (close_impl
    // ran before the user destroyed their GCBs), lock() returns null and the
    // destructor becomes a no-op — no UAF on the arena pointer. Same shape as
    // MeshBuffer::deallocate() locking its weak_ptr<MeshDevice>.
    if (auto arena = arena_.lock()) {
        arena->release(base_, size_, core_);
    }
}

}  // namespace tt::tt_metal
