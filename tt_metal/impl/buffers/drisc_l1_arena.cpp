// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/drisc_l1_arena.hpp"

#include <algorithm>

#include <tt-metalium/math.hpp>
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
    const bool has_receiver_zone = hal.has_cce_dram_cores();
    sender_state_zone_size_ = has_receiver_zone ? kCceSenderStateZoneSize : kSenderStateZoneSize;
    receiver_zone_size_ = has_receiver_zone ? kCceReceiverZoneSize : 0;
    TT_FATAL(
        sender_state_zone_size_ + receiver_zone_size_ < drisc_unreserved_size_,
        "DRISC L1 sender-state zone ({} B) and receiver zone ({} B) must leave room for the above-zone kernel "
        "working region (unreserved size: {} B)",
        sender_state_zone_size_,
        receiver_zone_size_,
        drisc_unreserved_size_);
}

DriscL1Arena::DriscL1Arena(DeviceAddr zone_base, uint32_t zone_size) :
    unreserved_base_(zone_base),
    drisc_unreserved_size_(zone_size),
    sender_state_zone_size_(zone_size),
    endpoint_l1_(EndpointL1::Shared) {}

std::shared_ptr<DriscL1Arena> DriscL1Arena::for_dispatch_engine(DeviceAddr zone_base, uint32_t zone_size) {
    TT_FATAL(zone_size > 0, "Dispatch-engine sender-state zone is empty");
    return std::shared_ptr<DriscL1Arena>(new DriscL1Arena(zone_base, zone_size));
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate_on(const CoreCoord& core, uint32_t size, uint32_t alignment) {
    return allocate_impl(core, unreserved_base_, sender_state_zone_size_, size, alignment);
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate(uint32_t size, uint32_t alignment) {
    return allocate_impl(/*core=*/std::nullopt, unreserved_base_, sender_state_zone_size_, size, alignment);
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate_receiver(uint32_t size, uint32_t alignment) {
    TT_FATAL(receiver_zone_size_ > 0, "This architecture's DRISC L1 arena has no receiver zone");
    return allocate_impl(/*core=*/std::nullopt, receiver_zone_base(), receiver_zone_size_, size, alignment);
}

std::shared_ptr<DriscL1Allocation> DriscL1Arena::allocate_impl(
    std::optional<CoreCoord> core, DeviceAddr zone_begin, uint32_t zone_size, uint32_t size, uint32_t alignment) {
    TT_FATAL(size > 0, "DriscL1Arena allocation requires size > 0");
    TT_FATAL(alignment > 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of two");

    const uint32_t aligned_size = tt::align(size, alignment);
    const DeviceAddr zone_end = zone_begin + zone_size;

    std::lock_guard<std::mutex> lock(mutex_);

    // First-fit over the ranges sharing an L1 with this one; ranges on other cores may share
    // addresses. live_ is sorted ascending by base, so a single forward pass that steps past every
    // overlap lands on the lowest free candidate.
    DeviceAddr candidate = tt::align(zone_begin, alignment);
    for (const LiveRange& range : live_) {
        if (!same_l1(range.core, core)) {
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
        "DRISC L1 {} zone full: requested {} B (aligned {} B) {}; zone is {} B starting at 0x{:x}",
        zone_begin == unreserved_base_ ? "sender-state" : "receiver",
        size,
        aligned_size,
        core.has_value() ? fmt::format("on DRAM sender core ({}, {})", core->x, core->y) : "on every DRAM core",
        zone_size,
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
    // ran before the user destroyed their pipes), lock() returns null and the
    // destructor becomes a no-op — no UAF on the arena pointer. Same shape as
    // MeshBuffer::deallocate() locking its weak_ptr<MeshDevice>.
    if (auto arena = arena_.lock()) {
        arena->release(base_, size_, core_);
    }
}

}  // namespace tt::tt_metal
