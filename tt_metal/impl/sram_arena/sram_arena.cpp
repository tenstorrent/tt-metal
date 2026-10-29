// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sram_arena/sram_arena.hpp>

#include <utility>

#include <tt_stl/assert.hpp>

#include "experimental/sram_arena/sram_arena_properties.hpp"
#include "impl/metal2_host_api/helpers.hpp"
#include "impl/sram_arena/sram_arena_impl.hpp"

namespace tt::tt_metal::experimental {

SRAMArena::SRAMArena(SRAMArenaImpl&& impl) : pimpl_(std::make_unique<SRAMArenaImpl>(std::move(impl))) {}

SRAMArena::~SRAMArena() = default;

SRAMArena::SRAMArena(SRAMArena&&) noexcept = default;

SRAMArena& SRAMArena::operator=(SRAMArena&&) noexcept = default;

SRAMArenaImpl& SRAMArena::impl() {
    TT_FATAL(pimpl_ != nullptr, "SRAMArena is in a moved-from state.");
    return *pimpl_;
}

const SRAMArenaImpl& SRAMArena::impl() const {
    TT_FATAL(pimpl_ != nullptr, "SRAMArena is in a moved-from state.");
    return *pimpl_;
}

bool SRAMArena::is_valueless_after_move() const { return pimpl_ == nullptr; }

DeviceAddr SRAMArenaImpl::address() const {
    return std::visit([](const auto& allocation) { return allocation.addr; }, layout);
}

bool SRAMArenaImpl::matches_properties(const SRAMArenaProperties& properties) const {
    if (const auto* lockstep = std::get_if<LockstepSRAMArena>(&layout)) {
        const auto* expected = std::get_if<LockstepSRAMArenaProperties>(&properties);
        return expected != nullptr && lockstep->config.per_bank_size == expected->per_bank_size;
    }
    const auto& range = std::get<RangeLockstepSRAMArena>(layout);
    const auto* expected = std::get_if<RangeLockstepSRAMArenaProperties>(&properties);
    return expected != nullptr && range.config.per_bank_size == expected->per_bank_size &&
           same_node_set(range.config.banks_to_be_allocated_on, expected->banks_to_be_allocated_on);
}

SRAMArena SRAMArena::allocate(distributed::MeshDevice& device, LockstepSRAMArenaProperties properties) {
    return SRAMArena(SRAMArenaImpl::allocate(device, properties));
}

SRAMArena SRAMArena::allocate(distributed::MeshDevice& device, RangeLockstepSRAMArenaProperties properties) {
    return SRAMArena(SRAMArenaImpl::allocate(device, std::move(properties)));
}

}  // namespace tt::tt_metal::experimental
