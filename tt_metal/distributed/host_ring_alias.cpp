// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/internal/host_ring_alias.hpp>

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fmt/format.h>

#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>

namespace tt::tt_metal::experimental {

namespace {

uint64_t arena_offset_for(AliasArena a, uint32_t core) {
    return a == AliasArena::Tx ? tx_arena_offset(core) : rx_arena_offset(core);
}

}  // namespace

struct RingAlias::Impl {
    AliasArena arena = AliasArena::Tx;
    std::vector<uint8_t*> base;
    std::vector<size_t> bytes;

    void unmap() {
        // Declarations first and unconditionally: this also runs from map()'s rollback,
        // where a half-declared set leaves provision() skipping bytes nothing is mapped over.
        HostRegion::clear_aliases();
        for (size_t c = 0; c < base.size(); ++c) {
            if (base[c] == nullptr || bytes[c] == 0) {
                continue;
            }
            // Put something back rather than leaving a hole: the region is one contiguous
            // object that unpinning and late readers still address.
            void* const restored =
                ::mmap(base[c], bytes[c], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            if (restored == MAP_FAILED) {
                ::munmap(base[c], bytes[c]);
            }
            base[c] = nullptr;
            bytes[c] = 0;
        }
    }
};

using RingAliasImpl = RingAlias::Impl;

RingAlias::RingAlias() : impl_(std::make_unique<RingAliasImpl>()) {}
RingAlias::~RingAlias() { impl_->unmap(); }

std::unique_ptr<RingAlias> RingAlias::map(
    uint8_t* region_base, AliasArena arena, const std::vector<Slot>& slots, std::string& err) {
    err.clear();
    if (region_base == nullptr) {
        err = "ring-alias: no region base; there is nothing to overlay onto";
        return nullptr;
    }
    // Pinning captures the physical pages; MAP_FIXED afterwards swaps them out from under
    // both the pin and the MR, with nothing reporting it.
    if (HostRegion::is_provisioned()) {
        err = "ring-alias: the region is already provisioned and therefore pinned; the overlay "
              "must precede HostRegion::provision()";
        return nullptr;
    }
    if (slots.size() > kProvisionedCores) {
        err = "ring-alias: more rings than the region has arenas";
        return nullptr;
    }

    std::unique_ptr<RingAlias> a(new RingAlias());
    RingAliasImpl& im = *a->impl_;
    im.arena = arena;
    im.base.assign(slots.size(), nullptr);
    im.bytes.assign(slots.size(), 0);

    for (uint32_t c = 0; c < slots.size(); ++c) {
        const Slot& s = slots[c];
        uint8_t* const target = region_base + arena_offset_for(arena, c);

        // What follows an arena slot is another core's arena, so an oversized overlay
        // silently replaces memory someone else owns. Refuse rather than clamp.
        const size_t rounded = (s.shm_size + kPageBytes - 1) & ~(static_cast<size_t>(kPageBytes) - 1);
        if (rounded > kArenaBytes) {
            err = fmt::format(
                "ring-alias: core {} ring is {} B ({} B page-rounded) but an arena slot is only {} B", c,
                s.shm_size, rounded, kArenaBytes);
            return nullptr;
        }

        const int fd = ::shm_open(s.shm_name.c_str(), O_RDWR, 0);
        if (fd < 0) {
            err = "ring-alias: shm_open(" + s.shm_name + ") failed: " + std::strerror(errno);
            return nullptr;
        }
        // MAP_FIXED deliberately: the point is to put these pages at an address the peer
        // already targets. The mapping holds its own reference, so the fd can go.
        void* const p = ::mmap(target, s.shm_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) {
            err = "ring-alias: mmap(" + s.shm_name + ") failed: " + std::strerror(errno);
            return nullptr;
        }
        if (p != target) {
            err = "ring-alias: MAP_FIXED did not honour the requested address";
            return nullptr;
        }

        im.base[c] = static_cast<uint8_t*>(p);
        im.bytes[c] = rounded;

        // Without this, provision() zeroes and reset_banks_and_arenas() complement-fills
        // over the socket's own counter word and the connector state behind it.
        HostRegion::declare_alias(arena, c, static_cast<uint64_t>(s.data_offset) + s.fifo_size, rounded);
    }
    return a;
}

uint8_t* RingAlias::base(uint32_t core) const {
    return core < impl_->base.size() ? impl_->base[core] : nullptr;
}

uint32_t RingAlias::count() const { return static_cast<uint32_t>(impl_->base.size()); }

std::string RingAlias::describe() const {
    return fmt::format("{} arenas aliased for {} ring(s)", impl_->arena == AliasArena::Tx ? "TX" : "RX", count());
}

}  // namespace tt::tt_metal::experimental
