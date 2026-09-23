// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_ring_alias.hpp"

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fmt/format.h>

#include <tt-metalium/experimental/sockets/host_uva_layout.hpp>

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
        // This arena only, and unconditionally: it also runs from map()'s rollback, where a
        // half-declared set leaves provision() skipping bytes nothing is mapped over.
        HostRegion::storage().clear_aliases(arena);
        for (size_t c = 0; c < base.size(); ++c) {
            if (base[c] == nullptr || bytes[c] == 0) {
                continue;
            }
            // Put something back rather than leaving a hole: the region is one contiguous
            // object that unpinning and late readers still address.
            void* const restored =
                ::mmap(base[c], bytes[c], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            if (restored == MAP_FAILED) {
                // Deliberately NOT munmap: that would punch the very hole the comment above
                // rules out. Leaving the shm mapped keeps the region addressable.
                continue;
            }
            // A fresh anonymous VMA does not inherit what provision() advised.
            (void)::madvise(base[c], bytes[c], MADV_DONTFORK);
            base[c] = nullptr;
            bytes[c] = 0;
        }
    }
};

RingAlias::RingAlias() : impl_(std::make_unique<Impl>()) {}
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
    HostRegion& region = HostRegion::storage();
    if (region.is_provisioned()) {
        err =
            "ring-alias: the region is already provisioned and therefore pinned; the overlay "
            "must precede HostRegion::provision()";
        return nullptr;
    }
    // Against what the region mapped, not kProvisionedCores: that bound was only right for
    // the fixed-size array this replaced.
    if (slots.size() > region.reserved_cores()) {
        err = fmt::format(
            "ring-alias: {} rings but the region was mapped for {} cores", slots.size(), region.reserved_cores());
        return nullptr;
    }

    std::unique_ptr<RingAlias> a(new RingAlias());
    Impl& im = *a->impl_;
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
                "ring-alias: core {} ring is {} B ({} B page-rounded) but an arena slot is only {} B",
                c,
                s.shm_size,
                rounded,
                kArenaBytes);
            return nullptr;
        }

        // MAP_FIXED past the end of the mapping SUCCEEDS, unmapping whatever VMA is there.
        // mmap cannot report it and the p != target check below cannot see it.
        const uint64_t arena_end = arena_offset_for(arena, c) + rounded;
        if (arena_end > region.region_bytes()) {
            err = fmt::format(
                "ring-alias: core {} arena ends at {} B, past the {} B the region mapped for {} cores",
                c,
                arena_end,
                region.region_bytes(),
                region.reserved_cores());
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
        region.declare_alias(arena, c, static_cast<uint64_t>(s.data_offset) + s.fifo_size, rounded);
    }
    return a;
}

uint8_t* RingAlias::base(uint32_t core) const { return core < impl_->base.size() ? impl_->base[core] : nullptr; }

uint32_t RingAlias::count() const { return static_cast<uint32_t>(impl_->base.size()); }

std::string RingAlias::describe() const {
    return fmt::format("{} arenas aliased for {} ring(s)", impl_->arena == AliasArena::Tx ? "TX" : "RX", count());
}

}  // namespace tt::tt_metal::experimental
