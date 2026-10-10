// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/erisc_bridge_region.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_fabric::erisc_bridge {

namespace {

using tt::tt_metal::experimental::PinnedMemory;

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) & ~(a - 1); }

}  // namespace

// Named, not anonymous: Impl has external linkage and holds a std::vector<AliasSpan> member, so
// internal linkage here is -Wsubobject-linkage, an error under gcc -Werror. Clang does not warn.
namespace erisc_region_detail {

// An overlay owns [0, mapped); the region may still fill [0, fill) and everything past
// mapped. The gap between them is the socket's own metadata and is never ours to touch.
struct AliasSpan {
    std::uint64_t fill = 0;
    std::uint64_t mapped = 0;
};

}  // namespace erisc_region_detail

struct EriscBridgeRegion::Impl {
    std::uint8_t* base = nullptr;
    std::uint64_t bytes = 0;
    std::vector<LinkBinding> links;
    Geometry geom{};
    bool provisioned = false;
    std::uint64_t pinned_bytes = 0;
    DeviceView device{};
    std::shared_ptr<PinnedMemory> pinned;
    // [arena][arena_idx]; declared before provisioning, cleared when an overlay is dropped.
    std::vector<erisc_region_detail::AliasSpan> alias[kBridgeArenas];

    std::uint64_t segment_bytes() const { return bridge_segment_bytes(geom.ring_pages, geom.packet_capacity); }
    std::uint32_t arenas() const {
        return bridge_arena_count(static_cast<std::uint32_t>(links.size()), geom.chans_per_link);
    }
    bool in_range(std::uint32_t arena_idx) const { return arena_idx < arenas(); }

    // Residency and clearing ahead of the pin, skipping the spans an overlay owns.
    void zero_around_aliases() {
        const std::uint32_t n = arenas();
        std::memset(base, 0, bridge_control_bytes(n, geom.ring_pages));
        const std::uint64_t seg = segment_bytes();
        for (std::uint32_t i = 0; i < n; ++i) {
            for (std::uint32_t a = 0; a < kBridgeArenas; ++a) {
                std::uint8_t* p = base + bridge_segment_offset(
                                             i, static_cast<BridgeArena>(a), n, geom.ring_pages, geom.packet_capacity);
                const erisc_region_detail::AliasSpan s = alias[a][i];
                std::memset(p, EriscBridgeRegion::kArenaFill, static_cast<size_t>(s.fill));
                if (s.mapped < seg) {
                    std::memset(p + s.mapped, EriscBridgeRegion::kArenaFill, static_cast<size_t>(seg - s.mapped));
                }
            }
        }
    }
};

EriscBridgeRegion::EriscBridgeRegion() : impl_(std::make_unique<Impl>()) {}

EriscBridgeRegion::~EriscBridgeRegion() {
    release();
    // The mapping outlives every pin and is only dropped here, after the pin is gone.
    if (impl_->base != nullptr) {
        ::munmap(impl_->base, impl_->bytes);
    }
}

BridgePinLimits bridge_query_pin_limits(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    BridgePinLimits out;
    rlimit rl{};
    if (::getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        out.rlimit_memlock = (rl.rlim_cur == RLIM_INFINITY) ? UINT64_MAX : static_cast<std::uint64_t>(rl.rlim_cur);
    }
    const auto params = tt::tt_metal::experimental::GetMemoryPinningParameters(*mesh_device);
    out.max_pins = params.max_pins;
    out.max_total_pin = params.max_total_pin_size;
    out.can_map_to_noc = params.can_map_to_noc;
    return out;
}

std::uint8_t* EriscBridgeRegion::reserve(
    const std::vector<LinkBinding>& links, const Geometry& geom, std::string& err) {
    if (impl_->base != nullptr) {
        err = "erisc bridge region: already reserved";
        return nullptr;
    }
    if (links.empty() || geom.packet_capacity == 0 || geom.ring_pages == 0 || geom.chans_per_link == 0) {
        err = "erisc bridge region: links, packet_capacity, ring_pages and chans_per_link must all be non-zero";
        return nullptr;
    }
    // chans_per_link is FLAT across every VC. A per-VC count collides arenas silently --
    // index(1,0,1) == index(0,1,1) -- and this is the only place that can be caught.
    const auto n = bridge_arena_count(static_cast<std::uint32_t>(links.size()), geom.chans_per_link);
    const std::uint64_t want =
        align_up(bridge_region_bytes(n, geom.ring_pages, geom.packet_capacity), kBridgeRegionAlign);

    // Reported before anything is pinned, so an over-large geometry fails with a number
    // rather than inside an ioctl.
    void* p = ::mmap(
        nullptr, want + kBridgeRegionAlign, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        err = "erisc bridge region: mmap of " + std::to_string(want) + " B failed: " + std::strerror(errno);
        return nullptr;
    }
    // Trim to a 2 MiB-aligned base; the slack either side is returned so nothing leaks.
    auto raw = reinterpret_cast<std::uintptr_t>(p);
    auto aligned = static_cast<std::uintptr_t>(align_up(raw, kBridgeRegionAlign));
    if (aligned != raw) {
        ::munmap(p, aligned - raw);
    }
    const std::uintptr_t tail = aligned + want;
    const std::uintptr_t end = raw + want + kBridgeRegionAlign;
    if (end > tail) {
        ::munmap(reinterpret_cast<void*>(tail), end - tail);
    }

    impl_->base = reinterpret_cast<std::uint8_t*>(aligned);
    impl_->bytes = want;
    impl_->links = links;
    impl_->geom = geom;
    for (auto& v : impl_->alias) {
        v.assign(n, erisc_region_detail::AliasSpan{});
    }
    return impl_->base;
}

bool EriscBridgeRegion::declare_alias(
    BridgeArena arena,
    std::uint32_t arena_idx,
    std::uint64_t fill_bytes,
    std::uint64_t mapped_bytes,
    std::string& err) {
    if (impl_->provisioned) {
        err = "erisc bridge region: declare_alias after provision";
        return false;
    }
    if (!impl_->in_range(arena_idx)) {
        err = "erisc bridge region: arena_idx " + std::to_string(arena_idx) + " out of range";
        return false;
    }
    // MAP_FIXED past a segment succeeds and unmaps whatever is there, and neither mmap nor
    // the caller can see it, so the extent is checked here instead.
    const std::uint64_t seg = impl_->segment_bytes();
    if (mapped_bytes > seg || fill_bytes > mapped_bytes) {
        err = "erisc bridge region: alias span " + std::to_string(mapped_bytes) + " exceeds segment " +
              std::to_string(seg);
        return false;
    }
    impl_->alias[static_cast<std::uint32_t>(arena)][arena_idx] =
        erisc_region_detail::AliasSpan{fill_bytes, mapped_bytes};
    return true;
}

void EriscBridgeRegion::clear_aliases(BridgeArena arena) {
    auto& v = impl_->alias[static_cast<std::uint32_t>(arena)];
    std::fill(v.begin(), v.end(), erisc_region_detail::AliasSpan{});
}

std::uint64_t EriscBridgeRegion::alias_fill_bytes(BridgeArena arena, std::uint32_t arena_idx) const {
    return impl_->in_range(arena_idx) ? impl_->alias[static_cast<std::uint32_t>(arena)][arena_idx].fill : 0;
}
std::uint64_t EriscBridgeRegion::alias_tail_offset(BridgeArena arena, std::uint32_t arena_idx) const {
    return impl_->in_range(arena_idx) ? impl_->alias[static_cast<std::uint32_t>(arena)][arena_idx].mapped : 0;
}

bool EriscBridgeRegion::provision(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, std::uint32_t chip, std::string& err) {
    (void)chip;
    if (impl_->base == nullptr) {
        err = "erisc bridge region: provision before reserve";
        return false;
    }
    if (impl_->provisioned) {
        err = "erisc bridge region: already provisioned";
        return false;
    }
    const std::uint64_t want = impl_->bytes;
    const BridgePinLimits lim = bridge_query_pin_limits(mesh_device);
    if (lim.max_total_pin != 0 && want > lim.max_total_pin) {
        err = "erisc bridge region: " + std::to_string(want) + " B exceeds max_total_pin " +
              std::to_string(lim.max_total_pin);
        return false;
    }
    if (want > lim.rlimit_memlock) {
        err = "erisc bridge region: " + std::to_string(want) + " B exceeds RLIMIT_MEMLOCK " +
              std::to_string(lim.rlimit_memlock);
        return false;
    }

    // Only acts on a 2 MiB range it covers fully, which is what `want` is rounded up for.
    (void)::madvise(impl_->base, want, MADV_HUGEPAGE);
    (void)::madvise(impl_->base, want, MADV_DONTFORK);
    impl_->zero_around_aliases();

    // PinnedMemory pins what this points at; it does not allocate. The no-op deleter is
    // because this region owns the mapping and unmaps it only in the destructor.
    auto borrowed =
        std::shared_ptr<std::uint32_t[]>(reinterpret_cast<std::uint32_t*>(impl_->base), [](std::uint32_t*) {});
    tt::tt_metal::HostBuffer view(
        ttsl::Span<std::uint32_t>(borrowed.get(), want / sizeof(std::uint32_t)), tt::tt_metal::MemoryPin(borrowed));

    const auto coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
    tt::tt_metal::distributed::MeshCoordinateRangeSet range;
    range.merge(tt::tt_metal::distributed::MeshCoordinateRange(coord, coord));

    auto pinned = PinnedMemory::Create(*mesh_device, range, view, /*map_to_noc=*/true);
    if (!pinned) {
        err = "erisc bridge region: PinnedMemory::Create returned null (is vIOMMU enabled?)";
        return false;
    }
    const auto ids = mesh_device->get_device_ids();
    if (ids.empty()) {
        err = "erisc bridge region: mesh device reports no device ids";
        return false;
    }
    // get_noc_addr(), not usable_from_noc(): on Blackhole the latter is false by
    // construction while the former still returns the address the device uses.
    const auto noc = pinned->get_noc_addr(ids.front());
    if (!noc.has_value()) {
        err = "erisc bridge region: no NOC address -- the device cannot reach the region";
        return false;
    }

    // get_noc_addr() returns a NocAddr{pcie_xy_enc, addr, device_id}, not a bare address. The
    // encoding comes free here; it used to have to be looked up separately.
    impl_->device.pcie_xy_enc = noc->pcie_xy_enc;
    impl_->device.io_base = noc->addr;
    impl_->pinned_bytes = want;
    impl_->pinned = std::move(pinned);
    impl_->provisioned = true;
    return true;
}

void EriscBridgeRegion::release() {
    // Dropping the PinnedMemory is what unpins; the flag is what lets an overlay map again.
    impl_->pinned.reset();
    impl_->pinned_bytes = 0;
    impl_->provisioned = false;
}

bool EriscBridgeRegion::is_provisioned() const { return impl_->provisioned; }
std::uint8_t* EriscBridgeRegion::base() const { return impl_->base; }
std::uint64_t EriscBridgeRegion::region_bytes() const { return impl_->bytes; }
std::uint32_t EriscBridgeRegion::link_count() const { return static_cast<std::uint32_t>(impl_->links.size()); }
std::uint32_t EriscBridgeRegion::arena_count() const { return impl_->arenas(); }
const EriscBridgeRegion::Geometry& EriscBridgeRegion::geometry() const { return impl_->geom; }
const EriscBridgeRegion::DeviceView& EriscBridgeRegion::device() const { return impl_->device; }

// Bound to this region's geometry so no caller repeats the multiply with the wrong count.
std::uint32_t EriscBridgeRegion::arena_index(std::uint32_t link_idx, std::uint32_t recv_chan) const {
    return bridge_arena_index(link_idx, recv_chan, impl_->geom.chans_per_link);
}
std::uint32_t EriscBridgeRegion::link_for_arena(std::uint32_t arena_idx) const {
    return impl_->in_range(arena_idx) ? arena_idx / impl_->geom.chans_per_link : npos;
}
std::uint32_t EriscBridgeRegion::chan_for_arena(std::uint32_t arena_idx) const {
    return impl_->in_range(arena_idx) ? arena_idx % impl_->geom.chans_per_link : npos;
}

const LinkBinding* EriscBridgeRegion::binding(std::uint32_t link_idx) const {
    return link_idx < impl_->links.size() ? &impl_->links[link_idx] : nullptr;
}

const LinkBinding* EriscBridgeRegion::binding_for_arena(std::uint32_t arena_idx) const {
    const std::uint32_t link = link_for_arena(arena_idx);
    return link == npos ? nullptr : binding(link);
}

// The send loop iterates arenas, not links: the flush unit is the rank and several arenas
// share a peer. Ascending, so a peer's puts are posted in a deterministic order.
std::vector<std::uint32_t> EriscBridgeRegion::arenas_to_rank(std::uint32_t peer_rank) const {
    std::vector<std::uint32_t> out;
    const std::uint32_t n = impl_->arenas();
    for (std::uint32_t a = 0; a < n; ++a) {
        if (impl_->links[a / impl_->geom.chans_per_link].peer_rank == peer_rank) {
            out.push_back(a);
        }
    }
    return out;
}

std::vector<std::uint32_t> EriscBridgeRegion::links_to_rank(std::uint32_t peer_rank) const {
    std::vector<std::uint32_t> out;
    for (std::uint32_t i = 0; i < impl_->links.size(); ++i) {
        if (impl_->links[i].peer_rank == peer_rank) {
            out.push_back(i);
        }
    }
    return out;
}

// nullptr rather than an address for a bad index, so a wrong arena cannot reach another's span.
std::uint8_t* EriscBridgeRegion::slot(std::uint32_t arena_idx, BridgeArena arena, std::uint32_t slot_idx) const {
    if (!impl_->in_range(arena_idx) || slot_idx >= impl_->geom.ring_pages) {
        return nullptr;
    }
    return impl_->base +
           bridge_segment_offset(arena_idx, arena, arena_count(), impl_->geom.ring_pages, impl_->geom.packet_capacity) +
           bridge_slot_offset_in_segment(slot_idx, impl_->geom.packet_capacity);
}

std::uint8_t* EriscBridgeRegion::desc(std::uint32_t arena_idx, BridgeArena arena, std::uint32_t slot_idx) const {
    if (!impl_->in_range(arena_idx) || slot_idx >= impl_->geom.ring_pages) {
        return nullptr;
    }
    return impl_->base + bridge_desc_offset(arena_idx, arena, slot_idx, impl_->geom.ring_pages);
}

std::uint8_t* EriscBridgeRegion::credits(std::uint32_t arena_idx) const {
    if (!impl_->in_range(arena_idx)) {
        return nullptr;
    }
    return impl_->base + bridge_credit_offset(arena_idx, impl_->geom.ring_pages);
}

// ---------------------------------------------------------------------------------------

struct BridgeArenaAlias::Impl {
    EriscBridgeRegion* region = nullptr;
    struct Mapped {
        std::uint8_t* base = nullptr;
        std::uint64_t bytes = 0;
        std::uint32_t arena_idx = 0;
        BridgeArena arena = BridgeArena::Tx;
    };
    std::vector<Mapped> mapped;
};

BridgeArenaAlias::BridgeArenaAlias() : impl_(std::make_unique<Impl>()) {}

std::unique_ptr<BridgeArenaAlias> BridgeArenaAlias::map(
    EriscBridgeRegion& region, const std::vector<Slot>& slots, std::string& err) {
    if (region.base() == nullptr) {
        err = "bridge alias: region not reserved";
        return nullptr;
    }
    if (region.is_provisioned()) {
        err = "bridge alias: region already provisioned -- the pin would name the old pages";
        return nullptr;
    }
    auto self = std::unique_ptr<BridgeArenaAlias>(new BridgeArenaAlias());
    self->impl_->region = &region;
    const auto& geom = region.geometry();
    const std::uint64_t seg = bridge_segment_bytes(geom.ring_pages, geom.packet_capacity);

    for (const auto& s : slots) {
        // The fifo, not the whole shm: it is [metadata | fifo], so mapping from 0 puts the
        // header where slot 0 belongs. data_offset is page aligned, hence a legal mmap offset.
        const std::uint64_t bytes = s.fifo_size != 0 ? s.fifo_size : (s.shm_size - s.data_offset);
        if (bytes > seg) {
            err = "bridge alias: fifo of " + s.shm_name + " is " + std::to_string(bytes) + " B, exceeds segment " +
                  std::to_string(seg);
            return nullptr;
        }
        std::uint8_t* at = region.slot(s.arena_idx, s.arena, 0);
        if (at == nullptr) {
            err = "bridge alias: arena_idx " + std::to_string(s.arena_idx) + " out of range";
            return nullptr;
        }
        const int fd = ::shm_open(s.shm_name.c_str(), O_RDWR, 0);
        if (fd < 0) {
            err = "bridge alias: shm_open(" + s.shm_name + ") failed: " + std::strerror(errno);
            return nullptr;
        }
        // MAP_FIXED over the segment so the NIC touches the socket's own pages, and from
        // data_offset so segment[0] is fifo slot 0 with no correction elsewhere.
        void* p = ::mmap(
            at,
            static_cast<size_t>(bytes),
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_FIXED,
            fd,
            static_cast<off_t>(s.data_offset));
        ::close(fd);
        if (p == MAP_FAILED) {
            err = "bridge alias: MAP_FIXED of " + s.shm_name + " failed: " + std::strerror(errno);
            return nullptr;
        }
        // mmap maps whole pages: the tail page holds the socket's bytes_sent, so never paint it.
        const std::uint64_t mapped = bridge_align_up(bytes, kBridgeSegmentAlign);
        if (!region.declare_alias(s.arena, s.arena_idx, /*fill_bytes=*/0, mapped, err)) {
            return nullptr;
        }
        self->impl_->mapped.push_back({reinterpret_cast<std::uint8_t*>(p), mapped, s.arena_idx, s.arena});
    }
    return self;
}

BridgeArenaAlias::~BridgeArenaAlias() {
    if (impl_->region == nullptr) {
        return;
    }
    // Anonymous pages back over each slot, so the region's mapping stays whole.
    for (const auto& m : impl_->mapped) {
        (void)::mmap(
            m.base,
            static_cast<size_t>(m.bytes),
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
            -1,
            0);
    }
    impl_->region->clear_aliases(BridgeArena::Tx);
    impl_->region->clear_aliases(BridgeArena::Rx);
}

std::uint8_t* BridgeArenaAlias::base(std::uint32_t arena_idx, BridgeArena arena) const {
    for (const auto& m : impl_->mapped) {
        if (m.arena_idx == arena_idx && m.arena == arena) {
            return m.base;
        }
    }
    return nullptr;
}

std::uint32_t BridgeArenaAlias::count() const { return static_cast<std::uint32_t>(impl_->mapped.size()); }

std::string BridgeArenaAlias::describe() const {
    std::string s = "BridgeArenaAlias{" + std::to_string(impl_->mapped.size()) + " overlays";
    for (const auto& m : impl_->mapped) {
        s += ", arena " + std::to_string(m.arena_idx) + (m.arena == BridgeArena::Tx ? " tx " : " rx ") +
             std::to_string(m.bytes) + "B";
    }
    return s + "}";
}

}  // namespace tt::tt_fabric::erisc_bridge
