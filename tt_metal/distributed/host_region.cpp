// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/internal/host_region.hpp>

#include <sys/mman.h>
#include <sys/resource.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <fmt/format.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::experimental {

namespace {

// Sized to kRegionBytesMax even though only a prefix is pinned: the offsets must be
// compile-time constants for every legal core index. In .bss, so it costs nothing on disk.
alignas(kAlign2M) uint8_t g_region[kRegionBytesMax];

bool g_provisioned = false;

constexpr uint32_t kArenas = static_cast<uint32_t>(AliasArena::Count);
uint64_t g_alias_fill[kArenas][kProvisionedCores] = {};
uint64_t g_alias_mapped[kArenas][kProvisionedCores] = {};

uint64_t arena_offset_of(AliasArena a, uint32_t core) {
    return a == AliasArena::Tx ? tx_arena_offset(core) : rx_arena_offset(core);
}

// Makes every page resident so the pin does not fault the whole region inside an ioctl,
// and clears it -- minus the spans a socket's own metadata owns.
void zero_around_aliases(uint8_t* base, uint64_t want) {
    uint64_t cursor = 0;
    bool past_end = false;
    // TX precedes RX within a core and arenas ascend with the index, so one cursor is enough.
    for (uint32_t c = 0; c < kProvisionedCores && !past_end; ++c) {
        for (uint32_t a = 0; a < kArenas && !past_end; ++a) {
            if (g_alias_fill[a][c] == 0) {
                continue;
            }
            const uint64_t arena = arena_offset_of(static_cast<AliasArena>(a), c);
            const uint64_t hole_start = arena + g_alias_fill[a][c];
            if (hole_start >= want) {
                past_end = true;
                break;
            }
            if (hole_start > cursor) {
                std::memset(base + cursor, 0, hole_start - cursor);
            }
            cursor = std::min(arena + g_alias_mapped[a][c], want);
        }
    }
    if (cursor < want) {
        std::memset(base + cursor, 0, want - cursor);
    }
}

void validate_shape(uint32_t cores_in_use, HostTopology topology, HostRegion::Grid grid) {
    std::string m;
    // The only bound the arenas need: both legs assign core i to logical{i%w, i/w}, so
    // tt_uva_core_index() of a used core IS i. The full grid's extent is irrelevant -- do not re-add it.
    if (cores_in_use == 0 || cores_in_use > kProvisionedCores) {
        m = fmt::format("cores_in_use {} is outside 1..{}", cores_in_use, kProvisionedCores);
    } else if (!host_topology_ok(topology)) {
        m = fmt::format(
            "topology ident={} num={} chips_per_host={} does not fit the 12-bit UVA selector", topology.ident,
            topology.num, topology.chips_per_host);
    } else if (grid.width == 0 || grid.height == 0) {
        m = "grid width/height must be non-zero; they are part of the wire contract";
    } else if (cores_in_use > grid.width * grid.height) {
        m = fmt::format(
            "cores_in_use {} exceeds the {} a {}x{} grid has", cores_in_use, grid.width * grid.height, grid.width,
            grid.height);
    } else {
        return;
    }
    throw std::runtime_error(m);
}

// Published LAST, and with a release store: everything else must be true before a peer can
// find the magic and start computing offsets against it.
void publish_header(
    RegionHeader* h, uint32_t cores_in_use, HostTopology topology, HostRegion::Grid grid, uint32_t chip,
    uint64_t pinned_bytes, const HostRegion::DeviceView& dev) {
    std::memset(h, 0, sizeof(*h));
    h->version = kRegionVersion;
    h->provisioned_cores = kProvisionedCores;
    h->arena_bytes = kArenaBytes;
    h->arena_stride = kArenaStride;
    h->credit_array_bytes = kCreditArrayBytes;
    h->done_array_bytes = kDoneArrayBytes;
    h->arena_array_offset = kArenaArrayOffset;
    h->cores_in_use = cores_in_use;
    h->host_id = topology.ident;
    h->chips_per_host = topology.chips_per_host;
    h->chip = chip;
    h->grid_width = grid.width;
    h->grid_height = grid.height;
    h->pinned_bytes = pinned_bytes;
    h->device_io_base = dev.io_base;
    h->pcie_xy_enc = dev.pcie_xy_enc;
    __atomic_store_n(&h->magic, kRegionMagic, __ATOMIC_RELEASE);
}

}  // namespace

uint8_t* HostRegion::reserved_base() { return g_region; }
bool HostRegion::is_provisioned() { return g_provisioned; }

// Dropping the PinnedMemory is what unpins; the flag is what lets RingAlias overlay again.
void HostRegion::release() {
    HostRegion& r = storage();
    r.pinned_.reset();
    r.base_ = nullptr;
    r.pinned_bytes_ = 0;
    g_provisioned = false;
}

HostRegion& HostRegion::storage() {
    static HostRegion r;
    return r;
}

void HostRegion::declare_alias(AliasArena arena, uint32_t core, uint64_t fill_bytes, uint64_t mapped_bytes) {
    const uint32_t a = static_cast<uint32_t>(arena);
    if (core >= kProvisionedCores || a >= kArenas) {
        throw std::runtime_error("declare_alias: core or arena is outside the provisioned range");
    }
    // An overlay declared after the pin leaves the pin naming pages that are no longer here.
    if (g_provisioned) {
        throw std::runtime_error("declare_alias called after provisioning; the overlay must precede the pin");
    }
    if (fill_bytes != 0 && (fill_bytes > mapped_bytes || mapped_bytes > kArenaBytes)) {
        // Refused, not clamped: a clamp would quietly fill part of a ring's metadata.
        throw std::runtime_error(fmt::format(
            "declare_alias(core {}, fill {}, mapped {}) is not 0 < fill <= mapped <= {}", core, fill_bytes,
            mapped_bytes, kArenaBytes));
    }
    g_alias_fill[a][core] = fill_bytes;
    g_alias_mapped[a][core] = fill_bytes == 0 ? 0 : mapped_bytes;
}

void HostRegion::clear_aliases() {
    std::memset(g_alias_fill, 0, sizeof(g_alias_fill));
    std::memset(g_alias_mapped, 0, sizeof(g_alias_mapped));
}

uint64_t HostRegion::alias_fill_bytes(AliasArena arena, uint32_t core) {
    const uint32_t a = static_cast<uint32_t>(arena);
    return (core >= kProvisionedCores || a >= kArenas || g_alias_fill[a][core] == 0) ? kArenaBytes
                                                                                     : g_alias_fill[a][core];
}

uint64_t HostRegion::alias_tail_offset(AliasArena arena, uint32_t core) {
    const uint32_t a = static_cast<uint32_t>(arena);
    return (core >= kProvisionedCores || a >= kArenas || g_alias_fill[a][core] == 0) ? kArenaBytes
                                                                                     : g_alias_mapped[a][core];
}

PinLimits query_pin_limits(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    PinLimits out;
    rlimit rl{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        out.rlimit_memlock = (rl.rlim_cur == RLIM_INFINITY) ? UINT64_MAX : static_cast<uint64_t>(rl.rlim_cur);
    }
    const auto params = GetMemoryPinningParameters(*mesh_device);
    out.max_pins = params.max_pins;
    out.max_total_pin = params.max_total_pin_size;
    out.can_map_to_noc = params.can_map_to_noc;
    return out;
}

HostRegion& HostRegion::provision(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t chip,
    uint32_t cores_in_use,
    HostTopology topology,
    Grid grid) {
    if (g_provisioned) {
        throw std::runtime_error("HostRegion::provision called twice; reuse the reference the first call returned");
    }
    validate_shape(cores_in_use, topology, grid);

    const uint64_t want = pinned_bytes_for(cores_in_use);

    // Checked before pinning: RLIMIT_MEMLOCK surfaces from inside the ioctl as a bare
    // errno with no indication of how much was asked for.
    const PinLimits limits = query_pin_limits(mesh_device);
    if (limits.rlimit_memlock != UINT64_MAX && want > limits.rlimit_memlock) {
        throw std::runtime_error(fmt::format(
            "need {} MiB pinned for {} cores but RLIMIT_MEMLOCK is {} MiB (ulimit -l); each core costs {} MiB",
            want >> 20, cores_in_use, limits.rlimit_memlock >> 20, kArenaStride >> 20));
    }
    if (limits.max_total_pin != 0 && want > limits.max_total_pin) {
        throw std::runtime_error(fmt::format(
            "need {} MiB but the driver's max_total_pin_size is {} MiB", want >> 20, limits.max_total_pin >> 20));
    }

    uint8_t* const base = g_region;
    if (reinterpret_cast<uintptr_t>(base) % kAlign2M != 0 || want % kPageBytes != 0) {
        throw std::runtime_error("the region is misaligned or the pinned length is not a page multiple");
    }

    // Both advisory. Hugepages save TLB walks; DONTFORK matches what ibv_reg_mr will set
    // on the same range, so a child can never inherit pinned pages.
    (void)madvise(base, want, MADV_HUGEPAGE);
    (void)madvise(base, want, MADV_DONTFORK);
    zero_around_aliases(base, want);

    // PinnedMemory pins what this points at; it does not allocate. The no-op deleter is
    // because the storage is static and outlives everything.
    auto borrowed = std::shared_ptr<uint32_t[]>(reinterpret_cast<uint32_t*>(base), [](uint32_t*) {});
    HostBuffer view(ttsl::Span<uint32_t>(borrowed.get(), want / sizeof(uint32_t)), MemoryPin(borrowed));

    const auto coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
    tt::tt_metal::distributed::MeshCoordinateRangeSet range;
    range.merge(tt::tt_metal::distributed::MeshCoordinateRange(coord, coord));

    auto pinned = PinnedMemory::Create(*mesh_device, range, view, /*map_to_noc=*/true);
    if (!pinned) {
        throw std::runtime_error("PinnedMemory::Create returned null (is vIOMMU enabled?)");
    }
    const auto ids = mesh_device->get_device_ids();
    if (ids.empty()) {
        throw std::runtime_error("mesh device reports no device ids");
    }
    // get_noc_addr(), not usable_from_noc(): on Blackhole the latter is false by
    // construction while the former still returns the address the device uses.
    const auto noc = pinned->get_noc_addr(ids.front());
    if (!noc.has_value()) {
        throw std::runtime_error("PinnedMemory has no NOC address -- the device cannot reach the region");
    }

    HostRegion& r = storage();
    r.base_ = base;
    r.pinned_bytes_ = want;
    r.cores_in_use_ = cores_in_use;
    r.chip_ = chip;
    r.topology_ = topology;
    r.grid_ = grid;
    r.device_ = DeviceView{noc->pcie_xy_enc, noc->addr};
    r.pinned_ = std::move(pinned);

    r.reset_arenas();
    publish_header(r.header(), cores_in_use, topology, grid, chip, want, r.device_);
    g_provisioned = true;
    return r;
}

void HostRegion::reset_arenas(uint8_t fill) {
    // Each arena in two pieces: a ring may be mapped over its front, and the bytes between
    // the ring's data region and the end of that mapping belong to the socket.
    const auto fill_arena = [fill](uint8_t* arena, AliasArena which, uint32_t core) {
        const uint64_t tail = alias_tail_offset(which, core);
        std::memset(arena, fill, alias_fill_bytes(which, core));
        if (tail < kArenaBytes) {
            std::memset(arena + tail, fill, kArenaBytes - tail);
        }
    };
    for (uint32_t core = 0; core < cores_in_use_; ++core) {
        fill_arena(tx_arena(core), AliasArena::Tx, core);
        fill_arena(rx_arena(core), AliasArena::Rx, core);
    }
    // Every line of both arrays, not just the cores in use: a stale count in an unused entry
    // is what makes a sender's gate open on a message that was never consumed.
    std::memset(base_ + kCreditArrayOffset, 0, kCreditArrayBytes + kDoneArrayBytes);
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

std::string HostRegion::verify_header() const {
    const RegionHeader* h = header();
    if (__atomic_load_n(&h->magic, __ATOMIC_ACQUIRE) != kRegionMagic) {
        return "region magic absent -- unprovisioned, or a pointer into the wrong mapping";
    }
    // chips_per_host and grid_width are the dangerous two: a mismatch does not corrupt an
    // offset, it silently names a different core, and that core reads legitimately idle.
    const struct {
        const char* what;
        uint64_t got;
        uint64_t want;
    } checks[] = {
        {"version", h->version, kRegionVersion},
        {"provisioned_cores", h->provisioned_cores, kProvisionedCores},
        {"arena_bytes", h->arena_bytes, kArenaBytes},
        {"arena_stride", h->arena_stride, kArenaStride},
        {"credit_array_bytes", h->credit_array_bytes, kCreditArrayBytes},
        {"done_array_bytes", h->done_array_bytes, kDoneArrayBytes},
        {"arena_array_offset", h->arena_array_offset, kArenaArrayOffset},
        {"chips_per_host", h->chips_per_host, topology_.chips_per_host},
        {"grid_width", h->grid_width, grid_.width},
    };
    for (const auto& c : checks) {
        if (c.got != c.want) {
            return fmt::format(
                "region header disagrees on {}: published {}, this build has {}. Rebuild both sides from the "
                "same commit.",
                c.what, c.got, c.want);
        }
    }
    return {};
}

}  // namespace tt::tt_metal::experimental
