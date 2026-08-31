// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "host_region.hpp"

#include <sys/mman.h>
#include <sys/resource.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::experimental {

namespace {

// THE REGION ITSELF.
//
// One object, file-scope, 2 MiB aligned. Not a pointer someone can reassign, not a
// factory that could hand out a second one. `alignas` on a static array of this size is
// ordinary -- it lands in .bss, so it contributes nothing to the binary on disk, and
// pages become resident only as they are touched.
//
// Size is kRegionBytesMax (386 MiB at 128 provisioned cores) even though only a prefix
// is ever pinned, because the OFFSETS must be compile-time constants for every legal
// core index. A region sized to the cores actually in use would make bank_offset() a
// function of a runtime value, and the Tensix kernel computes those offsets too.
alignas(kAlign2M) uint8_t g_region[kRegionBytesMax];

bool g_provisioned = false;

// RX arenas that an external mapping has been laid over -- see declare_rx_alias(). Two
// numbers per core and nothing else: how much of the arena this region may fill, and where
// the part it may fill again begins. Zero means "not aliased", which is why every read goes
// through the accessors below rather than touching the arrays.
uint64_t g_rx_alias_fill[kProvisionedCores] = {};
uint64_t g_rx_alias_mapped[kProvisionedCores] = {};

static_assert(sizeof(g_region) == kRegionBytesMax, "the region is exactly the provisioned size");

// The pre-pin touch-and-zero, minus whatever an H2D ring overlay owns.
//
// The zeroing has two jobs and only one of them survives aliasing. It makes every page
// resident so the pin does not fault 386 MiB inside an ioctl -- and an overlaid page is
// resident already, because tt-metal faulted and pinned it when it built the socket. And it
// clears the region, which must not extend into a ring's bytes_acked. So the overlay's
// tt-metal half is skipped on both counts rather than zeroed carefully.
//
// Cores are walked in index order and arenas ascend with the index, so the holes come out
// sorted and one cursor is enough. Aliasing off: one hole-free memset, exactly as before.
void zero_around_aliases(uint8_t* base, uint64_t want) {
    uint64_t cursor = 0;
    for (uint32_t c = 0; c < kProvisionedCores; ++c) {
        if (g_rx_alias_fill[c] == 0) {
            continue;
        }
        const uint64_t hole_start = rx_arena_offset(c) + g_rx_alias_fill[c];
        const uint64_t hole_end = rx_arena_offset(c) + g_rx_alias_mapped[c];
        if (hole_start >= want) {
            break;
        }
        if (hole_start > cursor) {
            std::memset(base + cursor, 0, hole_start - cursor);
        }
        cursor = std::min(hole_end, want);
    }
    if (cursor < want) {
        std::memset(base + cursor, 0, want - cursor);
    }
}

}  // namespace

// Outside the anonymous namespace above, where g_region lives, but still inside
// t6_host_uva -- a member definition has to sit in the namespace enclosing its class.
uint8_t* HostRegion::reserved_base() { return g_region; }

void HostRegion::declare_rx_alias(uint32_t core, uint64_t fill_bytes, uint64_t mapped_bytes) {
    if (core >= kProvisionedCores) {
        throw std::runtime_error("declare_rx_alias: core index is outside the provisioned banks");
    }
    if (g_provisioned) {
        // The same refusal map_rings() makes, restated where the state actually lives. An
        // overlay declared after the pin is either a caller that skipped map_rings()' guard
        // or one that mapped twice; both leave the pin naming pages that are no longer here.
        throw std::runtime_error(
            "declare_rx_alias called after the region was provisioned; the overlay must precede the pin");
    }
    if (fill_bytes == 0) {
        g_rx_alias_fill[core] = 0;
        g_rx_alias_mapped[core] = 0;
        return;
    }
    if (fill_bytes > mapped_bytes || mapped_bytes > kArenaBytes) {
        // Refused, not clamped. A clamp here would quietly fill part of a ring's metadata,
        // which is the exact failure this whole mechanism exists to prevent.
        std::ostringstream m;
        m << "declare_rx_alias(core " << core << ", fill " << fill_bytes << ", mapped " << mapped_bytes
          << ") is not 0 < fill <= mapped <= " << kArenaBytes << " (one arena)";
        throw std::runtime_error(m.str());
    }
    g_rx_alias_fill[core] = fill_bytes;
    g_rx_alias_mapped[core] = mapped_bytes;
}

void HostRegion::clear_rx_aliases() {
    for (uint32_t c = 0; c < kProvisionedCores; ++c) {
        g_rx_alias_fill[c] = 0;
        g_rx_alias_mapped[c] = 0;
    }
}

uint64_t HostRegion::rx_fill_bytes(uint32_t core) {
    if (core >= kProvisionedCores || g_rx_alias_fill[core] == 0) {
        return kArenaBytes;
    }
    return g_rx_alias_fill[core];
}

uint64_t HostRegion::rx_tail_offset(uint32_t core) {
    if (core >= kProvisionedCores || g_rx_alias_fill[core] == 0) {
        return kArenaBytes;
    }
    return g_rx_alias_mapped[core];
}

HostRegion& HostRegion::storage() {
    static HostRegion instance;
    return instance;
}

PinLimits query_pin_limits(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    PinLimits out;

    rlimit rl{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        out.rlimit_memlock = (rl.rlim_cur == RLIM_INFINITY) ? UINT64_MAX : static_cast<uint64_t>(rl.rlim_cur);
    }

    const auto params = tt::tt_metal::experimental::GetMemoryPinningParameters(*mesh_device);
    out.max_pins = params.max_pins;
    out.max_total_pin = params.max_total_pin_size;
    out.can_map_to_noc = params.can_map_to_noc;
    return out;
}

namespace {

// The checks that do not involve a device, shared by both provisioning paths so the
// unpinned one cannot drift into accepting a geometry the pinned one rejects.
void validate_shape(uint32_t cores_in_use, HostTopology topology, HostRegion::Grid grid) {
    if (cores_in_use == 0 || cores_in_use > kProvisionedCores) {
        std::ostringstream m;
        m << "cores_in_use " << cores_in_use << " is outside 1.." << kProvisionedCores
          << " (kProvisionedCores in host_uva_layout.hpp)";
        throw std::runtime_error(m.str());
    }
    if (!host_topology_ok(topology)) {
        std::ostringstream m;
        m << "topology ident=" << topology.ident << " num=" << topology.num
          << " chips_per_host=" << topology.chips_per_host
          << " does not fit the 12-bit UVA selector; see host_uva.hpp host_topology_ok()";
        throw std::runtime_error(m.str());
    }
    if (grid.width == 0 || grid.height == 0) {
        throw std::runtime_error("grid width/height must be non-zero; they are part of the wire contract");
    }
    const uint32_t max_index = core_index(grid.width - 1, grid.height - 1, grid.width);
    if (max_index >= kProvisionedCores) {
        std::ostringstream m;
        m << "a " << grid.width << "x" << grid.height << " grid reaches core index " << max_index
          << ", which does not fit kProvisionedCores=" << kProvisionedCores << " in host_uva_layout.hpp";
        throw std::runtime_error(m.str());
    }
    if (cores_in_use > grid.width * grid.height) {
        std::ostringstream m;
        m << "cores_in_use " << cores_in_use << " exceeds the " << (grid.width * grid.height) << " cores a "
          << grid.width << "x" << grid.height << " grid has";
        throw std::runtime_error(m.str());
    }
}

}  // namespace

void publish_header(HostRegion& r, uint32_t cores_in_use, HostTopology topology, HostRegion::Grid grid,
                    uint32_t chip, uint64_t pinned_bytes, uint64_t io_base, uint32_t pcie_xy_enc);

HostRegion& HostRegion::provision_unpinned(uint32_t cores_in_use, HostTopology topology, Grid grid) {
    if (g_provisioned) {
        throw std::runtime_error("HostRegion::provision_unpinned called after the region was provisioned");
    }
    validate_shape(cores_in_use, topology, grid);

    uint8_t* const base = g_region;
    const uint64_t want = pinned_bytes_for(cores_in_use);
    (void)madvise(base, want, MADV_HUGEPAGE);
    // Through the same helper as the pinned path, though nothing device-free can have an
    // overlay: the two provisioning paths are kept from drifting on purpose, and a zeroing
    // rule that holds in one and not the other is exactly the kind of divergence that makes
    // a self-test stop being evidence about the real thing.
    zero_around_aliases(base, want);

    storage().base_ = base;
    storage().pinned_bytes_ = want;
    storage().cores_in_use_ = cores_in_use;
    storage().chip_ = 0;
    storage().topology_ = topology;
    storage().grid_ = grid;
    storage().device_ = DeviceView{};  // no device: io_base stays 0, and is_pinned() is false
    storage().reset_banks_and_arenas();
    publish_header(storage(), cores_in_use, topology, grid, 0, want, 0, 0);
    g_provisioned = true;
    return storage();
}

HostRegion& HostRegion::provision(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t chip,
    uint32_t cores_in_use,
    HostTopology topology,
    Grid grid) {
    if (g_provisioned) {
        throw std::runtime_error("HostRegion::provision called twice in one process; use attached()");
    }
    validate_shape(cores_in_use, topology, grid);

    const uint64_t want = pinned_bytes_for(cores_in_use);

    // Check the limits BEFORE pinning. An RLIMIT_MEMLOCK failure surfaces from inside
    // TENSTORRENT_IOCTL_PIN_PAGES as a bare errno with no indication of how much was
    // asked for, which is a poor way to learn you need `ulimit -l`.
    const PinLimits limits = query_pin_limits(mesh_device);
    if (limits.rlimit_memlock != UINT64_MAX && want > limits.rlimit_memlock) {
        std::ostringstream m;
        m << "need " << (want >> 20) << " MiB pinned for " << cores_in_use << " cores but RLIMIT_MEMLOCK is "
          << (limits.rlimit_memlock >> 20) << " MiB. Raise it (ulimit -l) or run fewer cores: each core costs "
          << (kArenaStride >> 20) << " MiB of arena.";
        throw std::runtime_error(m.str());
    }
    if (limits.max_total_pin != 0 && want > limits.max_total_pin) {
        std::ostringstream m;
        m << "need " << (want >> 20) << " MiB pinned but the driver's max_total_pin_size is "
          << (limits.max_total_pin >> 20) << " MiB";
        throw std::runtime_error(m.str());
    }

    uint8_t* const base = g_region;
    if (reinterpret_cast<uintptr_t>(base) % kAlign2M != 0) {
        throw std::runtime_error("the static region is not 2 MiB aligned; alignas was dropped");
    }
    // map_for_dma() throws unless the VA is page-aligned and the length is a page
    // multiple. Both are guaranteed by the layout's static_asserts, so a failure here
    // means the layout header and this file disagree -- worth catching as an assertion
    // rather than as an ioctl error.
    if (want % kPageBytes != 0) {
        throw std::runtime_error("pinned length is not a page multiple; host_uva_layout.hpp is inconsistent");
    }

    // Ask for hugepages. Advisory: if the kernel declines, everything still works, the
    // poller just walks more TLB entries. Failure is not an error.
    (void)madvise(base, want, MADV_HUGEPAGE);
    // Keep the pinned prefix out of any child's address space. Nothing here forks, and
    // this makes that a property of the mapping rather than a convention -- ibv_reg_mr
    // will set the same flag on the same range when the transport registers it.
    (void)madvise(base, want, MADV_DONTFORK);

    // Touch every page before pinning. Pinning a page that is not yet resident forces
    // the fault inside the ioctl, and a 386 MiB region's worth of faults there is both
    // slow and attributed to the wrong place in a profile. Skips the spans an H2D ring
    // overlay owns -- see zero_around_aliases().
    zero_around_aliases(base, want);

    // The borrowed-HostBuffer pattern from tt-metal's own D2HSocket: a shared_ptr with a
    // no-op deleter (the storage is static and outlives everything), a MemoryPin over it,
    // and a HostBuffer viewing the span. PinnedMemory::Create pins what this points at;
    // it does not allocate.
    auto borrowed = std::shared_ptr<uint32_t[]>(reinterpret_cast<uint32_t*>(base), [](uint32_t*) {});
    tt::tt_metal::HostBuffer view(
        ttsl::Span<uint32_t>(borrowed.get(), want / sizeof(uint32_t)), tt::tt_metal::MemoryPin(borrowed));

    const auto device_coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
    tt::tt_metal::distributed::MeshCoordinateRangeSet range;
    range.merge(tt::tt_metal::distributed::MeshCoordinateRange(device_coord, device_coord));

    auto pinned = tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, range, view, /*map_to_noc=*/true);
    if (!pinned) {
        throw std::runtime_error("PinnedMemory::Create returned null (vIOMMU enabled?)");
    }

    // get_device_ids() rather than get_device(coord)->id(): the latter is deprecated
    // ("retrieving physical devices can fail in distributed contexts", removed after
    // 2026-02-28) and this is a unit mesh, so its single id is the one we want.
    const auto device_ids = mesh_device->get_device_ids();
    if (device_ids.empty()) {
        throw std::runtime_error("mesh device reports no device ids");
    }
    const auto device_id = device_ids.front();
    const auto noc = pinned->get_noc_addr(device_id);
    // Gate on get_noc_addr(), NOT usable_from_noc(): on Blackhole the latter is false by
    // construction because 64-bit PCIe addressing makes PinnedMemory force map_to_noc
    // off, while the former still returns the address the device actually uses.
    if (!noc.has_value()) {
        throw std::runtime_error(
            "PinnedMemory has no NOC address for this device -- the T6 cannot reach the region. "
            "vIOMMU must be enabled for pinned memory to be device-visible.");
    }

    storage().base_ = base;
    storage().pinned_bytes_ = want;
    storage().cores_in_use_ = cores_in_use;
    storage().chip_ = chip;
    storage().topology_ = topology;
    storage().grid_ = grid;
    storage().device_ = DeviceView{noc->pcie_xy_enc, noc->addr};
    storage().pinned_ = std::move(pinned);

    storage().reset_banks_and_arenas();

    publish_header(storage(), cores_in_use, topology, grid, chip, want, storage().device_.io_base,
                   storage().device_.pcie_xy_enc);

    g_provisioned = true;
    return storage();
}

// Publish the header LAST, in both paths. Everything else must be true before a peer or
// a tool can find the magic and start computing offsets against it -- the magic store is
// a release for exactly that reason.
void publish_header(HostRegion& r, uint32_t cores_in_use, HostTopology topology, HostRegion::Grid grid,
                    uint32_t chip, uint64_t pinned_bytes, uint64_t io_base, uint32_t pcie_xy_enc) {
    RegionHeader* h = r.header();
    std::memset(h, 0, sizeof(*h));
    h->version = static_cast<uint32_t>(kCtrlVersion);
    h->provisioned_cores = kProvisionedCores;
    h->arena_bytes = kArenaBytes;
    h->arena_stride = kArenaStride;
    h->bank_bytes = kBankBytes;
    h->arena_array_offset = kArenaArrayOffset;
    h->cores_in_use = cores_in_use;
    h->host_id = topology.ident;
    h->chips_per_host = topology.chips_per_host;
    h->chip = chip;
    h->grid_width = grid.width;
    h->grid_height = grid.height;
    h->pinned_bytes = pinned_bytes;
    h->device_io_base = io_base;
    h->pcie_xy_enc = pcie_xy_enc;
    __atomic_store_n(&h->magic, kRegionMagic, __ATOMIC_RELEASE);
}

HostRegion& HostRegion::attached() {
    if (!g_provisioned) {
        throw std::runtime_error("HostRegion::attached() before provision()");
    }
    return storage();
}

bool HostRegion::is_provisioned() { return g_provisioned; }

void HostRegion::reset_banks_and_arenas(uint8_t fill) {
    // Arenas first, control words after -- see the header comment. A reader that races
    // this sees an idle bank, never a live control word over a half-filled arena.
    for (uint32_t core = 0; core < cores_in_use_; ++core) {
        std::memset(tx_arena(core), fill, kArenaBytes);
        // The RX arena in two pieces, because an H2D ring may be mapped over its front and
        // the bytes between the ring's data region and the end of that mapping belong to
        // tt-metal -- bytes_acked and the connector state. Both spans are kArenaBytes and 0
        // long respectively when nothing is aliased, so the unaliased case is one memset as
        // it always was. See declare_rx_alias().
        const uint64_t fill_bytes = rx_fill_bytes(core);
        const uint64_t tail_off = rx_tail_offset(core);
        std::memset(rx_arena(core), fill, fill_bytes);
        if (tail_off < kArenaBytes) {
            std::memset(rx_arena(core) + tail_off, fill, kArenaBytes - tail_off);
        }
    }
    // Every bank, not just the cores in use: an armed word in a bank nobody is running
    // is exactly the kind of leftover that makes a poller service a phantom request.
    // The earlier design paid 144 phantom transfers to learn this.
    std::memset(base_ + kHeaderBytes, 0, kBankArrayBytes);
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

std::string HostRegion::verify_header() const {
    const RegionHeader* h = header();
    std::ostringstream m;
    if (__atomic_load_n(&h->magic, __ATOMIC_ACQUIRE) != kRegionMagic) {
        m << "region magic is " << std::hex << h->magic << " not " << kRegionMagic
          << " -- unprovisioned region, or a pointer into the wrong mapping";
        return m.str();
    }
    auto mismatch = [&m](const char* what, uint64_t got, uint64_t want) {
        m << "region header disagrees on " << what << ": published " << got << ", this build has " << want
          << ". Two parties with different geometry read different 64-byte lines for the same core and both "
             "see idle. Rebuild both sides from the same commit.";
    };
    if (h->version != kCtrlVersion) {
        mismatch("version", h->version, kCtrlVersion);
        return m.str();
    }
    if (h->provisioned_cores != kProvisionedCores) {
        mismatch("provisioned_cores", h->provisioned_cores, kProvisionedCores);
        return m.str();
    }
    if (h->arena_bytes != kArenaBytes) {
        mismatch("arena_bytes", h->arena_bytes, kArenaBytes);
        return m.str();
    }
    if (h->arena_stride != kArenaStride) {
        mismatch("arena_stride", h->arena_stride, kArenaStride);
        return m.str();
    }
    if (h->bank_bytes != kBankBytes) {
        mismatch("bank_bytes", h->bank_bytes, kBankBytes);
        return m.str();
    }
    if (h->arena_array_offset != kArenaArrayOffset) {
        mismatch("arena_array_offset", h->arena_array_offset, kArenaArrayOffset);
        return m.str();
    }
    // chips_per_host is the one CLAUDE.md singles out, because a mismatch here does not
    // corrupt an offset -- it silently names a different core on a different host.
    if (h->chips_per_host != topology_.chips_per_host) {
        mismatch("chips_per_host", h->chips_per_host, topology_.chips_per_host);
        return m.str();
    }
    // grid_width has the same failure shape as chips_per_host and gets the same check:
    // a disagreement names a different physical core for the same index, and the bank it
    // points at is legitimately idle, so nothing downstream reports it.
    if (h->grid_width != grid_.width) {
        mismatch("grid_width", h->grid_width, grid_.width);
        return m.str();
    }
    return {};
}

}  // namespace tt::tt_metal::experimental
