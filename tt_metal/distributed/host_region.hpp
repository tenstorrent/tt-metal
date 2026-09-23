// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The one region: a 2 MiB-aligned span holding the credit array and every core's arenas,
// mapped at runtime for the cores a run asks for, pinned once, exposed in the RMA window.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "tt_metal/distributed/host_uva_layout.hpp"
#include "tt_metal/distributed/host_uva.hpp"

namespace ttsl {
template <typename T>
class Indestructible;
}
namespace tt::tt_metal::distributed {
class MeshDevice;
}
namespace tt::tt_metal::experimental {
class PinnedMemory;
}

namespace tt::tt_metal::experimental {

// Which of a core's two arenas an overlay covers: TX is the D2H FIFO, RX the H2D ring.
enum class AliasArena : uint32_t { Tx = 0, Rx = 1, Count = 2 };

class HostRegion {
public:
    struct Grid {
        uint32_t width = 0;
        uint32_t height = 0;
    };

    // The one instance. Public because provision() and reserved_base() are members now:
    // a caller needs the region before it can ask the region for anything.
    static HostRegion& storage();

    // storage() hands out a reference and the private default ctor does NOT make the copy
    // ctor private: dropping the & would fork provisioned_ and the pin off the singleton.
    HostRegion(const HostRegion&) = delete;
    HostRegion& operator=(const HostRegion&) = delete;

    void provision(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
        uint32_t chip,
        uint32_t cores_in_use,
        HostTopology topology,
        Grid grid);

    bool is_provisioned() const { return provisioned_; }

    // Unpins and retracts the published magic; a later provision() at the same core count
    // may follow. Must run BEFORE the overlays are unmapped -- the pin names those pages.
    void release();

    // Maps the region for `cores_in_use` and hands back its 2 MiB-aligned base; must
    // precede the overlays, hence the count here. Not thread-safe, like the rest of this.
    uint8_t* reserved_base(uint32_t cores_in_use);

    // An overlay declares how much of an arena this region may still fill, and where the
    // tail it may fill again begins. Refused after provisioning: see RingAlias.
    void declare_alias(AliasArena arena, uint32_t core, uint64_t fill_bytes, uint64_t mapped_bytes);
    // Per arena: an overlay owns one of the two, and clearing both would erase the other
    // overlay's still-live declarations.
    void clear_aliases(AliasArena arena);
    uint64_t alias_fill_bytes(AliasArena arena, uint32_t core) const;
    uint64_t alias_tail_offset(AliasArena arena, uint32_t core) const;

    // The mapping's 2 MiB-aligned base, valid from reservation. Reading it does not
    // allocate -- reserved_base() is the entry point that maps.
    uint8_t* base() const { return region_; }

    // The mapping's real extent. An overlay MUST stay inside it: MAP_FIXED past the end
    // succeeds, unmapping whatever is there, and neither mmap nor the caller can see it.
    uint64_t region_bytes() const { return region_bytes_; }

    // What reserved_base() sized the mapping for. Set at reservation, so an overlay can
    // read it; cores_in_use() is still 0 at that point, being set by provision().
    uint32_t reserved_cores() const { return reserved_cores_; }

    uint64_t pinned_bytes() const { return pinned_bytes_; }
    uint32_t cores_in_use() const { return cores_in_use_; }

    // How the device addresses this region; both halves go to the socket metadata.
    struct DeviceView {
        uint32_t pcie_xy_enc = 0;
        uint64_t io_base = 0;
    };
    const DeviceView& device() const { return device_; }

    // Self-check only: reads the header THIS object published, so it catches a broken
    // publish, not a peer built from another commit. That would need the peer's header.
    std::string verify_header() const;

    static constexpr uint8_t kArenaFill = 0xA5;

private:
    HostRegion() = default;
    // Constructs the one instance in place; see storage(). Needed because the ctor above
    // is private and the destructor must never run -- the pin outlives the cluster.
    friend class ttsl::Indestructible<HostRegion>;

    // Complement-fills the arenas this region owns, so an unwritten byte always differs
    // and a test cannot pass by accident. Private: provision() is the only safe moment.
    void reset_arenas(uint8_t fill = kArenaFill);

    // Off region_, set at reservation. Callers must have reserved: storage() is public, so
    // being mapped is not something these three can assume.
    RegionHeader* header() const { return reinterpret_cast<RegionHeader*>(region_); }
    uint8_t* tx_arena(uint32_t core) const { return region_ + tx_arena_offset(core); }
    uint8_t* rx_arena(uint32_t core) const { return region_ + rx_arena_offset(core); }

    // Residency and clearing ahead of the pin, skipping the spans an overlay owns.
    void zero_around_aliases(uint64_t want);

    static constexpr uint32_t kArenas = static_cast<uint32_t>(AliasArena::Count);

    // The mapping, sized for the cores the run asked for. region_bytes_ is what a munmap
    // would need alongside region_; nothing frees it, so the mapping outlives every pin.
    uint8_t* region_ = nullptr;
    uint64_t region_bytes_ = 0;
    uint32_t reserved_cores_ = 0;
    bool provisioned_ = false;

    // What each overlay left this region free to write: fill_ is where a socket's own
    // metadata starts, mapped_ where the overlay ends. The gap between them is not ours.
    uint64_t alias_fill_[kArenas][kProvisionedCores]{};
    uint64_t alias_mapped_[kArenas][kProvisionedCores]{};

    uint64_t pinned_bytes_ = 0;
    uint32_t cores_in_use_ = 0;
    HostTopology topology_{};
    Grid grid_{};
    DeviceView device_{};
    std::shared_ptr<PinnedMemory> pinned_;
};

// Reported before anything is pinned, so an over-large request fails with a number rather
// than inside an ioctl.
struct PinLimits {
    uint64_t rlimit_memlock = 0;
    uint32_t max_pins = 0;
    uint64_t max_total_pin = 0;
    bool can_map_to_noc = false;
};
PinLimits query_pin_limits(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);

}  // namespace tt::tt_metal::experimental
