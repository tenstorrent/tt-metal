// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// One 2 MiB-aligned span holding a TX/RX arena pair per (LINK, RECEIVER CHANNEL), pinned once.
// Instantiable, unlike HostRegion: the bridge owns its own. ERISC_HOST_BRIDGE_CONTRACT.md §3.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hostdevcommon/erisc_bridge_layout.h"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_fabric::erisc_bridge {

using tt::tt_fabric::BridgeArena;

// One entry per cabled intermesh channel. A channel has exactly one peer, so peer_rank is an
// attribute, never an index dimension -- the arena index is the flattened (link, chan) pair.
struct LinkBinding {
    std::uint32_t peer_rank = 0;
    std::uint16_t local_mesh_id = 0;
    std::uint16_t local_chip_id = 0;
    std::uint16_t peer_mesh_id = 0;
    std::uint16_t peer_chip_id = 0;
    std::uint8_t local_chan = 0;
    std::uint8_t peer_chan = 0;
};

class EriscBridgeRegion {
public:
    EriscBridgeRegion();
    ~EriscBridgeRegion();
    EriscBridgeRegion(const EriscBridgeRegion&) = delete;
    EriscBridgeRegion& operator=(const EriscBridgeRegion&) = delete;

    // Each (link, receiver channel) gets one arena: a TX and an RX segment of ring_pages slots
    // each. Symmetric across ranks, so a peer computes the same offsets without an exchange.
    struct Geometry {
        std::uint32_t packet_capacity = 0;
        // TT_BRIDGE_RING_PAGES, default kBridgeDefaultRingPages. Tunable per platform like
        // TT_H2H_FLUSH_KB; depth buys rate and costs latency roughly linearly.
        std::uint32_t ring_pages = kBridgeDefaultRingPages;
        // FLAT across every VC. A per-VC count collides arenas silently -- index(1,0,1) ==
        // index(0,1,1) -- so take this from the builder's resolved receiver-channel count.
        std::uint32_t chans_per_link = 1;
    };

    // MUST run before any overlay: MAP_FIXED after the pin replaces pages the NIC was told
    // about. Fails with a number, not inside an ioctl, when the request exceeds the pin limits.
    std::uint8_t* reserve(const std::vector<LinkBinding>& links, const Geometry& geom, std::string& err);

    // Pins and publishes, via tt_metal's shared PinnedMemory. It pins what the view points at
    // and does not allocate, so the mapping must outlive every pin.
    bool provision(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
        std::uint32_t chip,
        std::string& err);
    bool is_provisioned() const;

    // Unpins and retracts the published magic. MUST run BEFORE the overlays are unmapped:
    // the pin names those pages. Dropping the PinnedMemory is what unpins.
    void release();

    // An overlay declares how much of an arena this region may still fill and where the tail
    // it may fill again begins. Refused after provisioning, like HostRegion::declare_alias.
    bool declare_alias(
        BridgeArena arena,
        std::uint32_t arena_idx,
        std::uint64_t fill_bytes,
        std::uint64_t mapped_bytes,
        std::string& err);
    void clear_aliases(BridgeArena arena);
    std::uint64_t alias_fill_bytes(BridgeArena arena, std::uint32_t arena_idx) const;
    std::uint64_t alias_tail_offset(BridgeArena arena, std::uint32_t arena_idx) const;

    std::uint8_t* base() const;
    std::uint64_t region_bytes() const;
    std::uint32_t link_count() const;
    // links x chans_per_link -- what every arena_idx below is bounded by.
    std::uint32_t arena_count() const;
    const Geometry& geometry() const;

    // The two directions of the flattening. arena_index() is bridge_arena_index() bound to
    // this region's geometry, so no caller repeats the multiply with the wrong count.
    std::uint32_t arena_index(std::uint32_t link_idx, std::uint32_t recv_chan) const;
    std::uint32_t link_for_arena(std::uint32_t arena_idx) const;
    std::uint32_t chan_for_arena(std::uint32_t arena_idx) const;

    // Resolve an arena's slot in the region; all return nullptr for an out-of-range index
    // rather than an address, so a bad arena_idx cannot reach another arena's span.
    std::uint8_t* slot(std::uint32_t arena_idx, BridgeArena arena, std::uint32_t slot_idx) const;
    std::uint8_t* desc(std::uint32_t arena_idx, BridgeArena arena, std::uint32_t slot_idx) const;
    std::uint8_t* credits(std::uint32_t arena_idx) const;

    // ARENAS reaching a rank -- what a send loop iterates, since several arenas share a peer.
    // links_to_rank() stays: a link is still one cable, just not the indexing dimension.
    static constexpr std::uint32_t npos = 0xFFFFFFFFu;
    std::vector<std::uint32_t> arenas_to_rank(std::uint32_t peer_rank) const;
    std::vector<std::uint32_t> links_to_rank(std::uint32_t peer_rank) const;
    const LinkBinding* binding(std::uint32_t link_idx) const;
    // The cable behind an arena, for the peer rank a put targets.
    const LinkBinding* binding_for_arena(std::uint32_t arena_idx) const;

    // How the device addresses this region; both halves go to the ERISC's compile-time args.
    struct DeviceView {
        std::uint32_t pcie_xy_enc = 0;
        std::uint64_t io_base = 0;
    };
    const DeviceView& device() const;

    // Complement fill, so an unwritten byte always differs and a test cannot pass by accident.
    static constexpr std::uint8_t kArenaFill = 0xA5;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Overlays an arena with a socket's shm ring so neither device leg needs a staging copy.
// After reserve(), before provision(). Its FIFO page MUST equal bridge_slot_size(capacity).
class BridgeArenaAlias {
public:
    // One per socket, taken straight off its descriptor: TX from the D2H socket the ERISC
    // writes, RX from the H2D socket it reads. The bridge never owns these pages.
    struct Slot {
        std::uint32_t arena_idx = 0;  // bridge_arena_index(link, recv_chan, chans_per_link)
        BridgeArena arena = BridgeArena::Tx;
        std::string shm_name;
        std::uint64_t shm_size = 0;
        std::uint32_t data_offset = 0;
        std::uint32_t fifo_size = 0;
    };

    static std::unique_ptr<BridgeArenaAlias> map(
        EriscBridgeRegion& region, const std::vector<Slot>& slots, std::string& err);

    // Restores anonymous pages over each slot and clears the region's declarations.
    ~BridgeArenaAlias();
    BridgeArenaAlias(const BridgeArenaAlias&) = delete;
    BridgeArenaAlias& operator=(const BridgeArenaAlias&) = delete;

    std::uint8_t* base(std::uint32_t arena_idx, BridgeArena arena) const;
    std::uint32_t count() const;
    std::string describe() const;

private:
    BridgeArenaAlias();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Reported before anything is pinned, so an over-large geometry fails with a number rather
// than inside an ioctl. Mirrors the D2H leg's query_pin_limits for the bridge's own region.
struct BridgePinLimits {
    std::uint64_t rlimit_memlock = 0;
    std::uint32_t max_pins = 0;
    std::uint64_t max_total_pin = 0;
    bool can_map_to_noc = false;
};
BridgePinLimits bridge_query_pin_limits(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);

}  // namespace tt::tt_fabric::erisc_bridge
