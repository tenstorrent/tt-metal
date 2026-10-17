// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Picks virtual cables (experimental); pure over global descriptors, so every rank agrees.

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

namespace tt::tt_fabric::experimental {

enum class BridgeReason : std::uint8_t {
    Downed,    // the factory system descriptor expects it; the live system does not have it
    Injected,  // up, but a test asked to treat it as broken (TT_METAL_E2H_BRIDGE_BREAK_LINKS)
};

// One end of a cable: a chip, by its live ASIC id, and the eth channel the cable leaves it through.
struct CableEnd {
    tt::tt_metal::AsicID asic{0};
    std::uint8_t chan = 0;
    friend bool operator==(const CableEnd&, const CableEnd&) = default;
};

// One undirected cable; `a` is the end with the lower physical address, so every rank names it the same way round.
struct BridgedCable {
    CableEnd a;
    CableEnd b;
    bool cross_rank = false;
    tt::tt_metal::PortType port_type = tt::tt_metal::PortType::UNKNOWN;
    BridgeReason reason = BridgeReason::Downed;
    friend bool operator==(const BridgedCable&, const BridgedCable&) = default;
};

// One end of a bridged cable, as the router on it sees it.
struct BridgedCableEnd {
    FabricNodeId peer_node{MeshId{0}, 0};
    chan_id_t peer_chan = 0;
    std::uint32_t rank = 0;       // MPI rank owning this end's chip
    std::uint32_t peer_rank = 0;  // and the peer's
    bool downed = false;          // physically absent, not broken by a test
};

// One bridged router as built (not the fabric-wide default config), recorded by the router builder for the host.
struct BridgedRouterLayout {
    std::uint32_t block_addr = 0;         // the E2H block: status, then one socket config per sender channel
    std::uint32_t recv_credits_addr = 0;  // the receive-side credit block, under the E2H block
    std::uint32_t slot_bytes = 0;         // the router's channel slot
    // By sender channel, in the router's order: its VC and its index within that VC.
    std::vector<std::uint32_t> sender_vc;
    std::vector<std::uint32_t> sender_index_in_vc;
    // By VC: the receiver channel the host feeds (dense index, or kNoReceiver), its L1 buffer and slot count.
    static constexpr std::uint32_t kNoReceiver = 0xFFFFFFFFu;
    std::vector<std::uint32_t> receiver_channel;
    std::vector<std::uint32_t> receiver_base;
    std::vector<std::uint32_t> receiver_slots;
    // Sender credit counters (counter form), one uint32 per sender channel from these bases.
    std::uint32_t completion_counters_addr = 0;
    std::uint32_t ack_counters_addr = 0;
};

// ControlPlane's virtual-cable state (ControlPlane::bridged_cables()); empty unless TT_METAL_E2H_BRIDGE_* asks.
struct BridgedCables {
    using Key = std::pair<FabricNodeId, chan_id_t>;
    std::vector<BridgedCable> cables;     // identical on every rank
    std::map<Key, BridgedCableEnd> ends;  // only cables with both chips in the mesh
    std::optional<BridgedCableEnd> end(FabricNodeId node, chan_id_t chan) const {
        const auto it = ends.find({node, chan});
        return it == ends.end() ? std::nullopt : std::optional{it->second};
    }
    // Each bridged router's layout as built, recorded by the router builder (thread-safe) for the host bridge.
    void record_layout(FabricNodeId node, chan_id_t chan, BridgedRouterLayout layout);
    std::optional<BridgedRouterLayout> layout(FabricNodeId node, chan_id_t chan) const;

private:
    mutable std::mutex layouts_mutex_;
    std::map<Key, BridgedRouterLayout> layouts_;
};

// TT_METAL_E2H_BRIDGE_BREAK_LINKS: list | all | cable:<i>[,<j>...] | random:<n>[:<seed>].
struct BreakSpec {
    enum class Mode : std::uint8_t { None, List, All, Cables, Random };
    Mode mode = Mode::None;
    std::vector<std::uint32_t> cable_indices;  // Cables
    std::uint32_t count = 0;                   // Random
    std::uint64_t seed = 0;                    // Random
};

// Throws std::invalid_argument on anything else. Empty is Mode::None.
BreakSpec parse_break_spec(std::string_view spec);

// Every live cable once, ordered by physical address: what `list` indexes and `random` draws from.
std::vector<BridgedCable> enumerate_live_cables(const tt::tt_metal::PhysicalSystemDescriptor& live);

// The cables `spec` picks, marked Injected, in enumeration order. Throws std::invalid_argument on a bad index or count.
std::vector<BridgedCable> select_injected_cables(const std::vector<BridgedCable>& live_cables, const BreakSpec& spec);

// Cables `fsd` expects and `live` lacks, marked Downed and named by live ASIC id; absent chips skipped.
std::vector<BridgedCable> find_downed_cables(
    const tt::tt_metal::PhysicalSystemDescriptor& fsd, const tt::tt_metal::PhysicalSystemDescriptor& live);

// Downed then injected; a cable in both is kept once, as Downed.
std::vector<BridgedCable> merge_bridged_cables(
    std::vector<BridgedCable> downed, const std::vector<BridgedCable>& injected);

// Records each Downed cable in `live` as discovery would have. Idempotent.
void add_bridged_cables_to_live(tt::tt_metal::PhysicalSystemDescriptor& live, const std::vector<BridgedCable>& cables);
// Test only: drops each cable from `live` as if it were unplugged before discovery.
void remove_cables_from_live(tt::tt_metal::PhysicalSystemDescriptor& live, const std::vector<BridgedCable>& cables);

// For logs.
std::string describe_cable(const tt::tt_metal::PhysicalSystemDescriptor& live, const BridgedCable& cable);

}  // namespace tt::tt_fabric::experimental
