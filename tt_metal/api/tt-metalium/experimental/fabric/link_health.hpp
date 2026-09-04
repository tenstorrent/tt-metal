// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hostdevcommon/fabric_common.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_node_id.hpp>

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}

namespace tt::tt_fabric {

class TopologyMapper;

enum class LinkScope {
    IntraMesh,
    InterMesh,
    // The mesh graph solve never placed one of this cable's ASICs, so there is no logical view of it
    // to classify. Never guessed at.
    Unknown,
};

// One downed link: a connection the factory system descriptor declares that the live descriptor does
// not have. Carries both identities of the same cable -- the logical one a reroute planner works in,
// and the physical one a datacenter technician works in.
//
// A cable is stored as two of these, one per direction, because every index below is keyed on the
// source end. The two ends of a cable rarely share a direction, so each record also carries the far
// end's direction rather than making callers find the mirrored record to get it.
struct LinkInfo {
    // Logical, resolved through the topology mapper and the mesh graph. Meaningless unless
    // logical_resolved -- check that before reading any of it.
    FabricNodeId src_node{MeshId{0}, 0};
    chan_id_t src_chan = 0;
    FabricNodeId dst_node{MeshId{0}, 0};
    chan_id_t dst_chan = 0;
    RoutingDirection src_direction = RoutingDirection::NONE;
    RoutingDirection dst_direction = RoutingDirection::NONE;
    LinkScope scope = LinkScope::Unknown;

    // False when the solve did not place one of these ASICs. The record is still reported, but only
    // through the physical and per-ASIC queries -- never the per-node, per-direction or per-mesh
    // ones, which have nothing to key on.
    bool logical_resolved = false;

    // Physical: the two ends' addresses, taken from the expected descriptor, which is the side that
    // knows what should be there. host_id is an accelerator-group id whose value today is a
    // hostname, canonicalized, so it compares equal however either side spelled it.
    std::string src_host_id;
    tt::tt_metal::TrayID src_tray{0};
    tt::tt_metal::ASICLocation src_loc{0};
    tt::tt_metal::AsicID src_asic{0};
    std::string dst_host_id;
    tt::tt_metal::TrayID dst_tray{0};
    tt::tt_metal::ASICLocation dst_loc{0};
    tt::tt_metal::AsicID dst_asic{0};
    tt::tt_metal::PortType medium = tt::tt_metal::PortType::UNKNOWN;

    // No routing plane. A plane index is a position in the live ordered channel list, and a downed
    // channel is by definition not in that list. Lost capacity is counted per (node, direction) --
    // see get_num_downed_routing_planes_in_direction.

    bool is_intramesh() const { return logical_resolved && scope == LinkScope::IntraMesh; }
    bool is_intermesh() const { return logical_resolved && scope == LinkScope::InterMesh; }

    MeshId src_mesh() const { return src_node.mesh_id; }
    MeshId dst_mesh() const { return dst_node.mesh_id; }
};

// Routing planes per (node, direction), before and after fabric's own downgrade. Filled by the
// control plane, which is the only thing that knows the post-trim numbers; LinkHealth just compares
// them. Expected comes from the mesh graph, live from the channel map after the row/column minimum,
// the trim and the cross-host merge.
struct RoutingPlaneSnapshot {
    std::unordered_map<FabricNodeId, std::unordered_map<RoutingDirection, std::size_t>> expected_planes;
    std::unordered_map<FabricNodeId, std::unordered_map<RoutingDirection, std::size_t>> live_planes;
};

// The set of downed links for one (expected descriptor, live descriptor) pair, with indexes over it.
//
// Globally complete on every rank without communicating: discovery already gathers and broadcasts a
// system-wide live descriptor, the factory descriptor is the same file everywhere, and the mapper
// resolves any ASIC in the system. So this purely local comparison is already the whole-system
// answer. That holds only while discovery is global -- with local-only discovery a rank sees remote
// chips as absent.
class LinkHealth {
public:
    LinkHealth(const TopologyMapper& mapper, const tt::tt_metal::PhysicalSystemDescriptor& live);

    // Recompute against a new mapper and/or live descriptor. nullptr keeps the current binding.
    // Invalidates every reference and pointer previously handed out.
    void refresh(const TopologyMapper* mapper = nullptr, const tt::tt_metal::PhysicalSystemDescriptor* live = nullptr);

    // Holds pointers to both descriptors, so copying or moving one of these would quietly alias
    // them. There is no default: the comparison has no meaning without both sides.
    LinkHealth() = delete;
    LinkHealth(const LinkHealth&) = delete;
    LinkHealth& operator=(const LinkHealth&) = delete;
    LinkHealth(LinkHealth&&) = delete;
    LinkHealth& operator=(LinkHealth&&) = delete;

    // Move the intra-mesh holes that sit on routing planes fabric already downgraded away out of the
    // active set. They are real unplugged cables, so they stay documented, but fabric does not route
    // on those planes and must not reroute for them. Runs after the plane trim and the cross-host
    // merge, so every rank classifies identically. Rebuilds the indexes.
    void classify_unused_from_routing_planes(const RoutingPlaneSnapshot& snapshot);

    bool has_downed_links() const { return !downed_.empty(); }
    // Static rerouting is on only for links fabric still routes on, so unused-plane holes do not
    // count here even though they are missing cables.
    bool fsd_rerouting_active() const { return !downed_.empty(); }

    const std::vector<LinkInfo>& get_downed_links() const { return downed_; }
    const std::vector<LinkInfo>& get_unused_downed_links() const { return unused_downed_; }

    // Directed endpoints the factory descriptor declares: two per cable, matching how downed links
    // are counted, so the ratio of the two is meaningful. Both the active and the unused sets count
    // as missing for that ratio.
    std::size_t fsd_expected_count() const { return fsd_expected_.size(); }

    // Logical view.
    //
    // Presence against the live descriptor, and nothing else: it does not ask the cluster whether
    // ethernet is up, which is what lets it work on a mock descriptor. Throws std::out_of_range when
    // the factory descriptor never declared this endpoint, since "healthy" would be a claim about
    // something there is no expectation for.
    bool is_link_healthy(const FabricNodeId& node, chan_id_t chan) const;
    std::optional<LinkInfo> find_downed_link(const FabricNodeId& node, chan_id_t chan) const;

    std::vector<LinkInfo> get_downed_links(const FabricNodeId& node) const;
    std::vector<chan_id_t> get_downed_eth_chans(const FabricNodeId& node) const;

    std::vector<chan_id_t> get_downed_eth_chans_in_direction(const FabricNodeId& node, RoutingDirection dir) const;
    bool has_downed_link_in_direction(const FabricNodeId& node, RoutingDirection dir) const;
    // Capacity lost on planes fabric still uses. Zero for a direction that was downgraded, because
    // its holes moved to the unused set. Intra-mesh only: an intermesh record has no direction, so
    // this is always zero for one, and there is no intermesh equivalent -- that would be a
    // requested-versus-resolved shortfall at a mesh boundary, which is not a cable-level number.
    std::size_t get_num_downed_routing_planes_in_direction(const FabricNodeId& node, RoutingDirection dir) const;

    std::vector<chan_id_t> get_downed_intramesh_eth_chans(const FabricNodeId& node) const;
    std::vector<chan_id_t> get_downed_intermesh_eth_chans(const FabricNodeId& node) const;

    std::vector<LinkInfo> get_downed_links(LinkScope scope) const;
    std::vector<LinkInfo> get_downed_intramesh_links() const;
    // Not empty when factory-declared intermesh cables are gone. Pairing only ever chose among live
    // cables, so it never assigned these a port and their directions stay NONE -- but they are still
    // downed links, and deliberately not filtered against the post-pairing mesh graph, which knows
    // only about links that came up.
    std::vector<LinkInfo> get_downed_intermesh_links() const;

    std::vector<LinkInfo> get_downed_links_between(const FabricNodeId& src, const FabricNodeId& dst) const;
    std::vector<chan_id_t> get_downed_forwarding_eth_chans_to_chip(
        const FabricNodeId& src, const FabricNodeId& dst) const;

    std::vector<LinkInfo> get_downed_intermesh_links(MeshId src_mesh, MeshId dst_mesh) const;
    std::vector<FabricNodeId> get_exit_nodes_with_downed_links(MeshId src_mesh, MeshId dst_mesh) const;

    // Physical view. Host ids are canonicalized on the way in, so any spelling of the same host
    // works.
    bool is_link_healthy(
        const std::string& host_id, tt::tt_metal::TrayID tray, tt::tt_metal::ASICLocation loc, chan_id_t chan) const;
    bool is_link_healthy(tt::tt_metal::AsicID asic, chan_id_t chan) const;

    std::vector<LinkInfo> get_downed_links_for_host(const std::string& host_id) const;
    std::vector<LinkInfo> get_downed_links_for_asic(tt::tt_metal::AsicID asic) const;
    // Both directions, and both scopes: a cross-host cable between two meshes belongs here too.
    std::vector<LinkInfo> get_downed_links_between_hosts(
        const std::string& a_host_id, const std::string& b_host_id) const;

private:
    // A cable endpoint, addressed rather than labelled. Keying these on AsicID would compare the
    // expected descriptor's labels against the live one's, and on the factory path those are file
    // order versus UMD chip ids -- nothing would ever match, and every expected link would read as
    // down.
    struct EndpointKey {
        tt::tt_metal::PhysicalNodeId node{};
        chan_id_t chan = 0;

        friend bool operator==(const EndpointKey&, const EndpointKey&) = default;

        struct Hash {
            std::size_t operator()(const EndpointKey& key) const noexcept {
                const std::size_t seed = std::hash<tt::tt_metal::PhysicalNodeId>{}(key.node);
                return seed ^ (std::hash<chan_id_t>{}(key.chan) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
            }
        };
    };

    struct NodeChanKey {
        FabricNodeId node{MeshId{0}, 0};
        chan_id_t chan = 0;

        friend bool operator==(const NodeChanKey& a, const NodeChanKey& b) {
            return a.node == b.node && a.chan == b.chan;
        }

        struct Hash {
            std::size_t operator()(const NodeChanKey& key) const noexcept {
                const std::size_t seed = std::hash<FabricNodeId>{}(key.node);
                return seed ^ (std::hash<chan_id_t>{}(key.chan) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
            }
        };
    };

    struct NodeDirKey {
        FabricNodeId node{MeshId{0}, 0};
        RoutingDirection dir = RoutingDirection::NONE;

        friend bool operator==(const NodeDirKey& a, const NodeDirKey& b) { return a.node == b.node && a.dir == b.dir; }

        struct Hash {
            std::size_t operator()(const NodeDirKey& key) const noexcept {
                const std::size_t seed = std::hash<FabricNodeId>{}(key.node);
                return seed ^ (std::hash<int>{}(static_cast<int>(key.dir)) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
            }
        };
    };

    struct MeshPairKey {
        MeshId src{0};
        MeshId dst{0};

        friend bool operator==(const MeshPairKey& a, const MeshPairKey& b) { return a.src == b.src && a.dst == b.dst; }

        struct Hash {
            std::size_t operator()(const MeshPairKey& key) const noexcept {
                const std::size_t seed = std::hash<uint32_t>{}(*key.src);
                return seed ^ (std::hash<uint32_t>{}(*key.dst) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
            }
        };
    };

    void rebuild_indexes();
    // Endpoint key for a logical channel, or nullopt when the node is not mapped to an address.
    std::optional<EndpointKey> endpoint_for(const FabricNodeId& node, chan_id_t chan) const;
    // The address an ASIC label refers to, whichever descriptor's label space it came from. Callers
    // in the datacenter hold live UMD ids; callers holding a factory descriptor hold file-order
    // labels. Neither is asked to know which.
    std::optional<tt::tt_metal::PhysicalNodeId> address_of(tt::tt_metal::AsicID asic) const;
    bool healthy(const EndpointKey& endpoint) const;
    static std::vector<LinkInfo> copy_records(const std::vector<const LinkInfo*>& records);

    const TopologyMapper* mapper_;
    const tt::tt_metal::PhysicalSystemDescriptor* live_;

    std::vector<LinkInfo> downed_;
    std::vector<LinkInfo> unused_downed_;

    std::unordered_set<EndpointKey, EndpointKey::Hash> fsd_expected_;
    std::unordered_set<EndpointKey, EndpointKey::Hash> live_present_;

    // Address-to-label translation for each side, kept so a query can accept either descriptor's
    // ASIC labels.
    tt::tt_metal::PhysicalNodeIdIndex expected_ids_;
    tt::tt_metal::PhysicalNodeIdIndex live_ids_;

    // All of these point into downed_, so every one is rebuilt whenever that vector changes. The
    // unused set is deliberately not indexed: it is reachable only as a whole, through
    // get_unused_downed_links().
    std::unordered_map<NodeChanKey, const LinkInfo*, NodeChanKey::Hash> by_node_chan_;
    std::unordered_map<FabricNodeId, std::vector<const LinkInfo*>> by_node_;
    std::unordered_map<NodeDirKey, std::vector<const LinkInfo*>, NodeDirKey::Hash> by_node_dir_;
    std::unordered_map<LinkScope, std::vector<const LinkInfo*>> by_scope_;
    std::unordered_map<MeshPairKey, std::vector<const LinkInfo*>, MeshPairKey::Hash> by_mesh_pair_;
    // Keyed on the source address rather than an ASIC label, so it does not matter which
    // descriptor's labels a caller has.
    std::unordered_map<tt::tt_metal::PhysicalNodeId, std::vector<const LinkInfo*>> by_src_address_;
    std::unordered_map<std::string, std::vector<const LinkInfo*>> by_host_;
};

}  // namespace tt::tt_fabric
