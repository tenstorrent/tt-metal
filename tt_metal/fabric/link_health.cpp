// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/link_health.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::experimental {

namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::experimental::PhysicalNodeId;
using tt::tt_metal::PhysicalSystemDescriptor;

// Which way out of each end the cable leaves, for an intra-mesh link. The two ends almost never
// agree -- east faces west -- so both are read independently, and either side that the mesh graph
// does not describe stays NONE rather than being guessed from the other.
std::pair<RoutingDirection, RoutingDirection> directions_from_mesh_graph(
    const MeshGraph& mesh_graph, const FabricNodeId& src, const FabricNodeId& dst) {
    const auto& connectivity = mesh_graph.get_intra_mesh_connectivity();
    auto direction = [&connectivity](const FabricNodeId& from, const FabricNodeId& to) {
        const auto mesh_index = *from.mesh_id;
        if (mesh_index >= connectivity.size() || from.chip_id >= connectivity[mesh_index].size()) {
            return RoutingDirection::NONE;
        }
        const auto& neighbors = connectivity[mesh_index][from.chip_id];
        const auto neighbor = neighbors.find(to.chip_id);
        return neighbor == neighbors.end() ? RoutingDirection::NONE : neighbor->second.port_direction;
    };
    return {direction(src, dst), direction(dst, src)};
}

// The mesh graph lists a connection in the direction the descriptor wrote it. A downed cable is
// stored in both directions, so either spelling means the graph uses that mesh pair.
bool mesh_pair_requested(const MeshGraph& mesh_graph, MeshId src, MeshId dst) {
    const auto contains_pair = [](const auto& table, uint32_t from, uint32_t to) {
        const auto by_src = table.find(from);
        return by_src != table.end() && by_src->second.contains(to);
    };
    const auto src_id = *src;
    const auto dst_id = *dst;
    const auto& relaxed = mesh_graph.get_requested_intermesh_connections();
    const auto& strict = mesh_graph.get_requested_intermesh_ports();
    return contains_pair(relaxed, src_id, dst_id) || contains_pair(relaxed, dst_id, src_id) ||
           contains_pair(strict, src_id, dst_id) || contains_pair(strict, dst_id, src_id);
}

// Active downed links are the missing cables the mesh graph routes over. An intra-mesh cable is
// one of those when the two chips share a grid edge. An inter-mesh cable is one of those when the
// descriptor requests a connection between its two meshes.
bool used_by_mesh_graph(const MeshGraph& mesh_graph, const LinkInfo& record) {
    if (record.is_intramesh()) {
        return record.src_direction != RoutingDirection::NONE;
    }
    if (record.is_intermesh()) {
        return mesh_pair_requested(mesh_graph, record.src_mesh(), record.dst_mesh());
    }
    return false;
}

std::size_t count_directed(const tt::tt_metal::AsicTopology& links) {
    std::size_t total = 0;
    for (const auto& [src, edges] : links) {
        for (const auto& [dst, connections] : edges) {
            total += connections.size();
        }
    }
    return total;
}

// One direction of one chip. Missing factory cables the mesh graph still needs stay downed. Missing
// factory cables past that count are unused. A live cable is not a downed link, so a live factory
// cable past the mesh-graph count stays out of both sets. When the mesh graph asks for more than
// the factory descriptor has, every missing factory cable stays downed. The extra channels are not created.
void split_edge_by_mesh_graph_count(
    std::vector<LinkInfo> missing,
    const std::vector<tt::tt_metal::EthConnection>& live_connections,
    std::size_t mesh_graph_count,
    std::optional<std::size_t> psd_override,
    std::vector<LinkInfo>& downed,
    std::vector<LinkInfo>& unused) {
    std::sort(
        missing.begin(), missing.end(), [](const LinkInfo& a, const LinkInfo& b) { return a.src_chan < b.src_chan; });

    const std::size_t psd = psd_override.value_or(live_connections.size());
    const std::size_t fsd = psd + missing.size();

    if (mesh_graph_count > fsd) {
        for (auto& record : missing) {
            downed.push_back(std::move(record));
        }
        return;
    }

    const std::size_t still_needed = mesh_graph_count > psd ? mesh_graph_count - psd : 0;
    for (std::size_t i = 0; i < missing.size(); ++i) {
        auto& destination = i < still_needed ? downed : unused;
        destination.push_back(std::move(missing[i]));
    }
}

}  // namespace

LinkHealth::LinkHealth(const TopologyMapper& mapper, const PhysicalSystemDescriptor& live) :
    mapper_(&mapper), live_(&live) {
    refresh();
}

void LinkHealth::refresh(const TopologyMapper* mapper, const PhysicalSystemDescriptor* live) {
    if (mapper != nullptr) {
        mapper_ = mapper;
    }
    if (live != nullptr) {
        live_ = live;
    }

    downed_.clear();
    unused_downed_.clear();
    fsd_expected_.clear();
    live_present_.clear();

    // The mapper was built on the expected graph, so its descriptor is the golden side.
    const auto& expected = mapper_->get_physical_system_descriptor();

    // Presence, as declared by each side. Addressed, not labelled: comparing the expected
    // descriptor's ASIC labels against the live one's would match nothing on the factory path,
    // where one side counts from one and the other carries UMD chip ids, and every expected link
    // would read as down.
    // The whole cable per endpoint, not endpoint presence alone: health has to see that A:1 reaches
    // the peer the expected descriptor says it reaches. Presence of A:1 by itself would read a
    // miswired cable (expected A:1 <-> B:2, live A:1 <-> C:3) as healthy.
    auto collect_endpoints = [](const PhysicalSystemDescriptor& descriptor) {
        std::unordered_map<EndpointKey, EndpointKey, EndpointKey::Hash> endpoints;
        for (const auto& [host, topology] : descriptor.get_system_graph().asic_connectivity_graph) {
            for (const auto& [asic, edges] : topology) {
                const auto address = descriptor.find_physical_node_id(asic);
                if (!address.has_value()) {
                    continue;
                }
                for (const auto& [peer, connections] : edges) {
                    const auto peer_address = descriptor.find_physical_node_id(peer);
                    if (!peer_address.has_value()) {
                        continue;
                    }
                    for (const auto& connection : connections) {
                        endpoints.insert_or_assign(
                            EndpointKey{*address, connection.src_chan},
                            EndpointKey{*peer_address, connection.dst_chan});
                    }
                }
            }
        }
        return endpoints;
    };
    fsd_expected_ = collect_endpoints(expected);
    live_present_ = collect_endpoints(*live_);

    const auto delta = tt::tt_metal::experimental::diff_physical_system_descriptors(expected, *live_);

    // Reserved up front because the indexes below hold pointers into this vector.
    downed_.reserve(count_directed(delta.missing_links));
    for (const auto& [src_asic, edges] : delta.missing_links) {
        const auto src_address = expected.find_physical_node_id(src_asic);
        TT_FATAL(
            src_address.has_value(),
            "A missing link names ASIC {}, which the expected descriptor does not describe.",
            src_asic);
        for (const auto& [dst_asic, connections] : edges) {
            const auto dst_address = expected.find_physical_node_id(dst_asic);
            TT_FATAL(
                dst_address.has_value(),
                "A missing link names ASIC {}, which the expected descriptor does not describe.",
                dst_asic);
            for (const auto& connection : connections) {
                LinkInfo record;

                // Physical identity from the expected side, which is the side that knows what
                // should be there. The ASIC label, though, is the live UMD id where that chip
                // exists, since that is the id anything outside this module can act on.
                record.src_cluster_id = std::string(tt::tt_metal::experimental::cluster_id_view(*src_address));
                record.src_tray = src_address->tray;
                record.src_loc = src_address->loc;
                record.src_chan = connection.src_chan;
                record.dst_cluster_id = std::string(tt::tt_metal::experimental::cluster_id_view(*dst_address));
                record.dst_tray = dst_address->tray;
                record.dst_loc = dst_address->loc;
                record.dst_chan = connection.dst_chan;
                record.medium = connection.port_type;

                const auto live_src = live_->find_asic_id(*src_address);
                record.src_asic = live_src.has_value() ? *live_src : src_asic;
                const auto live_dst = live_->find_asic_id(*dst_address);
                record.dst_asic = live_dst.has_value() ? *live_dst : dst_asic;

                const auto src_node = mapper_->find_fabric_node_id_from_physical_node_id(*src_address);
                const auto dst_node = mapper_->find_fabric_node_id_from_physical_node_id(*dst_address);
                record.logical_resolved = src_node.has_value() && dst_node.has_value();
                if (record.logical_resolved) {
                    record.src_node = *src_node;
                    record.dst_node = *dst_node;
                    if (src_node->mesh_id == dst_node->mesh_id) {
                        record.scope = LinkScope::IntraMesh;
                        std::tie(record.src_direction, record.dst_direction) =
                            directions_from_mesh_graph(mapper_->get_mesh_graph(), *src_node, *dst_node);
                    } else {
                        // Still a downed link. Pairing chose among live cables only, so it never
                        // gave this one a logical port and there is no direction to report. It is
                        // deliberately not filtered against the post-pairing mesh graph, which
                        // knows only about links that came up and would empty this set.
                        record.scope = LinkScope::InterMesh;
                    }
                }
                downed_.push_back(std::move(record));
            }
        }
    }

    // Cables the mesh graph does not route over stay documented, in the unused set. The downed
    // API is only the holes a reroute can act on.
    std::vector<LinkInfo> used;
    used.reserve(downed_.size());
    for (auto& record : downed_) {
        const bool used_by_graph = used_by_mesh_graph(mapper_->get_mesh_graph(), record);
        if (used_by_graph) {
            used.push_back(std::move(record));
        } else {
            unused_downed_.push_back(std::move(record));
        }
    }
    downed_ = std::move(used);

    rebuild_indexes();
}

void LinkHealth::rebuild_indexes() {
    by_node_chan_.clear();
    by_node_.clear();
    by_node_dir_.clear();
    by_scope_.clear();
    by_mesh_pair_.clear();
    by_src_address_.clear();
    by_host_.clear();

    for (const LinkInfo& record : downed_) {
        const LinkInfo* pointer = &record;
        by_src_address_[tt::tt_metal::experimental::make_physical_node_id(record.src_cluster_id, record.src_tray, record.src_loc)]
            .push_back(pointer);
        by_host_[record.src_cluster_id].push_back(pointer);
        if (!record.logical_resolved) {
            // Nothing logical to key on. The record is still reachable physically, which is the
            // point of reporting it at all.
            continue;
        }
        by_node_chan_[NodeChanKey{record.src_node, record.src_chan}] = pointer;
        by_node_[record.src_node].push_back(pointer);
        by_scope_[record.scope].push_back(pointer);
        if (record.src_direction != RoutingDirection::NONE) {
            by_node_dir_[NodeDirKey{record.src_node, record.src_direction}].push_back(pointer);
        }
        if (record.is_intermesh()) {
            by_mesh_pair_[MeshPairKey{record.src_mesh(), record.dst_mesh()}].push_back(pointer);
        }
    }
}

void LinkHealth::classify_unused_from_routing_planes(const RoutingPlaneSnapshot& snapshot) {
    auto mesh_graph_count_for = [](const auto& table, const FabricNodeId& node, RoutingDirection dir) {
        const auto by_node = table.find(node);
        if (by_node == table.end()) {
            return std::optional<std::size_t>{};
        }
        const auto by_dir = by_node->second.find(dir);
        return by_dir == by_node->second.end() ? std::optional<std::size_t>{} : std::optional{by_dir->second};
    };

    // Live cables between two meshes, counted in both directions. A cable is stored once each way.
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> live_intermesh;
    for (const auto& [host, topology] : live_->get_system_graph().asic_connectivity_graph) {
        (void)host;
        for (const auto& [asic, edges] : topology) {
            const auto src_address = live_->find_physical_node_id(asic);
            if (!src_address.has_value()) {
                continue;
            }
            const auto src_node = mapper_->find_fabric_node_id_from_physical_node_id(*src_address);
            if (!src_node.has_value()) {
                continue;
            }
            for (const auto& [peer, connections] : edges) {
                const auto dst_address = live_->find_physical_node_id(peer);
                if (!dst_address.has_value()) {
                    continue;
                }
                const auto dst_node = mapper_->find_fabric_node_id_from_physical_node_id(*dst_address);
                if (!dst_node.has_value() || src_node->mesh_id == dst_node->mesh_id) {
                    continue;
                }
                live_intermesh[{*src_node->mesh_id, *dst_node->mesh_id}] += connections.size();
            }
        }
    }
    auto live_between = [&](std::uint32_t src_mesh, std::uint32_t dst_mesh) {
        const auto forward = live_intermesh.find({src_mesh, dst_mesh});
        const auto backward = live_intermesh.find({dst_mesh, src_mesh});
        const std::size_t fwd = forward == live_intermesh.end() ? 0 : forward->second;
        const std::size_t back = backward == live_intermesh.end() ? 0 : backward->second;
        return std::max(fwd, back);
    };
    // The mesh graph stores a connection in one direction. Both ends of the cable use that count.
    auto intermesh_count = [&](std::uint32_t src_mesh, std::uint32_t dst_mesh) {
        const auto& table = mapper_->get_mesh_graph().get_requested_intermesh_connections();
        auto lookup = [&](std::uint32_t from, std::uint32_t to) {
            const auto by_src = table.find(from);
            if (by_src == table.end()) {
                return std::size_t{0};
            }
            const auto by_dst = by_src->second.find(to);
            return by_dst == by_src->second.end() ? std::size_t{0} : by_dst->second;
        };
        return std::max(lookup(src_mesh, dst_mesh), lookup(dst_mesh, src_mesh));
    };

    // One direction of one chip, toward one neighbor. The two ends of a cable are separate groups.
    struct EdgeKey {
        FabricNodeId node{MeshId{0}, 0};
        RoutingDirection dir = RoutingDirection::NONE;
        AsicID dst{0};
        bool operator==(const EdgeKey& other) const {
            return node == other.node && dir == other.dir && dst == other.dst;
        }
    };
    std::vector<std::pair<EdgeKey, std::vector<LinkInfo>>> edges;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<LinkInfo>> intermesh_edges;

    std::vector<LinkInfo> downed;
    downed.reserve(downed_.size());
    for (auto& record : downed_) {
        if (record.is_intermesh() && record.logical_resolved) {
            intermesh_edges[{*record.src_mesh(), *record.dst_mesh()}].push_back(std::move(record));
            continue;
        }
        // Direction NONE (a wrap the topology does not route) was already split in refresh().
        const auto mesh_graph_count =
            mesh_graph_count_for(snapshot.expected_planes, record.src_node, record.src_direction);
        if (!record.is_intramesh() || record.src_direction == RoutingDirection::NONE || !mesh_graph_count.has_value()) {
            downed.push_back(std::move(record));
            continue;
        }
        const EdgeKey key{record.src_node, record.src_direction, record.dst_asic};
        auto edge = std::find_if(edges.begin(), edges.end(), [&](const auto& entry) { return entry.first == key; });
        if (edge == edges.end()) {
            edges.push_back({key, {}});
            edge = edges.end() - 1;
        }
        edge->second.push_back(std::move(record));
    }

    for (auto& [key, missing] : edges) {
        const auto live = live_->get_eth_connections(missing.front().src_asic, key.dst);
        split_edge_by_mesh_graph_count(
            std::move(missing),
            live,
            *mesh_graph_count_for(snapshot.expected_planes, key.node, key.dir),
            mesh_graph_count_for(snapshot.psd_cables, key.node, key.dir),
            downed,
            unused_downed_);
    }

    // Intermesh has no routing planes. A missing factory cable the mesh graph still needs stays
    // downed. The rest of the factory-to-live mismatch is unused. Channels the factory never had
    // are not registered and are not link records.
    for (auto& [meshes, missing] : intermesh_edges) {
        const auto [src_mesh, dst_mesh] = meshes;
        const std::size_t psd = live_between(src_mesh, dst_mesh);
        split_edge_by_mesh_graph_count(
            std::move(missing),
            std::vector<tt::tt_metal::EthConnection>{},
            intermesh_count(src_mesh, dst_mesh),
            psd,
            downed,
            unused_downed_);
    }

    downed_ = std::move(downed);
    rebuild_indexes();
}

std::optional<LinkHealth::EndpointKey> LinkHealth::endpoint_for(const FabricNodeId& node, chan_id_t chan) const {
    const auto address = mapper_->find_physical_node_id_from_fabric_node_id(node);
    if (!address.has_value()) {
        return std::nullopt;
    }
    return EndpointKey{*address, chan};
}

std::optional<experimental::PhysicalNodeId> LinkHealth::address_of(AsicID asic) const {
    if (live_ != nullptr) {
        if (const auto live = live_->find_physical_node_id(asic); live.has_value()) {
            return live;
        }
    }
    if (mapper_ != nullptr) {
        return mapper_->get_physical_system_descriptor().find_physical_node_id(asic);
    }
    return std::nullopt;
}

bool LinkHealth::healthy(const EndpointKey& endpoint) const {
    const auto expected = fsd_expected_.find(endpoint);
    if (expected == fsd_expected_.end()) {
        throw std::out_of_range(fmt::format(
            "Channel {} on {} is not expected by the factory system descriptor, so it has no health to report.",
            endpoint.chan,
            endpoint.node));
    }
    // Healthy means the live cable from this endpoint reaches the expected peer. Mere presence of
    // the endpoint is not enough: a miswired cable keeps the endpoint live while the expected link
    // is down.
    const auto live = live_present_.find(endpoint);
    return live != live_present_.end() && live->second == expected->second;
}

std::vector<LinkInfo> LinkHealth::copy_records(const std::vector<const LinkInfo*>& records) {
    std::vector<LinkInfo> copies;
    copies.reserve(records.size());
    for (const LinkInfo* record : records) {
        copies.push_back(*record);
    }
    return copies;
}

bool LinkHealth::is_link_healthy(const FabricNodeId& node, chan_id_t chan) const {
    const auto endpoint = endpoint_for(node, chan);
    if (!endpoint.has_value()) {
        throw std::out_of_range(
            fmt::format("Fabric node {} has no physical address, so it has no expected links.", node));
    }
    return healthy(*endpoint);
}

bool LinkHealth::is_link_healthy(
    const std::string& cluster_id, tt::tt_metal::TrayID tray, tt::tt_metal::ASICLocation loc, chan_id_t chan) const {
    return healthy(EndpointKey{tt::tt_metal::experimental::make_physical_node_id(cluster_id, tray, loc), chan});
}

bool LinkHealth::is_link_healthy(AsicID asic, chan_id_t chan) const {
    const auto address = address_of(asic);
    if (!address.has_value()) {
        throw std::out_of_range(fmt::format("ASIC {} is in neither descriptor, so it has no expected links.", asic));
    }
    return healthy(EndpointKey{*address, chan});
}

std::optional<LinkInfo> LinkHealth::find_downed_link(const FabricNodeId& node, chan_id_t chan) const {
    const auto record = by_node_chan_.find(NodeChanKey{node, chan});
    return record == by_node_chan_.end() ? std::nullopt : std::optional{*record->second};
}

std::vector<LinkInfo> LinkHealth::get_downed_links(const FabricNodeId& node) const {
    const auto records = by_node_.find(node);
    return records == by_node_.end() ? std::vector<LinkInfo>{} : copy_records(records->second);
}

std::vector<chan_id_t> LinkHealth::get_downed_eth_chans(const FabricNodeId& node) const {
    std::vector<chan_id_t> chans;
    const auto records = by_node_.find(node);
    if (records == by_node_.end()) {
        return chans;
    }
    chans.reserve(records->second.size());
    for (const LinkInfo* record : records->second) {
        chans.push_back(record->src_chan);
    }
    std::sort(chans.begin(), chans.end());
    return chans;
}

std::vector<chan_id_t> LinkHealth::get_downed_eth_chans_in_direction(
    const FabricNodeId& node, RoutingDirection dir) const {
    std::vector<chan_id_t> chans;
    const auto records = by_node_dir_.find(NodeDirKey{node, dir});
    if (records == by_node_dir_.end()) {
        return chans;
    }
    chans.reserve(records->second.size());
    for (const LinkInfo* record : records->second) {
        chans.push_back(record->src_chan);
    }
    std::sort(chans.begin(), chans.end());
    return chans;
}

bool LinkHealth::has_downed_link_in_direction(const FabricNodeId& node, RoutingDirection dir) const {
    return by_node_dir_.contains(NodeDirKey{node, dir});
}

std::size_t LinkHealth::get_num_downed_routing_planes_in_direction(
    const FabricNodeId& node, RoutingDirection dir) const {
    const auto records = by_node_dir_.find(NodeDirKey{node, dir});
    return records == by_node_dir_.end() ? 0 : records->second.size();
}

std::vector<chan_id_t> LinkHealth::get_downed_intramesh_eth_chans(const FabricNodeId& node) const {
    std::vector<chan_id_t> chans;
    const auto records = by_node_.find(node);
    if (records == by_node_.end()) {
        return chans;
    }
    for (const LinkInfo* record : records->second) {
        if (record->is_intramesh()) {
            chans.push_back(record->src_chan);
        }
    }
    std::sort(chans.begin(), chans.end());
    return chans;
}

std::vector<chan_id_t> LinkHealth::get_downed_intermesh_eth_chans(const FabricNodeId& node) const {
    std::vector<chan_id_t> chans;
    const auto records = by_node_.find(node);
    if (records == by_node_.end()) {
        return chans;
    }
    for (const LinkInfo* record : records->second) {
        if (record->is_intermesh()) {
            chans.push_back(record->src_chan);
        }
    }
    std::sort(chans.begin(), chans.end());
    return chans;
}

std::vector<LinkInfo> LinkHealth::get_downed_links(LinkScope scope) const {
    // Unknown is not a bucket: an unresolved record has no logical view, so it is in neither scope.
    if (scope == LinkScope::Unknown) {
        return {};
    }
    const auto records = by_scope_.find(scope);
    return records == by_scope_.end() ? std::vector<LinkInfo>{} : copy_records(records->second);
}

std::vector<LinkInfo> LinkHealth::get_downed_intramesh_links() const { return get_downed_links(LinkScope::IntraMesh); }

std::vector<LinkInfo> LinkHealth::get_downed_intermesh_links() const { return get_downed_links(LinkScope::InterMesh); }

std::vector<LinkInfo> LinkHealth::get_downed_links_between(const FabricNodeId& src, const FabricNodeId& dst) const {
    std::vector<LinkInfo> between;
    const auto records = by_node_.find(src);
    if (records == by_node_.end()) {
        return between;
    }
    for (const LinkInfo* record : records->second) {
        if (record->dst_node == dst) {
            between.push_back(*record);
        }
    }
    return between;
}

std::vector<chan_id_t> LinkHealth::get_downed_forwarding_eth_chans_to_chip(
    const FabricNodeId& src, const FabricNodeId& dst) const {
    std::vector<chan_id_t> chans;
    for (const auto& record : get_downed_links_between(src, dst)) {
        chans.push_back(record.src_chan);
    }
    std::sort(chans.begin(), chans.end());
    return chans;
}

std::vector<LinkInfo> LinkHealth::get_downed_intermesh_links(MeshId src_mesh, MeshId dst_mesh) const {
    const auto records = by_mesh_pair_.find(MeshPairKey{src_mesh, dst_mesh});
    return records == by_mesh_pair_.end() ? std::vector<LinkInfo>{} : copy_records(records->second);
}

std::vector<FabricNodeId> LinkHealth::get_exit_nodes_with_downed_links(MeshId src_mesh, MeshId dst_mesh) const {
    std::vector<FabricNodeId> nodes;
    const auto records = by_mesh_pair_.find(MeshPairKey{src_mesh, dst_mesh});
    if (records == by_mesh_pair_.end()) {
        return nodes;
    }
    for (const LinkInfo* record : records->second) {
        nodes.push_back(record->src_node);
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

std::vector<LinkInfo> LinkHealth::get_downed_links_for_host(const std::string& cluster_id) const {
    const auto records = by_host_.find(tt::tt_metal::experimental::canonical_cluster_id_for_node_id(cluster_id));
    return records == by_host_.end() ? std::vector<LinkInfo>{} : copy_records(records->second);
}

std::vector<LinkInfo> LinkHealth::get_downed_links_for_asic(AsicID asic) const {
    const auto address = address_of(asic);
    if (!address.has_value()) {
        return {};
    }
    const auto records = by_src_address_.find(*address);
    return records == by_src_address_.end() ? std::vector<LinkInfo>{} : copy_records(records->second);
}

std::vector<LinkInfo> LinkHealth::get_downed_links_between_hosts(
    const std::string& a_cluster_id, const std::string& b_cluster_id) const {
    const auto a = tt::tt_metal::experimental::canonical_cluster_id_for_node_id(a_cluster_id);
    const auto b = tt::tt_metal::experimental::canonical_cluster_id_for_node_id(b_cluster_id);

    // Both directions, since each cable is stored once per end and a caller asking about a host pair
    // wants the cable, not one arbitrary half of it.
    std::vector<LinkInfo> between;
    auto collect = [this, &between](const std::string& from, const std::string& to) {
        const auto records = by_host_.find(from);
        if (records == by_host_.end()) {
            return;
        }
        for (const LinkInfo* record : records->second) {
            if (record->dst_cluster_id == to) {
                between.push_back(*record);
            }
        }
    };
    collect(a, b);
    if (a != b) {
        collect(b, a);
    }
    return between;
}

}  // namespace tt::tt_fabric::experimental
