// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/link_health.hpp>

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::PhysicalNodeId;
using tt::tt_metal::PhysicalNodeIdIndex;
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

std::size_t count_directed(const tt::tt_metal::AsicTopology& links) {
    std::size_t total = 0;
    for (const auto& [src, edges] : links) {
        for (const auto& [dst, connections] : edges) {
            total += connections.size();
        }
    }
    return total;
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
    expected_ids_ = tt::tt_metal::build_physical_node_id_index(expected);
    live_ids_ = tt::tt_metal::build_physical_node_id_index(*live_);

    // Presence, as declared by each side. Addressed, not labelled: comparing the expected
    // descriptor's ASIC labels against the live one's would match nothing on the factory path,
    // where one side counts from one and the other carries UMD chip ids, and every expected link
    // would read as down.
    auto collect_endpoints = [](const PhysicalSystemDescriptor& descriptor, const PhysicalNodeIdIndex& index) {
        std::unordered_set<EndpointKey, EndpointKey::Hash> endpoints;
        for (const auto& [host, topology] : descriptor.get_system_graph().asic_connectivity_graph) {
            for (const auto& [asic, edges] : topology) {
                const auto address = index.asic_id_to_node_id.find(asic);
                if (address == index.asic_id_to_node_id.end()) {
                    continue;
                }
                for (const auto& [peer, connections] : edges) {
                    for (const auto& connection : connections) {
                        endpoints.insert(EndpointKey{address->second, connection.src_chan});
                    }
                }
            }
        }
        return endpoints;
    };
    fsd_expected_ = collect_endpoints(expected, expected_ids_);
    live_present_ = collect_endpoints(*live_, live_ids_);

    const auto delta = tt::tt_metal::diff_physical_system_descriptors(expected, *live_);

    // Reserved up front because the indexes below hold pointers into this vector.
    downed_.reserve(count_directed(delta.missing_links));
    for (const auto& [src_asic, edges] : delta.missing_links) {
        const auto src_address = expected_ids_.asic_id_to_node_id.find(src_asic);
        TT_FATAL(
            src_address != expected_ids_.asic_id_to_node_id.end(),
            "A missing link names ASIC {}, which the expected descriptor does not describe.",
            src_asic);
        for (const auto& [dst_asic, connections] : edges) {
            const auto dst_address = expected_ids_.asic_id_to_node_id.find(dst_asic);
            TT_FATAL(
                dst_address != expected_ids_.asic_id_to_node_id.end(),
                "A missing link names ASIC {}, which the expected descriptor does not describe.",
                dst_asic);
            for (const auto& connection : connections) {
                LinkInfo record;

                // Physical identity from the expected side, which is the side that knows what
                // should be there. The ASIC label, though, is the live UMD id where that chip
                // exists, since that is the id anything outside this module can act on.
                record.src_host_id = std::string(tt::tt_metal::host_id_view(src_address->second));
                record.src_tray = src_address->second.tray;
                record.src_loc = src_address->second.loc;
                record.src_chan = connection.src_chan;
                record.dst_host_id = std::string(tt::tt_metal::host_id_view(dst_address->second));
                record.dst_tray = dst_address->second.tray;
                record.dst_loc = dst_address->second.loc;
                record.dst_chan = connection.dst_chan;
                record.medium = connection.port_type;

                const auto live_src = live_ids_.node_id_to_asic_id.find(src_address->second);
                record.src_asic = live_src == live_ids_.node_id_to_asic_id.end() ? src_asic : live_src->second;
                const auto live_dst = live_ids_.node_id_to_asic_id.find(dst_address->second);
                record.dst_asic = live_dst == live_ids_.node_id_to_asic_id.end() ? dst_asic : live_dst->second;

                const auto src_node = mapper_->find_fabric_node_id_from_physical_node_id(src_address->second);
                const auto dst_node = mapper_->find_fabric_node_id_from_physical_node_id(dst_address->second);
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
        by_src_address_[tt::tt_metal::make_physical_node_id(record.src_host_id, record.src_tray, record.src_loc)]
            .push_back(pointer);
        by_host_[record.src_host_id].push_back(pointer);
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
    auto planes = [](const auto& table, const FabricNodeId& node, RoutingDirection dir) -> std::optional<std::size_t> {
        const auto by_node = table.find(node);
        if (by_node == table.end()) {
            return std::nullopt;
        }
        const auto by_dir = by_node->second.find(dir);
        return by_dir == by_node->second.end() ? std::nullopt : std::optional{by_dir->second};
    };

    // Fabric has already dropped a plane when it ends up routing on fewer than the mesh graph asked
    // for. A hole on such a plane is a real unplugged cable that fabric will never route over, so it
    // must not drive rerouting -- but it still gets documented.
    auto downgraded = [&](const LinkInfo& record) {
        if (!record.is_intramesh() || record.src_direction == RoutingDirection::NONE) {
            // Intermesh holes stay active by decision, and a record with no direction has no plane
            // count to compare against.
            return false;
        }
        const auto expected = planes(snapshot.expected_planes, record.src_node, record.src_direction);
        const auto live = planes(snapshot.live_planes, record.src_node, record.src_direction);
        if (!expected.has_value() || !live.has_value()) {
            // Nothing said about this direction, so nothing was downgraded.
            return false;
        }
        return *live < *expected;
    };

    std::vector<LinkInfo> active;
    active.reserve(downed_.size());
    for (auto& record : downed_) {
        if (downgraded(record)) {
            unused_downed_.push_back(std::move(record));
        } else {
            active.push_back(std::move(record));
        }
    }
    downed_ = std::move(active);

    rebuild_indexes();
}

std::optional<LinkHealth::EndpointKey> LinkHealth::endpoint_for(const FabricNodeId& node, chan_id_t chan) const {
    const auto address = mapper_->find_physical_node_id_from_fabric_node_id(node);
    if (!address.has_value()) {
        return std::nullopt;
    }
    return EndpointKey{*address, chan};
}

std::optional<PhysicalNodeId> LinkHealth::address_of(AsicID asic) const {
    const auto live = live_ids_.asic_id_to_node_id.find(asic);
    if (live != live_ids_.asic_id_to_node_id.end()) {
        return live->second;
    }
    const auto expected = expected_ids_.asic_id_to_node_id.find(asic);
    if (expected != expected_ids_.asic_id_to_node_id.end()) {
        return expected->second;
    }
    return std::nullopt;
}

bool LinkHealth::healthy(const EndpointKey& endpoint) const {
    if (!fsd_expected_.contains(endpoint)) {
        throw std::out_of_range(fmt::format(
            "Channel {} on {} is not expected by the factory system descriptor, so it has no health to report.",
            endpoint.chan,
            endpoint.node));
    }
    return live_present_.contains(endpoint);
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
    const std::string& host_id, tt::tt_metal::TrayID tray, tt::tt_metal::ASICLocation loc, chan_id_t chan) const {
    return healthy(EndpointKey{tt::tt_metal::make_physical_node_id(host_id, tray, loc), chan});
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

std::vector<LinkInfo> LinkHealth::get_downed_links_for_host(const std::string& host_id) const {
    const auto records = by_host_.find(tt::tt_metal::canonical_host_for_node_id(host_id));
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
    const std::string& a_host_id, const std::string& b_host_id) const {
    const auto a = tt::tt_metal::canonical_host_for_node_id(a_host_id);
    const auto b = tt::tt_metal::canonical_host_for_node_id(b_host_id);

    // Both directions, since each cable is stored once per end and a caller asking about a host pair
    // wants the cable, not one arbitrary half of it.
    std::vector<LinkInfo> between;
    auto collect = [this, &between](const std::string& from, const std::string& to) {
        const auto records = by_host_.find(from);
        if (records == by_host_.end()) {
            return;
        }
        for (const LinkInfo* record : records->second) {
            if (record->dst_host_id == to) {
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

}  // namespace tt::tt_fabric
