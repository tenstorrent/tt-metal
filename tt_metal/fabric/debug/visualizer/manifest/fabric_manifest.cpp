// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include <llrt/tt_cluster.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <variant>
#include <unistd.h>

namespace tt::tt_fabric {

namespace {

using json = nlohmann::ordered_json;

using manifest::chip_key;
using manifest::direction_letter;
using manifest::kind_name;
using manifest::lower_enum_name;
using manifest::mesh_key;
using manifest::router_key;

// Returns the current UTC time in ISO 8601 format.
std::string utc_now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&now_time, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

// Returns a JSON object with the "run" information for the fabric instance.
json make_run_json(const ControlPlane& control_plane, const tt::Cluster& cluster) {
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    json run;
    run["arch"] = lower_enum_name(cluster.arch());
    run["fabric_config"] = lower_enum_name(control_plane.get_fabric_config());
    run["reliability_mode"] = lower_enum_name(control_plane.get_fabric_reliability_mode());
    run["tensix_config"] = lower_enum_name(control_plane.get_fabric_tensix_config());
    run["udm_mode"] = lower_enum_name(control_plane.get_fabric_udm_mode());
    run["host_rank"] = *control_plane.get_local_host_rank_id_binding();
    run["mpi_rank"] = *distributed_context->rank();
    run["world_size"] = *distributed_context->size();
    run["written_at"] = utc_now_iso8601();
    return run;
}

// Returns a JSON object with the fabric context block information.
json make_fabric_context_json(const FabricContext& fabric_context) {
    json block;
    block["topology"] = lower_enum_name(fabric_context.get_fabric_topology());
    block["is_2d_routing"] = fabric_context.is_2D_routing_enabled();
    block["packet_header_size_bytes"] = fabric_context.get_fabric_packet_header_size_bytes();
    block["max_payload_size_bytes"] = fabric_context.get_fabric_max_payload_size_bytes();
    block["channel_buffer_size_bytes"] = fabric_context.get_fabric_channel_buffer_size_bytes();
    if (fabric_context.is_2D_routing_enabled()) {
        block["routing_2d_route_buffer_size"] = fabric_context.get_2d_pkt_hdr_route_buffer_size();
    } else {
        block["routing_1d_extension_words"] = fabric_context.get_1d_pkt_hdr_extension_words();
    }
    block["multi_txq"] = fabric_context.get_builder_context().get_fabric_router_config().multi_txq_enabled();
    return block;
}

// ============ Credit transport ============

// Each fabric VC's credit transport backing on the mesh. VCs no router in the fabric has senders on are left out.
json make_credit_transport_json(const FabricBuilderContext& builder_context, MeshId mesh_id) {
    const auto& plan = builder_context.get_stream_assignment(mesh_id).plan();
    json transport = json::object();
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        if (builder_context.get_max_sender_channels_per_vc()[vc] == 0) {
            continue;
        }
        json reasons = json::array();
        for (const auto reason : plan.reasons(vc)) {
            reasons.push_back(lower_enum_name(reason));
        }
        json entry;
        entry["backing"] = plan.vc_uses_counters(vc) ? "l1_counter" : "stream_register";
        entry["reasons"] = std::move(reasons);
        transport[fmt::format("vc{}", vc)] = std::move(entry);
    }
    return transport;
}

// ============ Regions ============

json l1_region_json(const manifest::L1Region& region) {
    json out;
    out["address"] = region.address;
    out["size"] = region.size;
    if (region.num_elements.has_value()) {
        out["num_elements"] = *region.num_elements;
    }
    if (region.size_per_element.has_value()) {
        out["size_per_element"] = *region.size_per_element;
    }
    out["schema"] = region.schema;
    out["cleared_by_host"] = region.cleared_by_host;
    return out;
}

// ============ Paths ============

// One part of the manifest refers to a router elsewhere by its path, e.g. "M0/C7/E0".
std::string router_path(FabricNodeId node, const std::string& key) {
    return fmt::format("{}/{}/{}", mesh_key(node.mesh_id), chip_key(node.chip_id), key);
}

// ============ Link facts ============

// The path of the router on `peer_chan`, or nullopt when ControlPlane has no active router there (the
// channel was trimmed from the peer's routing planes).
std::optional<std::string> peer_router_path(
    const ControlPlane& control_plane, FabricNodeId peer_node, chan_id_t peer_chan) {
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(peer_node)) {
        if (chan == peer_chan) {
            return router_path(peer_node, router_key(direction, control_plane.get_routing_plane_id(peer_node, chan)));
        }
    }
    return std::nullopt;
}

// Fails unless the physical system descriptor has a cable from `chan` on `node` to `peer_chan` on `peer_node`.
void check_link_is_cabled(
    const ControlPlane& control_plane, FabricNodeId node, chan_id_t chan, FabricNodeId peer_node, chan_id_t peer_chan) {
    const auto connections = control_plane.get_physical_system_descriptor().get_eth_connections(
        control_plane.get_asic_id_from_fabric_node_id(node), control_plane.get_asic_id_from_fabric_node_id(peer_node));
    const bool cabled = std::ranges::any_of(connections, [&](const auto& connection) {
        return connection.src_chan == chan && connection.dst_chan == peer_chan;
    });
    TT_FATAL(
        cabled,
        "Fabric manifest: ControlPlane pairs {} channel {} with {} channel {}, but no cable connects them",
        node,
        chan,
        peer_node,
        peer_chan);
}

// Wrap edges are resolved here rather than inferred by the viewer: a link wraps when its axis genuinely
// closes and its coordinate delta spans the mesh.
bool is_wrap_link(
    FabricType fabric_type,
    const MeshGraph& mesh_graph,
    FabricNodeId node,
    eth_chan_directions direction,
    const std::optional<std::pair<FabricNodeId, chan_id_t>>& peer) {
    const MeshShape mesh_shape = mesh_graph.get_mesh_shape(node.mesh_id);
    const bool is_east_west = direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST;
    const bool is_north_south = direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH;

    // Wrap links require a 2D mesh, a peer in the same mesh, and an east-west or north-south direction
    if (mesh_shape.dims() != 2 || !peer.has_value() || peer->first.mesh_id != node.mesh_id ||
        !(is_east_west || is_north_south)) {
        return false;
    }
    const uint32_t axis = is_east_west ? 1 : 0;
    if (!has_genuine_torus_axis(fabric_type, mesh_shape, axis)) {
        return false;
    }
    const uint32_t here = mesh_graph.chip_to_coordinate(node.mesh_id, node.chip_id)[axis];
    const uint32_t there = mesh_graph.chip_to_coordinate(node.mesh_id, peer->first.chip_id)[axis];
    const uint32_t delta = here > there ? here - there : there - here;
    return delta == mesh_shape[axis] - 1;
}

// ============ Router ============

// The router's mesh and chip are its path, so they are not repeated here.
json router_identity_json(const manifest::RouterIdentity& identity, json logical_core, json virtual_core) {
    json out;
    out["eth_chan"] = identity.eth_chan;
    out["logical_core"] = std::move(logical_core);
    out["virtual_core"] = std::move(virtual_core);
    return out;
}

// Direction and routing plane are the router's key, so they are not repeated here.
json eth_link_json(const manifest::EthLink& link, const std::optional<std::string>& peer, bool cross_host, bool wrap) {
    json out;
    out["edge_capability"] = lower_enum_name(link.edge_capability);
    out["peer"] = peer.has_value() ? json(*peer) : json(nullptr);
    out["cross_host"] = cross_host;
    out["wrap"] = wrap;
    out["dispatch_link"] = link.is_dispatch_link;
    return out;
}

json router_shape_json(const manifest::RouterShape& shape) {
    json out;
    out["num_vcs"] = shape.num_vcs;
    out["senders_per_vc"] = shape.senders_per_vc;
    out["receivers_per_vc"] = shape.receivers_per_vc;
    out["num_active_eriscs"] = shape.num_active_eriscs;
    out["channel_trimming_overrides_applied"] = shape.channel_trimming_overrides_applied;
    out["vc0_bubble_flow_control"] = shape.vc0_bubble_flow_control;
    return out;
}

// index_space says whose sender channels an element belongs to: the to_sender arrays are indexed by this router's
// sender compact index, and the receiver arrays by the peer router's sender compact index.
json credit_counters_json(const manifest::L1CreditCounters& counters) {
    using manifest::CreditCounterArray;
    json out;
    const auto add = [&out](CreditCounterArray array, const manifest::L1Region& region, const char* index_space) {
        json entry = l1_region_json(region);
        entry["index_space"] = index_space;
        out[lower_enum_name(array)] = std::move(entry);
    };
    add(CreditCounterArray::TO_SENDER_ACK, counters.to_sender_ack, "own_sender_compact");
    add(CreditCounterArray::TO_SENDER_COMPLETION, counters.to_sender_completion, "own_sender_compact");
    add(CreditCounterArray::RECEIVER_ACK, counters.receiver_ack, "peer_sender_compact");
    add(CreditCounterArray::RECEIVER_COMPLETION, counters.receiver_completion, "peer_sender_compact");
    return out;
}

// ============ Channels ============

json stream_ref_json(const manifest::StreamRef& stream) {
    json out;
    out["stream_id"] = stream.stream_id;
    out["register"] = lower_enum_name(stream.reg);
    out["schema"] = stream.schema;
    return out;
}

// A counter element names its array by its path within the router, e.g. "credit_counters/to_sender_ack".
json credit_ref_json(const manifest::CreditRef& credit) {
    if (const auto* stream = std::get_if<manifest::StreamRef>(&credit)) {
        return stream_ref_json(*stream);
    }
    const auto& counter = std::get<manifest::CounterRef>(credit);
    json out;
    out["array"] = fmt::format("credit_counters/{}", lower_enum_name(counter.array));
    out["index"] = counter.index;
    return out;
}

// Keyed by field. Each carries its category and kind, then its region, its stream register or its value.
json fields_json(const std::vector<manifest::Field>& fields) {
    json out = json::object();
    for (const auto& field : fields) {
        json entry;
        entry["category"] = lower_enum_name(field.category);
        entry["kind"] = kind_name(field.kind);
        const auto schema = [](const manifest::ElementType& element) {
            return manifest::schema_name(element.type, element.size);
        };
        std::visit(
            ttsl::overloaded{
                [&](const manifest::kind::L1Value& kind) {
                    entry.update(l1_region_json(manifest::L1Region{
                        .address = field.arg,
                        .size = kind.element.size,
                        .schema = schema(kind.element),
                        .cleared_by_host = field.cleared_by_host,
                    }));
                },
                [&](const manifest::kind::Stream& kind) {
                    entry.update(stream_ref_json(
                        manifest::StreamRef{.stream_id = field.arg, .reg = kind.reg, .schema = schema(kind.element)}));
                },
                [&](const manifest::kind::Number&) { entry["value"] = field.arg; },
                [&](const manifest::kind::Flag&) { entry["value"] = field.arg != 0; },
                [&](const manifest::kind::Enum& kind) {
                    entry["value"] = field.arg;
                    entry["schema"] = schema(kind.element);
                },
            },
            field.kind);
        const std::string key(field.key);
        TT_FATAL(!out.contains(key), "Fabric manifest: two fields are keyed {}", key);
        out[key] = std::move(entry);
    }
    return out;
}

json serviced_by_json(const std::vector<uint32_t>& risc_ids) {
    json out = json::array();
    for (const auto risc_id : risc_ids) {
        out.push_back(fmt::format("erisc{}", risc_id));
    }
    return out;
}

// The channel of the router facing `direction` on the same chip and routing plane as the router on `chan`.
chan_id_t sibling_chan(
    const ControlPlane& control_plane, FabricNodeId node, chan_id_t chan, eth_chan_directions direction) {
    const auto plane = control_plane.get_routing_plane_id(node, chan);
    for (const auto& [sibling, sibling_direction] : control_plane.get_active_fabric_eth_channels(node)) {
        if (sibling_direction == direction && control_plane.get_routing_plane_id(node, sibling) == plane) {
            return sibling;
        }
    }
    TT_THROW(
        "Fabric manifest: {} channel {} has a sibling facing {} on routing plane {}, but no router is there",
        node,
        chan,
        direction_letter(direction),
        plane);
}

// The path of the router facing `direction` on the same chip and routing plane as the router on `chan`.
std::string sibling_router_path(
    const ControlPlane& control_plane, FabricNodeId node, chan_id_t chan, eth_chan_directions direction) {
    const auto sibling = sibling_chan(control_plane, node, chan, direction);
    return router_path(node, router_key(direction, control_plane.get_routing_plane_id(node, sibling)));
}

tt::tt_metal::CoreCoord router_virtual_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    const auto logical_core =
        cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        physical_chip_id, tt::tt_metal::CoreCoord(logical_core.x, logical_core.y), CoreType::ETH);
}

// A worker producer is "worker", the tensix mux is "tensix_mux", and a sibling router producer is that router's path.
json sender_producer_json(
    const std::optional<manifest::SenderChannelProducer>& producer,
    const ControlPlane& control_plane,
    FabricNodeId node,
    chan_id_t chan) {
    if (!producer.has_value()) {
        return nullptr;
    }
    if (std::holds_alternative<manifest::LocalWorker>(*producer)) {
        return "worker";
    }
    if (std::holds_alternative<manifest::LocalTensixMux>(*producer)) {
        return "tensix_mux";
    }
    return sibling_router_path(control_plane, node, chan, std::get<manifest::SiblingRouterRef>(*producer).direction);
}

json sender_channel_json(
    const manifest::SenderChannel& sender, const ControlPlane& control_plane, FabricNodeId node, chan_id_t chan) {
    json credits;
    if (sender.credits.acked.has_value()) {
        credits["acked"] = credit_ref_json(*sender.credits.acked);
    }
    credits["completed"] = credit_ref_json(sender.credits.completed);

    json out;
    out["status"] = lower_enum_name(sender.status);
    out["serviced_by"] = serviced_by_json(sender.serviced_by);
    out["producer"] = sender_producer_json(sender.producer, control_plane, node, chan);
    out["ring_buffer"] = l1_region_json(sender.ring_buffer);
    out["credits"] = std::move(credits);
    out["fields"] = fields_json(sender.fields);
    return out;
}

json receiver_channel_json(const manifest::ReceiverChannel& receiver) {
    json out;
    out["status"] = lower_enum_name(receiver.status);
    out["serviced_by"] = serviced_by_json(receiver.serviced_by);
    out["forwards_on"] =
        receiver.forwards_on.has_value() ? json(fmt::format("vc{}", *receiver.forwards_on)) : json(nullptr);
    out["ring_buffer"] = l1_region_json(receiver.ring_buffer);
    out["fields"] = fields_json(receiver.fields);
    return out;
}

// Keyed vc<N> then ch<M>. VCs the router has no channels of this kind on are left out.
template <typename Channel, typename ChannelJson>
json channels_by_vc_json(const std::vector<std::vector<Channel>>& channels_by_vc, ChannelJson channel_json) {
    json out = json::object();
    for (size_t vc = 0; vc < channels_by_vc.size(); ++vc) {
        const auto& channels = channels_by_vc[vc];
        if (channels.empty()) {
            continue;
        }
        json vc_json;
        for (size_t channel = 0; channel < channels.size(); ++channel) {
            vc_json[fmt::format("ch{}", channel)] = channel_json(channels[channel]);
        }
        out[fmt::format("vc{}", vc)] = std::move(vc_json);
    }
    return out;
}

// Keyed vc<N> then edge<N>, the kernel's EDGE_<N>. VCs without edges are left out. An edge names the sender channel
// it lands on. It goes through the tensix mux when it writes to a core other than the sibling's ERISC: in mux mode,
// the mux stands in for the sibling's VC0 channels other than the worker channel.
json downstream_edges_json(
    const manifest::Router& router,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricNodeId node,
    ChipId physical_chip_id) {
    const chan_id_t chan = router.identity.eth_chan;
    json out = json::object();
    for (size_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
        const auto& edges = router.intra_chip_downstream_edges[vc];
        if (edges.empty()) {
            continue;
        }
        json vc_json;
        for (const auto& edge : edges) {
            const auto target = sibling_chan(control_plane, node, chan, edge.target.direction);
            json entry;
            entry["downstream_channel"] = fmt::format(
                "{}/channels/senders/vc{}/ch{}",
                sibling_router_path(control_plane, node, chan, edge.target.direction),
                vc,
                edge.landing_channel);
            entry["through_tensix_mux"] = edge.core != router_virtual_core(cluster, physical_chip_id, target);
            entry["free_slots"] = stream_ref_json(edge.free_slots);
            entry["teardown_sem"] = l1_region_json(edge.teardown_sem);
            vc_json[fmt::format("edge{}", edge.edge)] = std::move(entry);
        }
        out[fmt::format("vc{}", vc)] = std::move(vc_json);
    }
    return out;
}

// ============ Fields ============

// Keyed erisc<N>.
json eriscs_json(const std::vector<manifest::Erisc>& eriscs) {
    json out = json::object();
    for (size_t risc_id = 0; risc_id < eriscs.size(); ++risc_id) {
        json& erisc = out[fmt::format("erisc{}", risc_id)];
        erisc["processor"] = lower_enum_name(eriscs[risc_id].processor);
        erisc["fields"] = fields_json(eriscs[risc_id].fields);
    }
    return out;
}

template <typename E>
json enum_names_json() {
    json out = json::array();
    for (const auto value : enchantum::values<E>) {
        out.push_back(lower_enum_name(value));
    }
    return out;
}

template <typename... Kinds>
json kind_names_json(std::type_identity<std::variant<Kinds...>>) {
    return json::array({kind_name<Kinds>()...});
}

// Every category and kind a field can have, so readers can group and read fields without a copy of the enums.
json make_vocabulary_json() {
    json out;
    out["categories"] = enum_names_json<manifest::FieldCategory>();
    out["kinds"] = kind_names_json(std::type_identity<manifest::Kind>{});
    return out;
}

// A collected router with what ControlPlane and the cluster know about it: peer, cross-host, wrap and cores.
json make_router_json(
    const manifest::Router& router,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricType fabric_type,
    FabricNodeId node,
    ChipId physical_chip_id) {
    const chan_id_t chan = router.identity.eth_chan;

    const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);
    std::optional<std::string> peer_path;
    if (peer.has_value()) {
        check_link_is_cabled(control_plane, node, chan, peer->first, peer->second);
        peer_path = peer_router_path(control_plane, peer->first, peer->second);
    }

    const auto logical_core =
        cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
    const auto virtual_core = router_virtual_core(cluster, physical_chip_id, chan);

    json out;
    out["identity"] = router_identity_json(
        router.identity, json::array({logical_core.x, logical_core.y}), json::array({virtual_core.x, virtual_core.y}));
    out["link"] = eth_link_json(
        router.link,
        peer_path,
        control_plane.is_cross_host_eth_link(physical_chip_id, chan),
        is_wrap_link(fabric_type, control_plane.get_mesh_graph(), node, router.link.direction, peer));
    out["shape"] = router_shape_json(router.shape);
    out["credit_counters"] = credit_counters_json(router.credit_counters);
    out["channels"]["senders"] = channels_by_vc_json(
        router.channels.senders,
        [&](const manifest::SenderChannel& sender) { return sender_channel_json(sender, control_plane, node, chan); });
    out["channels"]["receivers"] = channels_by_vc_json(router.channels.receivers, receiver_channel_json);
    out["intra_chip_downstream_edges"] = downstream_edges_json(router, control_plane, cluster, node, physical_chip_id);
    out["fields"] = fields_json(router.fields);
    out["eriscs"] = eriscs_json(router.eriscs);
    return out;
}

// ============ Chip and mesh ============

json local_sync_json(const manifest::LocalSync& local_sync) {
    json out;
    out["master_eth_chan"] = local_sync.master_eth_chan;
    out["num_routers"] = local_sync.num_routers;
    out["router_channels_mask"] = local_sync.router_channels_mask;
    return out;
}

// Each edge lands on a sender channel its sibling has, at the compact index the builder recorded for it.
void check_edge_landings(const manifest::Chip& chip, const ControlPlane& control_plane, FabricNodeId node) {
    std::map<chan_id_t, const manifest::Router*> by_chan;
    for (const auto& router : chip.routers) {
        by_chan.emplace(router.identity.eth_chan, &router);
    }
    for (const auto& router : chip.routers) {
        const chan_id_t chan = router.identity.eth_chan;
        for (uint32_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
            for (const auto& edge : router.intra_chip_downstream_edges[vc]) {
                const auto target = sibling_chan(control_plane, node, chan, edge.target.direction);
                const auto it = by_chan.find(target);
                TT_FATAL(
                    it != by_chan.end(),
                    "Fabric manifest: {} channel {} has an edge to channel {}, which was not collected",
                    node,
                    chan,
                    target);
                const auto& counts = it->second->shape.senders_per_vc;
                TT_FATAL(
                    edge.landing_channel < counts.at(vc),
                    "Fabric manifest: {} channel {}'s VC{} edge {} lands on channel {}, but channel {} has {}",
                    node,
                    chan,
                    vc,
                    edge.edge,
                    edge.landing_channel,
                    target,
                    counts.at(vc));
                if (edge.landing_compact.has_value()) {
                    const uint32_t compact =
                        std::accumulate(counts.begin(), counts.begin() + vc, 0u) + edge.landing_channel;
                    TT_FATAL(
                        compact == *edge.landing_compact,
                        "Fabric manifest: {} channel {}'s VC{} edge {} lands on compact channel {}, but the builder "
                        "connected it to {}",
                        node,
                        chan,
                        vc,
                        edge.edge,
                        compact,
                        *edge.landing_compact);
                }
            }
        }
    }
}

// Every sender channel a sibling feeds has exactly one edge into it, from that sibling, and every edge lands on a
// channel its router feeds. An edge through the tensix mux lands on a channel the mux carries.
void check_edges_match_producers(const json& routers, FabricNodeId node) {
    std::map<std::string, std::string> fed_by;
    std::map<std::string, std::string> edges_into;
    for (const auto& [key, router] : routers.items()) {
        const auto path = router_path(node, key);
        for (const auto& [vc_key, vc_channels] : router.at("channels").at("senders").items()) {
            for (const auto& [ch_key, sender] : vc_channels.items()) {
                const auto& producer = sender.at("producer");
                if (producer.is_string() && producer != "worker" && producer != "tensix_mux") {
                    fed_by[fmt::format("{}/channels/senders/{}/{}", path, vc_key, ch_key)] =
                        producer.get<std::string>();
                }
            }
        }
    }
    for (const auto& [key, router] : routers.items()) {
        const auto path = router_path(node, key);
        for (const auto& [vc_key, vc_edges] : router.at("intra_chip_downstream_edges").items()) {
            for (const auto& [edge_key, edge] : vc_edges.items()) {
                const auto target = edge.at("downstream_channel").get<std::string>();
                const auto [it, inserted] = edges_into.emplace(target, path);
                TT_FATAL(
                    inserted,
                    "Fabric manifest: {} and {} both have a downstream edge into {}",
                    it->second,
                    path,
                    target);
                if (edge.at("through_tensix_mux").get<bool>()) {
                    // The target's path within this chip's routers: it drops the mesh and chip keys.
                    const json::json_pointer status(
                        "/" + target.substr(target.find('/', target.find('/') + 1) + 1) + "/status");
                    TT_FATAL(
                        routers.contains(status) && routers.at(status) == lower_enum_name(manifest::ChannelStatus::MUX),
                        "Fabric manifest: {}'s {} {} goes through the tensix mux, but {} is not carried by a mux",
                        path,
                        vc_key,
                        edge_key,
                        target);
                    continue;
                }
                const auto fed = fed_by.find(target);
                TT_FATAL(
                    fed != fed_by.end() && fed->second == path,
                    "Fabric manifest: {} has a downstream edge into {}, but that channel's producer is {}",
                    path,
                    target,
                    fed != fed_by.end() ? fed->second : "not a sibling router");
            }
        }
    }
    for (const auto& [channel, producer] : fed_by) {
        TT_FATAL(
            edges_into.contains(channel),
            "Fabric manifest: {} is fed by {}, but no downstream edge of that router lands on it",
            channel,
            producer);
    }
}

// The chip's routers keyed <direction><routing_plane>. ControlPlane's active channels and the collected routers
// must correspond one to one, and agree on each router's direction.
json make_chip_routers_json(
    const manifest::Chip& chip,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricType fabric_type,
    FabricNodeId node,
    ChipId physical_chip_id) {
    std::map<chan_id_t, const manifest::Router*> collected;
    for (const auto& router : chip.routers) {
        TT_FATAL(
            collected.emplace(router.identity.eth_chan, &router).second,
            "Fabric manifest: {} has two collected routers on channel {}",
            node,
            router.identity.eth_chan);
    }

    json routers = json::object();
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(node)) {
        const auto it = collected.find(chan);
        TT_FATAL(
            it != collected.end(),
            "Fabric manifest: {} channel {} is an active fabric router, but no router was collected for it",
            node,
            chan);
        const manifest::Router& router = *it->second;
        collected.erase(it);
        TT_FATAL(
            router.link.direction == direction,
            "Fabric manifest: {} channel {} was built facing {}, but ControlPlane has it facing {}",
            node,
            chan,
            direction_letter(router.link.direction),
            direction_letter(direction));

        const auto key = router_key(direction, control_plane.get_routing_plane_id(node, chan));
        TT_FATAL(!routers.contains(key), "Fabric manifest: {} has two routers keyed {}", node, key);
        routers[key] = make_router_json(router, control_plane, cluster, fabric_type, node, physical_chip_id);
    }
    TT_FATAL(
        collected.empty(),
        "Fabric manifest: {} has {} collected routers on channels that are not active fabric routers",
        node,
        collected.size());
    check_edge_landings(chip, control_plane, node);
    check_edges_match_producers(routers, node);
    return routers;
}

// Every chip in the mesh graph appears, local or not, so the viewer can draw the whole mesh and show the host
// boundary. Only local chips have routers.
json make_chip_json(
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    const FabricBuilderContext& builder_context,
    FabricType fabric_type,
    FabricNodeId node) {
    const MeshCoordinate mesh_coord = control_plane.get_mesh_graph().chip_to_coordinate(node.mesh_id, node.chip_id);
    const auto physical_chip_id = control_plane.try_get_physical_chip_id_from_fabric_node_id(node);
    // A chip this rank cannot map to a physical device is a chip it cannot peek, so resolvability is the
    // practical definition of locality.
    const bool is_local = physical_chip_id.has_value();

    json chip;
    json coord = json::array();
    for (size_t dim = 0; dim < mesh_coord.dims(); ++dim) {
        coord.push_back(mesh_coord[dim]);
    }
    chip["mesh_coord"] = std::move(coord);
    if (!is_local) {
        chip["physical_chip_id"] = json(nullptr);
        chip["asic_id"] = json(nullptr);
        chip["is_local"] = false;
        return chip;
    }

    // A chip that built no routers publishes nothing.
    static const manifest::Chip no_routers{};
    const manifest::Chip& collected = builder_context.has_manifest_chip(*physical_chip_id)
                                          ? builder_context.get_manifest_chip(*physical_chip_id)
                                          : no_routers;
    chip["physical_chip_id"] = *physical_chip_id;
    // Hex string: the value exceeds what JSON numbers represent exactly.
    chip["asic_id"] = fmt::format("0x{:016x}", *control_plane.get_asic_id_from_fabric_node_id(node));
    chip["is_local"] = true;
    chip["z_port_role"] = lower_enum_name(collected.z_port_role);
    chip["local_sync"] = collected.local_sync ? local_sync_json(*collected.local_sync) : json(nullptr);
    chip["routers"] = make_chip_routers_json(collected, control_plane, cluster, fabric_type, node, *physical_chip_id);
    return chip;
}

json make_mesh_json(
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    const FabricBuilderContext& builder_context,
    FabricType fabric_type,
    MeshId mesh_id) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);

    json mesh;
    json shape = json::array();
    for (size_t dim = 0; dim < mesh_shape.dims(); ++dim) {
        shape.push_back(mesh_shape[dim]);
    }
    mesh["shape"] = std::move(shape);
    // has_genuine_torus_axis() is defined only for a 2D shape, and deliberately reports false for a declared
    // torus axis whose extent is too small to realize a distinct wrap edge.
    if (mesh_shape.dims() == 2) {
        json torus;
        torus["y"] = has_genuine_torus_axis(fabric_type, mesh_shape, 0);
        torus["x"] = has_genuine_torus_axis(fabric_type, mesh_shape, 1);
        mesh["torus"] = std::move(torus);
    }
    mesh["express_routing"] = control_plane.express_routing_enabled(mesh_id);
    // The builder context plans credits only for the meshes on this host.
    const auto local_mesh_ids = control_plane.get_local_mesh_id_bindings();
    if (std::ranges::find(local_mesh_ids, mesh_id) != local_mesh_ids.end()) {
        mesh["credit_transport"] = make_credit_transport_json(builder_context, mesh_id);
    }

    json chips = json::object();
    for (const auto& [_, fabric_chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
        chips[chip_key(fabric_chip_id)] =
            make_chip_json(control_plane, cluster, builder_context, fabric_type, FabricNodeId(mesh_id, fabric_chip_id));
    }
    mesh["chips"] = std::move(chips);
    return mesh;
}

// Writes the manifest to a temporary name and renames it into place, so a reader never sees a partial manifest.
void serialize_fabric_manifest_to_file(
    const ControlPlane& control_plane, const std::filesystem::path& output_file_path) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const FabricType fabric_type = get_fabric_type(control_plane.get_fabric_config(), cluster.is_ubb_galaxy());
    const auto& fabric_context = control_plane.get_fabric_context();
    TT_FATAL(
        fabric_context.has_builder_context(), "Fabric manifest: must be written after the fabric routers are compiled");
    const auto& builder_context = fabric_context.get_builder_context();

    json manifest;
    manifest["manifest_version"] = FABRIC_MANIFEST_VERSION;
    manifest["kind"] = "fabric_manifest";
    manifest["run"] = make_run_json(control_plane, cluster);
    manifest["fabric_context"] = make_fabric_context_json(fabric_context);
    manifest["vocabulary"] = make_vocabulary_json();

    auto mesh_ids = control_plane.get_mesh_graph().get_all_mesh_ids();
    std::ranges::sort(mesh_ids, {}, [](const MeshId& mesh_id) { return *mesh_id; });
    json meshes = json::object();
    for (const auto& mesh_id : mesh_ids) {
        meshes[mesh_key(mesh_id)] = make_mesh_json(control_plane, cluster, builder_context, fabric_type, mesh_id);
    }
    manifest["meshes"] = std::move(meshes);

    std::filesystem::create_directories(output_file_path.parent_path());
    const std::filesystem::path temporary_path =
        output_file_path.string() + ".tmp." + std::to_string(static_cast<uint64_t>(::getpid()));
    try {
        std::ofstream out_file;
        out_file.exceptions(std::ios::badbit | std::ios::failbit);
        out_file.open(temporary_path);
        out_file << manifest.dump(2) << '\n';
        out_file.close();
        std::filesystem::rename(temporary_path, output_file_path);
    } catch (...) {
        std::error_code remove_error;
        std::filesystem::remove(temporary_path, remove_error);
        throw;
    }

    log_debug(tt::LogFabric, "Serialized fabric manifest to file: {}", output_file_path.string());
}

}  // namespace

std::filesystem::path fabric_manifest_path(const tt::llrt::RunTimeOptions& rtoptions) {
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const int rank = *distributed_context->rank();
    const int world_size = *distributed_context->size();
    return std::filesystem::path(rtoptions.get_logs_dir()) / "generated" / "fabric" /
           ("fabric_manifest_rank_" + std::to_string(rank + 1) + "_of_" + std::to_string(world_size) + ".json");
}

void remove_stale_fabric_manifest(const tt::llrt::RunTimeOptions& rtoptions) {
    const auto manifest_path = fabric_manifest_path(rtoptions);
    try {
        if (std::filesystem::remove(manifest_path)) {
            log_debug(tt::LogFabric, "Removed stale fabric manifest: {}", manifest_path.string());
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "Failed to remove stale fabric manifest {}: {}", manifest_path.string(), e.what());
    }
}

void write_fabric_manifest(const ControlPlane& control_plane, const tt::llrt::RunTimeOptions& rtoptions) {
    const auto manifest_path = fabric_manifest_path(rtoptions);
    try {
        serialize_fabric_manifest_to_file(control_plane, manifest_path);
    } catch (const std::exception& e) {
        TT_THROW(
            "Failed to write fabric manifest {} (unset TT_METAL_FABRIC_GENERATE_MANIFEST to skip): {}",
            manifest_path.string(),
            e.what());
    }
}

}  // namespace tt::tt_fabric
