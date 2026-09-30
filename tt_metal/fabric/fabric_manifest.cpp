// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_manifest.hpp"

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <llrt/tt_cluster.hpp>

#include "fabric_builder_context.hpp"
#include "fabric_context.hpp"
#include "fabric_host_utils.hpp"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_manifest_model.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/fabric_manifest_names.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <unistd.h>

namespace tt::tt_fabric {

namespace {

using json = nlohmann::ordered_json;

using manifest::chip_key;
using manifest::direction_letter;
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

// Two TX queues, which the second ERISC enables. Credits on VC0 and VC1 then travel in L1 counters.
bool uses_multi_txq(const FabricBuilderContext& builder_context) {
    const auto& router_config = builder_context.get_fabric_router_config();
    return router_config.sender_txq_id != router_config.receiver_txq_id;
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
    block["multi_txq"] = uses_multi_txq(fabric_context.get_builder_context());
    return block;
}

// ============ Credit transport ============

// Each fabric VC's credit transport backing on the mesh.
json make_credit_transport_json(const FabricBuilderContext& builder_context, MeshId mesh_id) {
    const auto& plan = builder_context.get_stream_assignment(mesh_id).plan();

    json transport = json::object();
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        // Skip VCs that have no senders
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

// What each allocated stream register on the mesh is used for, keyed by stream id.
json make_stream_registers_json(const FabricBuilderContext& builder_context, MeshId mesh_id) {
    std::map<uint32_t, json> uses_by_id;

    // For the mesh, create a JSON object for each stream register
    for (const auto& use : builder_context.get_stream_assignment(mesh_id).uses()) {
        auto& uses = uses_by_id[use.stream_id];
        // Two credit uses of one register would count into the same value.
        TT_FATAL(
            uses.empty(),
            "Fabric manifest: {} stream {} has two buf_space_available uses: {} and {}",
            mesh_key(mesh_id),
            use.stream_id,
            uses.front().at("role").get<std::string>(),
            lower_enum_name(use.role));
        json entry;
        entry["role"] = lower_enum_name(use.role);
        if (use.vc.has_value()) {
            entry["vc"] = *use.vc;
        }
        if (use.index.has_value()) {
            entry["index"] = *use.index;
        }
        entry["register"] = lower_enum_name(manifest::StreamRegister::BUF_SPACE_AVAILABLE);
        uses.push_back(std::move(entry));
    }

    json registers = json::object();
    for (auto& [stream_id, uses] : uses_by_id) {
        registers[std::to_string(stream_id)] = std::move(uses);
    }
    return registers;
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

    // Wrap links require a 2D mesh, a peer, and an east-west or north-south direction
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
    out["host_cleared"] = region.host_cleared;
    return out;
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
    out["dispatch_link"] = link.dispatch_link;
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

// index_space says whose sender channels an element belongs to: the to_sender arrays are indexed by this
// router's sender compact index, and the receiver arrays by the peer's.
json credit_counters_json(const manifest::L1CreditCounters& counters) {
    const auto counter_array = [](const manifest::L1Region& region, const char* index_space) {
        json out = l1_region_json(region);
        out["index_space"] = index_space;
        return out;
    };
    json out;
    out["to_sender_ack"] = counter_array(counters.to_sender_ack, "own_sender_compact");
    out["to_sender_completion"] = counter_array(counters.to_sender_completion, "own_sender_compact");
    out["receiver_ack"] = counter_array(counters.receiver_ack, "peer_sender_compact");
    out["receiver_completion"] = counter_array(counters.receiver_completion, "peer_sender_compact");
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

// An array reference is a path within the router, e.g. "credit_counters/to_sender_ack".
json credit_ref_json(const manifest::CreditRef& credit) {
    if (const auto* stream = std::get_if<manifest::StreamRef>(&credit)) {
        return stream_ref_json(*stream);
    }
    const auto& array = std::get<manifest::ArrayRef>(credit);
    json out;
    out["array"] = array.array;
    out["index"] = array.index;
    return out;
}

// The NoC is written as its number: NOC's RISCV_*_default aliases share its values, so it has no unique name.
json noc_write_config_json(const manifest::NocWriteConfig& config) {
    json out;
    out["noc"] = static_cast<uint32_t>(config.noc);
    out["cmd_buf"] = lower_enum_name(config.cmd_buf);
    return out;
}

// The path of the router facing `direction` on the same chip and routing plane as the router on `chan`.
std::string sibling_router_path(
    const ControlPlane& control_plane, FabricNodeId node, chan_id_t chan, eth_chan_directions direction) {
    const auto plane = control_plane.get_routing_plane_id(node, chan);
    for (const auto& [sibling_chan, sibling_direction] : control_plane.get_active_fabric_eth_channels(node)) {
        if (sibling_direction == direction && control_plane.get_routing_plane_id(node, sibling_chan) == plane) {
            return router_path(node, router_key(direction, plane));
        }
    }
    TT_THROW(
        "Fabric manifest: {} channel {} has a producer facing {} on routing plane {}, but no router is there",
        node,
        chan,
        direction_letter(direction),
        plane);
}

// A worker producer is "worker" and a sibling router producer is that router's path.
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
    return sibling_router_path(
        control_plane, node, chan, std::get<manifest::SiblingRouterRef>(*producer).direction);
}

json noc_forward_config_json(const manifest::NocForwardConfig& config) {
    json out;
    out["noc"] = static_cast<uint32_t>(config.noc);
    out["data_cmd_buf"] = lower_enum_name(config.data_cmd_buf);
    out["sync_cmd_buf"] = lower_enum_name(config.sync_cmd_buf);
    return out;
}

json serviced_by_json(const std::vector<uint32_t>& risc_ids) {
    json out = json::array();
    for (const auto risc_id : risc_ids) {
        out.push_back(fmt::format("erisc{}", risc_id));
    }
    return out;
}

json sender_channel_json(
    const manifest::SenderChannel& sender, const ControlPlane& control_plane, FabricNodeId node, chan_id_t chan) {

    json credits;
    if (sender.credits.acked.has_value()) {
        credits["acked"] = credit_ref_json(*sender.credits.acked);
    }
    credits["completed"] = credit_ref_json(sender.credits.completed);

    json control_info;
    control_info["connection"] = l1_region_json(sender.control_info.connection);
    control_info["conn_info"] = l1_region_json(sender.control_info.conn_info);
    if (sender.control_info.buffer_index_sem.has_value()) {
        control_info["buffer_index_sem"] = l1_region_json(*sender.control_info.buffer_index_sem);
    }

    json out;
    out["serviced_by"] = serviced_by_json(sender.serviced_by);
    out["producer"] = sender_producer_json(sender.producer, control_plane, node, chan);
    out["is_injection_channel"] = sender.is_injection_channel;
    out["producer_credit_return"] = noc_write_config_json(sender.producer_credit_return);
    out["ring_buffer"] = l1_region_json(sender.ring_buffer);
    out["free_slots"] = stream_ref_json(sender.free_slots);
    out["credits"] = std::move(credits);
    out["control_info"] = std::move(control_info);
    return out;
}

// Keyed vc<N> then ch<M>. VCs this router has no senders on are left out.
json senders_json(const manifest::Router& router, const ControlPlane& control_plane, FabricNodeId node) {
    json out = json::object();
    for (size_t vc = 0; vc < router.channels.senders.size(); ++vc) {
        const auto& channels = router.channels.senders[vc];
        if (channels.empty()) {
            continue;
        }
        json vc_json;
        for (size_t channel = 0; channel < channels.size(); ++channel) {
            vc_json[fmt::format("ch{}", channel)] =
                sender_channel_json(channels[channel], control_plane, node, router.identity.eth_chan);
        }
        out[fmt::format("vc{}", vc)] = std::move(vc_json);
    }
    return out;
}

// A sender channel's path, e.g. "M0/C7/E0/senders/vc0/ch1".
std::string sender_channel_path(const std::string& router, uint32_t vc, uint32_t channel) {
    return fmt::format("{}/senders/vc{}/ch{}", router, vc, channel);
}

// Keyed edge<N>, the kernel's EDGE_<N>. The target router and landing channel are the downstream_channel path.
json downstream_edges_json(
    const std::vector<manifest::DownstreamEdge>& edges,
    const ControlPlane& control_plane,
    FabricNodeId node,
    chan_id_t chan) {
    json out = json::object();
    for (const auto& edge : edges) {
        json entry;
        entry["downstream_channel"] = sender_channel_path(
            sibling_router_path(control_plane, node, chan, edge.target.direction),
            edge.landing_vc,
            edge.landing_channel);
        entry["free_slots"] = stream_ref_json(edge.free_slots);
        entry["teardown_sem"] = l1_region_json(edge.teardown_sem);
        out[fmt::format("edge{}", edge.edge)] = std::move(entry);
    }
    return out;
}

// A receiver's producer is always the peer router, so it is the link's peer.
json receiver_channel_json(
    const manifest::ReceiverChannel& receiver,
    const std::optional<std::string>& peer_path,
    const ControlPlane& control_plane,
    FabricNodeId node,
    chan_id_t chan) {
    json out;
    out["serviced_by"] = serviced_by_json(receiver.serviced_by);
    out["producer"] = peer_path.has_value() ? json(*peer_path) : json(nullptr);
    out["forwarding_disabled"] = receiver.forwarding_disabled;
    out["intermesh_ingress"] = receiver.intermesh_ingress;
    out["forward_noc"] = noc_forward_config_json(receiver.forward_noc);
    out["local_write_noc"] = noc_write_config_json(receiver.local_write_noc);
    out["ring_buffer"] = l1_region_json(receiver.ring_buffer);
    out["pkts_sent"] = stream_ref_json(receiver.pkts_sent);
    if (receiver.free_slots.has_value()) {
        out["free_slots"] = stream_ref_json(*receiver.free_slots);
    }
    out["downstream_edges"] = downstream_edges_json(receiver.downstream_edges, control_plane, node, chan);
    return out;
}

// Keyed like senders_json. VCs this router has no receivers on are left out.
json receivers_json(
    const manifest::Router& router,
    const std::optional<std::string>& peer_path,
    const ControlPlane& control_plane,
    FabricNodeId node) {
    json out = json::object();
    for (size_t vc = 0; vc < router.channels.receivers.size(); ++vc) {
        const auto& channels = router.channels.receivers[vc];
        if (channels.empty()) {
            continue;
        }
        json vc_json;
        for (size_t channel = 0; channel < channels.size(); ++channel) {
            vc_json[fmt::format("ch{}", channel)] =
                receiver_channel_json(channels[channel], peer_path, control_plane, node, router.identity.eth_chan);
        }
        out[fmt::format("vc{}", vc)] = std::move(vc_json);
    }
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
    const auto virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
        physical_chip_id, tt::tt_metal::CoreCoord(logical_core.x, logical_core.y), CoreType::ETH);

    json out;
    out["identity"] = router_identity_json(
        router.identity,
        json::array({logical_core.x, logical_core.y}),
        json::array({virtual_core.x, virtual_core.y}));
    out["link"] = eth_link_json(
        router.link,
        peer_path,
        control_plane.is_cross_host_eth_link(physical_chip_id, chan),
        is_wrap_link(fabric_type, control_plane.get_mesh_graph(), node, router.link.direction, peer));
    out["shape"] = router_shape_json(router.shape);
    out["credit_counters"] = credit_counters_json(router.credit_counters);
    out["channels"]["senders"] = senders_json(router, control_plane, node);
    out["channels"]["receivers"] = receivers_json(router, peer_path, control_plane, node);
    return out;
}

// Every sender channel a sibling router feeds is exactly one downstream edge of that router, and every
// downstream edge lands on a sender channel fed by the router it leaves.
void check_edges_match_producers(const json& routers, FabricNodeId node) {
    std::map<std::string, std::string> fed_by;
    std::map<std::string, std::string> edges_into;
    for (const auto& [key, router] : routers.items()) {
        const auto path = router_path(node, key);
        const auto& channels = router.at("channels");
        for (const auto& [vc_key, vc_channels] : channels.at("senders").items()) {
            for (const auto& [ch_key, sender] : vc_channels.items()) {
                const auto& producer = sender.at("producer");
                if (producer.is_string() && producer != "worker") {
                    fed_by[fmt::format("{}/senders/{}/{}", path, vc_key, ch_key)] = producer.get<std::string>();
                }
            }
        }
        for (const auto& [vc_key, vc_channels] : channels.at("receivers").items()) {
            for (const auto& [ch_key, receiver] : vc_channels.items()) {
                for (const auto& [edge_key, edge] : receiver.at("downstream_edges").items()) {
                    const auto target = edge.at("downstream_channel").get<std::string>();
                    const auto [it, inserted] = edges_into.emplace(target, path);
                    TT_FATAL(
                        inserted,
                        "Fabric manifest: {} and {} both have a downstream edge into {}",
                        it->second,
                        path,
                        target);
                }
            }
        }
    }

    for (const auto& [target, source] : edges_into) {
        const auto it = fed_by.find(target);
        TT_FATAL(
            it != fed_by.end() && it->second == source,
            "Fabric manifest: {} has a downstream edge into {}, but that channel's producer is {}",
            source,
            target,
            it != fed_by.end() ? it->second : "not a sibling router");
    }
    for (const auto& [channel, producer] : fed_by) {
        TT_FATAL(
            edges_into.contains(channel),
            "Fabric manifest: {} is fed by {}, but no downstream edge of that router lands on it",
            channel,
            producer);
    }
}

// ============ Chip and mesh ============

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
    chip["routers"] =
        make_chip_routers_json(collected, control_plane, cluster, fabric_type, node, *physical_chip_id);
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
        mesh["stream_registers"] = make_stream_registers_json(builder_context, mesh_id);
    }

    json chips = json::object();
    for (const auto& [_, fabric_chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
        chips[chip_key(fabric_chip_id)] =
            make_chip_json(control_plane, cluster, builder_context, fabric_type, FabricNodeId(mesh_id, fabric_chip_id));
    }
    mesh["chips"] = std::move(chips);
    return mesh;
}

}  // namespace

std::filesystem::path fabric_manifest_path(const tt::llrt::RunTimeOptions& rtoptions) {
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const int rank = *distributed_context->rank();
    const int world_size = *distributed_context->size();
    return std::filesystem::path(rtoptions.get_logs_dir()) / "generated" / "fabric" /
           ("fabric_manifest_rank_" + std::to_string(rank + 1) + "_of_" + std::to_string(world_size) + ".json");
}

void serialize_fabric_manifest_to_file(
    const ControlPlane& control_plane, const std::filesystem::path& output_file_path) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const FabricType fabric_type = get_fabric_type(control_plane.get_fabric_config(), cluster.is_ubb_galaxy());
    const auto& fabric_context = control_plane.get_fabric_context();
    TT_FATAL(
        fabric_context.has_builder_context(), "fabric manifest must be serialized after routers are compiled");
    const auto& builder_context = fabric_context.get_builder_context();

    json manifest;
    manifest["manifest_version"] = FABRIC_MANIFEST_VERSION;
    manifest["kind"] = "fabric_manifest";
    manifest["run"] = make_run_json(control_plane, cluster);
    manifest["fabric_context"] = make_fabric_context_json(fabric_context);

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

}  // namespace tt::tt_fabric
