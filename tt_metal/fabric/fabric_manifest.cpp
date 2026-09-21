// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_manifest.hpp"

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/hal.hpp>
#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <llrt/tt_cluster.hpp>

#include "fabric_builder_context.hpp"
#include "fabric_context.hpp"
#include "fabric_edm_packet_header.hpp"
#include "fabric_host_utils.hpp"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <type_traits>
#include <unistd.h>

namespace tt::tt_fabric {

namespace {

// Retruns a string representation of the enum value.
//
// enchantum::to_string yields an empty view for a value that is not a named enumerator, which happens for
// bitmask combinations of FabricType such as MESH|TORUS_X, so in those cases we fall back to the numeric value.
template <typename E>
std::string enum_name(E value) {
    const auto name = enchantum::to_string(value);
    if (name.empty()) {
        return std::to_string(static_cast<std::underlying_type_t<E>>(value));
    }
    return std::string(name);
}

// (mesh, chip, channel) is the stable identity for a fabric router. The debug snapshot artifact keys its
// live values on the same triple, so the viewer can join a snapshot onto a manifest.
nlohmann::ordered_json manifest_endpoint_json(const FabricNodeId& node, chan_id_t chan) {
    nlohmann::ordered_json endpoint;
    endpoint["mesh_id"] = *node.mesh_id;
    endpoint["chip_id"] = node.chip_id;
    endpoint["eth_chan"] = chan;
    return endpoint;
}

using json = nlohmann::ordered_json;

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

// Returns a JSON object with the name of each enumerated value as the key, and the value as the numeric value.
template <typename E>
json enum_table() {
    json table = json::object();
    for (const auto& [value, name] : enchantum::entries_generator<E>) {
        table[std::string(name)] = static_cast<uint32_t>(value);
    }
    return table;
}

// Returns a JSON object with the base address and size of the specified L1 memory region.
json hal_l1_region_json(const tt::tt_metal::Hal& hal, tt::tt_metal::HalL1MemAddrType addr_type) {
    json region;
    region["base"] = hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, addr_type);
    region["size"] = hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, addr_type);
    return region;
}

// Returns a JSON object with the base address and size of the specified FabricRouterDiagnosticBufferMap region.
json diagnostic_region_json(const FabricRouterDiagnosticBufferMap::BufferRegion& region) {
    json out;
    out["base"] = region.l1_address;
    out["size"] = region.size_bytes;
    return out;
}

// Returns a JSON object with the "run" information for the fabric instance.
json make_run_json(const ControlPlane& control_plane, const tt::Cluster& cluster) {
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const FabricType fabric_type = get_fabric_type(control_plane.get_fabric_config(), cluster.is_ubb_galaxy());
    json run;
    run["arch"] = enum_name(cluster.arch());
    run["fabric_config"] = enum_name(control_plane.get_fabric_config());
    run["fabric_type"] = enum_name(fabric_type);
    run["reliability_mode"] = enum_name(control_plane.get_fabric_reliability_mode());
    run["tensix_config"] = enum_name(control_plane.get_fabric_tensix_config());
    run["udm_mode"] = enum_name(control_plane.get_fabric_udm_mode());
    run["host_rank"] = *control_plane.get_local_host_rank_id_binding();
    run["mpi_rank"] = *distributed_context->rank();
    run["world_size"] = *distributed_context->size();
    json local_mesh_ids = json::array();
    for (const auto& mesh_id : control_plane.get_local_mesh_id_bindings()) {
        local_mesh_ids.push_back(*mesh_id);
    }
    run["local_mesh_ids"] = std::move(local_mesh_ids);
    run["written_at"] = utc_now_iso8601();
    return run;
}

// Returns a JSON object with the hal block information.
json make_hal_json(const tt::tt_metal::Hal& hal) {
    using tt::tt_metal::HalL1MemAddrType;
    json hal_block;
    hal_block["unreserved"] = hal_l1_region_json(hal, HalL1MemAddrType::UNRESERVED);
    hal_block["go_msg"] = hal_l1_region_json(hal, HalL1MemAddrType::GO_MSG);
    hal_block["launch"] = hal_l1_region_json(hal, HalL1MemAddrType::LAUNCH);
    hal_block["fabric_telemetry"] = hal_l1_region_json(hal, HalL1MemAddrType::FABRIC_TELEMETRY);
    hal_block["routing_table"] = hal_l1_region_json(hal, HalL1MemAddrType::ROUTING_TABLE);
    hal_block["router_state"] = hal_l1_region_json(hal, HalL1MemAddrType::ROUTER_STATE);
    hal_block["router_command"] = hal_l1_region_json(hal, HalL1MemAddrType::ROUTER_COMMAND);
    hal_block["eth_fw_mailbox"] = hal_l1_region_json(hal, HalL1MemAddrType::ETH_FW_MAILBOX);
    return hal_block;
}

// Returns a JSON object with the heartbeat block information.
json make_heartbeat_json(tt::ARCH arch) {
    json heartbeat;
    heartbeat["address"] = arch == tt::ARCH::BLACKHOLE ? FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE
                                                       : FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE;
    heartbeat["magic"] = FABRIC_KERNEL_HEARTBEAT_MAGIC;
    heartbeat["magic_mask"] = FABRIC_KERNEL_HEARTBEAT_MAGIC_MASK;
    heartbeat["period_iters"] = FABRIC_KERNEL_HEARTBEAT_PERIOD_ITERS;
    return heartbeat;
}

// Returns a JSON object with the fabric context block information.
json make_fabric_context_json(const FabricContext& fabric_context) {
    json block;
    block["topology"] = enum_name(fabric_context.get_fabric_topology());
    block["is_2d_routing"] = fabric_context.is_2D_routing_enabled();
    block["packet_header_size_bytes"] = fabric_context.get_fabric_packet_header_size_bytes();
    block["max_payload_size_bytes"] = fabric_context.get_fabric_max_payload_size_bytes();
    block["channel_buffer_size_bytes"] = fabric_context.get_fabric_channel_buffer_size_bytes();
    if (fabric_context.is_2D_routing_enabled()) {
        block["routing_2d_route_buffer_size"] = fabric_context.get_2d_pkt_hdr_route_buffer_size();
    } else {
        block["routing_1d_extension_words"] = fabric_context.get_1d_pkt_hdr_extension_words();
    }
    block["tensix_enabled"] = fabric_context.is_tensix_enabled();
    block["bubble_flow_control"] = fabric_context.is_bubble_flow_control_enabled();
    return block;
}

// Returns a JSON object with the router template block information.
json make_router_template_json(const FabricBuilderContext& builder_context) {
    const auto& router_config = builder_context.get_fabric_router_config();
    const auto diagnostics = builder_context.get_telemetry_and_metadata_buffer_map();
    json block;
    block["edm_status_address"] = router_config.edm_status_address;
    block["termination_signal_address"] = router_config.termination_signal_address;
    block["edm_local_sync_address"] = router_config.edm_local_sync_address;
    block["handshake_address"] = tt::round_up(
        tt::tt_metal::hal::get_erisc_l1_unreserved_base(), FabricEriscDatamoverConfig::eth_channel_sync_size);
    block["unused_config_handshake_address"] = router_config.handshake_addr;
    block["edm_channel_ack_addr"] = router_config.edm_channel_ack_addr;
    json diagnostics_json;
    diagnostics_json["perf_telemetry"] = diagnostic_region_json(diagnostics.perf_telemetry);
    diagnostics_json["code_profiling"] = diagnostic_region_json(diagnostics.code_profiling);
    diagnostics_json["trimming"] = diagnostic_region_json(diagnostics.channel_trimming_capture);
    block["diagnostics"] = std::move(diagnostics_json);
    json addresses_to_clear = json::array();
    for (const auto address : builder_context.get_fabric_router_addresses_to_clear()) {
        addresses_to_clear.push_back(address);
    }
    block["addresses_to_clear"] = std::move(addresses_to_clear);
    block["router_buffer_clear_size_words"] = router_config.router_buffer_clear_size_words;
    return block;
}

// Returns a JSON object with the stream register assignment block information.
json make_stream_assignment_json(const ControlPlane& control_plane, const FabricBuilderContext& builder_context) {
    json assignment = json::object();
    auto local_mesh_ids = control_plane.get_local_mesh_id_bindings();
    std::sort(
        local_mesh_ids.begin(), local_mesh_ids.end(), [](const MeshId& lhs, const MeshId& rhs) { return *lhs < *rhs; });
    for (const auto& mesh_id : local_mesh_ids) {
        json named = json::object();
        for (const auto& [name, value] : builder_context.get_stream_assignment(mesh_id).named_args()) {
            named[name] = value;
        }
        assignment[std::to_string(*mesh_id)] = std::move(named);
    }
    return assignment;
}

// Returns a JSON object with the enum values for the fabric instance.
json make_enums_json() {
    json enums;
    // EDMStatus values are sparse 32-bit magic constants; enchantum cannot reflect them.
    json edm_status;
    edm_status["STARTED"] = EDMStatus::STARTED;
    edm_status["REMOTE_HANDSHAKE_COMPLETE"] = EDMStatus::REMOTE_HANDSHAKE_COMPLETE;
    edm_status["LOCAL_HANDSHAKE_COMPLETE"] = EDMStatus::LOCAL_HANDSHAKE_COMPLETE;
    edm_status["READY_FOR_TRAFFIC"] = EDMStatus::READY_FOR_TRAFFIC;
    edm_status["TERMINATED"] = EDMStatus::TERMINATED;
    edm_status["INITIALIZATION_STARTED"] = EDMStatus::INITIALIZATION_STARTED;
    edm_status["TXQ_INITIALIZED"] = EDMStatus::TXQ_INITIALIZED;
    edm_status["STREAM_REG_INITIALIZED"] = EDMStatus::STREAM_REG_INITIALIZED;
    edm_status["DOWNSTREAM_EDM_SETUP_STARTED"] = EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED;
    edm_status["EDM_VCS_SETUP_COMPLETE"] = EDMStatus::EDM_VCS_SETUP_COMPLETE;
    edm_status["WORKER_INTERFACES_INITIALIZED"] = EDMStatus::WORKER_INTERFACES_INITIALIZED;
    edm_status["ETHERNET_HANDSHAKE_COMPLETE"] = EDMStatus::ETHERNET_HANDSHAKE_COMPLETE;
    edm_status["VCS_OPENED"] = EDMStatus::VCS_OPENED;
    edm_status["ROUTING_TABLE_INITIALIZED"] = EDMStatus::ROUTING_TABLE_INITIALIZED;
    edm_status["INITIALIZATION_COMPLETE"] = EDMStatus::INITIALIZATION_COMPLETE;
    enums["EDMStatus"] = std::move(edm_status);
    enums["TerminationSignal"] = enum_table<TerminationSignal>();
    enums["RouterCommand"] = enum_table<RouterCommand>();
    json run_msg;
    run_msg["RUN_MSG_GO"] = 0x80;
    run_msg["RUN_MSG_DONE"] = 0;
    enums["RunMsg"] = std::move(run_msg);
    return enums;
}

std::string manifest_region_backing_to_str(ManifestRegionBacking backing) {
    switch (backing) {
        case ManifestRegionBacking::GROUP: return "group";
        case ManifestRegionBacking::UNRESERVED_L1: return "unreserved_l1";
        case ManifestRegionBacking::FIXED_L1: return "fixed_l1";
        case ManifestRegionBacking::STREAM_REG: return "stream_reg";
    }
    TT_THROW("Unknown manifest region backing {}", static_cast<uint32_t>(backing));
}

std::string manifest_region_writer_to_str(ManifestRegionWriter writer) {
    switch (writer) {
        case ManifestRegionWriter::NONE: return "none";
        case ManifestRegionWriter::ERISC0: return "erisc0";
        case ManifestRegionWriter::ERISC1: return "erisc1";
        case ManifestRegionWriter::ANY_ERISC: return "any_erisc";
        case ManifestRegionWriter::HOST: return "host";
        case ManifestRegionWriter::PEER: return "peer";
        case ManifestRegionWriter::WORKER: return "worker";
    }
    TT_THROW("Unknown manifest region writer {}", static_cast<uint32_t>(writer));
}

json make_manifest_router_region_json(const ManifestRouterRegion& region) {
    json out;
    out["id"] = region.id;
    out["parent"] = region.parent;
    out["backing"] = manifest_region_backing_to_str(region.backing);
    if (region.backing == ManifestRegionBacking::UNRESERVED_L1 || region.backing == ManifestRegionBacking::FIXED_L1) {
        out["address"] = region.address;
        out["size"] = region.size;
    }
    if (region.count.has_value()) {
        out["count"] = *region.count;
    }
    if (region.stride.has_value()) {
        out["stride"] = *region.stride;
    }
    if (region.stream_id.has_value()) {
        out["stream_id"] = *region.stream_id;
    }
    out["allocated"] = region.allocated;
    out["enabled"] = region.enabled;
    out["writer"] = manifest_region_writer_to_str(region.writer);
    if (!region.schema.empty()) {
        out["schema"] = region.schema;
    }
    if (!region.overlaps.empty()) {
        out["overlaps"] = region.overlaps;
    }
    return out;
}

json make_manifest_router_region_layout_json(const ManifestRouterRegionLayout& layout) {
    json regions = json::array();
    for (const auto& region : layout.regions) {
        regions.push_back(make_manifest_router_region_json(region));
    }
    json out;
    out["regions"] = std::move(regions);
    return out;
}

uint64_t fnv1a64(std::string_view value) {
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

using ManifestRouterBindingKey = std::pair<ChipId, chan_id_t>;

struct ManifestLayoutIndex {
    json layouts = json::object();
    std::map<ManifestRouterBindingKey, std::string> layout_ids;
    std::map<ManifestRouterBindingKey, const ManifestRouterInstance*> instances;
};

ManifestLayoutIndex build_manifest_layout_index(
    const ControlPlane& control_plane, const FabricBuilderContext& builder_context) {
    ManifestLayoutIndex data;
    std::map<std::string, std::string> canonical_to_id;
    std::map<std::string, std::string> id_to_canonical;

    auto mesh_ids = control_plane.get_mesh_graph().get_all_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end(), [](const MeshId& lhs, const MeshId& rhs) { return *lhs < *rhs; });
    for (const auto mesh_id : mesh_ids) {
        for (const auto& [_, fabric_chip_id] : control_plane.get_mesh_graph().get_chip_ids(mesh_id)) {
            const FabricNodeId node(mesh_id, fabric_chip_id);
            const auto physical_chip_id = control_plane.try_get_physical_chip_id_from_fabric_node_id(node);
            if (!physical_chip_id.has_value() || !builder_context.has_manifest_router_instances(*physical_chip_id)) {
                continue;
            }
            for (const auto& instance : builder_context.get_manifest_router_instances(*physical_chip_id)) {
                TT_FATAL(
                    instance.local_node == node,
                    "Manifest router instance for {} was published under {}",
                    instance.local_node,
                    node);
                const ManifestRouterBindingKey key{*physical_chip_id, instance.eth_chan};
                TT_FATAL(
                    data.instances.emplace(key, &instance).second,
                    "Duplicate manifest router instance for chip {} channel {}",
                    *physical_chip_id,
                    instance.eth_chan);

                const json layout = make_manifest_router_region_layout_json(instance.layout);
                const std::string canonical = layout.dump();
                auto canonical_it = canonical_to_id.find(canonical);
                std::string layout_id;
                if (canonical_it == canonical_to_id.end()) {
                    layout_id = fmt::format("L{:016x}", fnv1a64(canonical));
                    const auto [collision_it, inserted] = id_to_canonical.emplace(layout_id, canonical);
                    TT_FATAL(
                        inserted || collision_it->second == canonical,
                        "Fabric manifest layout hash collision for {}",
                        layout_id);
                    canonical_to_id.emplace(canonical, layout_id);
                    data.layouts[layout_id] = layout;
                    data.layouts[layout_id]["router_count"] = 0;
                } else {
                    layout_id = canonical_it->second;
                }
                data.layouts[layout_id]["router_count"] = data.layouts[layout_id]["router_count"].get<uint32_t>() + 1;
                TT_FATAL(
                    data.layout_ids.emplace(key, layout_id).second,
                    "Duplicate layout binding for chip {} channel {}",
                    *physical_chip_id,
                    instance.eth_chan);
            }
        }
    }
    return data;
}

json make_manifest_router_instance_json(
    const ManifestRouterInstance& instance, const std::optional<std::pair<FabricNodeId, chan_id_t>>& peer) {
    json out;
    out["peer"] = peer.has_value() ? manifest_endpoint_json(peer->first, peer->second) : json(nullptr);
    out["is_inter_mesh"] = instance.is_inter_mesh;
    out["is_dispatch_link"] = instance.is_dispatch_link;
    out["num_active_eriscs"] = instance.num_active_eriscs;
    out["sender_channels_per_vc"] = instance.sender_channels_per_vc;
    out["receiver_channels_per_vc"] = instance.receiver_channels_per_vc;
    out["worker_sender_channel"] = instance.worker_sender_channel;
    json producers = json::array();
    for (const auto& producer : instance.sender_producers) {
        producers.push_back(producer.has_value() ? json(*producer) : json(nullptr));
    }
    out["sender_producers"] = std::move(producers);
    out["credit_plan"] = {
        {"vc0_uses_counters", instance.credit_plan.vc0_uses_counters},
        {"vc1_uses_counters", instance.credit_plan.vc1_uses_counters},
        {"vc2_uses_counters", instance.credit_plan.vc2_uses_counters}};
    out["first_level_ack_vc0"] = instance.first_level_ack_vc0;
    out["downstream_edm_mask_vc0"] = instance.downstream_edm_mask_vc0;
    out["downstream_edm_mask_vc1"] = instance.downstream_edm_mask_vc1;
    const auto downstream_edges_json = [](const std::vector<ManifestDownstreamEdge>& edges) {
        json entries = json::array();
        for (const auto& edge : edges) {
            entries.push_back(
                {{"edge", edge.edge},
                 {"direction", direction_to_str(edge.direction)},
                 {"sender_channel", edge.sender_channel}});
        }
        return entries;
    };
    out["downstream_edges_vc0"] = downstream_edges_json(instance.downstream_edges_vc0);
    out["downstream_edges_vc1"] = downstream_edges_json(instance.downstream_edges_vc1);
    out["has_tensix_extension"] = instance.has_tensix_extension;
    out["udm_mode"] = instance.udm_mode;
    return out;
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
    using json = nlohmann::ordered_json;

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto fabric_config = control_plane.get_fabric_config();
    const FabricType fabric_type = get_fabric_type(fabric_config, cluster.is_ubb_galaxy());
    const auto& fabric_context = control_plane.get_fabric_context();
    TT_FATAL(
        fabric_context.has_builder_context(), "fabric manifest must be serialized after routers are compiled");
    const auto& builder_context = fabric_context.get_builder_context();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto layout_index = build_manifest_layout_index(control_plane, builder_context);

    json manifest;
    manifest["manifest_version"] = FABRIC_MANIFEST_VERSION;
    manifest["kind"] = "fabric_manifest";
    manifest["run"] = make_run_json(control_plane, cluster);
    manifest["hal"] = make_hal_json(hal);
    manifest["heartbeat"] = make_heartbeat_json(cluster.arch());
    manifest["fabric_context"] = make_fabric_context_json(fabric_context);
    manifest["router_template"] = make_router_template_json(builder_context);
    manifest["stream_assignment"] = make_stream_assignment_json(control_plane, builder_context);
    manifest["enums"] = make_enums_json();
    manifest["layouts"] = layout_index.layouts;

    // Chips and their routers describe what to peek; links describe what to draw. Both key on
    // (mesh, chip, channel) so a snapshot can be joined onto either.
    json meshes = json::array();
    json links = json::array();

    auto mesh_ids = mesh_graph.get_all_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end(), [](const MeshId& lhs, const MeshId& rhs) { return *lhs < *rhs; });

    for (const auto& mesh_id : mesh_ids) {
        const MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
        const bool is_2d_mesh = mesh_shape.dims() == 2;

        json mesh;
        mesh["mesh_id"] = *mesh_id;
        json shape = json::array();
        for (size_t dim = 0; dim < mesh_shape.dims(); ++dim) {
            shape.push_back(mesh_shape[dim]);
        }
        mesh["shape"] = std::move(shape);

        // has_genuine_torus_axis() is defined only for a 2D shape, and deliberately reports false for a
        // declared torus axis whose extent is too small to realize a distinct wrap edge.
        if (is_2d_mesh) {
            json torus;
            torus["y"] = has_genuine_torus_axis(fabric_type, mesh_shape, 0);
            torus["x"] = has_genuine_torus_axis(fabric_type, mesh_shape, 1);
            mesh["torus"] = std::move(torus);
        }

        json chips = json::array();
        for (const auto& [_, fabric_chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            const FabricNodeId node(mesh_id, fabric_chip_id);
            const MeshCoordinate mesh_coord = mesh_graph.chip_to_coordinate(mesh_id, fabric_chip_id);
            const auto physical_chip_id = control_plane.try_get_physical_chip_id_from_fabric_node_id(node);
            // A chip this rank cannot map to a physical device is a chip it cannot peek, so resolvability is
            // the practical definition of locality. Non-local chips still appear, so the viewer can draw the
            // whole mesh and show the host boundary.
            const bool is_local = physical_chip_id.has_value();

            json chip;
            chip["fabric_chip_id"] = fabric_chip_id;
            json coord = json::array();
            for (size_t dim = 0; dim < mesh_coord.dims(); ++dim) {
                coord.push_back(mesh_coord[dim]);
            }
            chip["mesh_coord"] = std::move(coord);
            if (is_local) {
                chip["physical_chip_id"] = *physical_chip_id;
                // Hex string: the value exceeds what JSON numbers represent exactly.
                chip["asic_id"] = fmt::format("0x{:016x}", *control_plane.get_asic_id_from_fabric_node_id(node));
            } else {
                chip["physical_chip_id"] = json(nullptr);
                chip["asic_id"] = json(nullptr);
            }
            chip["is_local"] = is_local;
            if (is_local) {
                chip["master_router_chan"] = builder_context.get_fabric_master_router_chan(*physical_chip_id);
            } else {
                chip["master_router_chan"] = json(nullptr);
            }

            json routers = json::array();
            if (is_local) {
                const auto intermesh_chan_list = control_plane.get_intermesh_facing_eth_chans(node);
                const std::set<chan_id_t> intermesh_chans(intermesh_chan_list.begin(), intermesh_chan_list.end());
                const auto& soc_desc = cluster.get_soc_desc(*physical_chip_id);

                // get_active_fabric_eth_channels() is the narrow router set (link up, assigned
                // EthRouterMode::FABRIC_ROUTER, survived routing-plane trimming) and is a std::set keyed on
                // channel, so iteration is already sorted.
                for (const auto& [chan, eth_direction] : control_plane.get_active_fabric_eth_channels(node)) {
                    const RoutingDirection direction = control_plane.eth_direction_to_routing_direction(eth_direction);
                    const char* link_class = intermesh_chans.contains(chan) ? "intermesh" : "intramesh";
                    const auto routing_plane = control_plane.get_routing_plane_id(node, chan);
                    const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);

                    json router;
                    router["eth_chan"] = chan;
                    router["direction"] = enum_name(direction);
                    router["routing_plane"] = routing_plane;
                    router["link_class"] = link_class;
                    const auto logical_core = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
                    router["logical_core"] = json::array({logical_core.x, logical_core.y});
                    const auto virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
                        *physical_chip_id, tt::tt_metal::CoreCoord(logical_core.x, logical_core.y), CoreType::ETH);
                    router["virtual_core"] = json::array({virtual_core.x, virtual_core.y});
                    const ManifestRouterBindingKey binding_key{*physical_chip_id, chan};
                    const auto layout_it = layout_index.layout_ids.find(binding_key);
                    const auto instance_it = layout_index.instances.find(binding_key);
                    TT_FATAL(
                        layout_it != layout_index.layout_ids.end() && instance_it != layout_index.instances.end(),
                        "No finalized manifest layout for active fabric router {} channel {}",
                        node,
                        chan);
                    TT_FATAL(
                        !peer.has_value() || instance_it->second->peer_node == peer->first,
                        "Manifest router instance peer disagrees with ControlPlane for {} channel {}",
                        node,
                        chan);
                    router["layout_id"] = layout_it->second;
                    router["instance"] = make_manifest_router_instance_json(*instance_it->second, peer);
                    routers.push_back(std::move(router));

                    // Wrap edges are resolved here rather than inferred by the viewer: a link wraps when its
                    // axis genuinely closes and its coordinate delta spans the mesh.
                    const bool is_east_west = direction == RoutingDirection::E || direction == RoutingDirection::W;
                    const bool is_north_south = direction == RoutingDirection::N || direction == RoutingDirection::S;
                    bool wrap = false;
                    if (is_2d_mesh && peer.has_value() && peer->first.mesh_id == mesh_id &&
                        (is_east_west || is_north_south)) {
                        const uint32_t axis = is_east_west ? 1 : 0;
                        if (has_genuine_torus_axis(fabric_type, mesh_shape, axis)) {
                            const auto peer_coord = mesh_graph.chip_to_coordinate(mesh_id, peer->first.chip_id);
                            const uint32_t here = mesh_coord[axis];
                            const uint32_t there = peer_coord[axis];
                            const uint32_t delta = here > there ? here - there : there - here;
                            wrap = delta == mesh_shape[axis] - 1;
                        }
                    }

                    json link;
                    link["src"] = manifest_endpoint_json(node, chan);
                    link["dst"] =
                        peer.has_value() ? manifest_endpoint_json(peer->first, peer->second) : json(nullptr);
                    link["direction"] = enum_name(direction);
                    link["routing_plane"] = routing_plane;
                    link["link_class"] = link_class;
                    link["wrap"] = wrap;
                    link["cross_host"] = control_plane.is_cross_host_eth_link(*physical_chip_id, chan);
                    links.push_back(std::move(link));
                }
            }
            chip["routers"] = std::move(routers);
            chips.push_back(std::move(chip));
        }
        mesh["chips"] = std::move(chips);
        meshes.push_back(std::move(mesh));
    }

    manifest["meshes"] = std::move(meshes);
    manifest["links"] = std::move(links);

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
