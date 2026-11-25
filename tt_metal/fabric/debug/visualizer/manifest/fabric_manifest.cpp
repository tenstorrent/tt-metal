// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>

#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_describe.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_topology_pass.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>
#include <unistd.h>

namespace tt::tt_fabric {

namespace {

using json = nlohmann::ordered_json;

using manifest::chip_key;
using manifest::kind_name;
using manifest::lower_enum_name;
using manifest::mesh_key;
using manifest::router_key;

// ============ Contents ============

json make_l1_json(const manifest::content::L1& l1) {
    json out;
    out["address"] = l1.address;
    out["num_elements"] = l1.num_elements;
    out["element_size"] = l1.element_size;
    out["type"] = std::string(l1.type);
    return out;
}

json make_optional_l1_json(const std::optional<manifest::content::L1>& l1) {
    return l1.has_value() ? make_l1_json(*l1) : json(nullptr);
}

json make_stream_reg_json(const manifest::content::StreamReg& stream) {
    json out;
    out["stream_id"] = stream.stream_id;
    out["address"] = stream.address;
    out["value_bits"] = stream.value_bits;
    out["type"] = std::string(stream.type);
    return out;
}

// ============ Run and fabric ============

// This process and the build it runs: ranks, host, when the manifest was written, and the commit built from.
json make_run_json(const manifest::RunInfo& run) {
    json out;
    out["run_id"] = run.run_id;
    out["mpi_rank"] = run.mpi_rank;
    out["world_size"] = run.world_size;
    if (run.subcontext.has_value()) {
        json subcontext;
        subcontext["id"] = run.subcontext->id;
        subcontext["rank"] = run.subcontext->rank;
        subcontext["size"] = run.subcontext->size;
        out["subcontext"] = std::move(subcontext);
    }
    out["host_name"] = run.host_name;
    out["written_at"] = run.written_at;
    json build;
    build["git_sha"] = std::string(run.build.git_sha);
    build["git_sha_short"] = std::string(run.build.git_sha_short);
    build["version"] = std::string(run.build.version);
    build["build_type"] = std::string(run.build.build_type);
    build["git_dirty"] = run.build.git_dirty;
    out["build"] = std::move(build);
    return out;
}

// This run's fabric: the settings it was set up with (config, SetFabricConfig's arguments), and what FabricContext
// derived from them.
json make_fabric_json(const manifest::FabricInfo& fabric) {
    json block;
    json config;
    config["fabric_config"] = lower_enum_name(fabric.config.fabric_config);
    config["reliability_mode"] = lower_enum_name(fabric.config.reliability_mode);
    config["num_routing_planes"] =
        fabric.config.num_routing_planes.has_value() ? json(*fabric.config.num_routing_planes) : json(nullptr);
    config["tensix_config"] = lower_enum_name(fabric.config.tensix_config);
    config["udm_mode"] = lower_enum_name(fabric.config.udm_mode);
    config["fabric_manager"] = lower_enum_name(fabric.config.fabric_manager);
    const auto& max_payload = fabric.config.router_config.max_packet_payload_size_bytes;
    json router_config;
    router_config["max_packet_payload_size_bytes"] = max_payload.has_value() ? json(*max_payload) : json(nullptr);
    config["router_config"] = std::move(router_config);
    block["config"] = std::move(config);
    block["topology"] = lower_enum_name(fabric.topology);
    block["is_2d_routing"] = fabric.is_2d_routing;
    block["packet_header"] = std::string(fabric.packet_header);
    block["packet_header_size_bytes"] = fabric.packet_header_size_bytes;
    block["max_payload_size_bytes"] = fabric.max_payload_size_bytes;
    block["channel_buffer_size_bytes"] = fabric.channel_buffer_size_bytes;
    if (fabric.routing_2d_route_buffer_size.has_value()) {
        block["routing_2d_route_buffer_size"] = *fabric.routing_2d_route_buffer_size;
    }
    if (fabric.routing_1d_extension_words.has_value()) {
        block["routing_1d_extension_words"] = *fabric.routing_1d_extension_words;
    }
    block["multi_txq"] = fabric.multi_txq;
    json host_clears = json::array();
    for (const auto& clear : fabric.host_clears) {
        host_clears.push_back({{"address", clear.address}, {"size_bytes", clear.size_bytes}});
    }
    block["host_clears"] = std::move(host_clears);
    const auto& intermesh_vc = fabric.intermesh_vc;
    block["intermesh_vc"] = {
        {"mode", lower_enum_name(intermesh_vc.mode)},
        {"requires_vc1", intermesh_vc.requires_vc1},
        {"requires_vc2", intermesh_vc.requires_vc2},
        {"intermesh_vc_disabled", intermesh_vc.intermesh_vc_disabled},
    };
    return block;
}

// ============ Credit transport ============

// Each fabric VC's credit transport backing on the mesh.
json make_credit_transport_json(const std::vector<manifest::VcCreditTransport>& transport) {
    json out = json::object();
    for (const auto& vc_transport : transport) {
        json entry;
        entry["backing"] = lower_enum_name(vc_transport.backing);
        out[fmt::format("vc{}", vc_transport.vc)] = std::move(entry);
    }
    return out;
}

// ============ Architecture ============

json make_heartbeat_json(const manifest::Heartbeat& heartbeat) {
    json out = make_l1_json(heartbeat.word);
    out["magic"] = heartbeat.magic;
    out["magic_mask"] = heartbeat.magic_mask;
    out["period_iters"] = heartbeat.period_iters;
    return out;
}

// An array, in the order the registers are read.
json make_core_registers_json(const manifest::CoreRegisters& registers) {
    const auto entry = [](const char* name, uint32_t address) {
        json out;
        out["name"] = name;
        out["address"] = address;
        out["type"] = std::string(manifest::get_manifest_type_name<uint32_t>());
        return out;
    };
    json soft_reset = entry("soft_reset", registers.soft_reset);
    soft_reset["reset_masks"] = registers.reset_masks;
    return json::array(
        {std::move(soft_reset),
         entry("wall_clock_low", registers.wall_clock_low),
         entry("wall_clock_high", registers.wall_clock_high)});
}

// What the architecture fixes on every router core: its L1, L1 areas at the HAL's addresses, and the register blocks.
json make_arch_json(const manifest::Arch& arch) {
    const auto& areas = arch.areas;
    json areas_json;
    areas_json["heartbeat"] = make_heartbeat_json(areas.heartbeat);
    areas_json["fabric_telemetry"] = make_l1_json(areas.fabric_telemetry);
    areas_json["routing_table"] = make_l1_json(areas.routing_table);
    areas_json["eth_fw_mailbox"] = make_optional_l1_json(areas.eth_fw_mailbox);
    areas_json["base_fw_port_status"] = make_optional_l1_json(areas.base_fw_port_status);
    areas_json["base_fw_train_status"] = make_optional_l1_json(areas.base_fw_train_status);
    areas_json["base_fw_retrain_count"] = make_optional_l1_json(areas.base_fw_retrain_count);
    areas_json["base_fw_rx_link_up"] = make_optional_l1_json(areas.base_fw_rx_link_up);
    areas_json["base_fw_postcode"] = make_optional_l1_json(areas.base_fw_postcode);
    json out;
    out["name"] = lower_enum_name(arch.arch);
    json l1;
    l1["address"] = arch.l1.address;
    l1["size_bytes"] = arch.l1.size_bytes;
    out["l1"] = std::move(l1);
    out["l1_areas"] = std::move(areas_json);
    out["core_registers"] = make_core_registers_json(arch.core_registers);
    out["fabric_telemetry_version"] = arch.fabric_telemetry_version;
    return out;
}

// ============ Paths ============

// One part of the manifest refers to a router elsewhere by its path, e.g. "M0/C7/E0".
std::string router_path(FabricNodeId node, eth_chan_directions direction, routing_plane_id_t routing_plane) {
    return fmt::format(
        "{}/{}/{}", mesh_key(node.mesh_id), chip_key(node.chip_id), router_key(direction, routing_plane));
}

std::string router_path(const manifest::RouterRef& router) {
    return router_path(router.node, router.direction, router.routing_plane);
}

// A sibling is on the same chip and routing plane as the router that refers to it.
std::string sibling_router_path(
    FabricNodeId node, const manifest::RouterIdentity& identity, const manifest::SiblingRouterRef& sibling) {
    return router_path(node, sibling.direction, identity.routing_plane);
}

// ============ Router ============

json make_core_json(const tt::tt_metal::CoreCoord& core) { return json::array({core.x, core.y}); }

// The router's mesh and chip are its path, and its direction and routing plane its key, so they are not repeated
// here.
json make_router_identity_json(const manifest::RouterIdentity& identity) {
    json out;
    out["eth_chan"] = identity.eth_chan;
    out["logical_core"] = make_core_json(identity.logical_core);
    out["virtual_core"] = make_core_json(identity.virtual_core);
    return out;
}

json make_eth_link_json(const manifest::EthLink& link) {
    json out;
    out["edge_capability"] = lower_enum_name(link.edge_capability);
    out["peer"] = link.peer.has_value() ? json(router_path(*link.peer)) : json(nullptr);
    out["cross_host"] = link.cross_host;
    out["wrap"] = link.wrap;
    out["dispatch_link"] = link.is_dispatch_link;
    return out;
}

json make_router_shape_json(const manifest::RouterShape& shape) {
    json out;
    out["num_vcs"] = shape.num_vcs;
    out["senders_per_vc"] = shape.senders_per_vc;
    out["receivers_per_vc"] = shape.receivers_per_vc;
    out["num_active_eriscs"] = shape.num_active_eriscs;
    out["channel_trimming_overrides_applied"] = shape.channel_trimming_overrides_applied;
    out["has_tensix_extension"] = shape.has_tensix_extension;
    return out;
}

// indexed_by says whose sender channel each element belongs to: the to_sender arrays are indexed by this router's
// sender compact index, and the receiver arrays by the peer router's sender compact index.
json make_credit_counters_json(const manifest::L1CreditCounters& counters) {
    using manifest::CreditCounterArray;
    using manifest::CreditCounterIndexedBy;
    json out;
    const auto add = [&out](
                         CreditCounterArray array, const manifest::content::L1& l1, CreditCounterIndexedBy indexed_by) {
        json entry = make_l1_json(l1);
        entry["indexed_by"] = lower_enum_name(indexed_by);
        out[lower_enum_name(array)] = std::move(entry);
    };
    add(CreditCounterArray::TO_SENDER_ACK, counters.to_sender_ack, CreditCounterIndexedBy::OWN_SENDER_CHANNEL);
    add(CreditCounterArray::TO_SENDER_COMPLETION,
        counters.to_sender_completion,
        CreditCounterIndexedBy::OWN_SENDER_CHANNEL);
    add(CreditCounterArray::RECEIVER_ACK, counters.receiver_ack, CreditCounterIndexedBy::PEER_SENDER_CHANNEL);
    add(CreditCounterArray::RECEIVER_COMPLETION,
        counters.receiver_completion,
        CreditCounterIndexedBy::PEER_SENDER_CHANNEL);
    return out;
}

// ============ Channels ============

// A counter element names its array by its path within the router, e.g. "credit_counters/to_sender_ack".
json make_credit_ref_json(const manifest::CreditRef& credit) {
    if (const auto* stream = std::get_if<std::optional<manifest::content::StreamReg>>(&credit)) {
        return stream->has_value() ? make_stream_reg_json(**stream) : json(nullptr);
    }
    const auto& counter = std::get<manifest::CounterRef>(credit);
    json out;
    out["array"] = fmt::format("credit_counters/{}", lower_enum_name(counter.array));
    out["index"] = counter.index;
    return out;
}

json make_fields_json(const std::vector<manifest::Field>& fields) {
    json out = json::object();
    for (const auto& field : fields) {
        const std::string key(field.key);
        TT_FATAL(!out.contains(key), "Fabric manifest: two fields are keyed {}", key);
        if (!field.content.has_value()) {
            out[key] = nullptr;
            continue;
        }
        json entry;
        entry["category"] = lower_enum_name(field.category);
        entry["kind"] = kind_name(*field.content);
        std::visit(
            ttsl::overloaded{
                [&](const manifest::content::L1& l1) { entry.update(make_l1_json(l1)); },
                [&](const manifest::content::StreamReg& stream) { entry.update(make_stream_reg_json(stream)); },
                [&](const manifest::content::Number& number) { entry["value"] = number.value; },
                [&](const manifest::content::Flag& flag) { entry["value"] = flag.value; },
                [&](const manifest::content::Enum& enumerator) {
                    entry["value"] = enumerator.value;
                    entry["value_name"] = manifest::lower_name(enumerator.value_name);
                },
            },
            *field.content);
        out[key] = std::move(entry);
    }
    return out;
}

// One entry per ERISC, keyed erisc<N>, each written by `entry_json`.
template <typename Entry, typename EntryJson>
json make_by_erisc_json(const std::vector<Entry>& entries_by_erisc, EntryJson entry_json) {
    json out = json::object();
    for (size_t risc_id = 0; risc_id < entries_by_erisc.size(); ++risc_id) {
        out[fmt::format("erisc{}", risc_id)] = entry_json(entries_by_erisc[risc_id]);
    }
    return out;
}

// Each ERISC's fields, keyed erisc<N>.
json make_fields_by_erisc_json(const manifest::FieldsByErisc& fields_by_erisc) {
    return make_by_erisc_json(
        fields_by_erisc, [](const std::vector<manifest::Field>& fields) { return make_fields_json(fields); });
}

json make_serviced_by_json(const std::vector<uint32_t>& risc_ids) {
    json out = json::array();
    for (const auto risc_id : risc_ids) {
        out.push_back(fmt::format("erisc{}", risc_id));
    }
    return out;
}

// A worker producer is "worker", the tensix mux is "tensix_mux", and a sibling router producer is that router's path.
json make_sender_producer_json(
    const std::optional<manifest::SenderChannelProducer>& producer,
    FabricNodeId node,
    const manifest::RouterIdentity& identity) {
    if (!producer.has_value()) {
        return nullptr;
    }
    if (std::holds_alternative<manifest::LocalWorker>(*producer)) {
        return "worker";
    }
    if (std::holds_alternative<manifest::LocalTensixMux>(*producer)) {
        return "tensix_mux";
    }
    return sibling_router_path(node, identity, std::get<manifest::SiblingRouterRef>(*producer));
}

json make_sender_channel_json(
    const manifest::SenderChannel& sender, FabricNodeId node, const manifest::RouterIdentity& identity) {
    json out;
    out["status"] = lower_enum_name(sender.status);
    out["serviced_by"] = make_serviced_by_json(sender.serviced_by);
    out["producer"] = make_sender_producer_json(sender.producer, node, identity);
    out["ring_buffer"] = make_optional_l1_json(sender.ring_buffer);
    out["credits"] = make_by_erisc_json(sender.credits_by_erisc, [](const manifest::SenderChannelCredits& credits) {
        json entry;
        if (credits.acked.has_value()) {
            entry["acked"] = make_credit_ref_json(*credits.acked);
        }
        entry["completed"] = make_credit_ref_json(credits.completed);
        return entry;
    });
    out["fields"] = make_fields_by_erisc_json(sender.fields_by_erisc);
    return out;
}

json make_receiver_channel_json(const manifest::ReceiverChannel& receiver) {
    json out;
    out["status"] = lower_enum_name(receiver.status);
    out["serviced_by"] = make_serviced_by_json(receiver.serviced_by);
    out["forwards_on"] =
        make_by_erisc_json(receiver.forwards_on_by_erisc, [](const std::optional<uint32_t>& forwards_on) {
            return forwards_on.has_value() ? json(fmt::format("vc{}", *forwards_on)) : json(nullptr);
        });
    out["ring_buffer"] = make_optional_l1_json(receiver.ring_buffer);
    out["fields"] = make_fields_by_erisc_json(receiver.fields_by_erisc);
    return out;
}

// Keyed vc<N> then ch<M>. VCs the router has no channels of this kind on are left out.
template <typename Channel, typename ChannelJson>
json make_channels_by_vc_json(const std::vector<std::vector<Channel>>& channels_by_vc, ChannelJson channel_json) {
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
// it lands on. In mux mode, the mux stands in for the sibling's VC0 channels other than the worker channel.
json make_downstream_edges_json(const manifest::Router& router, FabricNodeId node) {
    json out = json::object();
    for (size_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
        const auto& edges = router.intra_chip_downstream_edges[vc];
        if (edges.empty()) {
            continue;
        }
        json vc_json;
        for (const auto& edge : edges) {
            json entry;
            entry["downstream_channel"] = fmt::format(
                "{}/channels/senders/vc{}/ch{}",
                sibling_router_path(node, router.identity, edge.target),
                vc,
                edge.landing_channel);
            entry["through_tensix_mux"] = edge.through_tensix_mux;
            entry["fields"] = make_fields_by_erisc_json(edge.fields_by_erisc);
            vc_json[fmt::format("edge{}", edge.edge)] = std::move(entry);
        }
        out[fmt::format("vc{}", vc)] = std::move(vc_json);
    }
    return out;
}

// ============ Fields ============

// Keyed erisc<N>.
json make_eriscs_json(const std::vector<manifest::Erisc>& eriscs) {
    json out = json::object();
    for (size_t risc_id = 0; risc_id < eriscs.size(); ++risc_id) {
        json& erisc = out[fmt::format("erisc{}", risc_id)];
        erisc["processor"] = lower_enum_name(eriscs[risc_id].processor);
        erisc["physical_erisc"] = eriscs[risc_id].physical_erisc;
    }
    return out;
}

// Every name of a label type: an enum's enumerators, or the field kinds of Content's alternatives.
template <typename E>
    requires std::is_enum_v<E>
json make_label_names_json(std::type_identity<E>) {
    json out = json::array();
    for (const auto value : enchantum::values<E>) {
        out.push_back(lower_enum_name(value));
    }
    return out;
}

template <typename... Contents>
json make_label_names_json(std::type_identity<std::variant<Contents...>>) {
    return json::array({kind_name<Contents>()...});
}

// Every name each label can hold (manifest::k_vocabulary_lists), so readers know every value a label key can take
// without a copy of the C++ enums.
json make_vocabulary_json() {
    json out;
    std::apply(
        [&out](const auto&... lists) {
            ((out[std::string(lists.name)] =
                  make_label_names_json(std::type_identity<typename std::decay_t<decltype(lists)>::type>{})),
             ...);
        },
        manifest::k_vocabulary_lists);
    return out;
}

json make_router_json(const manifest::Router& router, FabricNodeId node) {
    json out;
    out["identity"] = make_router_identity_json(router.identity);
    out["link"] = make_eth_link_json(router.link);
    out["shape"] = make_router_shape_json(router.shape);
    out["credit_counters"] = make_credit_counters_json(router.credit_counters);
    out["channels"]["senders"] = make_channels_by_vc_json(
        router.channels.senders,
        [&](const manifest::SenderChannel& sender) { return make_sender_channel_json(sender, node, router.identity); });
    out["channels"]["receivers"] = make_channels_by_vc_json(
        router.channels.receivers,
        [&](const manifest::ReceiverChannel& receiver) { return make_receiver_channel_json(receiver); });
    out["intra_chip_downstream_edges"] = make_downstream_edges_json(router, node);
    out["fields"] = make_fields_by_erisc_json(router.fields_by_erisc);
    out["eriscs"] = make_eriscs_json(router.eriscs);
    out["leftover_l1"] = make_optional_l1_json(router.leftover_l1);
    return out;
}

// ============ Chip and mesh ============

json make_local_sync_json(const manifest::LocalSync& local_sync) {
    TT_FATAL(local_sync.master.has_value(), "Fabric manifest: the topology pass did not resolve the local sync master");
    json out;
    out["master"] = router_path(*local_sync.master);
    out["num_routers"] = local_sync.num_routers;
    out["router_channels_mask"] = local_sync.router_channels_mask;
    return out;
}

// The chip's routers keyed <direction><routing_plane>.
json make_chip_routers_json(const manifest::Chip& chip) {
    json routers = json::object();
    for (const auto& router : chip.routers) {
        routers[router_key(router.identity.direction, router.identity.routing_plane)] =
            make_router_json(router, chip.fabric_node_id);
    }
    return routers;
}

// A remote chip has only its coordinates, and nulls for the ids this rank cannot read.
json make_chip_json(const manifest::Chip& chip) {
    json out;
    out["mesh_coord"] = chip.mesh_coord;
    out["host_rank"] = chip.host_rank.has_value() ? json(*chip.host_rank) : json(nullptr);
    if (!chip.physical_chip_id.has_value()) {
        out["physical_chip_id"] = json(nullptr);
        out["asic_id"] = json(nullptr);
        out["is_local"] = false;
        return out;
    }
    TT_FATAL(chip.asic_id.has_value(), "Fabric manifest: local chip {} has no asic_id", chip.fabric_node_id);
    out["physical_chip_id"] = *chip.physical_chip_id;
    // Hex string: the value exceeds what JSON numbers represent exactly.
    out["asic_id"] = fmt::format("0x{:016x}", *chip.asic_id);
    out["is_local"] = true;
    out["z_port_role"] = lower_enum_name(chip.z_port_role);
    out["local_sync"] = chip.local_sync ? make_local_sync_json(*chip.local_sync) : json(nullptr);
    out["routers"] = make_chip_routers_json(chip);
    return out;
}

json make_mesh_json(const manifest::Mesh& mesh) {
    json out;
    out["shape"] = mesh.shape;
    if (mesh.torus.has_value()) {
        json torus;
        torus["y"] = mesh.torus->y;
        torus["x"] = mesh.torus->x;
        out["torus"] = std::move(torus);
    }
    out["express_routing"] = mesh.express_routing;
    if (mesh.credit_transport.has_value()) {
        out["credit_transport"] = make_credit_transport_json(*mesh.credit_transport);
    }
    json chips = json::object();
    for (const auto& chip : mesh.chips) {
        chips[chip_key(chip.fabric_node_id.chip_id)] = make_chip_json(chip);
    }
    out["chips"] = std::move(chips);
    return out;
}

// A new fabric run id: the UTC time to the second, then 8 random hex digits, such as "20261105T221500Z-1a2b3c4d". It
// sorts by time, and the digits tell apart runs started in the same second.
std::string make_fabric_run_id() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::array<char, 32> time_text{};
    std::strftime(time_text.data(), time_text.size(), "%Y%m%dT%H%M%SZ", &utc);
    std::random_device random_device;
    return fmt::format("{}-{:08x}", time_text.data(), std::uniform_int_distribution<uint32_t>()(random_device));
}

// The run id every rank of this fabric writes: rank 0 makes it and broadcasts it, so a run's files on every host share
// one id. The current world is this fabric's ranks: a job tt-run splits into sub-contexts has a fabric, and so a run,
// per sub-context.
std::string share_fabric_run_id() {
    constexpr size_t k_run_id_length = std::string_view("20261105T221500Z-1a2b3c4d").size();
    const auto& world = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    std::array<char, k_run_id_length> run_id{};
    if (*world->rank() == 0) {
        const std::string made = make_fabric_run_id();
        TT_FATAL(made.size() == run_id.size(), "Fabric manifest: run id {} isn't {} characters", made, run_id.size());
        std::ranges::copy(made, run_id.begin());
    }
    world->broadcast(
        ttsl::as_writable_bytes(ttsl::Span<char>{run_id.data(), run_id.size()}),
        tt_metal::distributed::multihost::Rank{0});
    return std::string(run_id.begin(), run_id.end());
}

// Writes the manifest to a temporary name and renames it into place, so a reader never sees a partial manifest.
void serialize_fabric_manifest_to_file(
    const ControlPlane& control_plane, const std::string& run_id, const std::filesystem::path& output_file_path) {
    const auto& fabric_context = control_plane.get_fabric_context();
    TT_FATAL(
        fabric_context.has_builder_context(), "Fabric manifest: must be written after the fabric routers are compiled");
    const manifest::ManifestFacts facts{
        .run = manifest::describe_run(control_plane, run_id),
        .fabric = manifest::describe_fabric(control_plane),
        .arch = manifest::describe_arch(fabric_context.get_hal()),
        .meshes = manifest::describe_meshes(
            control_plane, fabric_context.get_cluster(), fabric_context.get_builder_context()),
    };
    const json manifest_json = make_manifest_json(facts);

    std::filesystem::create_directories(output_file_path.parent_path());
    const std::filesystem::path temporary_path =
        output_file_path.string() + ".tmp." + std::to_string(static_cast<uint64_t>(::getpid()));
    try {
        std::ofstream out_file;
        out_file.exceptions(std::ios::badbit | std::ios::failbit);
        out_file.open(temporary_path);
        out_file << manifest_json.dump(2) << '\n';
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

json make_manifest_json(const manifest::ManifestFacts& facts) {
    json meshes = json::object();
    for (const auto& mesh : facts.meshes) {
        meshes[mesh_key(mesh.mesh_id)] = make_mesh_json(mesh);
    }

    json out;
    out["manifest_version"] = visualizer::k_manifest_version;
    out["file_kind"] = std::string(visualizer::k_manifest_file_kind);
    out["run"] = make_run_json(facts.run);
    out["fabric"] = make_fabric_json(facts.fabric);
    out["arch"] = make_arch_json(facts.arch);
    out["vocabulary"] = make_vocabulary_json();
    out["meshes"] = std::move(meshes);
    return out;
}

void write_fabric_manifest(const ControlPlane& control_plane, const tt::llrt::RunTimeOptions& rtoptions) {
    // Create and share a run id for this fabric run to all ranks in the world
    const std::string run_id = share_fabric_run_id();
    // Manifest file is named by the rank in the whole job so that every rank has a distinct file
    const auto world = tt_metal::distributed::multihost::DistributedContext::get_world_context();
    const std::filesystem::path manifest_path =
        visualizer::make_run_directory(rtoptions.get_logs_dir(), run_id) /
        visualizer::make_manifest_file_name(
            static_cast<uint32_t>(*world->rank()), static_cast<uint32_t>(*world->size()));
    try {
        serialize_fabric_manifest_to_file(control_plane, run_id, manifest_path);
    } catch (const std::exception& e) {
        TT_THROW(
            "Failed to write fabric manifest {} (unset TT_METAL_FABRIC_GENERATE_MANIFEST to skip): {}",
            manifest_path.string(),
            e.what());
    }
    log_info(tt::LogFabric, "Wrote fabric manifest {}", manifest_path.string());
}

}  // namespace tt::tt_fabric
