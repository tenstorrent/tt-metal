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

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_describe.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_chip_pass.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
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

// This process and the build it runs: ranks, when the manifest was written, and the commit built from.
json make_run_json(const manifest::RunInfo& run) {
    json out;
    out["host_rank"] = run.host_rank;
    out["mpi_rank"] = run.mpi_rank;
    out["world_size"] = run.world_size;
    out["written_at"] = run.written_at;
    json build;
    build["git_sha"] = std::string(run.build.git_sha);
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
    return block;
}

// ============ Credit transport ============

// Each fabric VC's credit transport backing on the mesh.
json make_credit_transport_json(const std::vector<manifest::VcCreditTransport>& transport) {
    json out = json::object();
    for (const auto& vc_transport : transport) {
        json reasons = json::array();
        for (const auto reason : vc_transport.l1_counter_reasons) {
            reasons.push_back(lower_enum_name(reason));
        }
        json entry;
        entry["backing"] = lower_enum_name(vc_transport.backing);
        entry["reasons"] = std::move(reasons);
        out[fmt::format("vc{}", vc_transport.vc)] = std::move(entry);
    }
    return out;
}

// ============ Contents ============

// A type's size and schema. An array also gets its element count and the size of one element.
void add_type_json(json& out, const layout::Type& type) {
    out["size"] = type.size;
    if (type.count > 0) {
        out["num_elements"] = type.count;
        out["size_per_element"] = manifest::element_size(type);
    }
    out["schema"] = manifest::schema_name(type);
}

json l1_json(const manifest::content::L1& l1) {
    json out;
    out["address"] = l1.address;
    add_type_json(out, l1.type);
    out["cleared_by_host"] = l1.cleared_by_host;
    return out;
}

json optional_l1_json(const std::optional<manifest::content::L1>& l1) {
    return l1.has_value() ? l1_json(*l1) : json(nullptr);
}

json stream_json(const manifest::content::Stream& stream) {
    TT_FATAL(
        stream.stream_id < StreamRegAssignments::num_eth_stream_registers,
        "Fabric manifest: stream id {} is not one of the {} stream registers",
        stream.stream_id,
        StreamRegAssignments::num_eth_stream_registers);
    json out;
    out["stream_id"] = stream.stream_id;
    out["register"] = lower_enum_name(stream.reg);
    out["schema"] = manifest::schema_name(stream.type);
    return out;
}

// ============ Architecture ============

json heartbeat_json(const manifest::Heartbeat& heartbeat) {
    json out = l1_json(heartbeat.word);
    out["magic"] = heartbeat.magic;
    out["magic_mask"] = heartbeat.magic_mask;
    out["period_iters"] = heartbeat.period_iters;
    return out;
}

json stream_registers_json(const manifest::StreamRegisters& registers) {
    json by_name = json::object();
    for (const auto& [reg, layout] : registers.registers) {
        json entry;
        entry["offset"] = layout.offset;
        entry["bits"] = layout.bits;
        by_name[lower_enum_name(reg)] = std::move(entry);
    }
    json out;
    out["base"] = registers.base;
    out["stride"] = registers.stride;
    out["count"] = registers.count;
    out["registers"] = std::move(by_name);
    return out;
}

// An array, in the order the registers are read.
json core_registers_json(const manifest::CoreRegisters& registers) {
    const auto entry = [](const char* name, uint32_t address) {
        json out;
        out["name"] = name;
        out["address"] = address;
        add_type_json(out, layout::type_of<uint32_t>());
        return out;
    };
    json soft_reset = entry("soft_reset", registers.soft_reset);
    soft_reset["reset_masks"] = registers.reset_masks;
    return json::array(
        {std::move(soft_reset),
         entry("wall_clock_low", registers.wall_clock_low),
         entry("wall_clock_high", registers.wall_clock_high)});
}

// Each struct by name, with its members in order.
json structs_json(const std::vector<manifest::StructType>& types) {
    json out = json::object();
    for (const auto& type : types) {
        const std::string name(type.name);
        TT_FATAL(!out.contains(name), "Fabric manifest: two structs are named {}, so its schema is ambiguous", name);
        json members = json::array();
        for (const auto& member : type.members) {
            json entry;
            entry["name"] = std::string(member.name);
            entry["offset"] = member.offset;
            add_type_json(entry, member.type);
            members.push_back(std::move(entry));
        }
        out[name]["size"] = type.size;
        out[name]["members"] = std::move(members);
    }
    return out;
}

// What the architecture fixes on every router core: L1 areas at the HAL's addresses, and the register blocks.
json make_arch_json(const manifest::Arch& arch) {
    const auto& areas = arch.areas;
    json areas_json;
    areas_json["heartbeat"] = heartbeat_json(areas.heartbeat);
    areas_json["fabric_telemetry"] = l1_json(areas.fabric_telemetry);
    areas_json["routing_table"] = l1_json(areas.routing_table);
    areas_json["eth_fw_mailbox"] = optional_l1_json(areas.eth_fw_mailbox);
    areas_json["base_fw_port_status"] = optional_l1_json(areas.base_fw_port_status);
    areas_json["base_fw_train_status"] = optional_l1_json(areas.base_fw_train_status);
    areas_json["base_fw_retrain_count"] = optional_l1_json(areas.base_fw_retrain_count);
    areas_json["base_fw_rx_link_up"] = optional_l1_json(areas.base_fw_rx_link_up);
    areas_json["base_fw_postcode"] = optional_l1_json(areas.base_fw_postcode);
    json out;
    out["name"] = lower_enum_name(arch.arch);
    out["l1_areas"] = std::move(areas_json);
    out["stream_registers"] = stream_registers_json(arch.stream_registers);
    out["core_registers"] = core_registers_json(arch.core_registers);
    return out;
}

// ============ Enums and schemas ============

// Each enum by name, with its values by their lower_name.
json make_enums_json(const std::vector<manifest::EnumType>& types) {
    json out = json::object();
    for (const auto& type : types) {
        const std::string name(type.name);
        TT_FATAL(!out.contains(name), "Fabric manifest: two enums are named {}, so its schema is ambiguous", name);
        json values = json::object();
        for (const auto& enumerator : type.enumerators) {
            values[manifest::lower_name(enumerator.name)] = enumerator.value;
        }
        out[name]["values"] = std::move(values);
    }
    return out;
}

// The type catalog: every struct and enum a schema names (struct:<Name>, enum:<Name>).
json make_types_json(const manifest::TypeCatalog& types) {
    json out;
    out["structs"] = structs_json(types.structs);
    out["enums"] = make_enums_json(types.enums);
    return out;
}

// Every struct and enum schema in `node` names a type in `structs` or `enums` (types.structs, types.enums), so decode
// can read everything the manifest hands it.
void check_schemas_resolve(const json& node, const json& structs, const json& enums) {
    if (node.is_array()) {
        for (const auto& element : node) {
            check_schemas_resolve(element, structs, enums);
        }
        return;
    }
    if (!node.is_object()) {
        return;
    }
    for (const auto& [key, value] : node.items()) {
        if (key != "schema" || !value.is_string()) {
            check_schemas_resolve(value, structs, enums);
            continue;
        }
        const auto& schema = value.get_ref<const std::string&>();
        const auto names_missing = [&](std::string_view prefix, const json& described) {
            return schema.starts_with(prefix) && !described.contains(schema.substr(prefix.size()));
        };
        const bool unresolved =
            names_missing(manifest::k_struct_schema, structs) || names_missing(manifest::k_enum_schema, enums);
        TT_FATAL(
            !unresolved,
            "Fabric manifest: the schema {} names a type the manifest does not describe; add it to "
            "layout::DescribedStructs (fabric_struct_layouts.hpp) or manifest::DescribedEnums "
            "(fabric_manifest_type_catalog.hpp)",
            schema);
    }
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

json core_json(const tt::tt_metal::CoreCoord& core) { return json::array({core.x, core.y}); }

// The router's mesh and chip are its path, and its direction and routing plane its key, so they are not repeated
// here.
json router_identity_json(const manifest::RouterIdentity& identity) {
    json out;
    out["eth_chan"] = identity.eth_chan;
    out["logical_core"] = core_json(identity.logical_core);
    out["virtual_core"] = core_json(identity.virtual_core);
    return out;
}

json eth_link_json(const manifest::EthLink& link) {
    json out;
    out["edge_capability"] = lower_enum_name(link.edge_capability);
    out["peer"] = link.peer.has_value() ? json(router_path(*link.peer)) : json(nullptr);
    out["cross_host"] = link.cross_host;
    out["wrap"] = link.wrap;
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

// indexed_by says whose sender channel each element belongs to: the to_sender arrays are indexed by this router's
// sender compact index, and the receiver arrays by the peer router's sender compact index.
json credit_counters_json(const manifest::L1CreditCounters& counters) {
    using manifest::CreditCounterArray;
    using manifest::CreditCounterIndexedBy;
    json out;
    const auto add = [&out](
                         CreditCounterArray array, const manifest::content::L1& l1, CreditCounterIndexedBy indexed_by) {
        json entry = l1_json(l1);
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
json credit_ref_json(const manifest::CreditRef& credit) {
    if (const auto* stream = std::get_if<std::optional<manifest::content::Stream>>(&credit)) {
        return stream->has_value() ? stream_json(**stream) : json(nullptr);
    }
    const auto& counter = std::get<manifest::CounterRef>(credit);
    json out;
    out["array"] = fmt::format("credit_counters/{}", lower_enum_name(counter.array));
    out["index"] = counter.index;
    return out;
}

json fields_json(const std::vector<manifest::Field>& fields) {
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
                [&](const manifest::content::L1& l1) { entry.update(l1_json(l1)); },
                [&](const manifest::content::Stream& stream) { entry.update(stream_json(stream)); },
                [&](const manifest::content::Number& number) { entry["value"] = number.value; },
                [&](const manifest::content::Flag& flag) { entry["value"] = flag.value; },
                [&](const manifest::content::Enum& enumerator) {
                    entry["value"] = enumerator.value;
                    entry["schema"] = manifest::schema_name(enumerator.type);
                },
            },
            *field.content);
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

// A worker producer is "worker", the tensix mux is "tensix_mux", and a sibling router producer is that router's path.
json sender_producer_json(
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

json sender_channel_json(
    const manifest::SenderChannel& sender, FabricNodeId node, const manifest::RouterIdentity& identity) {
    json credits;
    if (sender.credits.acked.has_value()) {
        credits["acked"] = credit_ref_json(*sender.credits.acked);
    }
    credits["completed"] = credit_ref_json(sender.credits.completed);

    json out;
    out["status"] = lower_enum_name(sender.status);
    out["serviced_by"] = serviced_by_json(sender.serviced_by);
    out["producer"] = sender_producer_json(sender.producer, node, identity);
    out["ring_buffer"] = optional_l1_json(sender.ring_buffer);
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
    out["ring_buffer"] = optional_l1_json(receiver.ring_buffer);
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
// it lands on. In mux mode, the mux stands in for the sibling's VC0 channels other than the worker channel.
json downstream_edges_json(const manifest::Router& router, FabricNodeId node) {
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
            entry["fields"] = fields_json(edge.fields);
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
        erisc["physical_erisc"] = eriscs[risc_id].physical_erisc;
        erisc["fields"] = fields_json(eriscs[risc_id].fields);
    }
    return out;
}

// Every name of a label type: an enum's enumerators, or the field kinds of Content's alternatives.
template <typename E>
    requires std::is_enum_v<E>
json label_names_json(std::type_identity<E>) {
    json out = json::array();
    for (const auto value : enchantum::values<E>) {
        out.push_back(lower_enum_name(value));
    }
    return out;
}

template <typename... Contents>
json label_names_json(std::type_identity<std::variant<Contents...>>) {
    return json::array({kind_name<Contents>()...});
}

// Every name each label can hold (manifest::k_vocabulary_lists), so readers know every value a label key can take
// without a copy of the C++ enums.
json make_vocabulary_json() {
    json out;
    std::apply(
        [&out](const auto&... lists) {
            ((out[std::string(lists.name)] =
                  label_names_json(std::type_identity<typename std::decay_t<decltype(lists)>::type>{})),
             ...);
        },
        manifest::k_vocabulary_lists);
    return out;
}

json make_router_json(const manifest::Router& router, FabricNodeId node) {
    json out;
    out["identity"] = router_identity_json(router.identity);
    out["link"] = eth_link_json(router.link);
    out["shape"] = router_shape_json(router.shape);
    out["credit_counters"] = credit_counters_json(router.credit_counters);
    out["channels"]["senders"] = channels_by_vc_json(
        router.channels.senders,
        [&](const manifest::SenderChannel& sender) { return sender_channel_json(sender, node, router.identity); });
    out["channels"]["receivers"] = channels_by_vc_json(router.channels.receivers, receiver_channel_json);
    out["intra_chip_downstream_edges"] = downstream_edges_json(router, node);
    out["fields"] = fields_json(router.fields);
    out["eriscs"] = eriscs_json(router.eriscs);
    out["leftover_l1"] = optional_l1_json(router.leftover_l1);
    return out;
}

// ============ Chip and mesh ============

json local_sync_json(const manifest::LocalSync& local_sync) {
    TT_FATAL(local_sync.master.has_value(), "Fabric manifest: the chip pass did not resolve the local sync master");
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
    out["local_sync"] = chip.local_sync ? local_sync_json(*chip.local_sync) : json(nullptr);
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

json make_manifest_json(const manifest::Manifest& manifest) {
    json out;
    out["manifest_version"] = FABRIC_MANIFEST_VERSION;
    out["file_kind"] = "fabric_manifest";
    out["run"] = make_run_json(manifest.run);
    out["fabric"] = make_fabric_json(manifest.fabric);
    out["arch"] = make_arch_json(manifest.arch);
    out["types"] = make_types_json(manifest.types);
    out["vocabulary"] = make_vocabulary_json();
    json meshes = json::object();
    for (const auto& mesh : manifest.meshes) {
        meshes[mesh_key(mesh.mesh_id)] = make_mesh_json(mesh);
    }
    out["meshes"] = std::move(meshes);
    return out;
}

// Writes the manifest to a temporary name and renames it into place, so a reader never sees a partial manifest.
void serialize_fabric_manifest_to_file(
    const ControlPlane& control_plane, const std::filesystem::path& output_file_path) {
    const auto& fabric_context = control_plane.get_fabric_context();
    TT_FATAL(
        fabric_context.has_builder_context(), "Fabric manifest: must be written after the fabric routers are compiled");
    const auto& metal_context = tt::tt_metal::MetalContext::instance();
    const manifest::Manifest manifest{
        .run = manifest::describe_run(control_plane),
        .fabric = manifest::describe_fabric(control_plane),
        .arch = manifest::describe_arch(metal_context.hal()),
        .types = manifest::describe_types(),
        .meshes =
            manifest::describe_meshes(control_plane, metal_context.get_cluster(), fabric_context.get_builder_context()),
    };
    const json manifest_json = make_manifest_json(manifest);
    check_schemas_resolve(
        manifest_json, manifest_json.at("types").at("structs"), manifest_json.at("types").at("enums"));

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
