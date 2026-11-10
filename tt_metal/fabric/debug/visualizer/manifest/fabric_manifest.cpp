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
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_json_reader.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_describe.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_topology_pass.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"
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
#include <unordered_map>
#include <variant>
#include <vector>
#include <unistd.h>

namespace tt::tt_fabric {

namespace {

using json = nlohmann::ordered_json;

using manifest::chip_key;
using manifest::downstream_channel_path;
using manifest::kind_name;
using manifest::lower_enum_name;
using manifest::make_asic_id_text;
using manifest::mesh_key;
using manifest::parse_asic_id_text;
using manifest::router_key;
using manifest::router_path;
using manifest::sibling_router_path;

// ============ Contents ============

json make_l1_json(const manifest::content::L1& l1) {
    json out;
    out["address"] = l1.address;
    out["num_elements"] = l1.num_elements;
    out["element_size"] = l1.element_size;
    out["type"] = l1.type;
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
    out["type"] = stream.type;
    return out;
}

// ============ Run and fabric ============

// The commit tt-metal was built from. A reader checks it against its own build, since the types the manifest names
// are the writing build's.
json make_build_json(const visualizer::BuildIdentity& build) {
    json out;
    out["git_sha"] = std::string(build.git_sha);
    out["git_sha_short"] = std::string(build.git_sha_short);
    out["version"] = std::string(build.version);
    out["build_type"] = std::string(build.build_type);
    out["git_dirty"] = build.git_dirty;
    return out;
}

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
    out["build"] = make_build_json(run.build);
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
    block["packet_header"] = fabric.packet_header;
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

// An array, in the order the registers are read (k_core_registers).
json make_core_registers_json(const manifest::CoreRegisters& registers) {
    json out = json::array();
    for (const auto& [name, address] : manifest::k_core_registers) {
        json entry;
        entry["name"] = std::string(name);
        entry["address"] = registers.*address;
        entry["type"] = std::string(manifest::get_manifest_type_name<uint32_t>());
        if (address == &manifest::CoreRegisters::soft_reset) {
            entry["reset_masks"] = registers.reset_masks;
        }
        out.push_back(std::move(entry));
    }
    return out;
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

// Each array keyed by its name, in k_credit_counter_arrays' order, with whose sender channel indexes it.
json make_credit_counters_json(const manifest::L1CreditCounters& counters) {
    json out;
    for (const auto& layout : manifest::k_credit_counter_arrays) {
        json entry = make_l1_json(counters.*layout.member);
        entry["indexed_by"] = lower_enum_name(layout.indexed_by);
        out[lower_enum_name(layout.array)] = std::move(entry);
    }
    return out;
}

// ============ Channels ============

// A counter element names its array by its path within the router, e.g. "credit_counters/to_sender_ack".
constexpr std::string_view k_credit_counters_path_prefix = "credit_counters/";

json make_credit_ref_json(const manifest::CounterRef& counter) {
    json out;
    out["array"] = fmt::format("{}{}", k_credit_counters_path_prefix, lower_enum_name(counter.array));
    out["index"] = counter.index;
    return out;
}

json make_fields_json(const std::vector<manifest::Field>& fields) {
    json out = json::object();
    for (const auto& field : fields) {
        const std::string& key = field.key;
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
                    entry["value_name"] = enumerator.value_name;
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

// Writes each fields block as `layout` says. Under SHARED it keeps each distinct block once, for field_sets.
class FieldsWriter {
public:
    explicit FieldsWriter(FieldsLayout layout) : layout_(layout) {}

    // The block for `fields_by_erisc`, or under SHARED the key of its entry in field_sets.
    json write(const manifest::FieldsByErisc& fields_by_erisc) {
        json block = make_fields_by_erisc_json(fields_by_erisc);
        if (layout_ == FieldsLayout::PER_ROUTER) {
            return block;
        }
        const auto [it, added] = keys_by_block_.try_emplace(block.dump(), manifest::field_set_key(field_sets_.size()));
        if (added) {
            field_sets_[it->second] = std::move(block);
        }
        return it->second;
    }

    // Every distinct block written, keyed by its field_sets key, in the order each was first written.
    json take_field_sets() { return std::move(field_sets_); }

private:
    FieldsLayout layout_;
    // Each distinct block, as its compact JSON text, and its field_sets key.
    std::unordered_map<std::string, std::string> keys_by_block_;
    json field_sets_ = json::object();
};

json make_serviced_by_json(const std::vector<uint32_t>& risc_ids) {
    json out = json::array();
    for (const auto risc_id : risc_ids) {
        out.push_back(fmt::format("erisc{}", risc_id));
    }
    return out;
}

// A worker producer is "worker", the tensix mux is "tensix_mux", and a sibling router producer is that router's path.
constexpr std::string_view k_local_worker_producer = "worker";
constexpr std::string_view k_local_tensix_mux_producer = "tensix_mux";

json make_sender_producer_json(
    const std::optional<manifest::SenderChannelProducer>& producer,
    FabricNodeId node,
    const manifest::RouterIdentity& identity) {
    if (!producer.has_value()) {
        return nullptr;
    }
    if (std::holds_alternative<manifest::LocalWorker>(*producer)) {
        return std::string(k_local_worker_producer);
    }
    if (std::holds_alternative<manifest::LocalTensixMux>(*producer)) {
        return std::string(k_local_tensix_mux_producer);
    }
    return sibling_router_path(node, identity, std::get<manifest::SiblingRouterRef>(*producer));
}

json make_sender_channel_json(
    const manifest::SenderChannel& sender,
    FabricNodeId node,
    const manifest::RouterIdentity& identity,
    FieldsWriter& fields_writer) {
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
    out["fields"] = fields_writer.write(sender.fields_by_erisc);
    return out;
}

json make_receiver_channel_json(const manifest::ReceiverChannel& receiver, FieldsWriter& fields_writer) {
    json out;
    out["status"] = lower_enum_name(receiver.status);
    out["serviced_by"] = make_serviced_by_json(receiver.serviced_by);
    out["forwards_on"] =
        make_by_erisc_json(receiver.forwards_on_by_erisc, [](const std::optional<uint32_t>& forwards_on) {
            return forwards_on.has_value() ? json(fmt::format("vc{}", *forwards_on)) : json(nullptr);
        });
    out["ring_buffer"] = make_optional_l1_json(receiver.ring_buffer);
    out["fields"] = fields_writer.write(receiver.fields_by_erisc);
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
json make_downstream_edges_json(const manifest::Router& router, FabricNodeId node, FieldsWriter& fields_writer) {
    json out = json::object();
    for (size_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
        const auto& edges = router.intra_chip_downstream_edges[vc];
        if (edges.empty()) {
            continue;
        }
        json vc_json;
        for (const auto& edge : edges) {
            json entry;
            entry["downstream_channel"] = downstream_channel_path(node, router.identity, vc, edge.downstream_channel);
            entry["through_tensix_mux"] = edge.through_tensix_mux;
            entry["fields"] = fields_writer.write(edge.fields_by_erisc);
            vc_json[fmt::format("edge{}", edge.edge)] = std::move(entry);
        }
        out[fmt::format("vc{}", vc)] = std::move(vc_json);
    }
    return out;
}

// Keyed vc<N> for VC0 and VC1: the directions the router forwards to on the VC, as lower-case names in enum order,
// empty when its downstream array is. Null in 1D.
json make_live_downstreams_json(const std::optional<manifest::LiveDownstreamMasks>& live_downstreams) {
    if (!live_downstreams.has_value()) {
        return json(nullptr);
    }
    json out = json::object();
    for (uint32_t vc = 0; vc < live_downstreams->size(); ++vc) {
        json directions = json::array();
        for (uint32_t index = 0; index < static_cast<uint32_t>(eth_chan_directions::COUNT); ++index) {
            const auto direction = static_cast<eth_chan_directions>(index);
            if (((*live_downstreams)[vc] & Routing2DCodec::action_bit(direction)) != 0) {
                directions.push_back(lower_enum_name(direction));
            }
        }
        out[fmt::format("vc{}", vc)] = std::move(directions);
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
        erisc["opt_level"] = lower_enum_name(eriscs[risc_id].opt_level);
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

json make_router_json(const manifest::Router& router, FabricNodeId node, FieldsWriter& fields_writer) {
    json out;
    out["identity"] = make_router_identity_json(router.identity);
    out["link"] = make_eth_link_json(router.link);
    out["shape"] = make_router_shape_json(router.shape);
    out["credit_counters"] = make_credit_counters_json(router.credit_counters);
    out["channels"]["senders"] =
        make_channels_by_vc_json(router.channels.senders, [&](const manifest::SenderChannel& sender) {
            return make_sender_channel_json(sender, node, router.identity, fields_writer);
        });
    out["channels"]["receivers"] = make_channels_by_vc_json(
        router.channels.receivers,
        [&](const manifest::ReceiverChannel& receiver) { return make_receiver_channel_json(receiver, fields_writer); });
    out["intra_chip_downstream_edges"] = make_downstream_edges_json(router, node, fields_writer);
    out["live_downstreams"] = make_live_downstreams_json(router.live_downstreams);
    out["fields"] = fields_writer.write(router.fields_by_erisc);
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
json make_chip_routers_json(const manifest::Chip& chip, FieldsWriter& fields_writer) {
    json routers = json::object();
    for (const auto& router : chip.routers) {
        routers[router_key(router.identity.key)] = make_router_json(router, chip.fabric_node_id, fields_writer);
    }
    return routers;
}

// A remote chip has only its coordinates, and nulls for the ids this rank cannot read.
json make_chip_json(const manifest::Chip& chip, FieldsWriter& fields_writer) {
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
    out["asic_id"] = make_asic_id_text(*chip.asic_id);
    out["is_local"] = true;
    out["z_port_role"] = lower_enum_name(chip.z_port_role);
    out["local_sync"] = chip.local_sync ? make_local_sync_json(*chip.local_sync) : json(nullptr);
    out["routers"] = make_chip_routers_json(chip, fields_writer);
    return out;
}

json make_mesh_json(const manifest::Mesh& mesh, FieldsWriter& fields_writer) {
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
        chips[chip_key(chip.fabric_node_id.chip_id)] = make_chip_json(chip, fields_writer);
    }
    out["chips"] = std::move(chips);
    return out;
}

// ============ Reading ============
//
// Each read_<name> is the inverse of make_<name>_json, read with the strict reader: every key the writer writes is
// taken, so a key either side has that the other doesn't fails, named by its JSON pointer.

using visualizer::JsonObjectReader;
using visualizer::JsonValueReader;

// The enum value a lower_enum_name spelling names.
template <typename E>
E read_enum(const JsonValueReader& value) {
    const std::string name = value.read<std::string>();
    const std::optional<E> parsed = manifest::parse_lower_enum_name<E>(name);
    if (!parsed.has_value()) {
        value.fail("is \"{}\", which names no {}", name, enchantum::type_name<E>);
    }
    return *parsed;
}

template <typename E>
E take_enum(JsonObjectReader& object, std::string_view key) {
    return read_enum<E>(object.take(key));
}

// A registered type's name. A name this build doesn't register is one decode couldn't read bytes as.
std::string take_type_name(JsonObjectReader& object, std::string_view key) {
    const JsonValueReader value = object.take(key);
    std::string name = value.read<std::string>();
    if (!manifest::is_registered_type_name(name)) {
        value.fail("is \"{}\", a type this build doesn't register", name);
    }
    return name;
}

// A key the writer makes from a prefix and a counter, such as ch0, ch1 and so on, which must be the next one.
void check_key_in_sequence(std::string_view key, std::string_view prefix, size_t index, const JsonValueReader& value) {
    if (key != fmt::format("{}{}", prefix, index)) {
        value.fail("is out of order: the key here is {}{}", prefix, index);
    }
}

// The keys make_l1_json writes, out of an object that may hold others.
manifest::content::L1 take_l1_keys(JsonObjectReader& object) {
    manifest::content::L1 l1;
    l1.address = object.take<uint32_t>("address");
    l1.num_elements = object.take<uint32_t>("num_elements");
    l1.element_size = object.take<uint32_t>("element_size");
    l1.type = take_type_name(object, "type");
    return l1;
}

manifest::content::L1 read_l1(const JsonValueReader& value) { return value.read_object(take_l1_keys); }

std::optional<manifest::content::L1> read_optional_l1(const JsonValueReader& value) {
    return value.read_nullable_object(take_l1_keys);
}

// The keys make_stream_reg_json writes, out of an object that may hold others.
manifest::content::StreamReg take_stream_reg_keys(JsonObjectReader& object) {
    manifest::content::StreamReg stream;
    stream.stream_id = object.take<uint32_t>("stream_id");
    stream.address = object.take<uint32_t>("address");
    stream.value_bits = object.take<uint32_t>("value_bits");
    stream.type = take_type_name(object, "type");
    return stream;
}

tt::tt_metal::CoreCoord read_core(const JsonValueReader& value) {
    const auto coordinates = value.read_array_of<size_t>();
    if (coordinates.size() != 2) {
        value.fail("holds {} numbers, not an [x, y] pair", coordinates.size());
    }
    return {coordinates[0], coordinates[1]};
}

// An ASIC id as make_asic_id_text writes it.
uint64_t read_asic_id(const JsonValueReader& value) {
    const std::string text = value.read<std::string>();
    const std::optional<uint64_t> asic_id = parse_asic_id_text(text);
    if (!asic_id.has_value()) {
        value.fail("is \"{}\", not an ASIC id written as 0x and 16 hex digits", text);
    }
    return *asic_id;
}

// ============ Reading run, fabric and arch ============

manifest::RunInfo read_run(JsonObjectReader& run_json) {
    manifest::RunInfo run;
    run.run_id = run_json.take<std::string>("run_id");
    run.mpi_rank = run_json.take<uint32_t>("mpi_rank");
    run.world_size = run_json.take<uint32_t>("world_size");
    if (const auto subcontext = run_json.take_optional("subcontext")) {
        run.subcontext = subcontext->read_object([](JsonObjectReader& subcontext_json) {
            manifest::Subcontext out;
            out.id = subcontext_json.take<uint32_t>("id");
            out.rank = subcontext_json.take<uint32_t>("rank");
            out.size = subcontext_json.take<uint32_t>("size");
            return out;
        });
    }
    run.host_name = run_json.take<std::string>("host_name");
    run.written_at = run_json.take<std::string>("written_at");
    // The types the manifest names are the writing build's, so only that build reads it.
    run.build = visualizer::get_build_identity();
    run_json.check("build", make_build_json(run.build));
    return run;
}

manifest::FabricInfo read_fabric(JsonObjectReader& block) {
    manifest::FabricInfo fabric;
    fabric.config = block.take("config").read_object([](JsonObjectReader& config) {
        manifest::FabricSettings settings;
        settings.fabric_config = take_enum<FabricConfig>(config, "fabric_config");
        settings.reliability_mode = take_enum<FabricReliabilityMode>(config, "reliability_mode");
        settings.num_routing_planes = config.take("num_routing_planes").read_nullable<uint8_t>();
        settings.tensix_config = take_enum<FabricTensixConfig>(config, "tensix_config");
        settings.udm_mode = take_enum<FabricUDMMode>(config, "udm_mode");
        settings.fabric_manager = take_enum<FabricManagerMode>(config, "fabric_manager");
        settings.router_config.max_packet_payload_size_bytes =
            config.take("router_config").read_object([](JsonObjectReader& router_config) {
                return router_config.take("max_packet_payload_size_bytes").read_nullable<size_t>();
            });
        return settings;
    });
    fabric.topology = take_enum<Topology>(block, "topology");
    fabric.is_2d_routing = block.take<bool>("is_2d_routing");
    fabric.packet_header = take_type_name(block, "packet_header");
    fabric.packet_header_size_bytes = block.take<size_t>("packet_header_size_bytes");
    fabric.max_payload_size_bytes = block.take<size_t>("max_payload_size_bytes");
    fabric.channel_buffer_size_bytes = block.take<size_t>("channel_buffer_size_bytes");
    if (const auto route_buffer_size = block.take_optional("routing_2d_route_buffer_size")) {
        fabric.routing_2d_route_buffer_size = route_buffer_size->read<uint32_t>();
    }
    if (const auto extension_words = block.take_optional("routing_1d_extension_words")) {
        fabric.routing_1d_extension_words = extension_words->read<uint32_t>();
    }
    fabric.multi_txq = block.take<bool>("multi_txq");
    block.take("host_clears").read_array([&fabric](const JsonValueReader& clear) {
        fabric.host_clears.push_back(clear.read_object([](JsonObjectReader& clear_json) {
            manifest::HostClear out;
            out.address = clear_json.take<uint32_t>("address");
            out.size_bytes = clear_json.take<uint32_t>("size_bytes");
            return out;
        }));
    });
    fabric.intermesh_vc = block.take("intermesh_vc").read_object([](JsonObjectReader& intermesh_vc) {
        manifest::IntermeshVc out;
        out.mode = take_enum<IntermeshVCMode>(intermesh_vc, "mode");
        out.requires_vc1 = intermesh_vc.take<bool>("requires_vc1");
        out.requires_vc2 = intermesh_vc.take<bool>("requires_vc2");
        out.intermesh_vc_disabled = intermesh_vc.take<bool>("intermesh_vc_disabled");
        return out;
    });
    return fabric;
}

// The array make_core_registers_json writes: each register in the order it's read (k_core_registers), each named, and
// each a word.
manifest::CoreRegisters read_core_registers(const JsonValueReader& value) {
    manifest::CoreRegisters registers;
    size_t index = 0;
    value.read_array([&](const JsonValueReader& entry) {
        if (index == manifest::k_core_registers.size()) {
            entry.fail("is a register past the {} the manifest has", manifest::k_core_registers.size());
        }
        const auto [name, address] = manifest::k_core_registers[index++];
        entry.read_object([&](JsonObjectReader& register_json) {
            register_json.check("name", std::string(name));
            registers.*address = register_json.take<uint32_t>("address");
            register_json.check("type", std::string(manifest::get_manifest_type_name<uint32_t>()));
            if (address == &manifest::CoreRegisters::soft_reset) {
                registers.reset_masks = register_json.take("reset_masks").read_array_of<uint32_t>();
            }
        });
    });
    if (index != manifest::k_core_registers.size()) {
        value.fail("holds {} registers, not {}", index, manifest::k_core_registers.size());
    }
    return registers;
}

manifest::ArchAreas read_arch_areas(JsonObjectReader& areas_json) {
    manifest::ArchAreas areas;
    areas.heartbeat = areas_json.take("heartbeat").read_object([](JsonObjectReader& heartbeat_json) {
        manifest::Heartbeat heartbeat;
        heartbeat.word = take_l1_keys(heartbeat_json);
        heartbeat.magic = heartbeat_json.take<uint32_t>("magic");
        heartbeat.magic_mask = heartbeat_json.take<uint32_t>("magic_mask");
        heartbeat.period_iters = heartbeat_json.take<uint32_t>("period_iters");
        return heartbeat;
    });
    areas.fabric_telemetry = read_l1(areas_json.take("fabric_telemetry"));
    areas.routing_table = read_l1(areas_json.take("routing_table"));
    areas.eth_fw_mailbox = read_optional_l1(areas_json.take("eth_fw_mailbox"));
    areas.base_fw_port_status = read_optional_l1(areas_json.take("base_fw_port_status"));
    areas.base_fw_train_status = read_optional_l1(areas_json.take("base_fw_train_status"));
    areas.base_fw_retrain_count = read_optional_l1(areas_json.take("base_fw_retrain_count"));
    areas.base_fw_rx_link_up = read_optional_l1(areas_json.take("base_fw_rx_link_up"));
    areas.base_fw_postcode = read_optional_l1(areas_json.take("base_fw_postcode"));
    return areas;
}

manifest::Arch read_arch(JsonObjectReader& block) {
    manifest::Arch arch;
    arch.arch = take_enum<tt::ARCH>(block, "name");
    arch.l1 = block.take("l1").read_object([](JsonObjectReader& l1) {
        manifest::EthCoreL1 out;
        out.address = l1.take<uint32_t>("address");
        out.size_bytes = l1.take<uint32_t>("size_bytes");
        return out;
    });
    arch.areas = block.take("l1_areas").read_object(read_arch_areas);
    arch.core_registers = read_core_registers(block.take("core_registers"));
    arch.fabric_telemetry_version = block.take<uint32_t>("fabric_telemetry_version");
    return arch;
}

// ============ Reading fields ============

// The content keys of each field kind, out of a field entry that holds its category and kind as well.
manifest::Content take_content_keys(JsonObjectReader& entry, std::type_identity<manifest::content::L1>) {
    return take_l1_keys(entry);
}

manifest::Content take_content_keys(JsonObjectReader& entry, std::type_identity<manifest::content::StreamReg>) {
    return take_stream_reg_keys(entry);
}

manifest::Content take_content_keys(JsonObjectReader& entry, std::type_identity<manifest::content::Number>) {
    return manifest::content::Number{.value = entry.take<uint32_t>("value")};
}

manifest::Content take_content_keys(JsonObjectReader& entry, std::type_identity<manifest::content::Flag>) {
    return manifest::content::Flag{.value = entry.take<bool>("value")};
}

manifest::Content take_content_keys(JsonObjectReader& entry, std::type_identity<manifest::content::Enum>) {
    manifest::content::Enum enumerator;
    enumerator.value = entry.take<uint32_t>("value");
    enumerator.value_name = entry.take<std::string>("value_name");
    return enumerator;
}

// The content of the kind `kind` names, or nullopt when it names none. Every alternative of Content needs a
// take_content_keys, so a new kind doesn't compile until it has a reader.
template <typename... Contents>
std::optional<manifest::Content> take_content_of_kind(
    std::string_view kind, JsonObjectReader& entry, std::type_identity<std::variant<Contents...>> /*content*/) {
    std::optional<manifest::Content> content;
    (void)((kind == manifest::kind_name<Contents>() &&
            (content = take_content_keys(entry, std::type_identity<Contents>{}), true)) ||
           ...);
    return content;
}

// A null field keeps the model's default category: make_fields_json writes no category for it.
std::vector<manifest::Field> read_fields(const JsonValueReader& value) {
    std::vector<manifest::Field> fields;
    value.read_object([&fields](JsonObjectReader& fields_json) {
        fields_json.take_each([&fields](std::string_view key, const JsonValueReader& field_value) {
            manifest::Field field{.key = std::string(key)};
            if (!field_value.is_null()) {
                field_value.read_object([&field](JsonObjectReader& entry) {
                    field.category = take_enum<manifest::FieldCategory>(entry, "category");
                    const JsonValueReader kind = entry.take("kind");
                    const std::string kind_text = kind.read<std::string>();
                    field.content = take_content_of_kind(kind_text, entry, std::type_identity<manifest::Content>{});
                    if (!field.content.has_value()) {
                        kind.fail("is \"{}\", which names no field kind", kind_text);
                    }
                });
            }
            fields.push_back(std::move(field));
        });
    });
    return fields;
}

// The entries make_by_erisc_json writes, keyed erisc0, erisc1 and so on, each read by `read_entry`.
template <typename ReadEntry>
auto read_by_erisc(const JsonValueReader& value, ReadEntry read_entry) {
    std::vector<std::invoke_result_t<ReadEntry&, const JsonValueReader&>> entries;
    value.read_object([&](JsonObjectReader& by_erisc) {
        by_erisc.take_each([&](std::string_view key, const JsonValueReader& entry) {
            check_key_in_sequence(key, "erisc", entries.size(), entry);
            entries.push_back(read_entry(entry));
        });
    });
    return entries;
}

manifest::FieldsByErisc read_fields_by_erisc(const JsonValueReader& value) { return read_by_erisc(value, read_fields); }

// Reads each fields block as make_manifest_json wrote it: an id into field_sets when the manifest has them (SHARED),
// and the block itself otherwise (PER_ROUTER).
class FieldsReader {
public:
    // Reads field_sets, when the manifest has them. They come before meshes, which name them.
    void take_field_sets(JsonObjectReader& manifest_json) {
        const auto field_sets = manifest_json.take_optional("field_sets");
        if (!field_sets.has_value()) {
            return;
        }
        auto& blocks = blocks_by_key_.emplace();
        field_sets->read_object([&blocks](JsonObjectReader& sets) {
            sets.take_each([&blocks](std::string_view key, const JsonValueReader& block) {
                blocks.emplace(std::string(key), read_fields_by_erisc(block));
            });
        });
    }

    manifest::FieldsByErisc read(const JsonValueReader& value) const {
        if (!blocks_by_key_.has_value()) {
            return read_fields_by_erisc(value);
        }
        const std::string key = value.read<std::string>();
        const auto block = blocks_by_key_->find(key);
        if (block == blocks_by_key_->end()) {
            value.fail("is \"{}\", which field_sets doesn't have", key);
        }
        return block->second;
    }

private:
    // field_sets' blocks by key. Null when the manifest has no field_sets.
    std::optional<std::unordered_map<std::string, manifest::FieldsByErisc>> blocks_by_key_;
};

// ============ Reading routers ============

// An object keyed vc<N> as make_channels_by_vc_json and make_downstream_edges_json write it: VCs in rising order,
// below MAX_NUM_VCS, with the ones that have nothing left out. Calls `read_vc(vc, value)` on each.
template <typename ReadVc>
void read_by_vc(const JsonValueReader& value, ReadVc read_vc) {
    value.read_object([&](JsonObjectReader& by_vc) {
        std::optional<uint32_t> previous_vc;
        by_vc.take_each([&](std::string_view key, const JsonValueReader& vc_value) {
            const uint32_t vc = manifest::parse_key_number(key, "vc");
            if (vc >= builder_config::MAX_NUM_VCS || (previous_vc.has_value() && vc <= *previous_vc)) {
                vc_value.fail("is out of order: VCs rise, below {}", builder_config::MAX_NUM_VCS);
            }
            previous_vc = vc;
            read_vc(vc, vc_value);
        });
    });
}

// Indexed [vc][channel], over every VC, as the router pass fills them.
template <typename Channel, typename ReadChannel>
std::vector<std::vector<Channel>> read_channels_by_vc(const JsonValueReader& value, ReadChannel read_channel) {
    std::vector<std::vector<Channel>> channels_by_vc(builder_config::MAX_NUM_VCS);
    read_by_vc(value, [&](uint32_t vc, const JsonValueReader& vc_value) {
        auto& channels = channels_by_vc[vc];
        vc_value.read_object([&](JsonObjectReader& vc_json) {
            vc_json.take_each([&](std::string_view key, const JsonValueReader& channel) {
                check_key_in_sequence(key, "ch", channels.size(), channel);
                channels.push_back(read_channel(channel));
            });
        });
    });
    return channels_by_vc;
}

std::vector<uint32_t> read_serviced_by(const JsonValueReader& value) {
    std::vector<uint32_t> risc_ids;
    value.read_array([&risc_ids](const JsonValueReader& risc) {
        risc_ids.push_back(manifest::parse_key_number(risc.read<std::string>(), "erisc"));
    });
    return risc_ids;
}

manifest::CounterRef read_credit_ref(const JsonValueReader& value) {
    return value.read_object([](JsonObjectReader& ref) {
        const JsonValueReader array = ref.take("array");
        const std::string path = array.read<std::string>();
        const auto counter_array = path.starts_with(k_credit_counters_path_prefix)
                                       ? manifest::parse_lower_enum_name<manifest::CreditCounterArray>(
                                             std::string_view(path).substr(k_credit_counters_path_prefix.size()))
                                       : std::nullopt;
        if (!counter_array.has_value()) {
            array.fail("is \"{}\", not a credit counter array such as credit_counters/to_sender_ack", path);
        }
        return manifest::CounterRef{.array = *counter_array, .index = ref.take<uint32_t>("index")};
    });
}

manifest::SenderChannelProducer parse_sender_producer(
    std::string_view producer, FabricNodeId node, const manifest::RouterIdentity& identity) {
    if (producer == k_local_worker_producer) {
        return manifest::LocalWorker{};
    }
    if (producer == k_local_tensix_mux_producer) {
        return manifest::LocalTensixMux{};
    }
    return manifest::parse_sibling_router_path(producer, node, identity);
}

manifest::SenderChannel read_sender_channel(
    const JsonValueReader& value,
    FabricNodeId node,
    const manifest::RouterIdentity& identity,
    const FieldsReader& fields_reader) {
    return value.read_object([&](JsonObjectReader& sender_json) {
        manifest::SenderChannel sender;
        sender.status = take_enum<manifest::ChannelStatus>(sender_json, "status");
        sender.serviced_by = read_serviced_by(sender_json.take("serviced_by"));
        if (const auto producer = sender_json.take("producer").read_nullable<std::string>()) {
            sender.producer = parse_sender_producer(*producer, node, identity);
        }
        sender.ring_buffer = read_optional_l1(sender_json.take("ring_buffer"));
        sender.credits_by_erisc = read_by_erisc(sender_json.take("credits"), [](const JsonValueReader& credits) {
            return credits.read_object([](JsonObjectReader& credits_json) {
                manifest::SenderChannelCredits out;
                if (const auto acked = credits_json.take_optional("acked")) {
                    out.acked = read_credit_ref(*acked);
                }
                out.completed = read_credit_ref(credits_json.take("completed"));
                return out;
            });
        });
        sender.fields_by_erisc = fields_reader.read(sender_json.take("fields"));
        return sender;
    });
}

manifest::ReceiverChannel read_receiver_channel(const JsonValueReader& value, const FieldsReader& fields_reader) {
    return value.read_object([&](JsonObjectReader& receiver_json) {
        manifest::ReceiverChannel receiver;
        receiver.status = take_enum<manifest::ChannelStatus>(receiver_json, "status");
        receiver.serviced_by = read_serviced_by(receiver_json.take("serviced_by"));
        receiver.forwards_on_by_erisc =
            read_by_erisc(receiver_json.take("forwards_on"), [](const JsonValueReader& forwards_on) {
                const auto vc = forwards_on.read_nullable<std::string>();
                return vc.has_value() ? std::optional<uint32_t>(manifest::parse_key_number(*vc, "vc")) : std::nullopt;
            });
        receiver.ring_buffer = read_optional_l1(receiver_json.take("ring_buffer"));
        receiver.fields_by_erisc = fields_reader.read(receiver_json.take("fields"));
        return receiver;
    });
}

std::vector<std::vector<manifest::DownstreamEdge>> read_downstream_edges(
    const JsonValueReader& value,
    FabricNodeId node,
    const manifest::RouterIdentity& identity,
    const FieldsReader& fields_reader) {
    std::vector<std::vector<manifest::DownstreamEdge>> edges_by_vc(builder_config::MAX_NUM_VCS);
    read_by_vc(value, [&](uint32_t vc, const JsonValueReader& vc_value) {
        vc_value.read_object([&](JsonObjectReader& vc_json) {
            vc_json.take_each([&](std::string_view key, const JsonValueReader& edge_value) {
                edges_by_vc[vc].push_back(edge_value.read_object([&](JsonObjectReader& edge_json) {
                    manifest::DownstreamEdge edge;
                    edge.edge = manifest::parse_key_number(key, "edge");
                    edge.downstream_channel = manifest::parse_downstream_channel_path(
                        edge_json.take<std::string>("downstream_channel"), node, identity, vc);
                    edge.through_tensix_mux = edge_json.take<bool>("through_tensix_mux");
                    edge.fields_by_erisc = fields_reader.read(edge_json.take("fields"));
                    return edge;
                }));
            });
        });
    });
    return edges_by_vc;
}

// What make_live_downstreams_json writes: null, or each of VC0's and VC1's directions by name, each once and in enum
// order.
std::optional<manifest::LiveDownstreamMasks> read_live_downstreams(const JsonValueReader& value) {
    return value.read_nullable_object([](JsonObjectReader& by_vc) {
        manifest::LiveDownstreamMasks masks{};
        for (uint32_t vc = 0; vc < masks.size(); ++vc) {
            std::optional<eth_chan_directions> previous;
            by_vc.take(fmt::format("vc{}", vc)).read_array([&](const JsonValueReader& name) {
                const auto direction = read_enum<eth_chan_directions>(name);
                if (direction >= eth_chan_directions::COUNT) {
                    name.fail("is \"{}\", which is not a direction", lower_enum_name(direction));
                }
                if (previous.has_value() && direction <= *previous) {
                    name.fail("is out of order: directions rise in enum order, each once");
                }
                previous = direction;
                masks[vc] |= Routing2DCodec::action_bit(direction);
            });
        }
        return masks;
    });
}

// The counts per VC, one for each of MAX_NUM_VCS.
std::array<uint32_t, builder_config::MAX_NUM_VCS> read_counts_per_vc(const JsonValueReader& value) {
    const auto counts = value.read_array_of<uint32_t>();
    if (counts.size() != builder_config::MAX_NUM_VCS) {
        value.fail("holds {} counts, not one for each of {} VCs", counts.size(), builder_config::MAX_NUM_VCS);
    }
    std::array<uint32_t, builder_config::MAX_NUM_VCS> out{};
    std::ranges::copy(counts, out.begin());
    return out;
}

manifest::RouterShape read_router_shape(JsonObjectReader& shape_json) {
    manifest::RouterShape shape;
    shape.num_vcs = shape_json.take<uint32_t>("num_vcs");
    shape.senders_per_vc = read_counts_per_vc(shape_json.take("senders_per_vc"));
    shape.receivers_per_vc = read_counts_per_vc(shape_json.take("receivers_per_vc"));
    shape.num_active_eriscs = shape_json.take<uint32_t>("num_active_eriscs");
    shape.channel_trimming_overrides_applied = shape_json.take<bool>("channel_trimming_overrides_applied");
    shape.has_tensix_extension = shape_json.take<bool>("has_tensix_extension");
    return shape;
}

manifest::EthLink read_eth_link(JsonObjectReader& link_json) {
    manifest::EthLink link;
    link.edge_capability = take_enum<EdgeCapability>(link_json, "edge_capability");
    if (const auto peer = link_json.take("peer").read_nullable<std::string>()) {
        link.peer = manifest::parse_router_path(*peer);
    }
    link.cross_host = link_json.take<bool>("cross_host");
    link.wrap = link_json.take<bool>("wrap");
    link.is_dispatch_link = link_json.take<bool>("dispatch_link");
    return link;
}

manifest::L1CreditCounters read_credit_counters(JsonObjectReader& counters_json) {
    manifest::L1CreditCounters counters;
    for (const auto& layout : manifest::k_credit_counter_arrays) {
        counters.*layout.member =
            counters_json.take(lower_enum_name(layout.array)).read_object([&layout](JsonObjectReader& entry) {
                manifest::content::L1 l1 = take_l1_keys(entry);
                entry.check("indexed_by", lower_enum_name(layout.indexed_by));
                return l1;
            });
    }
    return counters;
}

manifest::Erisc read_erisc(const JsonValueReader& value) {
    return value.read_object([](JsonObjectReader& erisc_json) {
        manifest::Erisc erisc;
        erisc.processor = take_enum<tt::tt_metal::DataMovementProcessor>(erisc_json, "processor");
        erisc.physical_erisc = erisc_json.take<uint32_t>("physical_erisc");
        erisc.opt_level = take_enum<tt::tt_metal::KernelBuildOptLevel>(erisc_json, "opt_level");
        return erisc;
    });
}

// The router `key` names on `node`.
manifest::Router read_router(
    const JsonValueReader& value,
    FabricNodeId node,
    const manifest::RouterKey& key,
    const FieldsReader& fields_reader) {
    return value.read_object([&](JsonObjectReader& router_json) {
        manifest::Router router;
        router.identity = router_json.take("identity").read_object([&key](JsonObjectReader& identity_json) {
            manifest::RouterIdentity identity;
            identity.eth_chan = identity_json.take<uint32_t>("eth_chan");
            identity.key = key;
            identity.logical_core = read_core(identity_json.take("logical_core"));
            identity.virtual_core = read_core(identity_json.take("virtual_core"));
            return identity;
        });
        router.link = router_json.take("link").read_object(read_eth_link);
        router.shape = router_json.take("shape").read_object(read_router_shape);
        router.credit_counters = router_json.take("credit_counters").read_object(read_credit_counters);
        router_json.take("channels").read_object([&](JsonObjectReader& channels) {
            router.channels.senders = read_channels_by_vc<manifest::SenderChannel>(
                channels.take("senders"), [&](const JsonValueReader& sender) {
                    return read_sender_channel(sender, node, router.identity, fields_reader);
                });
            router.channels.receivers = read_channels_by_vc<manifest::ReceiverChannel>(
                channels.take("receivers"),
                [&](const JsonValueReader& receiver) { return read_receiver_channel(receiver, fields_reader); });
        });
        router.intra_chip_downstream_edges = read_downstream_edges(
            router_json.take("intra_chip_downstream_edges"), node, router.identity, fields_reader);
        router.live_downstreams = read_live_downstreams(router_json.take("live_downstreams"));
        router.fields_by_erisc = fields_reader.read(router_json.take("fields"));
        router.eriscs = read_by_erisc(router_json.take("eriscs"), read_erisc);
        router.leftover_l1 = read_optional_l1(router_json.take("leftover_l1"));
        return router;
    });
}

// ============ Reading chips and meshes ============

manifest::LocalSync read_local_sync(JsonObjectReader& local_sync_json) {
    manifest::LocalSync local_sync;
    local_sync.master = manifest::parse_router_path(local_sync_json.take<std::string>("master"));
    local_sync.num_routers = local_sync_json.take<uint32_t>("num_routers");
    local_sync.router_channels_mask = local_sync_json.take<uint32_t>("router_channels_mask");
    return local_sync;
}

// The chip `node` names. A remote chip, one this rank has no ids for, has only its place.
manifest::Chip read_chip(const JsonValueReader& value, FabricNodeId node, const FieldsReader& fields_reader) {
    return value.read_object([&](JsonObjectReader& chip_json) {
        manifest::Chip chip{.fabric_node_id = node};
        chip.mesh_coord = chip_json.take("mesh_coord").read_array_of<uint32_t>();
        chip.host_rank = chip_json.take("host_rank").read_nullable<uint32_t>();
        chip.physical_chip_id = chip_json.take("physical_chip_id").read_nullable<ChipId>();
        const JsonValueReader asic_id = chip_json.take("asic_id");
        chip_json.check("is_local", chip.physical_chip_id.has_value());
        if (!chip.physical_chip_id.has_value()) {
            if (!asic_id.is_null()) {
                asic_id.fail("is set on a remote chip");
            }
            return chip;
        }
        chip.asic_id = read_asic_id(asic_id);
        chip.z_port_role = take_enum<ZPortRole>(chip_json, "z_port_role");
        chip.local_sync = chip_json.take("local_sync").read_nullable_object(read_local_sync);
        chip_json.take("routers").read_object([&](JsonObjectReader& routers) {
            routers.take_each([&](std::string_view key, const JsonValueReader& router) {
                chip.routers.push_back(read_router(router, node, manifest::parse_router_key(key), fields_reader));
            });
        });
        return chip;
    });
}

manifest::Mesh read_mesh(const JsonValueReader& value, MeshId mesh_id, const FieldsReader& fields_reader) {
    return value.read_object([&](JsonObjectReader& mesh_json) {
        manifest::Mesh mesh{.mesh_id = mesh_id};
        mesh.shape = mesh_json.take("shape").read_array_of<uint32_t>();
        if (const auto torus = mesh_json.take_optional("torus")) {
            mesh.torus = torus->read_object([](JsonObjectReader& torus_json) {
                manifest::MeshTorus out;
                out.y = torus_json.take<bool>("y");
                out.x = torus_json.take<bool>("x");
                return out;
            });
        }
        mesh.express_routing = mesh_json.take<bool>("express_routing");
        if (const auto credit_transport = mesh_json.take_optional("credit_transport")) {
            auto& transport = mesh.credit_transport.emplace();
            read_by_vc(*credit_transport, [&transport](uint32_t vc, const JsonValueReader& entry) {
                transport.push_back({
                    .vc = vc,
                    .backing = entry.read_object([](JsonObjectReader& entry_json) {
                        return take_enum<manifest::CreditBacking>(entry_json, "backing");
                    }),
                });
            });
        }
        mesh_json.take("chips").read_object([&](JsonObjectReader& chips) {
            chips.take_each([&](std::string_view key, const JsonValueReader& chip) {
                mesh.chips.push_back(
                    read_chip(chip, FabricNodeId(mesh_id, manifest::parse_chip_key(key)), fields_reader));
            });
        });
        return mesh;
    });
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
        // dump() doesn't use indentation, as opposed to dump(2), and saves lots of space in the file
        out_file << manifest_json.dump() << '\n';
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

json make_manifest_json(const manifest::ManifestFacts& facts, FieldsLayout fields_layout) {
    FieldsWriter fields_writer(fields_layout);
    json meshes = json::object();
    for (const auto& mesh : facts.meshes) {
        meshes[mesh_key(mesh.mesh_id)] = make_mesh_json(mesh, fields_writer);
    }

    json out;
    out["manifest_version"] = visualizer::k_manifest_version;
    out["file_kind"] = std::string(visualizer::k_manifest_file_kind);
    out["run"] = make_run_json(facts.run);
    out["fabric"] = make_fabric_json(facts.fabric);
    out["arch"] = make_arch_json(facts.arch);
    out["vocabulary"] = make_vocabulary_json();
    // field_sets only exists when we are using the shared layout. Otherwise, fields for routers / eriscs / chips
    // exist local to each router / erisc / chip block.
    if (fields_layout == FieldsLayout::SHARED) {
        out["field_sets"] = fields_writer.take_field_sets();
    }
    out["meshes"] = std::move(meshes);
    return out;
}

manifest::ManifestFacts parse_manifest_json(const json& manifest_json) {
    return visualizer::JsonValueReader(manifest_json, "Fabric manifest")
        .read_object([](visualizer::JsonObjectReader& top) {
            top.check("file_kind", std::string(visualizer::k_manifest_file_kind));
            top.check("manifest_version", visualizer::k_manifest_version);
            manifest::ManifestFacts facts;
            facts.run = top.take("run").read_object(read_run);
            facts.fabric = top.take("fabric").read_object(read_fabric);
            facts.arch = top.take("arch").read_object(read_arch);
            top.check("vocabulary", make_vocabulary_json());
            FieldsReader fields_reader;
            fields_reader.take_field_sets(top);
            top.take("meshes").read_object([&](visualizer::JsonObjectReader& meshes) {
                meshes.take_each([&](std::string_view key, const visualizer::JsonValueReader& mesh) {
                    facts.meshes.push_back(read_mesh(mesh, manifest::parse_mesh_key(key), fields_reader));
                });
            });
            return facts;
        });
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
