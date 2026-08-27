// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <regex>
#include <map>
#include <set>
#include <tt_stl/assert.hpp>

#include "protobuf/mesh_graph_descriptor.pb.h"
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-logger/tt-logger.hpp>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <unistd.h>

using namespace tt::tt_metal::distributed;

namespace tt::tt_fabric {

namespace {

// When DistributedContext is initialized (MPI / tt-run split layout), prefix instance names with mgd{id}_ using
// subcontext_id() so split-job ranks load disjoint logical names.
std::optional<int> subcontext_id_for_instance_name_uniquify() {
    using tt::tt_metal::distributed::multihost::DistributedContext;
    if (DistributedContext::is_initialized()) {
        const auto& world = DistributedContext::get_current_world();
        if (world != nullptr) {
            const auto sc = world->subcontext_id();
            if (sc.has_value()) {
                return *sc.value();
            }
        }
    }
    return std::nullopt;
}

std::string read_file_to_string(const std::filesystem::path& file_path) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

uint32_t get_max_dimensions_for_architecture(proto::Architecture arch) {
    switch (arch) {
        case proto::Architecture::WORMHOLE_B0: return 2;
        case proto::Architecture::BLACKHOLE: return 3;
        case proto::Architecture::INVALID_ARCHITECTURE:
        default: return 0;
    }
}

std::string get_validation_report(const std::vector<std::string>& error_messages) {
    if (error_messages.empty()) {
        return "No validation errors found.\n";
    }

    std::ostringstream report;
    report << "=== MeshGraphDescriptor Validation Report ===\n\n";
    report << "Errors:\n";
    for (const auto& error : error_messages) {
        report << "  - " << error << "\n";
    }
    report << "\n";

    return report.str();
}

LocalNodeId get_device_id(const MeshCoordinate& mesh_coord, const MeshShape& mesh_shape) {
    // Check that mesh_coord is within mesh_shape
    TT_FATAL(
        mesh_coord[0] < mesh_shape[0] && mesh_coord[1] < mesh_shape[1],
        "Mesh coordinate {} is out of bounds for mesh shape {}",
        mesh_coord,
        mesh_shape);
    return (mesh_coord[0] * mesh_shape[1]) + mesh_coord[1];
}

std::unordered_map<GlobalNodeId, std::vector<ConnectionData>> get_valid_connections(
    const MeshCoordinate& src_mesh_coord, const MeshCoordinateRange& mesh_coord_range, const InstanceData& instance) {
    std::unordered_map<GlobalNodeId, std::vector<ConnectionData>> connections;

    const proto::TorusTopology* device_topology = nullptr;
    const proto::Channels* channels = nullptr;

    if (instance.kind == NodeKind::Mesh) {
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
        device_topology = &mesh_desc->device_topology();
        channels = &mesh_desc->channels();
    } else if (instance.kind == NodeKind::Switch) {
        const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(instance.desc);
        device_topology = &switch_desc->device_topology();
        channels = &switch_desc->channels();
    } else {
        TT_THROW("get_valid_connections called on non-mesh/non-switch instance");
        return connections;
    }

    const auto& topology_types = device_topology->dim_types();
    const uint32_t channels_count = static_cast<uint32_t>(channels->count());
    const auto& policy = channels->policy();

    MeshShape mesh_shape = mesh_coord_range.shape();
    MeshCoordinate N(src_mesh_coord[0] - 1, src_mesh_coord[1]);
    MeshCoordinate E(src_mesh_coord[0], src_mesh_coord[1] + 1);
    MeshCoordinate S(src_mesh_coord[0] + 1, src_mesh_coord[1]);
    MeshCoordinate W(src_mesh_coord[0], src_mesh_coord[1] - 1);

    if (topology_types[0] == proto::TorusTopology::RING) {
        N = MeshCoordinate((src_mesh_coord[0] - 1 + mesh_shape[0]) % mesh_shape[0], src_mesh_coord[1]);
        S = MeshCoordinate((src_mesh_coord[0] + 1) % mesh_shape[0], src_mesh_coord[1]);
    }
    if (topology_types[1] == proto::TorusTopology::RING) {
        E = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] + 1) % mesh_shape[1]);
        W = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] - 1 + mesh_shape[1]) % mesh_shape[1]);
    }

    for (const auto& [coord, direction] :
         {std::pair{N, proto::RoutingDirection::N},
          std::pair{E, proto::RoutingDirection::E},
          std::pair{S, proto::RoutingDirection::S},
          std::pair{W, proto::RoutingDirection::W}}) {
        if (mesh_coord_range.contains(coord)) {
            const auto src_device_id =
                instance.sub_instances_local_id_to_global_id.at(get_device_id(src_mesh_coord, mesh_shape));
            const auto dst_device_id =
                instance.sub_instances_local_id_to_global_id.at(get_device_id(coord, mesh_shape));

            ConnectionData data{
                .nodes = {src_device_id, dst_device_id},
                .count = channels_count,
                .policy = policy,
                .parent_instance_id = instance.global_id,
                .routing_direction = direction,
            };

            connections[src_device_id].push_back(data);
        }
    }

    return connections;
}

}  // namespace

MeshGraphDescriptor::MeshGraphDescriptor(
    const std::string& text_proto, const bool backwards_compatible, std::string_view source_path) :
    top_level_id_(static_cast<GlobalNodeId>(-1)) {
    proto::MeshGraphDescriptor temp_proto;
    google::protobuf::TextFormat::Parser parser;

    // Allowing for back and forward compatibility for fields not currently in the proto file
    parser.AllowUnknownField(true);
    parser.AllowUnknownExtension(true);

    TT_FATAL(
        parser.ParseFromString(text_proto, &temp_proto),
        "Failed to parse MeshGraphDescriptor textproto{}{}",
        source_path.empty() ? "" : ": ",
        source_path);

    // Set defaults for missing fields
    set_defaults(temp_proto);

    // Validate the proto
    const auto errors = static_validate(temp_proto, backwards_compatible);
    TT_FATAL(
        errors.empty(),
        "Failed to validate MeshGraphDescriptor textproto{}{}: \n{}",
        source_path.empty() ? "" : " ",
        source_path,
        get_validation_report(errors));

    proto_ = std::make_shared<proto::MeshGraphDescriptor>(temp_proto);

    populate();

    // Prefix mgd{id}_ when DistributedContext reports a split-job sub-context id (MPI / tt-run).
    if (const auto sid = subcontext_id_for_instance_name_uniquify(); sid.has_value()) {
        const std::string prefix = "mgd" + std::to_string(*sid) + "_";
        instances_by_name_.clear();
        for (auto& [_, inst] : instances_) {
            inst.name = prefix + inst.name;
        }
        for (const auto& [gid, inst] : instances_) {
            instances_by_name_[inst.name].push_back(gid);
        }
    }
}

MeshGraphDescriptor::MeshGraphDescriptor(
    const std::filesystem::path& text_proto_file_path, const bool backwards_compatible) :
    MeshGraphDescriptor(
        read_file_to_string(text_proto_file_path.string()), backwards_compatible, text_proto_file_path.string()) {}

MeshGraphDescriptor::~MeshGraphDescriptor() = default;

proto::Architecture MeshGraphDescriptor::get_arch() const {
    // All meshes must have the same arch
    return proto_->mesh_descriptors(0).arch();
}

uint32_t MeshGraphDescriptor::get_num_eth_ports_per_direction() const {
    return proto_->mesh_descriptors(0).channels().count();
}

std::vector<std::string> MeshGraphDescriptor::get_all_mesh_names() const {
    std::unordered_set<std::string> names;
    names.reserve(mesh_instances_.size());
    for (GlobalNodeId id : mesh_instances_) {
        names.insert(get_instance(id).name);
    }
    std::vector<std::string> out(names.begin(), names.end());
    std::sort(out.begin(), out.end());
    return out;
}

uint32_t MeshGraphDescriptor::get_chip_count(GlobalNodeId mesh_instance_id) const {
    const auto& instance = get_instance(mesh_instance_id);
    return get_chip_count(instance);
}

uint32_t MeshGraphDescriptor::get_chip_count(const InstanceData& mesh_instance) const {
    TT_FATAL(is_mesh(mesh_instance), "get_chip_count() can only be called on mesh instances");

    const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(mesh_instance.desc);
    TT_FATAL(mesh_desc != nullptr, "Mesh descriptor is null for instance {}", mesh_instance.global_id);

    uint32_t chip_count = 1;
    for (const auto& dim : mesh_desc->device_topology().dims()) {
        chip_count *= dim;
    }

    return chip_count;
}

uint32_t MeshGraphDescriptor::get_switch_chip_count(GlobalNodeId switch_instance_id) const {
    const auto& instance = get_instance(switch_instance_id);
    return get_switch_chip_count(instance);
}

uint32_t MeshGraphDescriptor::get_switch_chip_count(const InstanceData& switch_instance) const {
    TT_FATAL(is_switch(switch_instance), "get_switch_chip_count() can only be called on switch instances");

    const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(switch_instance.desc);
    TT_FATAL(switch_desc != nullptr, "Switch descriptor is null for instance {}", switch_instance.global_id);

    uint32_t chip_count = 1;
    for (const auto& dim : switch_desc->device_topology().dims()) {
        chip_count *= dim;
    }

    return chip_count;
}

std::unordered_map<std::string, uint32_t> MeshGraphDescriptor::count_instances_by_type(
    const std::vector<std::string>& types) const {
    std::unordered_map<std::string, uint32_t> counts;

    for (const auto& type : types) {
        // Check if this type exists in instances_by_type_
        auto it = instances_by_type_.find(type);
        if (it != instances_by_type_.end()) {
            counts[type] = static_cast<uint32_t>(it->second.size());
        } else {
            counts[type] = 0;
        }
    }

    return counts;
}

namespace {

template <typename Descriptor>
FabricType infer_declared_fabric_type_from_dim_types(const Descriptor* descriptor) {
    const auto& dim_types = descriptor->device_topology().dim_types();
    if (dim_types.size() < 2) {
        return FabricType::MESH;
    }

    const bool y_is_ring = (dim_types[0] == proto::TorusTopology::RING);
    const bool x_is_ring = (dim_types[1] == proto::TorusTopology::RING);

    if (y_is_ring && x_is_ring) {
        return FabricType::TORUS_XY;
    }
    if (y_is_ring) {
        return FabricType::TORUS_Y;
    }
    if (x_is_ring) {
        return FabricType::TORUS_X;
    }
    return FabricType::MESH;
}

}  // namespace

FabricType MeshGraphDescriptor::infer_fabric_type_from_dim_types(const proto::MeshDescriptor* mesh_desc) {
    return infer_declared_fabric_type_from_dim_types(mesh_desc);
}

FabricType MeshGraphDescriptor::infer_fabric_type_from_dim_types(const proto::SwitchDescriptor* switch_desc) {
    return infer_declared_fabric_type_from_dim_types(switch_desc);
}

void MeshGraphDescriptor::set_defaults(proto::MeshGraphDescriptor& proto) {
    // Set the default for channel policy to strict if not specified
    for (auto& mesh : *proto.mutable_mesh_descriptors()) {
        if (mesh.has_channels() && !mesh.channels().has_policy()) {
            mesh.mutable_channels()->set_policy(proto::Policy::RELAXED);
        }
    }

    for (auto& switch_desc : *proto.mutable_switch_descriptors()) {
        if (switch_desc.has_channels() && !switch_desc.channels().has_policy()) {
            switch_desc.mutable_channels()->set_policy(proto::Policy::STRICT);
        }
    }

    for (auto& graph : *proto.mutable_graph_descriptors()) {
        // Set default policy for graph topology channels
        if (graph.has_graph_topology() && graph.graph_topology().has_channels() &&
            !graph.graph_topology().channels().has_policy()) {
            graph.mutable_graph_topology()->mutable_channels()->set_policy(proto::Policy::STRICT);
        }

        // Set default policy for connection channels
        for (auto& connection : *graph.mutable_connections()) {
            if (connection.has_channels() && !connection.channels().has_policy()) {
                connection.mutable_channels()->set_policy(proto::Policy::STRICT);
            }
        }
    }

    // Set dim_types to LINE if not specified for each dimension
    for (auto& mesh : *proto.mutable_mesh_descriptors()) {
        if (mesh.device_topology().dim_types_size() < mesh.device_topology().dims_size()) {
            for (int i = mesh.device_topology().dim_types_size(); i < mesh.device_topology().dims_size(); i++) {
                mesh.mutable_device_topology()->mutable_dim_types()->Add(proto::TorusTopology::LINE);
            }
        }
    }

    for (auto& switch_desc : *proto.mutable_switch_descriptors()) {
        if (switch_desc.device_topology().dim_types_size() < switch_desc.device_topology().dims_size()) {
            for (int i = switch_desc.device_topology().dim_types_size(); i < switch_desc.device_topology().dims_size();
                 i++) {
                switch_desc.mutable_device_topology()->mutable_dim_types()->Add(proto::TorusTopology::LINE);
            }
        }
    }

    // A RING dimension only realizes a distinct wrap edge at extent three or larger. Coerce RING on
    // smaller dimensions to LINE so nothing downstream (connection building, fabric-type inference,
    // deadlock-avoidance selection) treats the axis as a torus. See issue #54650: a declared-but-
    // degenerate torus axis enables bubble-flow/first-level-ACK on links that a rotated neighbor
    // mesh labels differently, hanging inter-mesh traffic.
    auto coerce_degenerate_rings = [](proto::TorusTopology& device_topology, const std::string& name) {
        for (int i = 0; i < device_topology.dim_types_size() && i < device_topology.dims_size(); i++) {
            if (device_topology.dim_types(i) == proto::TorusTopology::RING && device_topology.dims(i) <= 2) {
                log_warning(
                    tt::LogFabric,
                    "MeshGraphDescriptor: '{}' declares RING on dimension {} of extent {}; a ring needs more than 2 "
                    "devices, treating this dimension as LINE",
                    name,
                    i,
                    device_topology.dims(i));
                device_topology.set_dim_types(i, proto::TorusTopology::LINE);
            }
        }
    };
    for (auto& mesh : *proto.mutable_mesh_descriptors()) {
        coerce_degenerate_rings(*mesh.mutable_device_topology(), mesh.name());
    }
    for (auto& switch_desc : *proto.mutable_switch_descriptors()) {
        coerce_degenerate_rings(*switch_desc.mutable_device_topology(), switch_desc.name());
    }
}

std::vector<std::string> MeshGraphDescriptor::static_validate(
    const proto::MeshGraphDescriptor& proto, const bool backwards_compatible) {
    std::vector<std::string> all_errors;

    // Run validation groups with early exit checkpoints
    {
        validate_basic_structure(proto, all_errors);
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    {
        validate_names(proto, all_errors);
        validate_channels(proto, all_errors);
        validate_architecture_consistency(proto, all_errors);
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    {
        validate_mesh_topology(proto, all_errors);
        validate_express_connections(proto, all_errors);
        validate_switch_descriptors(proto, all_errors);
        validate_graph_descriptors(proto, all_errors);
        validate_graph_topology_and_connections(proto, all_errors);
        validate_pinnings(proto, all_errors);
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    {
        if (backwards_compatible) {
            validate_legacy_requirements(proto, all_errors);
        }
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    return all_errors;
}

void MeshGraphDescriptor::populate() {
    populate_descriptors();

    populate_top_level_instance();

    pre_populate_connections_lookups();

    populate_connections();

    populate_pinnings();
}

void MeshGraphDescriptor::populate_top_level_instance() {
    std::vector<GlobalNodeId> hierarchy;
    top_level_id_ = populate_instance(proto_->top_level_instance(), hierarchy);
}

void MeshGraphDescriptor::validate_basic_structure(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors) {
    // Allow either mesh descriptors or switch descriptors (switches can exist without meshes)
    if (proto.mesh_descriptors_size() == 0 && proto.switch_descriptors_size() == 0) {
        errors.push_back("There must be at least one mesh descriptor or switch descriptor");
    }
    if (!proto.has_top_level_instance()) {
        errors.push_back("Top level instance is required");
    }
}

void MeshGraphDescriptor::validate_names(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    unsigned int mesh_counter = 0;

    // Check that all mesh descriptors have a unique name
    std::unordered_set<std::string> mesh_names;
    for (const auto& mesh : proto.mesh_descriptors()) {
        mesh_counter++;
        if (mesh.name().empty()) {
            error_messages.push_back(fmt::format("Mesh descriptor {} has no name", mesh_counter));
            continue;
        }
        auto [it, inserted] = mesh_names.insert(mesh.name());
        if (!inserted) {
            error_messages.push_back(fmt::format("Mesh descriptor name is not unique (Mesh: {})", mesh.name()));
        }
    }

    unsigned int graph_counter = 0;

    // Check that all graph descriptors have a unique name
    std::unordered_set<std::string> graph_names;
    for (const auto& graph : proto.graph_descriptors()) {
        graph_counter++;
        if (graph.name().empty()) {
            error_messages.push_back(fmt::format("Graph descriptor {} has no name", graph_counter));
            continue;
        }
        auto [it, inserted] = graph_names.insert(graph.name());
        if (!inserted) {
            error_messages.push_back(fmt::format("Graph descriptor name is not unique (Graph: {})", graph.name()));
        }

        // TYPE name cannot be DEVICE, MESH, or SWITCH
        const auto& type = graph.type();
        if (type == "DEVICE" || type == "MESH" || type == "SWITCH") {
            error_messages.push_back(
                fmt::format("Graph descriptor type cannot be DEVICE, MESH, or SWITCH (Graph: {})", graph.name()));
        }
    }

    unsigned int switch_counter = 0;

    // Check that all switch descriptors have a unique name
    std::unordered_set<std::string> switch_names;
    for (const auto& switch_desc : proto.switch_descriptors()) {
        switch_counter++;
        if (switch_desc.name().empty()) {
            error_messages.push_back(fmt::format("Switch descriptor {} has no name", switch_counter));
            continue;
        }
        auto [it, inserted] = switch_names.insert(switch_desc.name());
        if (!inserted) {
            error_messages.push_back(
                fmt::format("Switch descriptor name is not unique (Switch: {})", switch_desc.name()));
        }
    }
}

void MeshGraphDescriptor::validate_mesh_topology(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Validate basic mesh properties (names and dimensions)
    for (const auto& mesh : proto.mesh_descriptors()) {
        // Check that all dims are positive
        for (const auto& dim : mesh.device_topology().dims()) {
            if (dim <= 0) {
                error_messages.push_back(
                    fmt::format("Device topology dimensions must be positive (Mesh: {})", mesh.name()));
                continue;
            }
        }

        // Check that device topology dimensions and types are the same size
        if (mesh.device_topology().dim_types_size() > 0) {
            if (mesh.device_topology().dims_size() != mesh.device_topology().dim_types_size()) {
                error_messages.push_back(
                    fmt::format("Device topology dimensions and types must be the same size (Mesh: {})", mesh.name()));
                continue;
            }
        }

        // Check that the device and host topology dimensions are the same size
        if (mesh.device_topology().dims_size() != mesh.host_topology().dims_size()) {
            error_messages.push_back(
                fmt::format("Device and host topology dimensions must be the same size (Mesh: {})", mesh.name()));
            continue;
        }

        // Check that the device topology dimensions are divisible by the host topology dimensions
        if (mesh.device_topology().dims_size() > 0) {
            for (int i = 0; i < mesh.device_topology().dims_size(); i++) {
                if (mesh.device_topology().dims(i) % mesh.host_topology().dims(i) != 0) {
                    error_messages.push_back(fmt::format(
                        "Device topology dimensions must be divisible by host topology dimensions (Mesh: {})",
                        mesh.name()));
                    continue;
                }
            }
        }
    }
}

void MeshGraphDescriptor::validate_architecture_consistency(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Check all architectures are the same across meshes and switches
    proto::Architecture first_arch = proto::Architecture::INVALID_ARCHITECTURE;
    if (proto.mesh_descriptors_size() > 0) {
        first_arch = proto.mesh_descriptors(0).arch();
        if (!std::all_of(
                proto.mesh_descriptors().begin(), proto.mesh_descriptors().end(), [first_arch](const auto& mesh) {
                    return mesh.arch() == first_arch;
                })) {
            error_messages.push_back("All mesh descriptors must have the same architecture");
            return;
        }
    }

    // Check switches match mesh architecture (if meshes exist)
    if (proto.switch_descriptors_size() > 0) {
        if (first_arch == proto::Architecture::INVALID_ARCHITECTURE && proto.mesh_descriptors_size() == 0) {
            // No meshes, use first switch arch as reference
            first_arch = proto.switch_descriptors(0).arch();
        }
        if (first_arch != proto::Architecture::INVALID_ARCHITECTURE) {
            if (!std::all_of(
                    proto.switch_descriptors().begin(),
                    proto.switch_descriptors().end(),
                    [first_arch](const auto& switch_desc) { return switch_desc.arch() == first_arch; })) {
                error_messages.push_back("All switch descriptors must have the same architecture as meshes");
                return;
            }
        }
    }

    // Verify that arch, device and host topology must exist in mesh descriptors
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.arch() == proto::Architecture::INVALID_ARCHITECTURE) {
            error_messages.push_back(
                fmt::format("Mesh descriptor must have a valid architecture (Mesh: {})", mesh.name()));
            continue;
        }

        // Validate architecture and dimension limits
        const uint32_t max_num_dims = get_max_dimensions_for_architecture(mesh.arch());
        if (max_num_dims == 0) {
            error_messages.push_back(fmt::format("Invalid architecture (Mesh: {})", mesh.name()));
            continue;
        }

        // Check that the number of dimensions is not greater than the maximum allowed for the architecture
        if (mesh.device_topology().dims_size() > max_num_dims) {
            error_messages.push_back(fmt::format(
                "Architecture devices allow a maximum of {} dimensions, but {} were provided (Mesh: {})",
                max_num_dims,
                mesh.device_topology().dims_size(),
                mesh.name()));
            continue;
        }
    }
}

void MeshGraphDescriptor::validate_switch_descriptors(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Validate basic switch properties (names and dimensions)
    for (const auto& switch_desc : proto.switch_descriptors()) {
        // Check that all dims are positive
        for (const auto& dim : switch_desc.device_topology().dims()) {
            if (dim <= 0) {
                error_messages.push_back(
                    fmt::format("Device topology dimensions must be positive (Switch: {})", switch_desc.name()));
                continue;
            }
        }

        // Check that device topology dimensions and types are the same size
        if (switch_desc.device_topology().dim_types_size() > 0) {
            if (switch_desc.device_topology().dims_size() != switch_desc.device_topology().dim_types_size()) {
                error_messages.push_back(fmt::format(
                    "Device topology dimensions and types must be the same size (Switch: {})", switch_desc.name()));
                continue;
            }
        }

        // Check that switch has valid architecture
        if (switch_desc.arch() == proto::Architecture::INVALID_ARCHITECTURE) {
            error_messages.push_back(
                fmt::format("Switch descriptor must have a valid architecture (Switch: {})", switch_desc.name()));
            continue;
        }

        // Validate architecture and dimension limits
        const uint32_t max_num_dims = get_max_dimensions_for_architecture(switch_desc.arch());
        if (max_num_dims == 0) {
            error_messages.push_back(fmt::format("Invalid architecture (Switch: {})", switch_desc.name()));
            continue;
        }

        // Check that the number of dimensions is not greater than the maximum allowed for the architecture
        if (switch_desc.device_topology().dims_size() > max_num_dims) {
            error_messages.push_back(fmt::format(
                "Architecture devices allow a maximum of {} dimensions, but {} were provided (Switch: {})",
                max_num_dims,
                switch_desc.device_topology().dims_size(),
                switch_desc.name()));
            continue;
        }

        // Validate express connections for switches
        uint32_t num_devices = 1;
        for (const auto& dim : switch_desc.device_topology().dims()) {
            num_devices *= dim;
        }
        for (const auto& express_conn : switch_desc.express_connections()) {
            if (express_conn.src() >= num_devices || express_conn.dst() >= num_devices) {
                error_messages.push_back(
                    fmt::format("Express connection destination is out of bounds (Switch: {})", switch_desc.name()));
            }
        }
    }
}

void MeshGraphDescriptor::validate_channels(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Check all channel counts > 0
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.channels().count() <= 0) {
            error_messages.push_back(fmt::format("Channel count must be positive (Mesh: {})", mesh.name()));
        }
    }

    // Check switch channel counts > 0
    for (const auto& switch_desc : proto.switch_descriptors()) {
        if (switch_desc.channels().count() <= 0) {
            error_messages.push_back(fmt::format("Channel count must be positive (Switch: {})", switch_desc.name()));
        }
    }

    // Check that channels in graph topology are positive
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.has_graph_topology() && graph.graph_topology().channels().count() <= 0) {
            error_messages.push_back(
                fmt::format("Graph topology channel count must be positive (Graph: {})", graph.name()));
        }
    }

    // Check all channel counts > 0 in graph descriptors and connections
    for (const auto& graph : proto.graph_descriptors()) {
        // Check connection-level channels and validate connection nodes
        for (const auto& connection : graph.connections()) {
            if (connection.channels().count() <= 0) {
                error_messages.push_back(
                    fmt::format("Connection channel count must be positive (Graph: {})", graph.name()));
            }
        }
    }
}

void MeshGraphDescriptor::validate_express_connections(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Validate express connections
    for (const auto& mesh : proto.mesh_descriptors()) {
        const uint32_t num_dims = mesh.device_topology().dims_size();

        // Calculate the number of devices in the mesh
        uint32_t num_devices = 1;
        for (uint32_t i = 0; i < num_dims; i++) {
            num_devices *= mesh.device_topology().dims(i);
        }

        // Check that express connections are valid and have the right number of devices
        for (const auto& express_connection : mesh.express_connections()) {
            if (express_connection.src() < 0 || express_connection.src() >= num_devices) {
                error_messages.push_back(
                    fmt::format("Express connection source is out of bounds (Mesh: {})", mesh.name()));
            }
            if (express_connection.dst() < 0 || express_connection.dst() >= num_devices) {
                error_messages.push_back(
                    fmt::format("Express connection destination is out of bounds (Mesh: {})", mesh.name()));
            }
        }
    }
}

void MeshGraphDescriptor::validate_graph_descriptors(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Check that there is at least one instance in the graph and validate references
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.instances_size() == 0) {
            error_messages.push_back(
                fmt::format("Graph descriptor must have at least one instance (Graph: {})", graph.name()));
        }
    }

    // Verify that type is set in graph descriptors
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.type().empty()) {
            error_messages.push_back(
                fmt::format("Graph descriptor must have a type specified (Graph: {})", graph.name()));
        }
    }
}

void MeshGraphDescriptor::validate_graph_topology_and_connections(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Combine all checks into a single loop over graph_descriptors
    for (const auto& graph : proto.graph_descriptors()) {
        // Check connections have at least 2 nodes
        for (const auto& connection : graph.connections()) {
            if (connection.nodes_size() < 2) {
                error_messages.push_back(
                    fmt::format("Connection must have at least two nodes (Graph: {})", graph.name()));
            }
        }
    }
}

void MeshGraphDescriptor::validate_legacy_requirements(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Topology must be 2D
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.device_topology().dims_size() != 2) {
            error_messages.push_back(fmt::format(
                "MGD 1.0 Compatibility requirement: There can only be 2 dimensions in the device topology (Mesh: {})",
                mesh.name()));
        }
    }

    // Express connections are currently not supported
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.express_connections_size() > 0) {
            error_messages.push_back(fmt::format(
                "MGD 1.0 Compatibility requirement: Express connections are not supported (Mesh: {})", mesh.name()));
        }
    }

    // Check that there are only 2 dimensions in the device topology and host topology
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.device_topology().dims_size() != 2 || mesh.host_topology().dims_size() != 2) {
            error_messages.push_back(fmt::format(
                "MGD 1.0 Compatibility requirement: There can only be 2 dimensions in the device topology and host "
                "topology (Mesh: {})",
                mesh.name()));
        }
    }

    // Check that there is only a FABRIC level graph
    if (proto.graph_descriptors_size() > 1) {
        error_messages.push_back(
            fmt::format("MGD 1.0 Compatibility requirement: There can only be one FABRIC level graph or less"));
    }

    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.type() != "FABRIC") {
            error_messages.push_back(fmt::format(
                "MGD 1.0 Compatibility requirement: There can only be one FABRIC level graph (Graph: {})",
                graph.name()));
        }
    }

    // Check that connections only have 2 nodes
    for (const auto& graph : proto.graph_descriptors()) {
        for (const auto& connection : graph.connections()) {
            if (connection.nodes_size() != 2) {
                error_messages.push_back(fmt::format(
                    "MGD 1.0 Compatibility requirement: Connections must have exactly 2 nodes (Graph: {})",
                    graph.name()));
            }
        }
    }

    // Check that connections in the same graph don't mix STRICT and RELAXED policies
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.connections_size() == 0) {
            continue;
        }

        // Determine the policy of the first connection (default to STRICT if not specified)
        proto::Policy first_policy = proto::Policy::STRICT;
        if (graph.connections(0).has_channels() && graph.connections(0).channels().has_policy()) {
            first_policy = graph.connections(0).channels().policy();
        }

        // Check all other connections have the same policy
        for (int i = 1; i < graph.connections_size(); ++i) {
            const auto& connection = graph.connections(i);
            proto::Policy connection_policy = proto::Policy::STRICT;
            if (connection.has_channels() && connection.channels().has_policy()) {
                connection_policy = connection.channels().policy();
            }

            if (connection_policy != first_policy) {
                error_messages.push_back(fmt::format(
                    "MGD 1.0 Compatibility requirement: Cannot mix STRICT and RELAXED policies in the same graph. "
                    "All connections in a graph must use the same policy (Graph: {})",
                    graph.name()));
                break;  // Only report once per graph
            }
        }
    }
}

void MeshGraphDescriptor::populate_descriptors() {
    mesh_desc_by_name_.clear();
    graph_desc_by_name_.clear();
    switch_desc_by_name_.clear();
    mesh_desc_by_name_.reserve(proto_->mesh_descriptors_size());
    graph_desc_by_name_.reserve(proto_->graph_descriptors_size());
    switch_desc_by_name_.reserve(proto_->switch_descriptors_size());
    // Use string_view into proto_ storage; safe as long as proto_ outlives maps
    for (int i = 0; i < proto_->mesh_descriptors_size(); ++i) {
        const auto& mesh = proto_->mesh_descriptors(i);
        mesh_desc_by_name_.emplace(mesh.name(), &mesh);
    }
    for (int i = 0; i < proto_->graph_descriptors_size(); ++i) {
        const auto& graph = proto_->graph_descriptors(i);
        graph_desc_by_name_.emplace(graph.name(), &graph);
    }
    for (int i = 0; i < proto_->switch_descriptors_size(); ++i) {
        const auto& switch_desc = proto_->switch_descriptors(i);
        switch_desc_by_name_.emplace(switch_desc.name(), &switch_desc);
    }
}

GlobalNodeId MeshGraphDescriptor::populate_instance(
    const proto::NodeRef& node_ref, std::vector<GlobalNodeId>& hierarchy) {
    GlobalNodeId global_id;
    if (node_ref.has_mesh()) {
        global_id = populate_mesh_instance(node_ref.mesh(), hierarchy);
    } else if (node_ref.has_graph()) {
        global_id = populate_graph_instance(node_ref.graph(), hierarchy);
    } else if (node_ref.has_switch_()) {
        global_id = populate_switch_instance(node_ref.switch_(), hierarchy);
    } else {
        TT_THROW("Invalid NodeRef: neither mesh, switch, nor graph set");
        return -1;
    }

    auto& instance = instances_.at(global_id);

    // Check that graph descriptor type is not already in the hierarchy
    for (const auto& id : hierarchy) {
        auto& instance_in_hierarchy = instances_.at(id);
        TT_FATAL(
            instance_in_hierarchy.type != instance.type,
            "Graph descriptor type {} already exists in hierarchy",
            instance.type);
    }

    return global_id;
}

GlobalNodeId MeshGraphDescriptor::populate_mesh_instance(
    const proto::MeshRef& mesh_ref, std::vector<GlobalNodeId>& hierarchy) {
    const std::string& descriptor_name = mesh_ref.mesh_descriptor();
    const auto it = mesh_desc_by_name_.find(descriptor_name);
    TT_FATAL(it != mesh_desc_by_name_.end(), "Mesh descriptor {} not found in instance", descriptor_name);
    const auto* mesh_desc = it->second;

    InstanceData data{
        .local_id = static_cast<LocalNodeId>(mesh_ref.mesh_id()),
        .name = mesh_desc->name(),
        .type = "MESH",
        .kind = NodeKind::Mesh,
        .desc = mesh_desc,
    };

    const auto& [instance_it, _] = instances_.emplace(data.global_id, std::move(data));
    auto& instance = instance_it->second;

    instance.hierarchy = hierarchy;

    // Populate devices in the mesh
    uint32_t num_devices = 1;
    for (const auto& dim : mesh_desc->device_topology().dims()) {
        num_devices *= dim;
    }

    instance.sub_instances.reserve(num_devices);
    instance.sub_instances_local_id_to_global_id.reserve(num_devices);

    hierarchy.push_back(instance.global_id);
    for (LocalNodeId i = 0; i < num_devices; ++i) {
        const auto device_id = populate_device_instance(i, hierarchy);

        instance.sub_instances.insert(device_id);
        instance.sub_instances_local_id_to_global_id.emplace(i, device_id);
    }
    hierarchy.pop_back();

    add_to_fast_lookups(instance);

    return instance.global_id;
}

GlobalNodeId MeshGraphDescriptor::populate_switch_instance(
    const proto::SwitchRef& switch_ref, std::vector<GlobalNodeId>& hierarchy) {
    const std::string& descriptor_name = switch_ref.switch_descriptor();
    const auto it = switch_desc_by_name_.find(descriptor_name);
    TT_FATAL(it != switch_desc_by_name_.end(), "Switch descriptor {} not found in instance", descriptor_name);
    const auto* switch_desc = it->second;

    InstanceData data{
        .local_id = static_cast<LocalNodeId>(switch_ref.switch_id()),
        .name = switch_desc->name(),
        .type = "SWITCH",
        .kind = NodeKind::Switch,
        .desc = switch_desc,
    };

    const auto& [instance_it, _] = instances_.emplace(data.global_id, std::move(data));
    auto& instance = instance_it->second;

    instance.hierarchy = hierarchy;

    // Populate devices in the switch
    uint32_t num_devices = 1;
    for (const auto& dim : switch_desc->device_topology().dims()) {
        num_devices *= dim;
    }

    instance.sub_instances.reserve(num_devices);
    instance.sub_instances_local_id_to_global_id.reserve(num_devices);

    hierarchy.push_back(instance.global_id);
    for (LocalNodeId i = 0; i < num_devices; ++i) {
        const auto device_id = populate_device_instance(i, hierarchy);

        instance.sub_instances.insert(device_id);
        instance.sub_instances_local_id_to_global_id.emplace(i, device_id);
    }
    hierarchy.pop_back();

    add_to_fast_lookups(instance);

    return instance.global_id;
}

GlobalNodeId MeshGraphDescriptor::populate_device_instance(LocalNodeId local_id, std::vector<GlobalNodeId>& hierarchy) {
    const std::string name = "D" + std::to_string(local_id);
    InstanceData data{
        .local_id = local_id,
        .name = name,
        .type = "DEVICE",
        .kind = NodeKind::Device,
    };
    const auto global_id = data.global_id;
    instances_.emplace(global_id, std::move(data));
    const auto it_instance = instances_.find(global_id);
    TT_FATAL(it_instance != instances_.end(), "Failed to create device instance for global id {}", global_id);
    auto& instance = it_instance->second;

    instance.hierarchy = hierarchy;

    device_instances_.push_back(instance.global_id);
    instances_by_type_["DEVICE"].push_back(instance.global_id);
    // Use stable storage for key to avoid dangling string_view
    instances_by_name_[instance.name].push_back(instance.global_id);

    return instance.global_id;
}

GlobalNodeId MeshGraphDescriptor::populate_graph_instance(
    const proto::GraphRef& graph_ref, std::vector<GlobalNodeId>& hierarchy) {
    const std::string& descriptor_name = graph_ref.graph_descriptor();
    const auto it = graph_desc_by_name_.find(descriptor_name);
    TT_FATAL(it != graph_desc_by_name_.end(), "Graph descriptor {} not found in instance", descriptor_name);
    const auto* graph_desc = it->second;

    InstanceData data{
        .local_id = static_cast<LocalNodeId>(graph_ref.graph_id()),
        .name = graph_desc->name(),
        .type = graph_desc->type(),
        .kind = NodeKind::Graph,
        .desc = graph_desc,
    };

    const auto emplace_result = instances_.emplace(data.global_id, std::move(data));
    auto& instance = emplace_result.first->second;

    instance.hierarchy = hierarchy;

    // Populate sub-instances from the graph descriptor
    std::unordered_set<GlobalNodeId> children_global_ids;
    children_global_ids.reserve(graph_desc->instances_size());
    instance.sub_instances_local_id_to_global_id.reserve(graph_desc->instances_size());

    hierarchy.push_back(instance.global_id);
    std::string_view child_graph_type;
    for (const auto& sub_ref : graph_desc->instances()) {
        GlobalNodeId child = populate_instance(sub_ref, hierarchy);

        const auto it_child = instances_.find(child);
        TT_FATAL(it_child != instances_.end(), "Child instance id {} not found while populating graph instance", child);
        const auto& child_instance = it_child->second;

        // Check that the child instance created has the same type as rest of the graph descriptor
        if (child_instance.kind == NodeKind::Graph) {
            if (child_graph_type.empty()) {
                child_graph_type = child_instance.type;
            } else {
                TT_FATAL(
                    child_graph_type == child_instance.type,
                    "Graph instance type {} does not match graph descriptor child type {}",
                    std::string(child_graph_type),
                    std::string(child_instance.type));
            }
        }

        TT_FATAL(
            !instance.sub_instances_local_id_to_global_id.contains(child_instance.local_id),
            "Graph instance id {} already exists in this graph",
            child_instance.local_id);

        children_global_ids.insert(child_instance.global_id);
        instance.sub_instances_local_id_to_global_id.emplace(child_instance.local_id, child_instance.global_id);
    }
    hierarchy.pop_back();

    instance.sub_instances = std::move(children_global_ids);

    graph_instances_.push_back(instance.global_id);
    instances_by_type_[graph_desc->type()].push_back(instance.global_id);
    instances_by_name_[graph_desc->name()].push_back(instance.global_id);

    return instance.global_id;
}

void MeshGraphDescriptor::populate_connections() {
    for (const auto& mesh_id : mesh_instances_) {
        populate_intra_mesh_connections(mesh_id);
        populate_intra_mesh_express_connections(mesh_id);
    }
    for (const auto& switch_id : switch_instances_) {
        populate_intra_mesh_connections(switch_id);
        populate_intra_mesh_express_connections(switch_id);
    }
    for (const auto& graph_id : graph_instances_) {
        populate_inter_mesh_connections(graph_id);
    }
}

void MeshGraphDescriptor::add_to_fast_lookups(const InstanceData& instance) {
    // Add to type-based lookup
    instances_by_type_[instance.type].push_back(instance.global_id);

    // Add to name-based lookup
    instances_by_name_[instance.name].push_back(instance.global_id);

    // Add to kind-specific lookups
    switch (instance.kind) {
        case NodeKind::Mesh: mesh_instances_.push_back(instance.global_id); break;
        case NodeKind::Graph: graph_instances_.push_back(instance.global_id); break;
        case NodeKind::Device: device_instances_.push_back(instance.global_id); break;
        case NodeKind::Switch: switch_instances_.push_back(instance.global_id); break;
    }
}

void MeshGraphDescriptor::pre_populate_connections_lookups() {
    for (const auto& [instance_id, instance] : instances_) {
        // Add empty vectors for the instance's type, instance id, and source device id
        if (!connections_by_type_.contains(instance.type)) {
            connections_by_type_.emplace(instance.type, std::vector<ConnectionId>());
        }
        if (!connections_by_instance_id_.contains(instance_id)) {
            connections_by_instance_id_.emplace(instance.global_id, std::vector<ConnectionId>());
        }
        if (!connections_by_source_device_id_.contains(instance_id)) {
            connections_by_source_device_id_.emplace(instance.global_id, std::vector<ConnectionId>());
        }
    }

    if (!connections_by_type_.contains("FABRIC")) {
        connections_by_type_.emplace("FABRIC", std::vector<ConnectionId>());
    }
}

void MeshGraphDescriptor::add_connection_to_fast_lookups(const ConnectionData& connection, const std::string& type) {
    // Add to instance-based lookup
    connections_by_instance_id_[connection.parent_instance_id].push_back(connection.connection_id);

    // Add to type-based lookup
    connections_by_type_[type].push_back(connection.connection_id);

    // Add to source device lookup
    if (!connection.nodes.empty()) {
        connections_by_source_device_id_[connection.nodes[0]].push_back(connection.connection_id);
    }
}

void MeshGraphDescriptor::populate_intra_mesh_connections(GlobalNodeId mesh_id) {
    auto& instance = instances_.at(mesh_id);

    const proto::TorusTopology* device_topology = nullptr;

    if (instance.kind == NodeKind::Mesh) {
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
        device_topology = &mesh_desc->device_topology();
    } else if (instance.kind == NodeKind::Switch) {
        const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(instance.desc);
        device_topology = &switch_desc->device_topology();
    } else {
        TT_THROW("populate_intra_mesh_connections called on non-mesh/non-switch instance");
        return;
    }

    TT_FATAL(device_topology->dims_size() == 2, "MGD currently only supports 2D meshes/switches");

    // TODO: Expand this for 2+ dimensional meshes/switches
    const std::uint32_t mesh_ns_size = device_topology->dims(0);
    const std::uint32_t mesh_ew_size = device_topology->dims(1);
    const auto mesh_shape = MeshShape(mesh_ns_size, mesh_ew_size);

    for (const auto& src_mesh_coord : MeshCoordinateRange(mesh_shape)) {
        const auto connections = get_valid_connections(src_mesh_coord, MeshCoordinateRange(mesh_shape), instance);

        for (const auto& [src_device_id, per_source_connections] : connections) {
            for (const auto& connection_data : per_source_connections) {
                const auto id = connection_data.connection_id;
                add_connection_to_fast_lookups(connection_data, instance.type);
                connections_.emplace(id, connection_data);
            }
        }
    }
}

void MeshGraphDescriptor::populate_intra_mesh_express_connections(GlobalNodeId mesh_id) {
    auto& instance = instances_.at(mesh_id);

    const proto::Channels* channels = nullptr;

    if (instance.kind == NodeKind::Mesh) {
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
        channels = &mesh_desc->channels();
        for (const auto& express_connection : mesh_desc->express_connections()) {
            const auto src_device_id = instance.sub_instances_local_id_to_global_id.at(express_connection.src());
            const auto dst_device_id = instance.sub_instances_local_id_to_global_id.at(express_connection.dst());

            ConnectionData data{
                .nodes = {src_device_id, dst_device_id},
                .count = static_cast<uint32_t>(channels->count()),
                .policy = channels->policy(),
                .parent_instance_id = mesh_id,
                .routing_direction = proto::RoutingDirection::C,  // TODO: Remove after MGD 1.0 is deprecated
            };

            add_connection_to_fast_lookups(data, instance.type);
            connections_.emplace(data.connection_id, std::move(data));

            ConnectionData data_reverse{
                .nodes = {dst_device_id, src_device_id},
                .count = static_cast<uint32_t>(channels->count()),
                .policy = channels->policy(),
                .parent_instance_id = mesh_id,
                .routing_direction = proto::RoutingDirection::C,  // TODO: Remove after MGD 1.0 is deprecated
            };

            add_connection_to_fast_lookups(data_reverse, instance.type);
            connections_.emplace(data_reverse.connection_id, std::move(data_reverse));
        }
    } else if (instance.kind == NodeKind::Switch) {
        const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(instance.desc);
        channels = &switch_desc->channels();
        for (const auto& express_connection : switch_desc->express_connections()) {
            const auto src_device_id = instance.sub_instances_local_id_to_global_id.at(express_connection.src());
            const auto dst_device_id = instance.sub_instances_local_id_to_global_id.at(express_connection.dst());

            ConnectionData data{
                .nodes = {src_device_id, dst_device_id},
                .count = static_cast<uint32_t>(channels->count()),
                .policy = channels->policy(),
                .parent_instance_id = mesh_id,
                .routing_direction = proto::RoutingDirection::C,  // TODO: Remove after MGD 1.0 is deprecated
            };

            add_connection_to_fast_lookups(data, instance.type);
            connections_.emplace(data.connection_id, std::move(data));

            ConnectionData data_reverse{
                .nodes = {dst_device_id, src_device_id},
                .count = static_cast<uint32_t>(channels->count()),
                .policy = channels->policy(),
                .parent_instance_id = mesh_id,
                .routing_direction = proto::RoutingDirection::C,  // TODO: Remove after MGD 1.0 is deprecated
            };

            add_connection_to_fast_lookups(data_reverse, instance.type);
            connections_.emplace(data_reverse.connection_id, std::move(data_reverse));
        }
    }
}

GlobalNodeId MeshGraphDescriptor::find_instance_by_ref(
    GlobalNodeId parent_instance_id, const proto::NodeRef& node_ref) {
    auto& parent_instance = instances_.at(parent_instance_id);

    if (node_ref.has_mesh()) {
        // Check the instance id exists References are indexed by local id
        const auto local_instance_id = node_ref.mesh().mesh_id();
        const auto it2 = parent_instance.sub_instances_local_id_to_global_id.find(local_instance_id);
        TT_FATAL(
            it2 != parent_instance.sub_instances_local_id_to_global_id.end(),
            "Mesh instance id {} not found in parent instance",
            local_instance_id);

        const auto global_instance_id = it2->second;
        auto& referenced_instance = instances_.at(global_instance_id);

        // Check if the mesh descriptor already exists
        const auto descriptor_name = node_ref.mesh().mesh_descriptor();
        TT_FATAL(
            descriptor_name == referenced_instance.name,
            "Mesh descriptor {} does not match referenced instance {}",
            descriptor_name,
            referenced_instance.name);

        // Check sub instance exists
        if (node_ref.mesh().has_device_id()) {
            const auto device_id = node_ref.mesh().device_id();
            auto& mesh_instance = instances_.at(global_instance_id);
            const auto it = mesh_instance.sub_instances_local_id_to_global_id.find(device_id);
            TT_FATAL(
                it != mesh_instance.sub_instances_local_id_to_global_id.end(),
                "Device id {} not found in mesh instance",
                device_id);
            return it->second;
        }

        return global_instance_id;
    }
    if (node_ref.has_switch_()) {
        // Check the instance id exists References are indexed by local id
        const auto local_instance_id = node_ref.switch_().switch_id();
        const auto it2 = parent_instance.sub_instances_local_id_to_global_id.find(local_instance_id);
        TT_FATAL(
            it2 != parent_instance.sub_instances_local_id_to_global_id.end(),
            "Switch instance id {} not found in parent instance",
            local_instance_id);

        const auto global_instance_id = it2->second;
        auto& referenced_instance = instances_.at(global_instance_id);

        // Check if the switch descriptor already exists
        const auto descriptor_name = node_ref.switch_().switch_descriptor();
        TT_FATAL(
            descriptor_name == referenced_instance.name,
            "Switch descriptor {} does not match referenced instance {}",
            descriptor_name,
            referenced_instance.name);

        // Check sub instance exists
        if (node_ref.switch_().has_device_id()) {
            const auto device_id = node_ref.switch_().device_id();
            auto& switch_instance = instances_.at(global_instance_id);
            const auto it = switch_instance.sub_instances_local_id_to_global_id.find(device_id);
            TT_FATAL(
                it != switch_instance.sub_instances_local_id_to_global_id.end(),
                "Device id {} not found in switch instance",
                device_id);
            return it->second;
        }

        return global_instance_id;
    }
    if (node_ref.has_graph()) {
        const auto instance_id = node_ref.graph().graph_id();
        const auto it = parent_instance.sub_instances_local_id_to_global_id.find(instance_id);
        TT_FATAL(
            it != parent_instance.sub_instances_local_id_to_global_id.end(),
            "Graph instance id {} not found in parent instance",
            instance_id);

        const auto global_instance_id = it->second;
        auto& referenced_instance = instances_.at(global_instance_id);

        const auto descriptor_name = node_ref.graph().graph_descriptor();
        TT_FATAL(
            descriptor_name == referenced_instance.name,
            "Graph descriptor {} does not match referenced instance {}",
            descriptor_name,
            referenced_instance.name);

        if (node_ref.graph().has_sub_ref()) {
            return find_instance_by_ref(global_instance_id, node_ref.graph().sub_ref());
        }

        return global_instance_id;
    }
    TT_THROW("Invalid NodeRef: neither mesh, switch, nor graph set");
    return -1;
}

void MeshGraphDescriptor::populate_inter_mesh_connections(GlobalNodeId graph_id) {
    const auto& graph_instance = get_instance(graph_id);
    const auto* const graph_desc = std::get<const proto::GraphDescriptor*>(graph_instance.desc);

    TT_FATAL(graph_desc, "Graph descriptor not found for graph instance {}", graph_id);

    populate_inter_mesh_manual_connections(graph_id);

    if (graph_desc->has_graph_topology()) {
        populate_inter_mesh_topology_connections(graph_id);
    }
}

void MeshGraphDescriptor::populate_inter_mesh_manual_connections(GlobalNodeId graph_id) {
    auto& instance = instances_.at(graph_id);

    const auto* const graph_desc = std::get<const proto::GraphDescriptor*>(instance.desc);

    TT_FATAL(graph_desc, "Graph descriptor not found for graph instance {}", graph_id);

    for (const auto& connection : graph_desc->connections()) {
        std::string_view type;

        std::vector<GlobalNodeId> nodes;

        for (const auto& node : connection.nodes()) {
            // Find the referenced instance
            GlobalNodeId ref_instance_id = find_instance_by_ref(graph_id, node);
            auto& ref_instance = instances_.at(ref_instance_id);

            // Check that the referenced instances have compatible types
            // Allow: MESH-MESH, SWITCH-SWITCH, and MESH-SWITCH connections
            if (type.empty()) {
                type = ref_instance.type;
            } else {
                // Allow connections between meshes and switches (via FABRIC connections)
                bool is_compatible =
                    (type == ref_instance.type) || ((type == "MESH" || type == "SWITCH") &&
                                                    (ref_instance.type == "MESH" || ref_instance.type == "SWITCH"));
                TT_FATAL(
                    is_compatible,
                    "Graph descriptor {} connections must reference instances within same type or compatible types "
                    "(MESH/SWITCH)",
                    instance.name);
            }

            nodes.push_back(ref_instance_id);
        }

        TT_ASSERT(nodes.size() >= 2, "Graph descriptor connections must have at least two nodes");

        // Directional inter-mesh connections are not yet supported end-to-end (issue #50292). Only the authored
        // direction is recorded, so in the control plane the peer endpoint never gathers the physical cable (the
        // two-sided connection_hash join needs both sides) and strict binding resolves 0 routers -> a hard-fatal
        // with a confusing "0 resolved" downstream. Surface it clearly here at parse time. Prefer directional:
        // false until directionality is tracked as a first-class property.
        if (connection.directional()) {
            std::string endpoints;
            for (const auto& node_global_id : nodes) {
                if (!endpoints.empty()) {
                    endpoints += " -> ";
                }
                endpoints += instances_.at(node_global_id).name;
            }
            TT_THROW(
                "Graph descriptor '{}' declares a directional inter-mesh connection ({}). Directional inter-mesh "
                "connections are not fully supported: only the authored direction is stored, so the peer endpoint "
                "will not gather the cable and strict binding will resolve 0 routers. Use directional: false "
                "instead. Tracking: https://github.com/tenstorrent/tt-metal/issues/50292.",
                instance.name,
                endpoints);
        }

        // Add the connection in every direction of the connection
        for (std::size_t i = 0; i < connection.nodes_size(); ++i) {
            // Create a copy of the nodes vector and swap the first and i-th elements so source is always first
            std::vector<GlobalNodeId> nodes_copy = nodes;
            std::swap(nodes_copy[0], nodes_copy[i]);

            ConnectionData data{
                .nodes = nodes_copy,
                .count = static_cast<uint32_t>(connection.channels().count()),
                .policy = connection.channels().policy(),
                .parent_instance_id = graph_id,
                .routing_direction = proto::RoutingDirection::NONE,
                .assign_z_direction = connection.has_assign_z_direction() && connection.assign_z_direction(),
            };

            add_connection_to_fast_lookups(data, instance.type);
            connections_.emplace(data.connection_id, std::move(data));

            if (connection.directional()) {
                break;
            }
        }
    }
}

void MeshGraphDescriptor::populate_inter_mesh_topology_connections(GlobalNodeId graph_id) {
    auto& instance = instances_.at(graph_id);
    const auto* const graph_desc = std::get<const proto::GraphDescriptor*>(instance.desc);

    TT_FATAL(graph_desc, "Graph descriptor not found for graph instance {}", graph_id);

    const auto& graph_topology = graph_desc->graph_topology();

    switch (graph_topology.layout_type()) {
        case proto::GraphTopology::ALL_TO_ALL: populate_inter_mesh_topology_connections_all_to_all(graph_id); break;
        case proto::GraphTopology::RING: populate_inter_mesh_topology_connections_ring(graph_id); break;
        case proto::GraphTopology::INVALID_TYPE:
        case proto::GraphTopology_Type_GraphTopology_Type_INT_MIN_SENTINEL_DO_NOT_USE_:
        case proto::GraphTopology_Type_GraphTopology_Type_INT_MAX_SENTINEL_DO_NOT_USE_:
            // These are protobuf sentinel values that should never be used
            TT_THROW("Invalid graph topology type: {}", graph_topology.layout_type());
            break;
    }
}

void MeshGraphDescriptor::populate_inter_mesh_topology_connections_all_to_all(GlobalNodeId graph_id) {
    // Iterate over all instances in graph
    auto& instance = instances_.at(graph_id);

    const auto* const graph_desc = std::get<const proto::GraphDescriptor*>(instance.desc);

    TT_FATAL(graph_desc, "Graph descriptor not found for graph instance {}", graph_id);

    for (const auto& sub_instance_a : instance.sub_instances) {
        for (const auto& sub_instance_b : instance.sub_instances) {
            if (sub_instance_a == sub_instance_b) {
                continue;
            }

            // Create a connection between the two instances
            ConnectionData data{
                .nodes = {sub_instance_a, sub_instance_b},
                .count = static_cast<uint32_t>(graph_desc->graph_topology().channels().count()),
                .policy = graph_desc->graph_topology().channels().policy(),
                .parent_instance_id = graph_id,
                .routing_direction = proto::RoutingDirection::NONE,
                .assign_z_direction = graph_desc->graph_topology().has_assign_z_direction() &&
                                      graph_desc->graph_topology().assign_z_direction(),
            };

            const auto id = data.connection_id;
            add_connection_to_fast_lookups(data, instance.type);
            connections_.emplace(id, std::move(data));
        }
    }
}

void MeshGraphDescriptor::populate_inter_mesh_topology_connections_ring(GlobalNodeId graph_id) {
    // Iterate over all instances in graph
    auto& instance = instances_.at(graph_id);

    const auto* const graph_desc = std::get<const proto::GraphDescriptor*>(instance.desc);

    TT_FATAL(graph_desc, "Graph descriptor not found for graph instance {}", graph_id);

    std::vector<GlobalNodeId> instances =
        std::vector<GlobalNodeId>(instance.sub_instances.begin(), instance.sub_instances.end());

    for (size_t i = 0; i < instances.size(); ++i) {
        const auto src_instance = instances[i];
        const auto dst_instance = instances[(i + 1) % instances.size()];

        ConnectionData data{
            .nodes = {src_instance, dst_instance},
            .count = static_cast<uint32_t>(graph_desc->graph_topology().channels().count()),
            .policy = graph_desc->graph_topology().channels().policy(),
            .parent_instance_id = graph_id,
            .routing_direction = proto::RoutingDirection::NONE,
            .assign_z_direction = graph_desc->graph_topology().has_assign_z_direction() &&
                                  graph_desc->graph_topology().assign_z_direction(),
        };

        const auto id = data.connection_id;
        add_connection_to_fast_lookups(data, instance.type);
        connections_.emplace(id, std::move(data));

        ConnectionData data_reverse{
            .nodes = {dst_instance, src_instance},
            .count = static_cast<uint32_t>(graph_desc->graph_topology().channels().count()),
            .policy = graph_desc->graph_topology().channels().policy(),
            .parent_instance_id = graph_id,
            .routing_direction = proto::RoutingDirection::NONE,
            .assign_z_direction = graph_desc->graph_topology().has_assign_z_direction() &&
                                  graph_desc->graph_topology().assign_z_direction(),
        };

        const auto id_reverse = data_reverse.connection_id;
        add_connection_to_fast_lookups(data_reverse, instance.type);
        connections_.emplace(id_reverse, std::move(data_reverse));
    }
}

void MeshGraphDescriptor::print_node(GlobalNodeId id, int indent_level) {
    std::string indent(indent_level * 2, ' ');
    std::stringstream ss;

    const auto it = instances_.find(id);
    if (it == instances_.end()) {
        ss << indent << "Unknown instance id: " << id << std::endl;
        log_debug(tt::LogFabric, "{}", ss.str());
        return;
    }

    const InstanceData& inst = it->second;
    if (inst.kind == NodeKind::Mesh) {
        ss << indent << "=== MESH INSTANCE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
        ss << indent << "Local ID: " << inst.local_id << std::endl;
        ss << indent << "Name: " << inst.name << std::endl;
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(inst.desc);
        ss << indent << "Device Topology Dimensions: [";
        for (int i = 0; i < mesh_desc->device_topology().dims_size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << mesh_desc->device_topology().dims(i);
        }
        ss << "]" << std::endl;
        ss << indent << "Host Topology Dimensions: [";
        for (int i = 0; i < mesh_desc->host_topology().dims_size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << mesh_desc->host_topology().dims(i);
        }
        ss << "]" << std::endl;
        ss << indent << "Channel Count: " << mesh_desc->channels().count() << std::endl;
        ss << indent << "Express Connections: " << mesh_desc->express_connections_size() << std::endl;
        if (!inst.sub_instances.empty()) {
            ss << indent << "Devices:" << std::endl;
            log_debug(tt::LogFabric, "{}", ss.str());
            ss.str(std::string());
            // Print devices in ascending local_id order
            std::vector<std::pair<LocalNodeId, GlobalNodeId>> ordered;
            ordered.reserve(inst.sub_instances_local_id_to_global_id.size());
            for (const auto& kv : inst.sub_instances_local_id_to_global_id) {
                ordered.emplace_back(kv.first, kv.second);
            }
            std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& kv : ordered) {
                print_node(kv.second, indent_level + 1);
            }
            return;  // children already printed with their own trailing separators
        }
    } else if (inst.kind == NodeKind::Graph) {
        ss << indent << "=== GRAPH INSTANCE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
        ss << indent << "Local ID: " << inst.local_id << std::endl;
        ss << indent << "Name: " << inst.name << std::endl;
        ss << indent << "Type: " << inst.type << std::endl;
        const auto* graph_desc = std::get<const proto::GraphDescriptor*>(inst.desc);
        ss << indent << "Total Instances in Descriptor: " << graph_desc->instances_size() << std::endl;
        ss << indent << "Connections: " << graph_desc->connections_size() << std::endl;
        if (graph_desc->has_graph_topology()) {
            ss << indent << "Has Graph Topology: Yes" << std::endl;
        }
        if (!inst.sub_instances.empty()) {
            ss << indent << "Sub-instances:" << std::endl;
            log_debug(tt::LogFabric, "{}", ss.str());
            ss.str(std::string());
            // Print children in ascending local_id order
            std::vector<std::pair<LocalNodeId, GlobalNodeId>> ordered;
            ordered.reserve(inst.sub_instances_local_id_to_global_id.size());
            for (const auto& kv : inst.sub_instances_local_id_to_global_id) {
                ordered.emplace_back(kv.first, kv.second);
            }
            std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& kv : ordered) {
                print_node(kv.second, indent_level + 1);
            }
            return;  // children already printed with their own trailing separators
        }
    } else if (inst.kind == NodeKind::Device) {
        ss << indent << "=== DEVICE INSTANCE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
        ss << indent << "Local ID: " << inst.local_id << std::endl;
        ss << indent << "Name: " << inst.name << std::endl;
        ss << indent << "Hierarchy Depth: " << inst.hierarchy.size() << std::endl;
    } else {
        ss << indent << "=== UNKNOWN NODE TYPE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
    }

    ss << indent << "---" << std::endl;
    log_debug(tt::LogFabric, "{}", ss.str());
}

void MeshGraphDescriptor::print_all_nodes() {
    std::stringstream ss;
    ss << "\n=== PRINTING ALL NODE INSTANCES (recursive from top-level) ===" << std::endl;
    ss << "Total instances: " << instances_.size() << std::endl;
    ss << "=====================================" << std::endl;
    log_debug(tt::LogFabric, "{}", ss.str());

    // Start from top-level and recursively print in local-id order
    print_node(top_level_id_, 0);
}

namespace {
struct IdPatternResult {
    std::vector<uint32_t> ids;
    std::string error;
};

std::string trim_id_pattern(const std::string& pattern) {
    auto b = pattern.find_first_not_of(" \t");
    auto e = pattern.find_last_not_of(" \t");
    return (b == std::string::npos) ? std::string{} : pattern.substr(b, e - b + 1);
}

bool is_range_list_pattern(const std::string& p) { return p.find_first_not_of("0123456789,- \t") == std::string::npos; }

// Validate range/list/regex syntax without expanding against a domain.
void validate_id_pattern_syntax(
    const std::string& pattern, const std::string& field_label, std::vector<std::string>& errors) {
    const std::string p = trim_id_pattern(pattern);
    if (p.empty()) {
        errors.push_back(fmt::format("{} is empty", field_label));
        return;
    }

    if (is_range_list_pattern(p)) {
        std::stringstream ss(p);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            const auto dash = tok.find('-');
            try {
                if (dash == std::string::npos) {
                    if (tok.find_first_not_of("0123456789 \t") != std::string::npos || tok.empty()) {
                        errors.push_back(fmt::format("{} has malformed token '{}'", field_label, tok));
                    } else {
                        (void)std::stoul(tok);
                    }
                } else {
                    if (dash == 0 || dash + 1 >= tok.size()) {
                        errors.push_back(fmt::format("{} has malformed range token '{}'", field_label, tok));
                        continue;
                    }
                    const auto lo = static_cast<uint32_t>(std::stoul(tok.substr(0, dash)));
                    const auto hi = static_cast<uint32_t>(std::stoul(tok.substr(dash + 1)));
                    if (lo > hi) {
                        errors.push_back(fmt::format("{} has inverted range '{}' in token '{}'", field_label, p, tok));
                    }
                }
            } catch (const std::exception&) {
                errors.push_back(fmt::format("{} has malformed token '{}'", field_label, tok));
            }
        }
        return;
    }

    try {
        (void)std::regex(p);
    } catch (const std::regex_error& e) {
        errors.push_back(fmt::format("{} has invalid regex '{}': {}", field_label, p, e.what()));
    }
}

// Expand a pinning id pattern against a domain of valid ids. Supports:
//   - inclusive numeric range   "0-8"          -> ids in domain within 0..8
//   - comma list of the above   "0,2,4-6"      -> ids in domain matching any token
//   - std::regex (full match)   "\\d*[02468]"  -> every id whose decimal string matches
// The result is the subset of `domain` (in domain order) selected by the pattern.
IdPatternResult expand_id_pattern(const std::string& pattern, const std::vector<uint32_t>& domain) {
    const std::string p = trim_id_pattern(pattern);
    IdPatternResult result;
    if (p.empty()) {
        result.error = "id pattern is empty";
        return result;
    }

    // Pure digits/commas/dashes -> treat as a range/list; anything else -> regex.
    if (is_range_list_pattern(p)) {
        struct TokenSpec {
            bool is_range = false;
            uint32_t single = 0;
            uint32_t lo = 0;
            uint32_t hi = 0;
        };
        std::vector<TokenSpec> specs;
        std::stringstream ss(p);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            const auto dash = tok.find('-');
            try {
                if (dash == std::string::npos) {
                    if (tok.find_first_not_of("0123456789 \t") != std::string::npos || tok.empty()) {
                        result.error = fmt::format("malformed token '{}'", tok);
                        return result;
                    }
                    specs.push_back({false, static_cast<uint32_t>(std::stoul(tok)), 0, 0});
                } else {
                    if (dash == 0 || dash + 1 >= tok.size()) {
                        result.error = fmt::format("malformed range token '{}'", tok);
                        return result;
                    }
                    const auto lo = static_cast<uint32_t>(std::stoul(tok.substr(0, dash)));
                    const auto hi = static_cast<uint32_t>(std::stoul(tok.substr(dash + 1)));
                    if (lo > hi) {
                        result.error = fmt::format("inverted range in token '{}'", tok);
                        return result;
                    }
                    specs.push_back({true, 0, lo, hi});
                }
            } catch (const std::exception&) {
                result.error = fmt::format("malformed token '{}'", tok);
                return result;
            }
        }

        for (uint32_t id : domain) {
            for (const auto& spec : specs) {
                const bool matches = spec.is_range ? (id >= spec.lo && id <= spec.hi) : (id == spec.single);
                if (matches) {
                    result.ids.push_back(id);
                    break;
                }
            }
        }
        return result;
    }

    try {
        const std::regex re(p);
        for (uint32_t id : domain) {
            if (std::regex_match(std::to_string(id), re)) {
                result.ids.push_back(id);
            }
        }
    } catch (const std::regex_error& e) {
        result.error = fmt::format("invalid regex '{}': {}", p, e.what());
    }
    return result;
}

constexpr uint32_t kPhysicalIdDomainMax = 255;

const std::vector<uint32_t>& physical_id_domain() {
    static const std::vector<uint32_t> domain = []() {
        std::vector<uint32_t> ids;
        ids.reserve(kPhysicalIdDomainMax + 1);
        for (uint32_t i = 0; i <= kPhysicalIdDomainMax; ++i) {
            ids.push_back(i);
        }
        return ids;
    }();
    return domain;
}

std::vector<AsicPosition> expand_physical_asic_positions(
    const google::protobuf::RepeatedPtrField<proto::PhysicalAsicPosition>& physical_positions, std::string& error) {
    const auto& domain = physical_id_domain();
    std::vector<AsicPosition> positions;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (const auto& physical_pos : physical_positions) {
        std::vector<uint32_t> trays;
        if (physical_pos.tray_id_regex().empty()) {
            trays = {physical_pos.tray_id()};
        } else {
            IdPatternResult tray_result = expand_id_pattern(physical_pos.tray_id_regex(), domain);
            if (!tray_result.error.empty()) {
                error = fmt::format("tray_id_regex: {}", tray_result.error);
                return {};
            }
            trays = std::move(tray_result.ids);
        }

        std::vector<uint32_t> asic_locs;
        if (physical_pos.asic_location_regex().empty()) {
            asic_locs = {physical_pos.asic_location()};
        } else {
            IdPatternResult loc_result = expand_id_pattern(physical_pos.asic_location_regex(), domain);
            if (!loc_result.error.empty()) {
                error = fmt::format("asic_location_regex: {}", loc_result.error);
                return {};
            }
            asic_locs = std::move(loc_result.ids);
        }

        for (uint32_t tray : trays) {
            for (uint32_t loc : asic_locs) {
                if (seen.insert({tray, loc}).second) {
                    positions.emplace_back(tt::tt_metal::TrayID{tray}, tt::tt_metal::ASICLocation{loc});
                }
            }
        }
    }
    return positions;
}

bool pinning_entry_uses_regex(const proto::AsicPinning& pinning) {
    for (const auto& n : pinning.logical_fabric_node_id()) {
        if (!n.mesh_id_regex().empty() || !n.chip_id_regex().empty()) {
            return true;
        }
    }
    for (const auto& p : pinning.physical_asic_position()) {
        if (!p.tray_id_regex().empty() || !p.asic_location_regex().empty()) {
            return true;
        }
    }
    return false;
}
}  // namespace

void MeshGraphDescriptor::populate_pinnings() {
    pinnings_.clear();

    // Domain for logical-id regex expansion: instantiated local mesh ids and their chip counts.
    std::map<uint32_t, uint32_t> mesh_chip_count;  // local mesh_id -> chip count
    for (GlobalNodeId gid : mesh_instances_) {
        const auto& inst = get_instance(gid);
        mesh_chip_count[static_cast<uint32_t>(inst.local_id)] = get_chip_count(inst);
    }
    std::vector<uint32_t> all_mesh_ids;
    all_mesh_ids.reserve(mesh_chip_count.size());
    for (const auto& [m, _] : mesh_chip_count) {
        all_mesh_ids.push_back(m);
    }

    // Extract pinnings from the top-level pinnings section, preserving the many-to-many grouping.
    //
    // Each AsicPinning entry may list multiple logical fabric nodes and multiple physical ASIC
    // positions (all-to-all). We keep the group intact here: any listed node may map to any listed
    // position. Downstream consumers enumerate each group into the existing 1:many pinning format --
    // one (fabric_node -> asic_positions) entry per node -- so no downstream interface changes. A
    // single-node/single-position entry reproduces the classic one-to-one pin.
    for (const auto& pinning : proto_->pinnings()) {
        std::string expand_error;
        // Physical positions are shared by every group produced from this entry (regex-expanded when used).
        const std::vector<AsicPosition> positions =
            expand_physical_asic_positions(pinning.physical_asic_position(), expand_error);
        TT_FATAL(expand_error.empty(), "Failed to expand physical ASIC positions: {}", expand_error);

        // Fast path: no regex fields. Still emit one group PER MESH so downstream can look up pins by
        // mesh id (same shape as the regex path) instead of filtering mixed-mesh groups later.
        if (!pinning_entry_uses_regex(pinning)) {
            std::map<uint32_t, std::vector<uint32_t>> mesh_to_chips;
            for (const auto& logical_node_id : pinning.logical_fabric_node_id()) {
                mesh_to_chips[logical_node_id.mesh_id()].push_back(logical_node_id.chip_id());
            }
            for (const auto& [m, chips] : mesh_to_chips) {
                AsicPinningGroup group;
                group.fabric_nodes.reserve(chips.size());
                for (uint32_t c : chips) {
                    group.fabric_nodes.emplace_back(MeshId{m}, c);
                }
                group.asic_positions = positions;
                pinnings_[MeshId{m}].push_back(std::move(group));
            }
            continue;
        }

        // Regex path: expand into concrete (mesh_id -> chips), grouped BY MESH so each matched mesh gets its
        // own all-to-all group (preserving the per-mesh bijection).
        std::map<uint32_t, std::vector<uint32_t>> mesh_to_chips;  // ordered mesh -> ordered unique chips
        std::map<uint32_t, std::set<uint32_t>> seen_chips;
        for (const auto& n : pinning.logical_fabric_node_id()) {
            std::vector<uint32_t> meshes;
            if (n.mesh_id_regex().empty()) {
                meshes = {n.mesh_id()};
            } else {
                IdPatternResult mesh_result = expand_id_pattern(n.mesh_id_regex(), all_mesh_ids);
                TT_FATAL(
                    mesh_result.error.empty(),
                    "Failed to expand mesh_id_regex '{}': {}",
                    n.mesh_id_regex(),
                    mesh_result.error);
                meshes = std::move(mesh_result.ids);
            }
            for (uint32_t m : meshes) {
                std::vector<uint32_t> chips;
                if (n.chip_id_regex().empty()) {
                    chips = {n.chip_id()};
                } else {
                    std::vector<uint32_t> chip_domain;
                    auto it = mesh_chip_count.find(m);
                    const uint32_t cc = (it != mesh_chip_count.end()) ? it->second : 0;
                    chip_domain.reserve(cc);
                    for (uint32_t c = 0; c < cc; ++c) {
                        chip_domain.push_back(c);
                    }
                    IdPatternResult chip_result = expand_id_pattern(n.chip_id_regex(), chip_domain);
                    TT_FATAL(
                        chip_result.error.empty(),
                        "Failed to expand chip_id_regex '{}': {}",
                        n.chip_id_regex(),
                        chip_result.error);
                    chips = std::move(chip_result.ids);
                }
                for (uint32_t c : chips) {
                    if (seen_chips[m].insert(c).second) {
                        mesh_to_chips[m].push_back(c);
                    }
                }
            }
        }
        for (const auto& [m, chips] : mesh_to_chips) {
            AsicPinningGroup group;
            group.fabric_nodes.reserve(chips.size());
            for (uint32_t c : chips) {
                group.fabric_nodes.emplace_back(MeshId{m}, c);
            }
            group.asic_positions = positions;
            pinnings_[MeshId{m}].push_back(std::move(group));
        }
    }

    // A logical fabric node may appear in several pinning groups. Each group is carried through as its own
    // constraint; the consumer filters out whatever is not present on the physical mesh being solved and
    // applies the rest, so a node listed in a wide group and in a 1:1 anchor is narrowed by the anchor.
}

void MeshGraphDescriptor::validate_pinnings(
    const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    for (const auto& pinning : proto.pinnings()) {
        // All-to-all entries must list at least one logical node and at least one physical position.
        if (pinning.logical_fabric_node_id().empty()) {
            error_messages.push_back("Pinning entry has no logical_fabric_node_id");
        }
        if (pinning.physical_asic_position().empty()) {
            error_messages.push_back("Pinning entry has no physical_asic_position");
        }

        // A single pinning entry must not mix regex and non-regex logical_fabric_node_id fields.
        // Allowing both would make expansion ambiguous (literal nodes vs pattern-expanded nodes in
        // the same group).
        bool has_regex_node = false;
        bool has_non_regex_node = false;
        for (const auto& logical_node_id : pinning.logical_fabric_node_id()) {
            if (!logical_node_id.mesh_id_regex().empty() || !logical_node_id.chip_id_regex().empty()) {
                has_regex_node = true;
            } else {
                has_non_regex_node = true;
            }
        }
        if (has_regex_node && has_non_regex_node) {
            error_messages.push_back(
                "Pinning entry mixes regex and non-regex logical_fabric_node_id fields; use separate entries");
        }

        for (const auto& logical_node_id : pinning.logical_fabric_node_id()) {
            if (!logical_node_id.mesh_id_regex().empty()) {
                validate_id_pattern_syntax(
                    logical_node_id.mesh_id_regex(), "logical_fabric_node_id.mesh_id_regex", error_messages);
            }
            if (!logical_node_id.chip_id_regex().empty()) {
                validate_id_pattern_syntax(
                    logical_node_id.chip_id_regex(), "logical_fabric_node_id.chip_id_regex", error_messages);
            }
            if (!logical_node_id.mesh_id_regex().empty() && logical_node_id.has_mesh_id()) {
                error_messages.push_back(
                    "logical_fabric_node_id sets both mesh_id_regex and mesh_id; use one or the other");
            }
            if (!logical_node_id.chip_id_regex().empty() && logical_node_id.has_chip_id()) {
                error_messages.push_back(
                    "logical_fabric_node_id sets both chip_id_regex and chip_id; use one or the other");
            }
        }

        // A single pinning entry must not mix regex and non-regex physical_asic_position fields.
        bool has_regex_physical = false;
        bool has_non_regex_physical = false;
        for (const auto& physical_pos : pinning.physical_asic_position()) {
            if (!physical_pos.tray_id_regex().empty() || !physical_pos.asic_location_regex().empty()) {
                has_regex_physical = true;
            } else {
                has_non_regex_physical = true;
            }
        }
        if (has_regex_physical && has_non_regex_physical) {
            error_messages.push_back(
                "Pinning entry mixes regex and non-regex physical_asic_position fields; use separate entries");
        }

        for (const auto& physical_pos : pinning.physical_asic_position()) {
            if (!physical_pos.tray_id_regex().empty()) {
                validate_id_pattern_syntax(
                    physical_pos.tray_id_regex(), "physical_asic_position.tray_id_regex", error_messages);
            }
            if (!physical_pos.asic_location_regex().empty()) {
                validate_id_pattern_syntax(
                    physical_pos.asic_location_regex(), "physical_asic_position.asic_location_regex", error_messages);
            }
            if (!physical_pos.tray_id_regex().empty() && physical_pos.has_tray_id()) {
                error_messages.push_back(
                    "physical_asic_position sets both tray_id_regex and tray_id; use one or the other");
            }
            if (!physical_pos.asic_location_regex().empty() && physical_pos.has_asic_location()) {
                error_messages.push_back(
                    "physical_asic_position sets both asic_location_regex and asic_location; use one or the other");
            }
        }
    }
}
}  // namespace tt::tt_fabric
