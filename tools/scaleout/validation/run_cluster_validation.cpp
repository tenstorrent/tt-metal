// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <optional>
#include <chrono>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#include <factory_system_descriptor/utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <cabling_generator/cabling_generator.hpp>
#include <tt-metalium/hal.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"
#include "protobuf/mesh_graph_descriptor.pb.h"
#include <google/protobuf/text_format.h>

namespace tt::scaleout_tools {

using tt::tt_metal::PhysicalSystemDescriptor;

// Captures current list of supported input args
struct InputArgs {
    std::optional<std::string> cabling_descriptor_path = std::nullopt;
    std::optional<std::string> deployment_descriptor_path = std::nullopt;
    std::optional<std::string> fsd_path = std::nullopt;
    std::optional<std::string> gsd_path = std::nullopt;
    std::filesystem::path output_path = "";
    bool fail_on_warning = false;
    bool log_ethernet_metrics = false;
    bool print_connectivity = false;
    bool help = false;
    bool send_traffic = false;
    uint32_t data_size = 0;
    uint32_t packet_size_bytes = 64;
    uint32_t num_iterations = 50;
    bool sweep_traffic_configs = false;
    bool validate_connectivity = true;
};

std::filesystem::path generate_output_dir() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string dir_name = ss.str();
    const auto& rt_options = tt::tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir_path = rt_options.get_root_dir() + "cluster_validation_logs/" + dir_name;
    std::filesystem::create_directories(output_dir_path);
    return output_dir_path;
}

InputArgs parse_input_args(const std::vector<std::string>& args_vec) {
    InputArgs input_args;

    if (test_args::has_command_option(args_vec, "--help")) {
        input_args.help = true;
        return input_args;
    }

    if (test_args::has_command_option(args_vec, "--cabling-descriptor-path")) {
        //     TT_FATAL(
        //         test_args::has_command_option(args_vec, "--deployment-descriptor-path"),
        //         "Deployment Descriptor Path is required when Cabling Descriptor Path is provided.");
        input_args.cabling_descriptor_path = test_args::get_command_option(args_vec, "--cabling-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--deployment-descriptor-path")) {
        // TT_FATAL(
        //     input_args.cabling_descriptor_path.has_value(),
        //     "Cabling Descriptor Path is required when Deployment Descriptor Path is provided.");
        input_args.deployment_descriptor_path = test_args::get_command_option(args_vec, "--deployment-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--factory-descriptor-path")) {
        // TT_FATAL(
        //     !(input_args.cabling_descriptor_path.has_value() || input_args.deployment_descriptor_path.has_value()),
        //     "Pass in either Cabling Spec + Deployment Spec or just Factory System Descriptor.");
        input_args.fsd_path = test_args::get_command_option(args_vec, "--factory-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--global-descriptor-path")) {
        input_args.gsd_path = test_args::get_command_option(args_vec, "--global-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--output-path")) {
        input_args.output_path = std::filesystem::path(test_args::get_command_option(args_vec, "--output-path"));
    } else {
        input_args.output_path = generate_output_dir();
    }
    if (test_args::has_command_option(args_vec, "--num-iterations")) {
        input_args.num_iterations = std::stoi(test_args::get_command_option(args_vec, "--num-iterations"));
    }
    if (test_args::has_command_option(args_vec, "--data-size")) {
        // input_args.data_size = std::stoi(test_args::get_command_option(args_vec, "--data-size"));
        // TT_FATAL(
        //     input_args.data_size <= tt::tt_metal::hal::get_erisc_l1_unreserved_size(),
        //     "Data size must be less than or equal to the L1 unreserved size: {} bytes",
        //     tt::tt_metal::hal::get_erisc_l1_unreserved_size());
    } else {
        input_args.data_size = align_down(tt::tt_metal::hal::get_erisc_l1_unreserved_size(), 64);
    }

    if (test_args::has_command_option(args_vec, "--packet-size-bytes")) {
        input_args.packet_size_bytes = std::stoi(test_args::get_command_option(args_vec, "--packet-size-bytes"));
        // TT_FATAL(
        //     input_args.data_size % input_args.packet_size_bytes == 0, "Data size must be divisible by packet size");
        // TT_FATAL(input_args.packet_size_bytes % 16 == 0, "Packet size must be divisible by 16");
    }
    log_output_rank0("Generating System Validation Logs in " + input_args.output_path.string());

    input_args.fail_on_warning = test_args::has_command_option(args_vec, "--hard-fail");
    input_args.log_ethernet_metrics = test_args::has_command_option(args_vec, "--log-ethernet-metrics");
    input_args.print_connectivity = test_args::has_command_option(args_vec, "--print-connectivity");
    input_args.send_traffic = test_args::has_command_option(args_vec, "--send-traffic");
    input_args.sweep_traffic_configs = test_args::has_command_option(args_vec, "--sweep-traffic-configs");
    input_args.validate_connectivity =
        input_args.cabling_descriptor_path.has_value() || input_args.fsd_path.has_value();

    return input_args;
}

std::string get_factory_system_descriptor_path(const InputArgs& input_args) {
    std::string fsd_path;
    if (input_args.cabling_descriptor_path.has_value()) {
        const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
        log_output_rank0("Creating Factory System Descriptor (Golden Representation)");
        tt::scaleout_tools::CablingGenerator cabling_generator(
            input_args.cabling_descriptor_path.value(), input_args.deployment_descriptor_path.value());
        std::string filename =
            "generated_factory_system_descriptor_" + std::to_string(*distributed_context.rank()) + ".textproto";
        fsd_path = input_args.output_path / filename;
        cabling_generator.emit_factory_system_descriptor(fsd_path);
    } else {
        fsd_path = input_args.fsd_path.value();
    }
    return fsd_path;
}

PhysicalSystemDescriptor generate_physical_system_descriptor(const InputArgs& input_args) {
    auto log_hostnames = [&](const std::vector<std::string>& hostnames) {
        std::stringstream ss;
        for (const auto& hostname : hostnames) {
            ss << hostname << ", ";
        }
        return ss.str();
    };

    if (input_args.gsd_path.has_value()) {
        auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(input_args.gsd_path.value());
        log_output_rank0("Detected Hosts: " + log_hostnames(physical_system_descriptor.get_all_hostnames()));
        return physical_system_descriptor;
    } else {
        log_output_rank0("Running Physical Discovery");
        constexpr bool run_discovery = true;
        auto& context = tt::tt_metal::MetalContext::instance();
        const auto& driver = context.get_cluster().get_driver();
        auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
            driver,
            context.get_distributed_context_ptr(),
            &context.hal(),
            context.rtoptions(),
            run_discovery
        );
        log_output_rank0("Physical Discovery Complete");
        log_output_rank0("Detected Hosts: " + log_hostnames(physical_system_descriptor.get_all_hostnames()));
        return physical_system_descriptor;
    }
}

void cleanup_metadata(const InputArgs& input_args, const std::string& gsd_file, const std::string& fsd_file) {
    // Remove GSD file
    std::filesystem::remove(gsd_file);
    if (!input_args.fsd_path.has_value()) {
        // Remove FSD file
        std::filesystem::remove(fsd_file);
    } else {
        TT_FATAL(fsd_file == input_args.fsd_path.value(), "Internal error: Expected FSD File Paths to match");
    }
}

AsicTopology validate_connectivity(const InputArgs& input_args, PhysicalSystemDescriptor& physical_system_descriptor) {
    if (!input_args.validate_connectivity) {
        return {};
    }
    // Set output path for the YAML file
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    std::string gsd_yaml_filename = "global_system_descriptor_" + std::to_string(*distributed_context.rank()) + ".yaml";
    std::string gsd_yaml_path = input_args.output_path / gsd_yaml_filename;
    // Dump the discovered system to YAML
    physical_system_descriptor.dump_to_yaml(gsd_yaml_path);
    log_output_rank0("Validating Factory System Descriptor (Golden Representation) against Global System Descriptor");
    bool log_output = *distributed_context.rank() == 0;
    const auto fsd_path = get_factory_system_descriptor_path(input_args);
    auto missing_physical_connections = tt::scaleout_tools::validate_fsd_against_gsd(
        fsd_path, gsd_yaml_path, true, input_args.fail_on_warning, log_output);
    log_output_rank0("Factory System Descriptor (Golden Representation) Validation Complete");
    // TODO (AS): We shouldn't need to dump files to disk for validation, once validate_fsd_against_gsd can support
    // comparing string representations of the FSD and GSD. For now, each rank dumps a file to disk, which gets deleted
    // post validation (for all ranks except rank 0).
    if (*distributed_context.rank() != 0) {
        cleanup_metadata(input_args, gsd_yaml_path, fsd_path);
    }
    return generate_asic_topology_from_connections(missing_physical_connections, physical_system_descriptor);
}

void print_usage_info() {
    std::cout << "Utility to validate Ethernet Links and Connections for a Multi-Node TT Cluster" << std::endl;
    std::cout << "Compares live system state against the requested Cabling and Deployment Specifications" << std::endl
              << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  --cabling-descriptor-path: Path to cabling descriptor" << std::endl;
    std::cout << "  --deployment-descriptor-path: Path to deployment descriptor" << std::endl;
    std::cout << "  --factory-descriptor-path: Path to factory descriptor" << std::endl;
    std::cout << "  --global-descriptor-path: Path to global descriptor" << std::endl;
    std::cout << "  --output-path: Path to output directory" << std::endl;
    std::cout << "  --hard-fail: Fail on warning" << std::endl;
    std::cout << "  --log-ethernet-metrics: Log live ethernet statistics" << std::endl;
    std::cout << "  --print-connectivity: Print Ethernet Connectivity between ASICs" << std::endl;
    std::cout << "  --send-traffic: Send traffic across detected links" << std::endl;
    std::cout << "  --num-iterations: Number of iterations to send traffic" << std::endl;
    std::cout << "  --data-size: Data size (bytes) sent across each link per iteration" << std::endl;
    std::cout << "  --packet-size-bytes: Packet size (bytes) sent across each link" << std::endl;
    std::cout << "  --sweep-traffic-configs: Sweep pre-generated traffic configurations across detected links (stress "
                 "testing)"
              << std::endl;
    std::cout << "  --help: Print usage information" << std::endl << std::endl;
    std::cout << "To run on a multi-node cluster, use mpirun with a --hostfile option" << std::endl;
}

void set_config_vars() {
    // This tool must be run with slow dispatch mode, since
    // it writes custom kernels to ethernet cores, which shouldn't
    // be running fabric routers
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);

    // Only set these if they are not already set
    if (getenv("TT_MESH_HOST_RANK") == nullptr) {
        setenv("TT_MESH_HOST_RANK", "0", 1);
    }
    if (getenv("TT_MESH_ID") == nullptr) {
        setenv("TT_MESH_ID", "0", 1);
    }
}

}  // namespace tt::scaleout_tools

int main(int argc, char* argv[]) {
    using namespace tt::scaleout_tools;

    set_config_vars();

    auto input_args = parse_input_args(std::vector<std::string>(argv, argv + argc));
    std::vector<std::string> hostnames = {
        "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15"};
    std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> intermesh_connections;

    auto cabling_descriptor =
        tt::scaleout_tools::CablingGenerator(input_args.cabling_descriptor_path.value(), hostnames);
    auto hosts = cabling_descriptor.get_deployment_hosts();
    auto chip_connections = cabling_descriptor.get_chip_connections();

    for (const auto& host : hosts) {
        std::cout << "Host: " << host.hostname << " Cluster Type: " << host.node_type << std::endl;
    }
    for (const auto& connection : chip_connections) {
        if (connection.first.host_id == connection.second.host_id) {
            continue;
        }
        intermesh_connections[hostnames[*connection.first.host_id]][hostnames[*connection.second.host_id]]++;
    }

    for (const auto& [src_host, conn] : intermesh_connections) {
        for (const auto& [dst_host, count] : conn) {
            std::cout << "Intermesh Connection: " << src_host << " -> " << dst_host << " Count: " << count << std::endl;
        }
    }

    // Create lookup table for node types to shapes, architectures, and channel counts
    struct NodeTypeInfo {
        std::vector<int> device_dims;
        tt::tt_fabric::proto::Architecture arch;
        int channel_count;
    };

    std::unordered_map<std::string, NodeTypeInfo> node_type_lookup = {
        // Wormhole architectures
        {"N300_LB_DEFAULT", {{2, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 2}},     // N300: 2 channels
        {"N300_QB_DEFAULT", {{2, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 2}},     // N300: 2 channels
        {"WH_GALAXY", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},           // WH Galaxy: 4 channels
        {"WH_GALAXY_X_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},   // WH Galaxy: 4 channels
        {"WH_GALAXY_Y_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},   // WH Galaxy: 4 channels
        {"WH_GALAXY_XY_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::WORMHOLE_B0, 4}},  // WH Galaxy: 4 channels

        // Blackhole architectures
        {"P150_LB", {{2, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 4}},             // P150: 4 channels
        {"P150_QB_AE_DEFAULT", {{2, 2}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 4}},  // P150: 4 channels
        {"P300_QB_GE", {{2, 2}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},          // P300: 2 channels
        {"BH_GALAXY", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},           // BH Galaxy: 2 channels
        {"BH_GALAXY_X_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},   // BH Galaxy: 2 channels
        {"BH_GALAXY_Y_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},   // BH Galaxy: 2 channels
        {"BH_GALAXY_XY_TORUS", {{8, 4}, tt::tt_fabric::proto::Architecture::BLACKHOLE, 2}},  // BH Galaxy: 2 channels
    };

    // Generate Mesh Graph Descriptor using protobuf
    tt::tt_fabric::proto::MeshGraphDescriptor mgd;

    // Determine common architecture, device topology, and channel count from first host
    tt::tt_fabric::proto::Architecture arch = tt::tt_fabric::proto::Architecture::WORMHOLE_B0;
    std::vector<int> device_dims = {1, 1};  // Default fallback
    int channel_count = 2;                  // Default channel count

    if (!hosts.empty()) {
        const auto& first_host = hosts[0];
        auto it = node_type_lookup.find(first_host.node_type);
        if (it != node_type_lookup.end()) {
            device_dims = it->second.device_dims;
            arch = it->second.arch;
            channel_count = it->second.channel_count;
        } else {
            std::cerr << "Warning: Unknown node type '" << first_host.node_type
                      << "', using default 1x1 topology and 2 channels" << std::endl;
        }
    }

    // Create a single mesh descriptor that will be reused for all instances
    auto* mesh_desc = mgd.add_mesh_descriptors();
    mesh_desc->set_name("M0");  // Use M0 like in the example
    mesh_desc->set_arch(arch);

    // Set device topology
    auto* device_topo = mesh_desc->mutable_device_topology();
    for (int dim : device_dims) {
        device_topo->add_dims(dim);
    }

    // Set host topology
    auto* host_topo = mesh_desc->mutable_host_topology();
    host_topo->add_dims(1);
    host_topo->add_dims(1);

    // Set channels using the value from the lookup table
    auto* channels = mesh_desc->mutable_channels();
    channels->set_count(channel_count);
    channels->set_policy(tt::tt_fabric::proto::Policy::STRICT);

    // Create graph descriptor
    auto* graph_desc = mgd.add_graph_descriptors();
    graph_desc->set_name("G0");
    graph_desc->set_type("FABRIC");

    // Add mesh instances (all using the same mesh descriptor)
    for (size_t i = 0; i < hosts.size(); ++i) {
        auto* instance = graph_desc->add_instances();
        auto* mesh_ref = instance->mutable_mesh();
        mesh_ref->set_mesh_descriptor("M0");
        mesh_ref->set_mesh_id(i);
    }

    // Add connections between meshes
    for (const auto& [src_host, conn] : intermesh_connections) {
        for (const auto& [dst_host, count] : conn) {
            // Find host indices
            int src_idx = -1, dst_idx = -1;
            for (size_t i = 0; i < hosts.size(); ++i) {
                if (hosts[i].hostname == src_host) {
                    src_idx = i;
                }
                if (hosts[i].hostname == dst_host) {
                    dst_idx = i;
                }
            }

            if (src_idx >= 0 && dst_idx >= 0) {
                auto* connection = graph_desc->add_connections();

                // Add source node
                auto* src_node = connection->add_nodes();
                auto* src_mesh_ref = src_node->mutable_mesh();
                src_mesh_ref->set_mesh_descriptor("M0");
                src_mesh_ref->set_mesh_id(src_idx);

                // Add destination node
                auto* dst_node = connection->add_nodes();
                auto* dst_mesh_ref = dst_node->mutable_mesh();
                dst_mesh_ref->set_mesh_descriptor("M0");
                dst_mesh_ref->set_mesh_id(dst_idx);

                // Set channels (no policy for cleaner output)
                auto* conn_channels = connection->mutable_channels();
                conn_channels->set_count(count);
            }
        }
    }

    // Set top-level instance
    auto* top_level = mgd.mutable_top_level_instance();
    auto* graph_ref = top_level->mutable_graph();
    graph_ref->set_graph_descriptor("G0");
    graph_ref->set_graph_id(0);

    // Write to file with custom formatting to match the expected style
    std::string mgd_output_path = "mesh_graph_descriptor.textproto";
    std::ofstream mgd_file(mgd_output_path);
    if (!mgd_file.is_open()) {
        std::cerr << "Failed to open file for writing: " << mgd_output_path << std::endl;
        return -1;
    }

    // Write Meshes section
    mgd_file << "# --- Meshes ---------------------------------------------------------------\n\n";
    mgd_file << "mesh_descriptors {\n";
    mgd_file << "  name: \"" << mesh_desc->name() << "\"\n";
    mgd_file << "  arch: " << tt::tt_fabric::proto::Architecture_Name(mesh_desc->arch()) << "\n";
    mgd_file << "  device_topology { dims: [ ";
    for (int i = 0; i < device_topo->dims_size(); ++i) {
        if (i > 0) {
            mgd_file << ", ";
        }
        mgd_file << device_topo->dims(i);
    }
    mgd_file << " ] }\n";
    mgd_file << "  host_topology   { dims: [ ";
    for (int i = 0; i < host_topo->dims_size(); ++i) {
        if (i > 0) {
            mgd_file << ", ";
        }
        mgd_file << host_topo->dims(i);
    }
    mgd_file << " ] }\n";
    mgd_file << "  channels {\n";
    mgd_file << "    count: " << channels->count() << "\n";
    mgd_file << "    policy: " << tt::tt_fabric::proto::Policy_Name(channels->policy()) << "\n";
    mgd_file << "  }\n";
    mgd_file << "}\n\n";

    // Write Graphs section
    mgd_file << "# --- Graphs ---------------------------------------------------------------\n\n";
    mgd_file << "graph_descriptors {\n";
    mgd_file << "  name: \"" << graph_desc->name() << "\"\n";
    mgd_file << "  type: \"" << graph_desc->type() << "\"\n";

    // Write instances in compact format
    mgd_file << "  # Instances: mesh ids 0";
    for (size_t i = 1; i < hosts.size(); ++i) {
        mgd_file << "," << i;
    }
    mgd_file << " (all " << device_dims[0] << "x" << device_dims[1] << ")\n";

    for (int i = 0; i < graph_desc->instances_size(); ++i) {
        const auto& instance = graph_desc->instances(i);
        mgd_file << "  instances { mesh { mesh_descriptor: \"" << instance.mesh().mesh_descriptor()
                 << "\" mesh_id: " << instance.mesh().mesh_id() << " } }\n";
    }

    // Collect and sort connections for ordered output
    struct ConnectionInfo {
        int src_id;
        int dst_id;
        int channel_count;
    };
    std::vector<ConnectionInfo> sorted_connections;

    for (int i = 0; i < graph_desc->connections_size(); ++i) {
        const auto& connection = graph_desc->connections(i);
        if (connection.nodes_size() == 2) {
            int id1 = connection.nodes(0).mesh().mesh_id();
            int id2 = connection.nodes(1).mesh().mesh_id();
            // Store with smaller id first for consistent ordering
            sorted_connections.push_back({std::min(id1, id2), std::max(id1, id2), connection.channels().count()});
        }
    }

    // Sort connections by source id first, then by destination id
    std::sort(
        sorted_connections.begin(), sorted_connections.end(), [](const ConnectionInfo& a, const ConnectionInfo& b) {
            if (a.src_id != b.src_id) {
                return a.src_id < b.src_id;
            }
            return a.dst_id < b.dst_id;
        });

    // Write sorted connections
    mgd_file << "\n";
    for (const auto& conn : sorted_connections) {
        mgd_file << "  # M" << conn.src_id << " <-> M" << conn.dst_id << "\n";
        mgd_file << "  connections {\n";
        mgd_file << "    nodes { mesh { mesh_descriptor: \"M0\" mesh_id: " << conn.src_id << " } }\n";
        mgd_file << "    nodes { mesh { mesh_descriptor: \"M0\" mesh_id: " << conn.dst_id << " } }\n";
        mgd_file << "    channels { count: " << conn.channel_count << " }\n";
        mgd_file << "  }\n";
    }

    mgd_file << "}\n\n";

    // Write Instantiation section
    mgd_file << "# --- Instantiation ----------------------------------------------------------\n";
    mgd_file << "top_level_instance { graph { graph_descriptor: \"" << top_level->graph().graph_descriptor()
             << "\" graph_id: " << top_level->graph().graph_id() << " } }\n";

    mgd_file.close();

    std::cout << "\nMesh Graph Descriptor written to: " << mgd_output_path << std::endl;

    exit(0);

    if (input_args.help) {
        print_usage_info();
        return 0;
    }

    bool eth_connections_healthy = true;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    // Create physical system descriptor and discover the system
    auto physical_system_descriptor = generate_physical_system_descriptor(input_args);

    AsicTopology missing_asic_topology = validate_connectivity(input_args, physical_system_descriptor);
    bool links_reset = false;
    // Ethernet Link Retraining through SW is currently only supported for Wormhole
    bool link_retrain_supported = tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::WORMHOLE_B0;
    constexpr uint32_t MAX_RETRAINS_BEFORE_FAILURE =
        5;  // If links don't come up after 5 retrains, the system is in an unrecoverable state.
    uint32_t num_retrains = 0;
    while (!missing_asic_topology.empty() && link_retrain_supported && num_retrains < MAX_RETRAINS_BEFORE_FAILURE) {
        reset_ethernet_links(physical_system_descriptor, missing_asic_topology);
        links_reset = true;
        num_retrains++;
        physical_system_descriptor.run_discovery(true, true);
        missing_asic_topology = validate_connectivity(input_args, physical_system_descriptor);
    }

    if (num_retrains == MAX_RETRAINS_BEFORE_FAILURE && !missing_asic_topology.empty()) {
        TT_THROW("Encountered unrecoverable state. Please check the system and try again.");
        return -1;
    }
    if (links_reset) {
        log_output_rank0("Ethernet Links were Retrained. Please run the validation tool again to issue traffic.");
        return 0;
    }

    eth_connections_healthy = generate_link_metrics(
        physical_system_descriptor,
        input_args.num_iterations,
        input_args.log_ethernet_metrics,
        input_args.send_traffic,
        input_args.sweep_traffic_configs,
        input_args.packet_size_bytes,
        input_args.data_size,
        input_args.output_path);

    if (*distributed_context.rank() == 0 && input_args.print_connectivity) {
        print_ethernet_connectivity(input_args.print_connectivity, physical_system_descriptor);
    }
    distributed_context.barrier();
    if (input_args.fail_on_warning && !eth_connections_healthy) {
        TT_THROW("Encountered unhealthy ethernet connections, listed above");
        return -1;
    }
    return 0;
}
