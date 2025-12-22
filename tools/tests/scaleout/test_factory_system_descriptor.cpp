// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <google/protobuf/text_format.h>
#include <yaml-cpp/yaml.h>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include <node/node_types.hpp>

// Include generated protobuf headers
#include "protobuf/deployment.pb.h"
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

// Helper function to create a modified GSD with some connections removed
// Returns path to the modified GSD file
// Removes connections until each ASIC pair has exactly min_connections_to_keep_per_pair remaining
std::string create_gsd_with_missing_connections(
    const std::string& original_gsd_path, const std::string& output_path, uint32_t min_connections_to_keep_per_pair) {
    YAML::Node gsd = YAML::LoadFile(original_gsd_path);

    auto make_asic_key = [](const YAML::Node& endpoint) {
        return endpoint["host_name"].as<std::string>() + "_" + std::to_string(endpoint["tray_id"].as<uint32_t>()) +
               "_" + std::to_string(endpoint["asic_location"].as<uint32_t>());
    };

    auto make_pair_key = [](const std::string& a, const std::string& b) {
        return (a < b) ? (a + "|" + b) : (b + "|" + a);
    };

    // Delete connections in GSD, keeping only min_connections_to_keep_per_pair per ASIC pair
    auto delete_connections_in_gsd = [&](const std::string& connection_type) {
        if (!gsd[connection_type] || gsd[connection_type].IsNull() || gsd[connection_type].size() == 0) {
            return;
        }

        YAML::Node original_connections = gsd[connection_type];
        YAML::Node modified_connections;

        // Track connections per ASIC pair
        std::map<std::string, uint32_t> asic_pair_counts;
        std::map<std::string, uint32_t> asic_pair_kept;

        // First pass: count connections per ASIC pair
        for (const auto& conn : original_connections) {
            std::string key_a = make_asic_key(conn[0]);
            std::string key_b = make_asic_key(conn[1]);
            std::string pair_key = make_pair_key(key_a, key_b);
            asic_pair_counts[pair_key]++;
            asic_pair_kept[pair_key] = 0;
        }

        // Second pass: keep only min_connections_to_keep_per_pair connections per ASIC pair
        for (const auto& conn : original_connections) {
            std::string key_a = make_asic_key(conn[0]);
            std::string key_b = make_asic_key(conn[1]);
            std::string pair_key = make_pair_key(key_a, key_b);

            uint32_t kept_for_pair = asic_pair_kept[pair_key];
            uint32_t total_for_pair = asic_pair_counts[pair_key];

            // Keep connection if we haven't reached the minimum yet
            // If total is less than min, keep all connections (kept_for_pair < total_for_pair)
            // Otherwise, keep up to min_connections_to_keep_per_pair
            uint32_t target_to_keep = std::min(min_connections_to_keep_per_pair, total_for_pair);
            if (kept_for_pair < target_to_keep) {
                modified_connections.push_back(conn);
                asic_pair_kept[pair_key]++;
            }
            // Otherwise, skip this connection (it will be removed)
        }

        gsd[connection_type] = modified_connections;
    };

    // Delete connections in both local and global eth connections
    delete_connections_in_gsd("local_eth_connections");
    delete_connections_in_gsd("global_eth_connections");

    // Write modified GSD to output file
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    std::ofstream out_file(output_path);
    out_file << gsd;
    out_file.close();

    return output_path;
}

static const std::string root_output_dir = "generated/tests/";

// Helper function to create deployment descriptor protobuf object
void create_deployment_descriptor(
    const std::string& node_type_string, tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment) {
    deployment.set_rack_capacity(1);

    auto* host = deployment.add_hosts();
    host->set_hall("0");
    host->set_aisle("0");
    host->set_rack(0);
    host->set_shelf_u(0);
    host->set_node_type(node_type_string);
    host->set_host("host");
}

// Helper function to create cluster config protobuf object for a single node
void create_cluster_config(
    const std::string& node_type_string, tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster) {
    // Create graph template
    auto& graph_template = cluster.mutable_graph_templates()->operator[]("single_" + node_type_string);
    auto* child = graph_template.add_children();
    child->set_name("node1");
    auto* node_ref = child->mutable_node_ref();
    node_ref->set_node_descriptor(node_type_string);

    // Create root instance
    auto* root_instance = cluster.mutable_root_instance();
    root_instance->set_template_name("single_" + node_type_string);
    auto& child_mapping = root_instance->mutable_child_mappings()->operator[]("node1");
    child_mapping.set_host_id(0);
}

// Helper function to serialize protobuf object to a temporary file
template <typename ProtoType>
std::string serialize_proto_to_temp_file(const ProtoType& proto, const std::string& filename) {
    std::string temp_path = root_output_dir + filename;
    std::filesystem::create_directories(std::filesystem::path(temp_path).parent_path());
    std::ofstream output_file(temp_path);
    if (!output_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + temp_path);
    }

    std::string output_string;
    google::protobuf::TextFormat::Printer printer;
    printer.SetUseShortRepeatedPrimitives(true);
    printer.SetUseUtf8StringEscaping(true);
    printer.SetSingleLineMode(false);
    printer.SetPrintMessageFieldsInIndexOrder(true);

    if (!printer.PrintToString(proto, &output_string)) {
        throw std::runtime_error("Failed to write textproto to file: " + temp_path);
    }

    output_file << output_string;
    output_file.close();
    return temp_path;
}

TEST(Cluster, TestFactorySystemDescriptorSingleNodeTypes) {
    for (auto node_type : enchantum::values_generator<NodeType>) {
        auto node_type_string = std::string(enchantum::to_string(node_type));

        // Create protobuf objects
        tt::scaleout_tools::deployment::proto::DeploymentDescriptor deployment;
        tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor cluster;

        create_deployment_descriptor(node_type_string, deployment);
        create_cluster_config(node_type_string, cluster);

        // Serialize to temporary files
        std::string deployment_file =
            serialize_proto_to_temp_file(deployment, node_type_string + "_deployment.textproto");
        std::string cluster_file = serialize_proto_to_temp_file(cluster, node_type_string + "_cluster.textproto");

        std::string fsd_file = root_output_dir + "fsd/factory_system_descriptor_" + node_type_string + ".textproto";

        // Create the cabling generator with temporary file paths
        CablingGenerator cabling_generator(cluster_file, deployment_file);

        // Generate the FSD
        cabling_generator.emit_factory_system_descriptor(fsd_file);
    }
}

TEST(Cluster, TestFactorySystemDescriptor16LB) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto",
        "tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto");

    const std::string fsd_file = root_output_dir + "fsd/factory_system_descriptor_16_n300_lb.textproto";
    cabling_generator.emit_factory_system_descriptor(fsd_file);
    cabling_generator.emit_cabling_guide_csv(root_output_dir + "fsd/cabling_guide_16_n300_lb.csv");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(fsd_file, "tools/tests/scaleout/global_system_descriptors/16_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5LB) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/5_n300_lb_superpod.textproto",
        "tools/tests/scaleout/deployment_descriptors/5_lb_deployment.textproto");

    const std::string fsd_file = root_output_dir + "fsd/factory_system_descriptor_5_n300_lb.textproto";
    cabling_generator.emit_factory_system_descriptor(fsd_file);
    cabling_generator.emit_cabling_guide_csv(root_output_dir + "fsd/cabling_guide_5_n300_lb.csv");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(fsd_file, "tools/tests/scaleout/global_system_descriptors/5_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5WHGalaxyYTorus) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/5_wh_galaxy_y_torus_superpod.textproto",
        "tools/tests/scaleout/deployment_descriptors/5_wh_galaxy_y_torus_deployment.textproto");

    const std::string fsd_file = root_output_dir + "fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto";
    cabling_generator.emit_factory_system_descriptor(fsd_file);

    cabling_generator.emit_cabling_guide_csv(root_output_dir + "fsd/cabling_guide_5_wh_galaxy_y_torus.csv");

    // Validate the FSD against the discovered GSD using the common utility function
    EXPECT_THROW(
        {
            try {
                validate_fsd_against_gsd(
                    fsd_file,
                    "tools/tests/scaleout/global_system_descriptors/"
                    "5_wh_galaxy_y_torus_physical_desc.yaml");
            } catch (const std::runtime_error& e) {
                std::cout << e.what() << std::endl;
                throw;
            }
        },
        std::runtime_error);
}

TEST(Cluster, TestGenerateClusterDescriptorFromFSD) {
    // Generate BH_GALAXY FSD first
    const std::string node_type_string = "BH_GALAXY";

    // Create protobuf objects
    tt::scaleout_tools::deployment::proto::DeploymentDescriptor deployment;
    tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor cluster;

    create_deployment_descriptor(node_type_string, deployment);
    create_cluster_config(node_type_string, cluster);

    std::string deployment_file = serialize_proto_to_temp_file(deployment, node_type_string + "_deployment.textproto");
    std::string cluster_file = serialize_proto_to_temp_file(cluster, node_type_string + "_cluster.textproto");

    std::string fsd_file = root_output_dir + "fsd/factory_system_descriptor_" + node_type_string + ".textproto";

    CablingGenerator cabling_generator(cluster_file, deployment_file);

    cabling_generator.emit_factory_system_descriptor(fsd_file);

    std::string output_dir = root_output_dir + "cluster_descs/";
    std::string base_filename = "cluster_descriptor_" + node_type_string;
    std::string result_file = generate_cluster_descriptor_from_fsd(fsd_file, output_dir, base_filename);

    EXPECT_TRUE(std::filesystem::exists(result_file));
    EXPECT_GT(std::filesystem::file_size(result_file), 0);

    std::cout << "Generated cluster descriptor written to: " << result_file << std::endl;
}

TEST(Cluster, TestGenerateMultiHostClusterDescriptorFromFSD) {
    const std::string cluster_file =
        "tools/tests/scaleout/cabling_descriptors/8x16_wh_galaxy_xy_torus_superpod.textproto";
    const std::string deployment_file =
        "tools/tests/scaleout/deployment_descriptors/8x16_wh_galaxy_xy_torus_deployment.textproto";
    const std::string fsd_file = root_output_dir + "fsd/factory_system_descriptor_8x16_wh_galaxy_xy_torus.textproto";

    CablingGenerator cabling_generator(cluster_file, deployment_file);

    cabling_generator.emit_factory_system_descriptor(fsd_file);

    std::string output_dir = root_output_dir + "cluster_descs/";
    std::string base_filename = "cluster_descriptor_8x16_wh_galaxy_xy_torus";
    std::string result_file = generate_cluster_descriptor_from_fsd(fsd_file, output_dir, base_filename);

    EXPECT_TRUE(std::filesystem::exists(result_file));
    EXPECT_GT(std::filesystem::file_size(result_file), 0);
}

}  // namespace tt::scaleout_tools
