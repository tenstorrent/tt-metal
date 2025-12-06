// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <google/protobuf/text_format.h>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include <node/node_types.hpp>

// Include generated protobuf headers
#include "protobuf/deployment.pb.h"
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

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

// ============================================================================
// Tests for multi-path and directory-based descriptor merging
// ============================================================================

// Helper to create a 4-host deployment descriptor for merge tests
void create_4host_deployment_descriptor(tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment) {
    deployment.set_rack_capacity(4);

    for (int i = 0; i < 4; ++i) {
        auto* host = deployment.add_hosts();
        host->set_hall("0");
        host->set_aisle("0");
        host->set_rack(0);
        host->set_shelf_u(i);
        host->set_node_type("BH_GALAXY");
        host->set_host("host" + std::to_string(i));
    }
}

TEST(Cluster, TestCablingGeneratorWithMultiplePaths) {
    // Test constructing CablingGenerator with multiple explicit descriptor paths
    std::vector<std::string> cluster_paths = {
        "tools/tests/scaleout/cabling_descriptors/merge_tests/base_intrapod.textproto",
        "tools/tests/scaleout/cabling_descriptors/merge_tests/additional_interpod.textproto"};

    // Create a deployment descriptor for 4 hosts
    tt::scaleout_tools::deployment::proto::DeploymentDescriptor deployment;
    create_4host_deployment_descriptor(deployment);
    std::string deployment_file = serialize_proto_to_temp_file(deployment, "merge_test_deployment.textproto");

    // Create the cabling generator with multiple paths
    CablingGenerator cabling_generator(cluster_paths, deployment_file);

    // Should have 4 hosts
    EXPECT_EQ(cabling_generator.get_deployment_hosts().size(), 4);

    // Should have chip connections (the merged connections)
    const auto& connections = cabling_generator.get_chip_connections();
    EXPECT_GT(connections.size(), 0) << "Expected merged connections to produce chip connections";

    // Generate FSD to verify it works end-to-end
    const std::string fsd_file = root_output_dir + "fsd/factory_system_descriptor_merged_multipaths.textproto";
    cabling_generator.emit_factory_system_descriptor(fsd_file);
    EXPECT_TRUE(std::filesystem::exists(fsd_file));
}

TEST(Cluster, TestCablingGeneratorWithMultiplePathsHostnames) {
    // Test constructing CablingGenerator with multiple paths and hostnames (no deployment descriptor)
    std::vector<std::string> cluster_paths = {
        "tools/tests/scaleout/cabling_descriptors/merge_tests/base_intrapod.textproto",
        "tools/tests/scaleout/cabling_descriptors/merge_tests/additional_interpod.textproto"};

    std::vector<std::string> hostnames = {"host0", "host1", "host2", "host3"};

    // Create the cabling generator with multiple paths and hostnames
    CablingGenerator cabling_generator(cluster_paths, hostnames);

    // Should have 4 hosts
    EXPECT_EQ(cabling_generator.get_deployment_hosts().size(), 4);

    // Verify hostnames are correctly set
    const auto& hosts = cabling_generator.get_deployment_hosts();
    for (size_t i = 0; i < hostnames.size(); ++i) {
        EXPECT_EQ(hosts[i].hostname, hostnames[i]);
    }
}

TEST(Cluster, TestCablingGeneratorWithConflictingDescriptors) {
    // Test that conflicting descriptors throw an error
    std::vector<std::string> cluster_paths = {
        "tools/tests/scaleout/cabling_descriptors/merge_tests/base_intrapod.textproto",
        "tools/tests/scaleout/cabling_descriptors/merge_tests/conflicting_connection.textproto"};

    tt::scaleout_tools::deployment::proto::DeploymentDescriptor deployment;
    create_4host_deployment_descriptor(deployment);
    std::string deployment_file = serialize_proto_to_temp_file(deployment, "merge_test_conflict_deployment.textproto");

    // Should throw due to conflicting connections
    EXPECT_THROW({ CablingGenerator cabling_generator(cluster_paths, deployment_file); }, std::runtime_error);
}

TEST(Cluster, TestCablingGeneratorMergeWithDifferentTemplates) {
    // Test merging descriptors that add completely different graph templates
    std::vector<std::string> cluster_paths = {
        "tools/tests/scaleout/cabling_descriptors/merge_tests/base_intrapod.textproto",
        "tools/tests/scaleout/cabling_descriptors/merge_tests/different_template.textproto"};

    tt::scaleout_tools::deployment::proto::DeploymentDescriptor deployment;
    create_4host_deployment_descriptor(deployment);
    std::string deployment_file =
        serialize_proto_to_temp_file(deployment, "merge_test_difftemplate_deployment.textproto");

    // Should succeed - different templates don't conflict
    CablingGenerator cabling_generator(cluster_paths, deployment_file);

    EXPECT_EQ(cabling_generator.get_deployment_hosts().size(), 4);
}

}  // namespace tt::scaleout_tools
