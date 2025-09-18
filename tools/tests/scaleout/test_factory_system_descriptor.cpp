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
std::string serialize_proto_to_temp_file(const ProtoType& proto, const std::string& suffix) {
    std::string temp_path = std::filesystem::temp_directory_path() / ("temp_" + suffix);

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

        std::string fsd_file = "fsd/factory_system_descriptor_" + node_type_string + ".textproto";

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

    cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_16_n300_lb.textproto");
    cabling_generator.emit_cabling_guide_csv("fsd/cabling_guide_16_n300_lb.csv");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_16_n300_lb.textproto",
        "tools/tests/scaleout/global_system_descriptors/16_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5LB) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/5_n300_lb_superpod.textproto",
        "tools/tests/scaleout/deployment_descriptors/5_lb_deployment.textproto");

    // Generate the FSD (textproto format)
    cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_5_n300_lb.textproto");
    cabling_generator.emit_cabling_guide_csv("fsd/cabling_guide_5_n300_lb.csv");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_5_n300_lb.textproto",
        "tools/tests/scaleout/global_system_descriptors/5_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5WHGalaxyYTorus) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/5_wh_galaxy_y_torus_superpod.textproto",
        "tools/tests/scaleout/deployment_descriptors/5_wh_galaxy_y_torus_deployment.textproto");

    // Generate the FSD (textproto format)
    cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto");

    cabling_generator.emit_cabling_guide_csv("fsd/cabling_guide_5_wh_galaxy_y_torus.csv");

    // Validate the FSD against the discovered GSD using the common utility function
    EXPECT_THROW(
        {
            try {
                validate_fsd_against_gsd(
                    "fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto",
                    "tools/tests/scaleout/global_system_descriptors/"
                    "5_wh_galaxy_y_torus_physical_desc.yaml");
            } catch (const std::runtime_error& e) {
                std::cout << e.what() << std::endl;
                throw;
            }
        },
        std::runtime_error);
}

}  // namespace tt::scaleout_tools
