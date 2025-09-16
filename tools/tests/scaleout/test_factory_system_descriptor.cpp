// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include <node/node_types.hpp>

namespace tt::scaleout_tools {

// Helper function to generate deployment descriptor textproto content
std::string generate_deployment_descriptor(const std::string& node_type_string) {
    std::ostringstream oss;
    oss << "hosts: {\n";
    oss << "  hall: \"0\"\n";
    oss << "  aisle: \"0\"\n";
    oss << "  rack: 0\n";
    oss << "  shelf_u: 0\n";
    oss << "  node_type: \"" << node_type_string << "\"\n";
    oss << "  host: \"host\"\n";
    oss << "}\n";
    return oss.str();
}

// Helper function to generate cluster config textproto content for a single node
std::string generate_cluster_config(const std::string& node_type_string) {
    std::ostringstream oss;
    oss << "graph_templates {\n";
    oss << "  key: \"single_" << node_type_string << "\"\n";
    oss << "  value {\n";
    oss << "    children {\n";
    oss << "      name: \"node1\"\n";
    oss << "      node_ref { node_descriptor: \"" << node_type_string << "\" }\n";
    oss << "    }\n";
    oss << "  }\n";
    oss << "}\n";
    oss << "\n";
    oss << "# Root instance with concrete host mapping\n";
    oss << "root_instance {\n";
    oss << "  template_name: \"single_" << node_type_string << "\"\n";
    oss << "  child_mappings {\n";
    oss << "    key: \"node1\"\n";
    oss << "    value { host_id: 0 }\n";
    oss << "  }\n";
    oss << "}\n";
    return oss.str();
}

// Helper function to write content to a temporary file
std::string write_temp_file(const std::string& content, const std::string& suffix) {
    std::string temp_path = std::filesystem::temp_directory_path() / ("temp_" + suffix);
    std::ofstream file(temp_path);
    file << content;
    file.close();
    return temp_path;
}

TEST(Cluster, TestFactorySystemDescriptorSingleNodeTypes) {
    for (auto node_type : enchantum::values_generator<NodeType>) {
        auto node_type_string = std::string(enchantum::to_string(node_type));

        // Generate temporary file names
        std::string deployment_file = write_temp_file(
            generate_deployment_descriptor(node_type_string), node_type_string + "_deployment.textproto");

        std::string cluster_file =
            write_temp_file(generate_cluster_config(node_type_string), node_type_string + "_cluster.textproto");

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
