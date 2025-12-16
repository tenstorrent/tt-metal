// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <stdexcept>
#include <google/protobuf/text_format.h>
#include <cxxopts.hpp>

#include "protobuf/cluster_config.pb.h"
#include "protobuf/node_config.pb.h"

struct InputConfig {
    int nodes_0{};
    int nodes_1{};
    bool torus_0{};
    bool torus_1{};
    std::string galaxy_structure;
    std::string topology;
    std::string galaxy_type;
};

InputConfig parse_arguments(int argc, char** argv) {
    cxxopts::Options options("2d_big_mesh_cabling_gen", "Generate 2D big mesh cabling configuration");

    options.add_options()
        ("g,galaxy-structure", "Galaxy structure in format 'NxM' where N and M are positive integers. N must be divisible by 8, M must be divisible by 4", cxxopts::value<std::string>())
        ("t,topology", "Torus configuration: '10' (torus in first dimension), '01' (torus in second dimension), '11' (torus in both dimensions), '00' (mesh topology)", cxxopts::value<std::string>())
        ("y,galaxy-type", "Galaxy type: 'WH_GALAXY' or 'BH_GALAXY'", cxxopts::value<std::string>()->default_value("WH_GALAXY"))
        ("h,help", "Print usage information");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (!result.count("galaxy-structure")) {
            throw std::invalid_argument("Galaxy structure is required");
        }

        if (!result.count("topology")) {
            throw std::invalid_argument("Topology is required");
        }

        InputConfig config;
        config.galaxy_structure = result["galaxy-structure"].as<std::string>();
        config.topology = result["topology"].as<std::string>();
        config.galaxy_type = result["galaxy-type"].as<std::string>();

        // Parse galaxy structure "NxM"
        size_t x_pos = config.galaxy_structure.find('x');
        if (x_pos == std::string::npos) {
            throw std::invalid_argument(
                "Galaxy structure must be in format 'NxM' (e.g., '8x4'), got: '" + config.galaxy_structure + "'");
        }

        try {
            config.nodes_0 = std::stoi(config.galaxy_structure.substr(0, x_pos));
            config.nodes_1 = std::stoi(config.galaxy_structure.substr(x_pos + 1));
        } catch (const std::exception& e) {
            throw std::invalid_argument(
                "Invalid numbers in galaxy structure '" + config.galaxy_structure + "': " + e.what());
        }

        // Validate node counts
        if (config.nodes_0 <= 0 || config.nodes_1 <= 0) {
            throw std::invalid_argument(
                "Node counts must be positive integers, got: " + std::to_string(config.nodes_0) + "x" +
                std::to_string(config.nodes_1));
        }

        if (config.nodes_0 % 8 != 0) {
            throw std::invalid_argument(
                "First dimension (" + std::to_string(config.nodes_0) + ") must be divisible by 8 for 8x4 Galaxy basis");
        }

        if (config.nodes_1 % 4 != 0) {
            throw std::invalid_argument(
                "Second dimension (" + std::to_string(config.nodes_1) + ") must be divisible by 4 for 8x4 Galaxy basis");
        }

        // Parse topology
        config.torus_0 = false;
        config.torus_1 = false;

        if (config.topology == "10") {
            config.torus_0 = true;
        } else if (config.topology == "01") {
            config.torus_1 = true;
        } else if (config.topology == "11") {
            config.torus_0 = true;
            config.torus_1 = true;
        } else if (config.topology == "00") {
            // Mesh topology - no torus
        } else {
            throw std::invalid_argument(
                "Invalid topology '" + config.topology + "'. Must be '10', '01', '11', or '00' (for mesh)");
        }

        // Validate galaxy type
        if (config.galaxy_type != "WH_GALAXY" && config.galaxy_type != "BH_GALAXY") {
            throw std::invalid_argument(
                "Invalid galaxy type '" + config.galaxy_type + "'. Must be 'WH_GALAXY' or 'BH_GALAXY'");
        }

        return config;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    try {
        InputConfig config = parse_arguments(argc, argv);

        int galaxy_nodes_0 = config.nodes_0 / 8;
        int galaxy_nodes_1 = config.nodes_1 / 4;

        std::string graph_template_name = "big_mesh_" + config.galaxy_structure;
        const std::string dim_1_graph_template_name = "big_mesh_dim1";

        tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor big_mesh_descriptor =
            tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor();

        auto* root_instance_ptr = big_mesh_descriptor.mutable_root_instance();
        auto* graph_templates_ptr = big_mesh_descriptor.mutable_graph_templates();
        root_instance_ptr->set_template_name(graph_template_name);

        graph_templates_ptr->insert(
            {dim_1_graph_template_name, tt::scaleout_tools::cabling_generator::proto::GraphTemplate()});
        graph_templates_ptr->insert(
            {graph_template_name, tt::scaleout_tools::cabling_generator::proto::GraphTemplate()});

        // Scope block for dim_1_graph_template setup
        {
            auto& graph_template_instance = graph_templates_ptr->at(dim_1_graph_template_name);

            graph_template_instance.mutable_internal_connections()->insert(
                {"QSFP_DD", tt::scaleout_tools::cabling_generator::proto::PortConnections()});
            auto& internal_connection = graph_template_instance.mutable_internal_connections()->at("QSFP_DD");

            for (int i = 0; i < galaxy_nodes_1; i++) {
                auto* child = graph_template_instance.mutable_children()->Add();
                child->set_name("dim1_node" + std::to_string(i));
                child->mutable_node_ref()->set_node_descriptor(config.galaxy_type);

                if (i < galaxy_nodes_1 - 1 || config.torus_1) {
                    tt::scaleout_tools::cabling_generator::proto::Connection connection;

                    connection.mutable_port_a()->add_path("dim1_node" + std::to_string(i));
                    connection.mutable_port_b()->add_path("dim1_node" + std::to_string((i + 1) % galaxy_nodes_1));

                    for (int port_id = 3; port_id <= 6; port_id++) {
                        int tray_a = 0, tray_b = 0;
                        if (config.galaxy_type == "WH_GALAXY") {
                            tray_a = 2;
                            tray_b = 1;
                        } else if (config.galaxy_type == "BH_GALAXY") {
                            tray_a = 3;
                            tray_b = 1;
                        }
                        connection.mutable_port_a()->set_tray_id(tray_a);
                        connection.mutable_port_b()->set_tray_id(tray_b);
                        connection.mutable_port_a()->set_port_id(port_id);
                        connection.mutable_port_b()->set_port_id(port_id);
                        auto* new_conn = internal_connection.add_connections();
                        new_conn->CopyFrom(connection);

                        if (config.galaxy_type == "WH_GALAXY") {
                            tray_a = 4;
                            tray_b = 3;
                        } else if (config.galaxy_type == "BH_GALAXY") {
                            tray_a = 4;
                            tray_b = 2;
                        }
                        connection.mutable_port_a()->set_tray_id(tray_a);
                        connection.mutable_port_b()->set_tray_id(tray_b);
                        new_conn = internal_connection.add_connections();
                        new_conn->CopyFrom(connection);
                    }
                }
            }
        }
        // Scope block for graph_template_instance setup
        {
            auto& graph_template_instance = graph_templates_ptr->at(graph_template_name);
            graph_template_instance.mutable_internal_connections()->insert(
                {"QSFP_DD", tt::scaleout_tools::cabling_generator::proto::PortConnections()});
            auto& internal_connection = graph_template_instance.mutable_internal_connections()->at("QSFP_DD");

            for (int i = 0; i < galaxy_nodes_0; i++) {
                auto* group = graph_template_instance.mutable_children()->Add();
                group->set_name("dim0_group" + std::to_string(i));
                group->mutable_graph_ref()->set_graph_template(dim_1_graph_template_name);

                if (i < galaxy_nodes_0 - 1 || config.torus_0) {
                    for (int j = 0; j < galaxy_nodes_1; j++) {
                        tt::scaleout_tools::cabling_generator::proto::Connection connection;
                        connection.mutable_port_a()->add_path("dim0_group" + std::to_string(i));
                        connection.mutable_port_a()->add_path("dim1_node" + std::to_string(j));
                        connection.mutable_port_b()->add_path("dim0_group" + std::to_string((i + 1) % galaxy_nodes_0));
                        connection.mutable_port_b()->add_path("dim1_node" + std::to_string(j));

                        for (int port_id = 1; port_id <= 2; port_id++) {
                            int tray_a = 0, tray_b = 0;
                            if (config.galaxy_type == "WH_GALAXY") {
                                tray_a = 3;
                                tray_b = 1;
                            } else if (config.galaxy_type == "BH_GALAXY") {
                                tray_a = 2;
                                tray_b = 1;
                            }
                            connection.mutable_port_a()->set_tray_id(tray_a);
                            connection.mutable_port_b()->set_tray_id(tray_b);
                            connection.mutable_port_a()->set_port_id(port_id);
                            connection.mutable_port_b()->set_port_id(port_id);
                            auto* new_conn = internal_connection.add_connections();
                            new_conn->CopyFrom(connection);

                            if (config.galaxy_type == "WH_GALAXY") {
                                tray_a = 4;
                                tray_b = 2;
                            } else if (config.galaxy_type == "BH_GALAXY") {
                                tray_a = 4;
                                tray_b = 3;
                            }
                            connection.mutable_port_a()->set_tray_id(tray_a);
                            connection.mutable_port_b()->set_tray_id(tray_b);
                            new_conn = internal_connection.add_connections();
                            new_conn->CopyFrom(connection);
                        }
                    }
                }
            }
        }

        // Root instance setup
        for (int i = 0; i < galaxy_nodes_0; i++) {
            root_instance_ptr->mutable_child_mappings()->insert(
                {"dim0_group" + std::to_string(i), tt::scaleout_tools::cabling_generator::proto::ChildMapping()});
            auto* sub_instance = root_instance_ptr->mutable_child_mappings()
                                     ->at("dim0_group" + std::to_string(i))
                                     .mutable_sub_instance();
            sub_instance->set_template_name(dim_1_graph_template_name);

            for (int j = 0; j < galaxy_nodes_1; j++) {
                sub_instance->mutable_child_mappings()->insert(
                    {"dim1_node" + std::to_string(j), tt::scaleout_tools::cabling_generator::proto::ChildMapping()});
                sub_instance->mutable_child_mappings()
                    ->at("dim1_node" + std::to_string(j))
                    .set_host_id((i * galaxy_nodes_1) + j);
            }
        }

        std::string output_path = "out/scaleout/" + std::string(config.galaxy_type) + "_" + "big_mesh_" + std::to_string(config.nodes_0) + "x" +
                                  std::to_string(config.nodes_1) + "_" + config.topology + ".textproto";

        std::filesystem::path output_file_path(output_path);
        if (output_file_path.has_parent_path()) {
            std::filesystem::create_directories(output_file_path.parent_path());
        }

        std::ofstream output_file(output_path);
        if (!output_file.is_open()) {
            throw std::runtime_error("Failed to open output file: " + output_path);
        }

        std::string output_string;
        google::protobuf::TextFormat::Printer printer;
        printer.SetUseShortRepeatedPrimitives(true);
        printer.SetUseUtf8StringEscaping(true);
        printer.SetSingleLineMode(false);

        if (!printer.PrintToString(big_mesh_descriptor, &output_string)) {
            throw std::runtime_error("Failed to write textproto to file: " + output_path);
        }
        output_file << output_string;
        output_file.close();

        std::cout << "Successfully generated cluster descriptor: " << output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
