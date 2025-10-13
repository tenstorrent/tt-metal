// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <stdexcept>
#include <google/protobuf/text_format.h>    

#include "protobuf/cluster_config.pb.h"
#include "protobuf/node_config.pb.h"

struct InputConfig {
    int nodes_0;
    int nodes_1;
    bool torus_0;
    bool torus_1;
    std::string galaxy_structure;
    std::string topology;
    std::string galaxy_type;
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <galaxy_structure> <topology> [galaxy_type]" << std::endl;
    std::cerr << "       " << program_name << " --help" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  galaxy_structure: Format 'NxM' where N and M are positive integers" << std::endl;
    std::cerr << "                   N must be divisible by 8, M must be divisible by 4" << std::endl;
    std::cerr << "                   Example: '8x4', '16x8', '32x16'" << std::endl;
    std::cerr << "  topology:        Torus configuration - '10', '01', or '11'" << std::endl;
    std::cerr << "                   '10' - torus in first dimension only" << std::endl;
    std::cerr << "                   '01' - torus in second dimension only" << std::endl;
    std::cerr << "                   '11' - torus in both dimensions" << std::endl;
    std::cerr << "                   '00' - no torus (mesh topology)" << std::endl;
    std::cerr << "  galaxy_type:     (Optional) Galaxy type - 'WH_GALAXY' or 'BH_GALAXY'" << std::endl;
    std::cerr << "                   Defaults to 'WH_GALAXY' if not specified" << std::endl;
}

InputConfig parse_arguments(int argc, char** argv) {
    // Handle help flag
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        exit(0);
    }
    
    // Validate argument count
    if (argc != 3 && argc != 4) {
        std::cerr << "Error: Expected 2 or 3 arguments, got " << (argc - 1) << std::endl;
        print_usage(argv[0]);
        exit(1);
    }

    InputConfig config;
    config.galaxy_structure = argv[1];
    config.topology = argv[2];
    
    // Set galaxy_type with default value if not provided
    if (argc == 3) {
        config.galaxy_type = "WH_GALAXY";
        std::cout << "Note: Using default galaxy type 'WH_GALAXY' (not specified)" << std::endl;
    } else {
        config.galaxy_type = argv[3];
    }
    
    // Parse galaxy structure "NxM"
    size_t x_pos = config.galaxy_structure.find('x');
    if (x_pos == std::string::npos) {
        throw std::invalid_argument("Galaxy structure must be in format 'NxM' (e.g., '8x4'), got: '" + config.galaxy_structure + "'");
    }
    
    try {
        config.nodes_0 = std::stoi(config.galaxy_structure.substr(0, x_pos));
        config.nodes_1 = std::stoi(config.galaxy_structure.substr(x_pos + 1));
    } catch (const std::exception& e) {
        throw std::invalid_argument("Invalid numbers in galaxy structure '" + config.galaxy_structure + "': " + e.what());
    }
    
    // Validate node counts
    if (config.nodes_0 <= 0 || config.nodes_1 <= 0) {
        throw std::invalid_argument("Node counts must be positive integers, got: " + std::to_string(config.nodes_0) + "x" + std::to_string(config.nodes_1));
    }
    
    if (config.nodes_0 % 8 != 0) {
        throw std::invalid_argument("First dimension (" + std::to_string(config.nodes_0) + ") must be divisible by 8 for 8x4 Galaxy basis");
    }
    
    if (config.nodes_1 % 4 != 0) {
        throw std::invalid_argument("Second dimension (" + std::to_string(config.nodes_1) + ") must be divisible by 4 for 8x4 Galaxy basis");
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
        throw std::invalid_argument("Invalid topology '" + config.topology + "'. Must be '10', '01', '11', or '00' (for mesh)");
    }
    
    // Validate galaxy type
    if (config.galaxy_type != "WH_GALAXY" && config.galaxy_type != "BH_GALAXY") {
        throw std::invalid_argument("Invalid galaxy type '" + config.galaxy_type + "'. Must be 'WH_GALAXY' or 'BH_GALAXY'");
    }
    
    return config;
}

int main(int argc, char** argv) {
    try {
        InputConfig config = parse_arguments(argc, argv);
        
        int galaxy_nodes_0 = config.nodes_0 / 8;
        int galaxy_nodes_1 = config.nodes_1 / 4; 


        std::string graph_template_name = "big_mesh_" + config.galaxy_structure;
    std::string dim_1_graph_template_name = "big_mesh_dim1";


    tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor big_mesh_descriptor = tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor();

    auto* root_instance_ptr = big_mesh_descriptor.mutable_root_instance();
    auto* graph_templates_ptr = big_mesh_descriptor.mutable_graph_templates();
    root_instance_ptr->set_template_name(graph_template_name); 

    graph_templates_ptr->insert({dim_1_graph_template_name, tt::scaleout_tools::cabling_generator::proto::GraphTemplate()});
    graph_templates_ptr->insert({graph_template_name, tt::scaleout_tools::cabling_generator::proto::GraphTemplate()});

    
    // Scope block for dim_1_graph_template setup
    {
        auto& graph_template_instance = graph_templates_ptr->at(dim_1_graph_template_name);
    
        graph_templates_ptr->insert({dim_1_graph_template_name, graph_template_instance});

        graph_template_instance.mutable_internal_connections()->insert({"QSFP_DD", tt::scaleout_tools::cabling_generator::proto::PortConnections()});
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
                    connection.mutable_port_a()->set_tray_id(2);
                    connection.mutable_port_b()->set_tray_id(1);
                    connection.mutable_port_a()->set_port_id(port_id);
                    connection.mutable_port_b()->set_port_id(port_id);
                    auto * new_conn = internal_connection.add_connections();
                    new_conn->CopyFrom(connection);
                    connection.mutable_port_a()->set_tray_id(4);
                    connection.mutable_port_b()->set_tray_id(3);
                    new_conn = internal_connection.add_connections();
                    new_conn->CopyFrom(connection);            
                }
            } 
        }
    }
    // Scope block for graph_template_instance setup
    {
        auto& graph_template_instance = graph_templates_ptr->at(graph_template_name);
        graph_template_instance.mutable_internal_connections()->insert({"QSFP_DD", tt::scaleout_tools::cabling_generator::proto::PortConnections()});
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
                        connection.mutable_port_a()->set_tray_id(1);
                        connection.mutable_port_b()->set_tray_id(3);
                        connection.mutable_port_a()->set_port_id(port_id);
                        connection.mutable_port_b()->set_port_id(port_id);
                        auto * new_conn = internal_connection.add_connections();
                        new_conn->CopyFrom(connection);
                        connection.mutable_port_a()->set_tray_id(2);
                        connection.mutable_port_b()->set_tray_id(4);
                        new_conn = internal_connection.add_connections();
                        new_conn->CopyFrom(connection);
                    }
                }
            }
        }
    }

    // Root instance setup
    for (int i = 0; i < galaxy_nodes_0; i++) {
        root_instance_ptr->mutable_child_mappings()->insert({"dim0_group" + std::to_string(i), tt::scaleout_tools::cabling_generator::proto::ChildMapping()});
        auto* sub_instance = root_instance_ptr->mutable_child_mappings()->at("dim0_group" + std::to_string(i)).mutable_sub_instance();
            sub_instance->set_template_name(dim_1_graph_template_name);
        
        for (int j = 0; j < galaxy_nodes_1; j++) {
            sub_instance->mutable_child_mappings()->insert({"dim1_node" + std::to_string(j), tt::scaleout_tools::cabling_generator::proto::ChildMapping()});
            sub_instance->mutable_child_mappings()->at("dim1_node" + std::to_string(j)).set_host_id(i * galaxy_nodes_1 + j);
        }
    }


        std::string output_path = "out/scaleout/big_mesh_" + std::to_string(config.nodes_0) + "x" + std::to_string(config.nodes_1) + "_" + config.topology + ".textproto";

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
