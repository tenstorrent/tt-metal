// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <google/protobuf/text_format.h>    
// #include <cabling_generator/cabling_generator.hpp>
// #include <factory_system_descriptor/utils.hpp>
// #include <node/node_types.hpp>

#include "protobuf/cluster_config.pb.h"
#include "protobuf/node_config.pb.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_galaxy_nodes> <topology>" << std::endl;
        return 1;
    }

    std::string num_galaxy_nodes_s = argv[1];
    std::string topology = argv[2];

    int num_galaxy_nodes = std::stoi(num_galaxy_nodes_s);

    bool populate_x = false, populate_y = false;

    if (topology.find('X') != std::string::npos) {
        populate_x = true;
    }
    if (topology.find('Y') != std::string::npos) {
        populate_y = true;
    }


    std::string graph_template_name = num_galaxy_nodes_s + "_" + topology + "_big_mesh";

    tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor big_mesh_descriptor = tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor();

    // auto big_mesh_graph_template = tt::scaleout_tools::cabling_generator::proto::GraphTemplate();

    auto* root_instance_ptr = big_mesh_descriptor.mutable_root_instance();
    auto* graph_templates_ptr = big_mesh_descriptor.mutable_graph_templates();

    graph_templates_ptr->insert({graph_template_name, tt::scaleout_tools::cabling_generator::proto::GraphTemplate()});

    auto& big_mesh_graph_template = graph_templates_ptr->at(graph_template_name);

    root_instance_ptr->set_template_name(graph_template_name); 

    graph_templates_ptr->insert({graph_template_name, big_mesh_graph_template});

    big_mesh_graph_template.mutable_internal_connections()->insert({"QSFP_DD", tt::scaleout_tools::cabling_generator::proto::PortConnections()});
    auto& internal_connection = big_mesh_graph_template.mutable_internal_connections()->at("QSFP_DD");

    for (int i = 0; i < num_galaxy_nodes; i++) {
        // big_mesh_descriptor.mutable_root_instance()->mutable_child_mappings()->insert({"node" + std::to_string(i), tt::scaleout_tools::cabling_generator::proto::ChildMapping()});
        // big_mesh_descriptor.mutable_root_instance()->mutable_child_mappings()->at("node" + std::to_string(i)).mutable_host_id()->set_host_id(i);
        auto* child = big_mesh_graph_template.mutable_children()->Add();
        child->set_name("node" + std::to_string(i));
        if (populate_y) {
            child->mutable_node_ref()->set_node_descriptor("WH_GALAXY_Y_TORUS");
        } else {
            child->mutable_node_ref()->set_node_descriptor("WH_GALAXY");
        }

        root_instance_ptr->mutable_child_mappings()->insert({"node" + std::to_string(i), tt::scaleout_tools::cabling_generator::proto::ChildMapping()});
        root_instance_ptr->mutable_child_mappings()->at("node" + std::to_string(i)).set_host_id(i);

        if (i < num_galaxy_nodes - 1) {
            tt::scaleout_tools::cabling_generator::proto::Connection connection;

            connection.mutable_port_a()->add_path("node" + std::to_string(i));
            connection.mutable_port_b()->add_path("node" + std::to_string(i + 1));
            
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
        } else if (populate_x) {
            tt::scaleout_tools::cabling_generator::proto::Connection connection;

            connection.mutable_port_a()->add_path("node" + std::to_string(i));
            connection.mutable_port_b()->add_path("node" + std::to_string(0));
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


    std::string output_path = "out/scaleout/big_mesh_cabling_gen.textproto";

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
    printer.SetPrintMessageFieldsInIndexOrder(true);

    if (!printer.PrintToString(big_mesh_descriptor, &output_string)) {
        throw std::runtime_error("Failed to write textproto to file: " + output_path);
    }
    output_file << output_string;
    output_file.close();
}
