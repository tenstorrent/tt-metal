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

// #include "protobuf/cluster_config.pb.h"
// #include "protobuf/node_config.pb.h"
using namespace tt::scaleout_tools;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <cluster_descriptor_path> <deployment_descriptor_path> <output_name>" << std::endl;
        return 1;
    }

    std::string cluster_descriptor_path = argv[1];
    std::string deployment_descriptor_path = argv[2];
    std::string output_name = argv[3];

    CablingGenerator cabling_generator(cluster_descriptor_path, deployment_descriptor_path);

    cabling_generator.emit_factory_system_descriptor("out/scaleout/factory_system_descriptor" + output_name + ".textproto");
    cabling_generator.emit_cabling_guide_csv("out/scaleout/cabling_guide" + output_name + ".csv");

    return 0;
    
}