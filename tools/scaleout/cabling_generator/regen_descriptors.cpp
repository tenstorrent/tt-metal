// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regen_descriptors.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include <board/board.hpp>
#include <cabling_generator/cabling_generator.hpp>

namespace tt::scaleout_tools {

namespace {

// Parse the YAML schema emitted by log_unretrainable_channels (cluster_validation_utils.cpp):
//   unretrainable_channels:
//     - host: <hostname>
//       tray_id: <uint>
//       asic_location: <uint>
//       channel: <uint>
//       asic_unique_id: <hex string>      # ignored here
std::set<PhysicalChannelEndpoint> load_unretrainable_channels(const std::string& yaml_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to parse unretrainable channels YAML at '" + yaml_path + "': " + e.what());
    }

    std::set<PhysicalChannelEndpoint> dead_channels;

    auto channels_node = root["unretrainable_channels"];
    if (!channels_node || channels_node.IsNull()) {
        return dead_channels;
    }
    if (!channels_node.IsSequence()) {
        throw std::runtime_error("Expected 'unretrainable_channels' in '" + yaml_path + "' to be a YAML sequence");
    }

    for (const auto& entry : channels_node) {
        try {
            auto host = entry["host"].as<std::string>();
            auto tray_id = entry["tray_id"].as<uint32_t>();
            auto asic_location = entry["asic_location"].as<uint32_t>();
            auto channel = entry["channel"].as<uint32_t>();

            dead_channels.insert(PhysicalChannelEndpoint{
                .hostname = std::move(host),
                .tray_id = TrayId(tray_id),
                .asic_channel =
                    AsicChannel{
                        .asic_location = asic_location,
                        .channel_id = ChanId(channel),
                    },
            });
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("Malformed entry in 'unretrainable_channels' (" + yaml_path + "): " + e.what());
        }
    }

    return dead_channels;
}

}  // namespace

RegenDescriptorsSummary regenerate_descriptors_excluding_dead_channels(
    const std::string& cabling_descriptor_path,
    const std::string& deployment_descriptor_path,
    const std::string& unretrainable_channels_yaml_path,
    const std::string& output_dir) {
    auto dead_channels = load_unretrainable_channels(unretrainable_channels_yaml_path);

    CablingGenerator gen(cabling_descriptor_path, deployment_descriptor_path);

    auto pruned_cables = gen.prune_dead_channels(dead_channels);

    std::filesystem::create_directories(output_dir);
    std::filesystem::path out(output_dir);
    auto fsd_path = (out / "factory_system_descriptor.textproto").string();
    auto cabling_path = (out / "cabling_descriptor.textproto").string();
    auto deployment_path = (out / "deployment_descriptor.textproto").string();

    gen.emit_factory_system_descriptor(fsd_path);
    gen.emit_cabling_descriptor(cabling_path);
    gen.emit_deployment_descriptor(deployment_path);

    return RegenDescriptorsSummary{
        .pruned_cables = std::move(pruned_cables),
        .channels_remaining = gen.get_chip_connections().size(),
        .input_dead_channels = dead_channels.size(),
        .output_fsd_path = std::move(fsd_path),
        .output_cabling_descriptor_path = std::move(cabling_path),
        .output_deployment_descriptor_path = std::move(deployment_path),
    };
}

}  // namespace tt::scaleout_tools
