// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "yaml_loader.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>

namespace tt::noc_estimator::loader {

using namespace common;

template <typename T>
T get_or_default(const YAML::Node& node, const std::string& key, T default_value) {
    if (node[key]) {
        return node[key].as<T>();
    }
    return default_value;
}

LoadedData load_latency_data_from_yaml(const std::string& yaml_path) {
    LoadedData result;

    try {
        YAML::Node root = YAML::LoadFile(yaml_path);

        result.transaction_sizes = root["transaction_sizes"].as<std::vector<uint32_t>>();
        std::cout << "Loading YAML with " << result.transaction_sizes.size() << " transaction sizes\n";

        const YAML::Node& entries = root["entries"];
        for (const auto& entry : entries) {
            const YAML::Node& key_node = entry["key"];
            common::GroupKey key{
                .mechanism = static_cast<NocMechanism>(get_or_default<int>(key_node, "mechanism", DEFAULT_MECHANISM)),
                .pattern = static_cast<NocPattern>(get_or_default<int>(key_node, "pattern", DEFAULT_PATTERN)),
                .memory = static_cast<MemoryType>(get_or_default<int>(key_node, "memory", DEFAULT_MEMORY)),
                .arch = static_cast<Architecture>(get_or_default<int>(key_node, "arch", DEFAULT_ARCH)),
                .num_transactions = get_or_default<uint32_t>(key_node, "num_transactions", DEFAULT_NUM_TRANSACTIONS),
                .num_subordinates = get_or_default<uint32_t>(key_node, "num_subordinates", DEFAULT_NUM_SUBORDINATES),
                .same_axis = get_or_default<bool>(key_node, "same_axis", DEFAULT_SAME_AXIS),
                .linked = get_or_default<bool>(key_node, "linked", DEFAULT_LINKED)};

            common::LatencyData latency_data;
            latency_data.latencies = entry["latencies"].as<std::vector<double>>();

            result.entries[key] = latency_data;
        }

        std::cout << "Loaded " << result.entries.size() << " entries from YAML\n";

    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return result;
}

}  // namespace tt::noc_estimator::loader
