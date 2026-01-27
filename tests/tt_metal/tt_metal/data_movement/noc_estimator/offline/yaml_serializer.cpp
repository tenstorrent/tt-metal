// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "yaml_serializer.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>

namespace tt::noc_estimator::offline {

using namespace common;

bool save_latency_data_to_yaml(
    const std::map<common::GroupKey, common::LatencyData>& data, const std::string& yaml_path) {
    YAML::Emitter out;
    out.SetDoublePrecision(6);

    out << YAML::BeginMap;

    // Write transaction sizes once at top
    out << YAML::Key << "transaction_sizes" << YAML::Value << YAML::Flow << STANDARD_TRANSACTION_SIZES;
    out << YAML::Key << "num_entries" << YAML::Value << data.size();

    out << YAML::Key << "entries" << YAML::Value << YAML::BeginSeq;

    for (const auto& [key, latency_data] : data) {
        out << YAML::BeginMap;

        out << YAML::Key << "key" << YAML::Value << YAML::BeginMap;

        // Memory optimization, only write values that are not default
        if (key.num_transactions != DEFAULT_NUM_TRANSACTIONS) {
            out << YAML::Key << "num_transactions" << YAML::Value << key.num_transactions;
        }
        if (key.num_subordinates != DEFAULT_NUM_SUBORDINATES) {
            out << YAML::Key << "num_subordinates" << YAML::Value << key.num_subordinates;
        }
        if (key.same_axis != DEFAULT_SAME_AXIS) {
            out << YAML::Key << "same_axis" << YAML::Value << key.same_axis;
        }
        if (key.linked != DEFAULT_LINKED) {
            out << YAML::Key << "linked" << YAML::Value << key.linked;
        }
        if (static_cast<int>(key.arch) != DEFAULT_ARCH) {
            out << YAML::Key << "arch" << YAML::Value << static_cast<int>(key.arch);
        }
        if (static_cast<int>(key.mechanism) != DEFAULT_MECHANISM) {
            out << YAML::Key << "mechanism" << YAML::Value << static_cast<int>(key.mechanism);
        }
        if (static_cast<int>(key.memory) != DEFAULT_MEMORY) {
            out << YAML::Key << "memory" << YAML::Value << static_cast<int>(key.memory);
        }
        if (static_cast<int>(key.pattern) != DEFAULT_PATTERN) {
            out << YAML::Key << "pattern" << YAML::Value << static_cast<int>(key.pattern);
        }
        out << YAML::EndMap;

        out << YAML::Key << "latencies" << YAML::Value << YAML::Flow << latency_data.latencies;

        out << YAML::EndMap;
    }

    out << YAML::EndSeq;
    out << YAML::EndMap;

    std::ofstream file(yaml_path);
    if (!file.is_open()) {
        std::cerr << "Failed to create YAML file: " << yaml_path << std::endl;
        return false;
    }

    file << out.c_str();
    file.flush();
    if (!file) {
        std::cerr << "Failed to write YAML data to file: " << yaml_path << std::endl;
        file.close();
        return false;
    }
    file.close();

    return true;
}

}  // namespace tt::noc_estimator::offline
