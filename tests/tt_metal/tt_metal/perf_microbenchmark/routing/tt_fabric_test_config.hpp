// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

#include "assert.hpp"
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"

#include "tt_fabric_test_config.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/mesh_graph.hpp>

// needed since the API throws exception if the option is not found and default is not set
const std::string no_default_test_yaml_config = "";

namespace tt::tt_fabric {
namespace fabric_tests {

struct TestFabricSetup {
    tt::tt_fabric::Topology topology;
    std::optional<std::vector<chip_id_t>> physical_devices;
    std::optional<std::vector<std::pair<MeshId, chip_id_t>>> logical_devices;
};

struct TestConfig {
    std::string name;
    std::optional<TestFabricSetup> fabric_setup;  // if not present, use the default aka device init based setup
};

// checks how many of the specified keys are present in the yaml node
size_t check_for_keys(const YAML::Node& yaml_node, const std::vector<std::string>& keys) {
    size_t num_keys_present = 0;
    for (const auto& key : keys) {
        if (yaml_node[key]) {
            num_keys_present++;
        }
    }

    return num_keys_present;
}

MeshId parse_mesh_id(const YAML::Node& yaml_node) {
    uint32_t mesh_id = yaml_node.as<uint32_t>();
    return MeshId{mesh_id};
}

template <typename T>
T parse_scalar(const YAML::Node& yaml_node) {
    if constexpr (std::is_same_v<T, MeshId>) {
        return parse_mesh_id(yaml_node);
    } else {
        return yaml_node.as<T>();
    }
}

template <typename T>
std::vector<T> parse_scalar_sequence(const YAML::Node& yaml_node) {
    std::vector<T> sequence;
    for (const auto& entry : yaml_node) {
        TT_FATAL(entry.IsScalar(), "Expected each entry to be scalar");
        sequence.push_back(parse_scalar<T>(entry));
    }

    return sequence;
}

template <typename T1, typename T2>
std::pair<T1, T2> parse_pair(const YAML::Node& yaml_sequence) {
    TT_FATAL(yaml_sequence.size() == 2, "Expected only 2 entries for the pair");
    return {parse_scalar<T1>(yaml_sequence[0]), parse_scalar<T2>(yaml_sequence[1])};
}

template <typename T1, typename T2>
std::vector<std::pair<T1, T2>> parse_pair_sequence(const YAML::Node& yaml_node) {
    std::vector<std::pair<T1, T2>> pair_sequence;
    for (const auto& entry : yaml_node) {
        TT_FATAL(entry.IsSequence(), "Expected each entry to be sequence");
        pair_sequence.push_back(parse_pair<T1, T2>(entry));
    }

    return pair_sequence;
}

tt::tt_fabric::Topology get_topology_from_string(const std::string& topology_string) {
    if (topology_string == "Ring") {
        return tt::tt_fabric::Topology::Ring;
    } else if (topology_string == "Linear") {
        return tt::tt_fabric::Topology::Linear;
    } else if (topology_string == "Mesh") {
        return tt::tt_fabric::Topology::Mesh;
    } else {
        TT_THROW("Unsupported topology: {}", topology_string);
    }
}

std::vector<std::pair<MeshId, chip_id_t>> get_logical_devices(
    MeshId mesh_id, const std::vector<chip_id_t>& logical_chip_ids) {
    std::vector<std::pair<MeshId, chip_id_t>> logical_devices;
    for (const auto& chip_id : logical_chip_ids) {
        logical_devices.push_back({mesh_id, chip_id});
    }

    return logical_devices;
}

template <typename T>
std::vector<T> get_elements_in_range(T start, T end) {
    std::vector<T> range(end - start + 1);
    std::iota(range.begin(), range.end(), start);
    return range;
}

TestFabricSetup parse_test_fabric_setup(const YAML::Node& test_yaml) {
    TT_FATAL(test_yaml["fabric_setup"].IsMap(), "Expected fabric_setup to be a map");
    const auto& fabric_setup_yaml = test_yaml["fabric_setup"];

    TestFabricSetup fabric_setup;

    // parse topology
    TT_FATAL(fabric_setup_yaml["topology"], "Expected topology in fabric_setup");
    fabric_setup.topology = get_topology_from_string(fabric_setup_yaml["topology"].as<std::string>());

    // parse devices
    const auto& devices_yaml = fabric_setup_yaml["devices"];
    TT_FATAL(devices_yaml, "Expected devices in fabric_setup");
    TT_FATAL(devices_yaml.IsMap(), "Expected devices to be a map");
    TT_FATAL(
        check_for_keys(devices_yaml, {"sequence", "range"}) == 1,
        "Expected (only) one of sequence or range for devices");

    if (devices_yaml["sequence"]) {
        TT_FATAL(devices_yaml["sequence"].IsSequence(), "Expected a sequence for device sequence");
        if (devices_yaml["mesh_id"]) {
            TT_FATAL(devices_yaml["mesh_id"].IsScalar(), "Expected mesh id to be scalar");
            // treat the device sequence as logical devices sharing the same mesh id
            const auto mesh_id = parse_scalar<MeshId>(devices_yaml["mesh_id"]);
            const auto& logical_chip_ids = parse_scalar_sequence<chip_id_t>(devices_yaml["sequence"]);
            fabric_setup.logical_devices = get_logical_devices(mesh_id, logical_chip_ids);
        } else {
            // the sequence could be either physical ids or a pair of mesh id and logical device id
            if (devices_yaml["sequence"].begin()->IsScalar()) {
                fabric_setup.physical_devices = parse_scalar_sequence<chip_id_t>(devices_yaml["sequence"]);
            } else if (devices_yaml["sequence"].begin()->IsSequence()) {
                fabric_setup.logical_devices = parse_pair_sequence<MeshId, chip_id_t>(devices_yaml["sequence"]);
            } else {
                TT_THROW("Unsupported device sequence");
            }
        }
    } else if (devices_yaml["range"]) {
        TT_FATAL(devices_yaml["range"].IsSequence(), "Expected a sequence for device range");
        if (devices_yaml["range"].begin()->IsScalar()) {
            const auto [phys_device_id_start, phys_device_id_end] =
                parse_pair<chip_id_t, chip_id_t>(devices_yaml["range"]);
            fabric_setup.physical_devices = get_elements_in_range<chip_id_t>(phys_device_id_start, phys_device_id_end);
        } else if (devices_yaml["range"].begin()->IsSequence()) {
            const auto& logical_range = parse_pair_sequence<MeshId, chip_id_t>(devices_yaml["range"]);
            TT_FATAL(logical_range[0].first == logical_range[1].first, "Expected same mesh ids for logical range");
            const auto& logical_chip_ids =
                get_elements_in_range<chip_id_t>(logical_range[0].second, logical_range[1].second);
            fabric_setup.logical_devices = get_logical_devices(logical_range[0].first, logical_chip_ids);
        }
    }

    if (fabric_setup.physical_devices.has_value()) {
        log_info(tt::LogTest, "physical devices: {}", fabric_setup.physical_devices.value());
    }

    if (fabric_setup.logical_devices.has_value()) {
        log_info(tt::LogTest, "logical devices: {}", fabric_setup.logical_devices.value());
    }

    return fabric_setup;
}

TestConfig parse_test_config(const YAML::Node& test_yaml) {
    TestConfig test_config;

    test_config.name = test_yaml["name"].as<std::string>();
    log_info(tt::LogTest, "name: {}", test_config.name);

    if (test_yaml["fabric_setup"]) {
        test_config.fabric_setup = parse_test_fabric_setup(test_yaml);
        log_info(tt::LogTest, "topology: {}", test_config.fabric_setup.value().topology);
    } else {
        log_info(
            tt::LogTest, "No custom fabric setup found for test name: {}, will use default setup", test_config.name);
    }

    return test_config;
}

void parse_yaml_config(const std::string& yaml_config_path) {
    std::ifstream yaml_config(yaml_config_path);
    TT_FATAL(not yaml_config.fail(), "Failed to open file: {}", yaml_config_path);

    YAML::Node yaml = YAML::LoadFile(yaml_config_path);

    TT_FATAL(yaml["Tests"].IsSequence(), "Expected Tests to be a sequence");

    std::vector<TestConfig> test_configs;
    for (const auto& test_yaml : yaml["Tests"]) {
        TT_FATAL(test_yaml.IsMap(), "Expected each test in Tests to be a map");

        test_configs.emplace_back(parse_test_config(test_yaml));
    }
}

void parse_config(const std::vector<std::string>& input_args) {
    std::string yaml_config = test_args::get_command_option(input_args, "--test_config", no_default_test_yaml_config);

    if (yaml_config != no_default_test_yaml_config) {
        std::filesystem::path fpath(yaml_config);
        if (!fpath.is_absolute()) {
            const auto& fname = fpath.filename();
            fpath = std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/" / fname;
            log_warning(tt::LogTest, "Relative fpath for config provided, using absolute path: {}", fpath);
        }
        parse_yaml_config(fpath);
    }

    /*
        uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
        num_packets = test_args::get_command_option_uint32(input_args, "--num_packets", default_num_packets);
        packet_payload_size_kb =
            test_args::get_command_option_uint32(input_args, "--packet_payload_size_kb",
       default_packet_payload_size_kb); num_routing_planes = test_args::get_command_option_uint32(input_args,
       "--num_routing_planes", default_num_routing_planes);

        uint32_t test_device_id_l =
            test_args::get_command_option_uint32(input_args, "--device_id", default_test_device_id_l);
        uint32_t test_device_id_r =
            test_args::get_command_option_uint32(input_args, "--device_id_r", default_test_device_id_r);

        uint32_t num_sender_chips =
            test_args::get_command_option_uint32(input_args, "--num_sender_chips", default_num_sender_chips);

        std::unordered_map<tt_fabric::RoutingDirection, uint32_t> global_hops_counts;
        global_hops_counts[tt_fabric::RoutingDirection::N] =
            test_args::get_command_option_uint32(input_args, "--n_hops", default_num_hops);
        global_hops_counts[tt_fabric::RoutingDirection::S] =
            test_args::get_command_option_uint32(input_args, "--s_hops", default_num_hops);
        global_hops_counts[tt_fabric::RoutingDirection::E] =
            test_args::get_command_option_uint32(input_args, "--e_hops", default_num_hops);
        global_hops_counts[tt_fabric::RoutingDirection::W] =
            test_args::get_command_option_uint32(input_args, "--w_hops", default_num_hops);

        mcast_mode = test_args::has_command_option(input_args, "--mcast_mode");
        bidirectional_mode = test_args::has_command_option(input_args, "--bidirectional_mode");
        all_to_all_mode = test_args::has_command_option(input_args, "--all_to_all_mode");
        benchmark_mode = test_args::has_command_option(input_args, "--benchmark_mode");
        verbose_mode = test_args::has_command_option(input_args, "--verbose");

        metal_fabric_init_level = test_args::get_command_option_uint32(input_args, "--metal_fabric_init_level", 0);
    */
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
