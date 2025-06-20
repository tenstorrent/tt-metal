// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <variant>
#include <random>
#include <algorithm>
#include <numeric>

#include "assert.hpp"
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"

#include "tests/tt_metal/test_utils/test_common.hpp"

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/routing_table_generator.hpp>

#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common_types.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

struct ParsedYamlConfig {
    std::vector<TestConfig> test_configs;
    std::optional<AllocatorPolicies> allocation_policies;
};

inline TrafficPatternConfig merge_patterns(const TrafficPatternConfig& base, const TrafficPatternConfig& specific) {
    TrafficPatternConfig merged;

    merged.ftype = specific.ftype.has_value() ? specific.ftype : base.ftype;
    merged.ntype = specific.ntype.has_value() ? specific.ntype : base.ntype;
    merged.size = specific.size.has_value() ? specific.size : base.size;
    merged.num_packets = specific.num_packets.has_value() ? specific.num_packets : base.num_packets;

    // Special handling for nested destination
    if (specific.destination.has_value()) {
        if (base.destination.has_value()) {
            // Both have destinations, merge them.
            DestinationConfig merged_dest;
            merged_dest.device =
                specific.destination->device.has_value() ? specific.destination->device : base.destination->device;
            merged.destination = merged_dest;
        } else {
            // Only specific has a destination, use it directly.
            merged.destination = specific.destination;
        }
    } else {
        // Specific has no destination, use the base one.
        merged.destination = base.destination;
    }

    return merged;
}

class YamlConfigParser {
public:
    YamlConfigParser(IDeviceInfoProvider& device_info_provider) : device_info_provider_(device_info_provider) {}

    ParsedYamlConfig parse_file(const std::string& yaml_config_path);

private:
    IDeviceInfoProvider& device_info_provider_;

    FabricNodeId parse_device_identifier(const YAML::Node& node);
    DestinationConfig parse_destination_config(const YAML::Node& dest_yaml);
    TrafficPatternConfig parse_traffic_pattern_config(const YAML::Node& pattern_yaml);
    SenderConfig parse_sender_config(const YAML::Node& sender_yaml, const TrafficPatternConfig& defaults);
    TestFabricSetup parse_fabric_setup(const YAML::Node& fabric_setup_yaml);
    TestConfig parse_test_config(const YAML::Node& test_yaml);
    AllocatorPolicies parse_allocator_policies(const YAML::Node& policies_yaml);
    CoreAllocationConfig parse_core_allocation_config(const YAML::Node& config_yaml, CoreAllocationConfig base_config);

    // Parsing helpers
    CoreCoord parse_core_coord(const YAML::Node& node);
    MeshId parse_mesh_id(const YAML::Node& yaml_node);
    template <typename T>
    T parse_scalar(const YAML::Node& yaml_node);
    template <typename T>
    std::vector<T> parse_scalar_sequence(const YAML::Node& yaml_node);
    template <typename T1, typename T2>
    std::pair<T1, T2> parse_pair(const YAML::Node& yaml_sequence);
    template <typename T1, typename T2>
    std::vector<std::pair<T1, T2>> parse_pair_sequence(const YAML::Node& yaml_node);
    template <typename T>
    std::vector<T> get_elements_in_range(T start, T end);
    ParametrizationOptionsMap parse_parametrization_params(const YAML::Node& params_yaml);
    HighLevelPatternConfig parse_high_level_pattern_config(const YAML::Node& pattern_yaml);
};

class CmdlineParser {
public:
    CmdlineParser(const std::vector<std::string>& input_args, IDeviceInfoProvider& device_info_provider) :
        input_args_(input_args), device_info_provider_(device_info_provider) {}

    std::optional<std::string> get_yaml_config_path();
    void apply_overrides(std::vector<TestConfig>& test_configs);
    std::vector<TestConfig> generate_default_configs();
    std::optional<uint32_t> get_master_seed();
    bool dump_built_tests();
    std::string get_built_tests_dump_file_path(const std::string& default_path);

private:
    const std::vector<std::string>& input_args_;
    IDeviceInfoProvider& device_info_provider_;
};

const std::string no_default_test_yaml_config = "";

const std::vector<std::string> supported_high_level_patterns = {"all_to_all_unicast", "full_device_random_pairing"};

// ======================================================================================
// Section 2: Enum and String Conversion Utilities
// ======================================================================================
// Helper functions and mappings for converting between string representations in YAML
// and their corresponding C++ enum types.

namespace detail {
template <typename T>
struct StringEnumMapper {
    std::unordered_map<std::string, T> to_enum;
    std::unordered_map<T, std::string> to_string_map;

    StringEnumMapper(const std::initializer_list<std::pair<const char*, T>>& mapping_data) {
        for (const auto& pair : mapping_data) {
            to_enum[pair.first] = pair.second;
            to_string_map[pair.second] = pair.first;
        }
    }

    const std::string& to_string(T value, const std::string& type_name) const {
        auto it = to_string_map.find(value);
        if (it == to_string_map.end()) {
            TT_THROW("Unknown enum value for {}", type_name);
        }
        return it->second;
    }

    T from_string(const std::string& s, const std::string& type_name) const {
        auto it = to_enum.find(s);
        if (it == to_enum.end()) {
            TT_THROW("Unsupported string value '{}' for {}", s, type_name);
        }
        return it->second;
    }
};

static const StringEnumMapper<ChipSendType> chip_send_type_mapper({
    {"mcast", ChipSendType::CHIP_MULTICAST},
    {"unicast", ChipSendType::CHIP_UNICAST},
});

static const StringEnumMapper<NocSendType> noc_send_type_mapper({
    {"unicast_write", NocSendType::NOC_UNICAST_WRITE},
    {"atomic_inc", NocSendType::NOC_UNICAST_ATOMIC_INC},
    {"fused_atomic_inc", NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC},
});

static const StringEnumMapper<RoutingDirection> routing_direction_mapper({
    {"N", RoutingDirection::N},
    {"S", RoutingDirection::S},
    {"E", RoutingDirection::E},
    {"W", RoutingDirection::W},
});

static const StringEnumMapper<Topology> topology_mapper({
    {"Ring", Topology::Ring},
    {"Linear", Topology::Linear},
    {"Mesh", Topology::Mesh},
});

static const StringEnumMapper<CoreAllocationPolicy> core_allocation_policy_mapper({
    {"RoundRobin", CoreAllocationPolicy::RoundRobin},
    {"ExhaustFirst", CoreAllocationPolicy::ExhaustFirst},
});

}  // namespace detail

inline ParsedYamlConfig YamlConfigParser::parse_file(const std::string& yaml_config_path) {
    std::ifstream yaml_config(yaml_config_path);
    TT_FATAL(not yaml_config.fail(), "Failed to open file: {}", yaml_config_path);

    YAML::Node yaml = YAML::LoadFile(yaml_config_path);

    ParsedYamlConfig result;
    if (yaml["allocation_policies"]) {
        result.allocation_policies = parse_allocator_policies(yaml["allocation_policies"]);
    }

    TT_FATAL(yaml["Tests"].IsSequence(), "Expected 'Tests' to be a sequence at the root of the YAML file.");

    for (const auto& test_yaml : yaml["Tests"]) {
        TT_FATAL(test_yaml.IsMap(), "Expected each test in Tests to be a map");

        result.test_configs.emplace_back(parse_test_config(test_yaml));
    }
    return result;
}

inline FabricNodeId YamlConfigParser::parse_device_identifier(const YAML::Node& node) {
    if (node.IsScalar()) {
        chip_id_t chip_id = parse_scalar<chip_id_t>(node);
        return this->device_info_provider_.get_fabric_node_id(chip_id);
    } else if (node.IsSequence() && node.size() == 2) {
        return FabricNodeId{parse_scalar<MeshId>(node[0]), parse_scalar<chip_id_t>(node[1])};
    }
    TT_THROW("Unsupported device identifier format. Expected scalar chip_id or sequence [mesh_id, chip_id].");
}

inline DestinationConfig YamlConfigParser::parse_destination_config(const YAML::Node& dest_yaml) {
    DestinationConfig config;
    if (dest_yaml["device"]) {
        config.device = parse_device_identifier(dest_yaml["device"]);
    }
    if (dest_yaml["core"]) {
        config.core = parse_core_coord(dest_yaml["core"]);
    }
    if (dest_yaml["hops"]) {
        TT_FATAL(dest_yaml["hops"].IsMap(), "Expected 'hops' to be a map.");
        std::unordered_map<RoutingDirection, uint32_t> hops_map;
        for (const auto& it : dest_yaml["hops"]) {
            std::string dir_str = parse_scalar<std::string>(it.first);
            RoutingDirection dir = detail::routing_direction_mapper.from_string(dir_str, "RoutingDirection");
            uint32_t num_hops = parse_scalar<uint32_t>(it.second);
            hops_map[dir] = num_hops;
        }
        config.hops = hops_map;
    }
    if (dest_yaml["target_address"]) {
        config.target_address = parse_scalar<uint32_t>(dest_yaml["target_address"]);
    }
    return config;
}

inline TrafficPatternConfig YamlConfigParser::parse_traffic_pattern_config(const YAML::Node& pattern_yaml) {
    TT_FATAL(pattern_yaml.IsMap(), "Expected pattern to be a map");

    TrafficPatternConfig config;
    if (pattern_yaml["ftype"]) {
        config.ftype =
            detail::chip_send_type_mapper.from_string(parse_scalar<std::string>(pattern_yaml["ftype"]), "ftype");
    }
    if (pattern_yaml["ntype"]) {
        config.ntype =
            detail::noc_send_type_mapper.from_string(parse_scalar<std::string>(pattern_yaml["ntype"]), "ntype");
    }
    if (pattern_yaml["size"]) {
        config.size = parse_scalar<uint32_t>(pattern_yaml["size"]);
    }
    if (pattern_yaml["num_packets"]) {
        config.num_packets = parse_scalar<uint32_t>(pattern_yaml["num_packets"]);
    }
    if (pattern_yaml["destination"]) {
        config.destination = parse_destination_config(pattern_yaml["destination"]);
    }
    return config;
}

inline SenderConfig YamlConfigParser::parse_sender_config(
    const YAML::Node& sender_yaml, const TrafficPatternConfig& defaults) {
    TT_FATAL(sender_yaml.IsMap(), "Expected sender to be a map");
    TT_FATAL(sender_yaml["device"] && sender_yaml["patterns"], "Sender config missing required keys");

    SenderConfig config;
    config.device = parse_device_identifier(sender_yaml["device"]);
    if (sender_yaml["core"]) {
        config.core = parse_core_coord(sender_yaml["core"]);
    }

    TT_FATAL(sender_yaml["patterns"].IsSequence(), "Expected patterns to be a sequence");
    for (const auto& pattern_node : sender_yaml["patterns"]) {
        TrafficPatternConfig specific_pattern = parse_traffic_pattern_config(pattern_node);
        config.patterns.push_back(merge_patterns(defaults, specific_pattern));
    }
    return config;
}

inline TestFabricSetup YamlConfigParser::parse_fabric_setup(const YAML::Node& fabric_setup_yaml) {
    TT_FATAL(fabric_setup_yaml.IsMap(), "Expected fabric setup to be a map");
    TT_FATAL(fabric_setup_yaml["topology"], "Fabric setup missing Topolgy key");
    TestFabricSetup fabric_setup;
    auto topology_str = parse_scalar<std::string>(fabric_setup_yaml["topology"]);
    fabric_setup.topology = detail::topology_mapper.from_string(topology_str, "Topology");
    return fabric_setup;
}

inline TestConfig YamlConfigParser::parse_test_config(const YAML::Node& test_yaml) {
    TestConfig test_config;

    test_config.name = parse_scalar<std::string>(test_yaml["name"]);
    log_info(tt::LogTest, "name: {}", test_config.name);

    TT_FATAL(test_yaml["fabric_setup"], "No fabric setup specified for test: {}", test_config.name);
    test_config.fabric_setup = parse_fabric_setup(test_yaml["fabric_setup"]);

    if (test_yaml["parametrization_params"]) {
        test_config.parametrization_params = parse_parametrization_params(test_yaml["parametrization_params"]);
    }

    if (test_yaml["on_missing_param_policy"]) {
        test_config.on_missing_param_policy = parse_scalar<std::string>(test_yaml["on_missing_param_policy"]);
    }

    if (test_yaml["defaults"]) {
        test_config.defaults = parse_traffic_pattern_config(test_yaml["defaults"]);
    }

    if (test_yaml["patterns"]) {
        TT_FATAL(test_yaml["patterns"].IsSequence(), "Expected 'patterns' to be a sequence.");
        std::vector<HighLevelPatternConfig> high_level_patterns;
        for (const auto& pattern_node : test_yaml["patterns"]) {
            high_level_patterns.push_back(parse_high_level_pattern_config(pattern_node));
        }
        test_config.patterns = high_level_patterns;
    }

    if (test_yaml["senders"]) {
        TT_FATAL(test_yaml["senders"].IsSequence(), "Expected senders to be a sequence");
        for (const auto& sender_node : test_yaml["senders"]) {
            test_config.senders.push_back(
                parse_sender_config(sender_node, test_config.defaults.value_or(TrafficPatternConfig{})));
        }
    }

    if (test_yaml["bw_calc_func"]) {
        test_config.bw_calc_func = parse_scalar<std::string>(test_yaml["bw_calc_func"]);
    }

    return test_config;
}

inline AllocatorPolicies YamlConfigParser::parse_allocator_policies(const YAML::Node& policies_yaml) {
    TT_FATAL(policies_yaml.IsMap(), "Expected 'allocation_policies' to be a map.");
    AllocatorPolicies policies;  // this will get defaults from constructor
    if (policies_yaml["sender"]) {
        policies.sender_config = parse_core_allocation_config(policies_yaml["sender"], policies.sender_config);
    }
    if (policies_yaml["receiver"]) {
        policies.receiver_config = parse_core_allocation_config(policies_yaml["receiver"], policies.receiver_config);
    }
    return policies;
}

inline CoreAllocationConfig YamlConfigParser::parse_core_allocation_config(
    const YAML::Node& config_yaml, CoreAllocationConfig base_config) {
    CoreAllocationConfig config = base_config;
    if (config_yaml["policy"]) {
        config.policy = detail::core_allocation_policy_mapper.from_string(
            parse_scalar<std::string>(config_yaml["policy"]), "CoreAllocationPolicy");
    }
    if (config_yaml["max_workers_per_core"]) {
        config.max_workers_per_core = parse_scalar<uint32_t>(config_yaml["max_workers_per_core"]);
    }
    if (config_yaml["default_payload_chunk_size"]) {
        config.default_payload_chunk_size = parse_scalar<uint32_t>(config_yaml["default_payload_chunk_size"]);
    }
    if (config_yaml["initial_pool_size"]) {
        config.initial_pool_size = parse_scalar<uint32_t>(config_yaml["initial_pool_size"]);
    }
    if (config_yaml["pool_refill_size"]) {
        config.pool_refill_size = parse_scalar<uint32_t>(config_yaml["pool_refill_size"]);
    }
    return config;
}

// CmdlineParser methods
inline std::optional<std::string> CmdlineParser::get_yaml_config_path() {
    std::string yaml_config = test_args::get_command_option(input_args_, "--test_config", "");

    if (!yaml_config.empty()) {
        std::filesystem::path fpath(yaml_config);
        if (!fpath.is_absolute()) {
            const auto& fname = fpath.filename();
            fpath = std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/" / fname;
            log_warning(tt::LogTest, "Relative fpath for config provided, using absolute path: {}", fpath);
        }
        return fpath.string();
    }

    return std::nullopt;
}

inline void CmdlineParser::apply_overrides(std::vector<TestConfig>& test_configs) {
    bool is_default_test = (test_configs.size() == 1 && test_configs[0].name == "DefaultCommandLineTest");

    if (!is_default_test) {
        if (test_args::has_command_option(input_args_, "--pattern") ||
            test_args::has_command_option(input_args_, "--chip-ids") ||
            test_args::has_command_option(input_args_, "--iterations") ||
            test_args::has_command_option(input_args_, "--src-device") ||
            test_args::has_command_option(input_args_, "--dst-device")) {
            log_warning(
                LogTest,
                "Ignoring structural command-line arguments (--pattern, --chip-ids, etc.) as a YAML "
                "configuration is being used. Only value overrides (--num-packets, --payload-size, etc.) are applied.");
        }
    }

    if (test_args::has_command_option(input_args_, "--num-packets")) {
        uint32_t num_packets_override = test_args::get_command_option_uint32(input_args_, "--num-packets", 1);
        log_info(LogTest, "Overriding num_packets for all patterns with: {}", num_packets_override);
        for (auto& config : test_configs) {
            if (!config.defaults.has_value()) {
                config.defaults = TrafficPatternConfig{};
            }
            config.defaults->num_packets = num_packets_override;
        }
    }

    if (test_args::has_command_option(input_args_, "--payload-size")) {
        uint32_t payload_size_override = test_args::get_command_option_uint32(input_args_, "--payload-size", 64);
        log_info(LogTest, "Overriding payload_size for all patterns with: {}", payload_size_override);
        for (auto& config : test_configs) {
            if (!config.defaults.has_value()) {
                config.defaults = TrafficPatternConfig{};
            }
            config.defaults->size = payload_size_override;
        }
    }
}

inline std::vector<TestConfig> CmdlineParser::generate_default_configs() {
    log_info(LogTest, "No YAML config provided. Generating a default test configuration from command-line args.");

    TestFabricSetup fabric_setup;
    TestConfig default_test;

    if (test_args::has_command_option(input_args_, "--pattern")) {
        log_info(LogTest, "Generating a high-level pattern test from command line.");
        std::string pattern_type = test_args::get_command_option(input_args_, "--pattern", "");
        TT_FATAL(
            std::find(supported_high_level_patterns.begin(), supported_high_level_patterns.end(), pattern_type) !=
                supported_high_level_patterns.end(),
            "Unsupported pattern type from command line: '{}'. Supported types are: {}",
            pattern_type,
            supported_high_level_patterns);

        HighLevelPatternConfig hlp_config;
        hlp_config.type = pattern_type;
        hlp_config.iterations = test_args::get_command_option_uint32(input_args_, "--iterations", 1);

        default_test.patterns = std::vector<HighLevelPatternConfig>{hlp_config};
    } else {
        log_info(LogTest, "Generating a simple unicast test from command line.");
        std::string src_device_str = test_args::get_command_option(input_args_, "--src-device", "0");
        std::string dst_device_str = test_args::get_command_option(input_args_, "--dst-device", "1");

        FabricNodeId src_device_id = this->device_info_provider_.get_fabric_node_id(std::stoul(src_device_str));
        FabricNodeId dst_device_id = this->device_info_provider_.get_fabric_node_id(std::stoul(dst_device_str));

        fabric_setup.topology = tt::tt_fabric::Topology::Linear;

        TrafficPatternConfig pattern = {.destination = DestinationConfig{.device = dst_device_id}};
        SenderConfig sender = {.device = src_device_id, .patterns = {pattern}};
        default_test.senders = {sender};
    }

    default_test.name = "DefaultCommandLineTest";
    default_test.fabric_setup = fabric_setup;
    default_test.defaults = TrafficPatternConfig{};

    return {default_test};
}

inline std::optional<uint32_t> CmdlineParser::get_master_seed() {
    if (test_args::has_command_option(input_args_, "--master-seed")) {
        uint32_t master_seed = test_args::get_command_option_uint32(input_args_, "--master-seed", 0);
        log_info(tt::LogTest, "Using master seed from command line: {}", master_seed);
        return std::make_optional(master_seed);
    }

    log_info(LogTest, "No master seed provided. Use --master-seed to reproduce.");
    return std::nullopt;
}

inline bool CmdlineParser::dump_built_tests() {
    return test_args::has_command_option(input_args_, "--dump-built-tests");
}

inline std::string CmdlineParser::get_built_tests_dump_file_path(const std::string& default_path) {
    auto dump_path = test_args::get_command_option(input_args_, "--built-tests-dump-file", default_path);
    return dump_path;
}

// YamlConfigParser private helpers
inline CoreCoord YamlConfigParser::parse_core_coord(const YAML::Node& node) {
    TT_FATAL(node.IsSequence() && node.size() == 2, "Expected core coordinates to be a sequence of [x, y]");
    return CoreCoord(parse_scalar<size_t>(node[0]), parse_scalar<size_t>(node[1]));
}

inline MeshId YamlConfigParser::parse_mesh_id(const YAML::Node& yaml_node) {
    uint32_t mesh_id = yaml_node.as<uint32_t>();
    return MeshId{mesh_id};
}

template <typename T>
inline T YamlConfigParser::parse_scalar(const YAML::Node& yaml_node) {
    TT_FATAL(yaml_node.IsScalar(), "Expected yaml node to be a scalar value");
    if constexpr (std::is_same_v<T, MeshId>) {
        return parse_mesh_id(yaml_node);
    } else {
        return yaml_node.as<T>();
    }
}

template <typename T>
inline std::vector<T> YamlConfigParser::parse_scalar_sequence(const YAML::Node& yaml_node) {
    std::vector<T> sequence;
    for (const auto& entry : yaml_node) {
        sequence.push_back(parse_scalar<T>(entry));
    }

    return sequence;
}

template <typename T1, typename T2>
inline std::pair<T1, T2> YamlConfigParser::parse_pair(const YAML::Node& yaml_sequence) {
    TT_FATAL(yaml_sequence.size() == 2, "Expected only 2 entries for the pair");
    return {parse_scalar<T1>(yaml_sequence[0]), parse_scalar<T2>(yaml_sequence[1])};
}

template <typename T1, typename T2>
inline std::vector<std::pair<T1, T2>> YamlConfigParser::parse_pair_sequence(const YAML::Node& yaml_node) {
    std::vector<std::pair<T1, T2>> pair_sequence;
    for (const auto& entry : yaml_node) {
        TT_FATAL(entry.IsSequence(), "Expected each entry to be sequence");
        pair_sequence.push_back(parse_pair<T1, T2>(entry));
    }

    return pair_sequence;
}

template <typename T>
inline std::vector<T> YamlConfigParser::get_elements_in_range(T start, T end) {
    std::vector<T> range(end - start + 1);
    std::iota(range.begin(), range.end(), start);
    return range;
}

inline ParametrizationOptionsMap YamlConfigParser::parse_parametrization_params(const YAML::Node& params_yaml) {
    ParametrizationOptionsMap options;
    TT_FATAL(params_yaml.IsMap(), "Expected 'parametrization_params' to be a map.");

    for (const auto& it : params_yaml) {
        std::string key = parse_scalar<std::string>(it.first);
        const auto& node = it.second;
        TT_FATAL(node.IsSequence(), "Parametrization option '{}' must be a sequence of values.", key);

        if (key == "ftype" || key == "ntype") {
            options[key] = parse_scalar_sequence<std::string>(node);
        } else if (key == "size" || key == "num_packets") {
            options[key] = parse_scalar_sequence<uint32_t>(node);
        } else {
            TT_THROW("Unsupported parametrization parameter: {}", key);
        }
    }
    return options;
}

inline HighLevelPatternConfig YamlConfigParser::parse_high_level_pattern_config(const YAML::Node& pattern_yaml) {
    HighLevelPatternConfig config;
    TT_FATAL(pattern_yaml["type"], "High-level pattern must have a 'type' key.");
    config.type = parse_scalar<std::string>(pattern_yaml["type"]);

    TT_FATAL(
        std::find(supported_high_level_patterns.begin(), supported_high_level_patterns.end(), config.type) !=
            supported_high_level_patterns.end(),
        "Unsupported pattern type: '{}'. Supported types are: {}",
        config.type,
        supported_high_level_patterns);

    if (pattern_yaml["iterations"]) {
        config.iterations = parse_scalar<uint32_t>(pattern_yaml["iterations"]);
    }
    return config;
}

class TestConfigBuilder {
public:
    TestConfigBuilder(IDeviceInfoProvider& device_info_provider, IRouteManager& route_manager, std::mt19937& gen) :
        device_info_provider_(device_info_provider), route_manager_(route_manager), gen_(gen) {}

    std::vector<TestConfig> build_tests(const std::vector<TestConfig>& raw_configs) {
        std::vector<TestConfig> built_tests;

        for (const auto& raw_config : raw_configs) {
            std::vector<TestConfig> parametrized_configs;
            parametrized_configs.push_back(raw_config);

            if (raw_config.parametrization_params.has_value()) {
                for (const auto& [param_name, values_variant] : raw_config.parametrization_params.value()) {
                    std::vector<TestConfig> next_level_configs;

                    for (const auto& current_config : parametrized_configs) {
                        // Handle string-based parameters
                        if (std::holds_alternative<std::vector<std::string>>(values_variant)) {
                            const auto& values = std::get<std::vector<std::string>>(values_variant);
                            for (const auto& value : values) {
                                TestConfig next_config = current_config;
                                next_config.name += "_" + param_name + "_" + value;
                                if (!next_config.defaults.has_value()) {
                                    next_config.defaults = TrafficPatternConfig{};
                                }

                                if (param_name == "ftype") {
                                    next_config.defaults->ftype =
                                        detail::chip_send_type_mapper.from_string(value, "ftype");
                                } else if (param_name == "ntype") {
                                    next_config.defaults->ntype =
                                        detail::noc_send_type_mapper.from_string(value, "ntype");
                                }
                                next_level_configs.push_back(next_config);
                            }
                        }
                        // Handle integer-based parameters
                        else if (std::holds_alternative<std::vector<uint32_t>>(values_variant)) {
                            const auto& values = std::get<std::vector<uint32_t>>(values_variant);
                            for (const auto& value : values) {
                                TestConfig next_config = current_config;
                                next_config.name += "_" + param_name + "_" + std::to_string(value);
                                if (!next_config.defaults.has_value()) {
                                    next_config.defaults = TrafficPatternConfig{};
                                }

                                if (param_name == "size") {
                                    next_config.defaults->size = value;
                                } else if (param_name == "num_packets") {
                                    next_config.defaults->num_packets = value;
                                }
                                next_level_configs.push_back(next_config);
                            }
                        }
                    }
                    // Move the newly generated configs to be the input for the next parameter loop.
                    parametrized_configs = std::move(next_level_configs);
                }
            }

            // For each newly generated parametrized config, expand its high-level patterns
            for (auto& p_config : parametrized_configs) {
                p_config.parametrization_params.reset();  // Clear now-used params before final expansion

                uint32_t max_iterations = 1;
                if (p_config.patterns) {
                    for (const auto& p : p_config.patterns.value()) {
                        max_iterations = std::max(max_iterations, p.iterations.value_or(1));
                    }
                }

                if (max_iterations > 1 && p_config.patterns.has_value() && p_config.patterns.value().size() > 1) {
                    log_warning(
                        LogTest,
                        "Test '{}' has multiple high-level patterns and specifies iterations. All patterns will be "
                        "expanded "
                        "together in each iteration. This may lead to a very large number of connections.",
                        p_config.name);
                }

                for (uint32_t i = 0; i < max_iterations; ++i) {
                    TestConfig iteration_test = p_config;
                    iteration_test.patterns.reset();  // Will be expanded into concrete senders.

                    if (max_iterations > 1) {
                        iteration_test.name += "_iter_" + std::to_string(i);
                    }

                    iteration_test.seed = std::uniform_int_distribution<uint32_t>()(this->gen_);

                    if (p_config.patterns.has_value()) {
                        if (!p_config.senders.empty()) {
                            log_warning(
                                LogTest,
                                "Test '{}' has both concrete 'senders' and high-level 'patterns'. The patterns will be "
                                "expanded and added to the existing senders.",
                                p_config.name);
                        }
                        expand_patterns_into_test(iteration_test, p_config.patterns.value(), i);
                    }

                    // After patterns are expanded, resolve any missing params based on policy
                    resolve_missing_params(iteration_test);

                    validate_test(iteration_test);
                    built_tests.push_back(iteration_test);
                }
            }
        }
        return built_tests;
    }

private:
    void validate_test(const TestConfig& test) const {
        const auto& fabric_setup = test.fabric_setup;

        // validate fabric
        // validate sender/patterns

        /*
        if (test.fabric_setup->topology == tt::tt_fabric::Topology::Linear) {
            TT_FATAL(
                this->route_manager_.are_devices_linear(physical_devices_in_test),
                "For a 'Linear' topology, all specified devices must be in the same row or column. Test: {}",
                test.name);
        }

        for (const auto& sender : test.senders) {
            chip_id_t src_chip = this->device_info_provider_.get_physical_chip_id(sender.device);
            for (const auto& pattern : sender.patterns) {
                TT_FATAL(
                    pattern.destination.has_value() && pattern.destination->device.has_value(),
                    "Pattern for sender {} is missing a destination.",
                    src_chip);
                chip_id_t dst_chip =
                    this->device_info_provider_.get_physical_chip_id(pattern.destination->device.value());
                TT_FATAL(src_chip != dst_chip, "Sender on chip {} cannot have itself as a destination.", src_chip);
            }
        } */
    }

    void expand_patterns_into_test(
        TestConfig& test, const std::vector<HighLevelPatternConfig>& patterns, uint32_t iteration_idx) {
        std::vector<FabricNodeId> devices = device_info_provider_.get_all_node_ids();
        TT_FATAL(!devices.empty(), "Cannot expand patterns because no devices were found.");

        const auto& defaults = test.defaults.value_or(TrafficPatternConfig{});

        for (const auto& pattern : patterns) {
            if (pattern.iterations.has_value() && iteration_idx >= pattern.iterations.value()) {
                continue;
            }

            if (pattern.type == "all_to_all_unicast") {
                expand_all_to_all_unicast(test, devices, defaults);
            }
            // Add other patterns like "all_to_all_multicast" here
            else if (pattern.type == "full_device_random_pairing") {
                expand_full_device_random_pairing(test, devices, defaults);
            } else {
                TT_THROW("Unsupported pattern type: {}", pattern.type);
            }
        }
    }

    void expand_all_to_all_unicast(
        TestConfig& test, const std::vector<FabricNodeId>& devices, const TrafficPatternConfig& base_pattern) {
        log_info(LogTest, "Expanding all_to_all_unicast pattern for test: {}", test.name);

        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;
        for (const auto& src_node : devices) {
            for (const auto& dst_node : devices) {
                if (src_node == dst_node) {
                    continue;
                }
                // For linear topology, only allow connections between chips on the same line.
                if (test.fabric_setup.topology == Topology::Linear) {
                    if (!this->route_manager_.are_devices_linear({src_node, dst_node})) {
                        continue;
                    }
                }
                pairs.push_back({src_node, dst_node});
            }
        }
        add_senders_from_pairs(test, pairs, base_pattern);
    }

    void expand_full_device_random_pairing(
        TestConfig& test,
        std::vector<FabricNodeId> devices,  // Pass by value to allow shuffling
        const TrafficPatternConfig& base_pattern) {
        log_info(LogTest, "Expanding full_device_random_pairing pattern for test: {}", test.name);

        TT_FATAL(
            devices.size() % 2 == 0,
            "full_device_random_pairing pattern requires an even number of devices, but found {}.",
            devices.size());

        if (devices.empty()) {
            return;
        }

        // 1. Shuffle the list of devices
        std::shuffle(devices.begin(), devices.end(), this->gen_);

        // 2. Create pairs
        // TODO: random pairs need consideration for the topology as well, cannot assign any device pair for 1D
        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;
        for (size_t i = 0; i < devices.size(); i += 2) {
            pairs.push_back({devices[i], devices[i + 1]});
        }

        add_senders_from_pairs(test, pairs, base_pattern);
    }

    // Helper to reduce code duplication
    void add_senders_from_pairs(
        TestConfig& test,
        const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
        const TrafficPatternConfig& base_pattern) {
        std::map<FabricNodeId, std::vector<TrafficPatternConfig>> generated_senders;

        for (const auto& pair : pairs) {
            FabricNodeId src_node = pair.first;
            FabricNodeId dst_node = pair.second;

            TrafficPatternConfig specific_pattern;
            specific_pattern.destination = DestinationConfig{.device = dst_node};
            specific_pattern.ftype = ChipSendType::CHIP_UNICAST;

            generated_senders[src_node].push_back(merge_patterns(base_pattern, specific_pattern));
        }

        for (const auto& [src_node, patterns] : generated_senders) {
            test.senders.push_back(SenderConfig{.device = src_node, .patterns = patterns});
        }
    }

    void resolve_missing_params(TestConfig& test) {
        if (!test.on_missing_param_policy.has_value() || test.on_missing_param_policy.value() != "randomize") {
            return;  // No policy or not 'randomize', do nothing.
        }

        for (auto& sender : test.senders) {
            for (auto& pattern : sender.patterns) {
                if (!pattern.ftype.has_value()) {
                    pattern.ftype = get_random_choice<ChipSendType>({ChipSendType::CHIP_UNICAST});
                }
                if (!pattern.ntype.has_value()) {
                    pattern.ntype = get_random_choice<NocSendType>(
                        {NocSendType::NOC_UNICAST_WRITE, NocSendType::NOC_UNICAST_ATOMIC_INC});
                }
                if (!pattern.size.has_value()) {
                    pattern.size = get_random_in_range(64, 4096);
                }
                if (!pattern.num_packets.has_value()) {
                    pattern.num_packets = get_random_in_range(10, 1000);
                }
            }
        }
    }

    IDeviceInfoProvider& device_info_provider_;
    IRouteManager& route_manager_;
    std::mt19937& gen_;

    // Randomization helpers
    template <typename T>
    T get_random_choice(const std::vector<T>& choices) {
        if (choices.empty()) {
            TT_THROW("Cannot make a random choice from an empty list.");
        }
        std::uniform_int_distribution<> distrib(0, choices.size() - 1);
        return choices[distrib(this->gen_)];
    }

    uint32_t get_random_in_range(uint32_t min, uint32_t max) {
        if (min > max) {
            std::swap(min, max);
        }
        std::uniform_int_distribution<uint32_t> distrib(min, max);
        return distrib(this->gen_);
    }
};

// ======================================================================================
// Section 7: Serialization to YAML
// ======================================================================================
// Functions to convert the C++ TestConfig objects back into a YAML representation.

class YamlTestConfigSerializer {
public:
    static void dump(
        const std::vector<TestConfig>& test_configs, const AllocatorPolicies& policies, const std::string& dump_path) {
        YAML::Emitter out;
        out << YAML::BeginMap;

        out << YAML::Key << "allocation_policies";
        out << YAML::Value;
        to_yaml(out, policies);

        out << YAML::Key << "Tests";
        out << YAML::Value;

        out << YAML::BeginSeq;
        for (const auto& test_config : test_configs) {
            to_yaml(out, test_config);
        }
        out << YAML::EndSeq;

        out << YAML::EndMap;

        std::ofstream fout(dump_path);
        fout << out.c_str();
    }

private:
    static void to_yaml(YAML::Emitter& out, const FabricNodeId& id) {
        out << YAML::Flow;
        out << YAML::BeginSeq << *id.mesh_id << id.chip_id << YAML::EndSeq;
    }

    static void to_yaml(YAML::Emitter& out, const CoreCoord& core) {
        out << YAML::Flow;
        out << YAML::BeginSeq << core.x << core.y << YAML::EndSeq;
    }

    static void to_yaml(YAML::Emitter& out, const DestinationConfig& config) {
        out << YAML::BeginMap;
        if (config.device) {
            out << YAML::Key << "device";
            out << YAML::Value;
            to_yaml(out, config.device.value());
        }
        if (config.core) {
            out << YAML::Key << "core";
            out << YAML::Value;
            to_yaml(out, config.core.value());
        }
        if (config.hops) {
            out << YAML::Key << "hops";
            out << YAML::Value;
            out << YAML::BeginMap;
            for (const auto& [dir, count] : config.hops.value()) {
                out << YAML::Key << to_string(dir);
                out << YAML::Value << count;
            }
            out << YAML::EndMap;
        }
        if (config.target_address) {
            out << YAML::Key << "target_address";
            out << YAML::Value << config.target_address.value();
        }
        out << YAML::EndMap;
    }

    static std::string to_string(ChipSendType ftype) {
        return detail::chip_send_type_mapper.to_string(ftype, "ChipSendType");
    }

    static std::string to_string(NocSendType ntype) {
        return detail::noc_send_type_mapper.to_string(ntype, "NocSendType");
    }

    static std::string to_string(RoutingDirection dir) {
        return detail::routing_direction_mapper.to_string(dir, "RoutingDirection");
    }

    static void to_yaml(YAML::Emitter& out, const TrafficPatternConfig& config) {
        out << YAML::BeginMap;

        if (config.ftype) {
            out << YAML::Key << "ftype";
            out << YAML::Value << to_string(config.ftype.value());
        }
        if (config.ntype) {
            out << YAML::Key << "ntype";
            out << YAML::Value << to_string(config.ntype.value());
        }
        if (config.size) {
            out << YAML::Key << "size";
            out << YAML::Value << config.size.value();
        }
        if (config.num_packets) {
            out << YAML::Key << "num_packets";
            out << YAML::Value << config.num_packets.value();
        }
        if (config.destination) {
            out << YAML::Key << "destination";
            out << YAML::Value;
            to_yaml(out, config.destination.value());
        }

        out << YAML::EndMap;
    }

    static void to_yaml(YAML::Emitter& out, const SenderConfig& config) {
        out << YAML::BeginMap;
        out << YAML::Key << "device";
        out << YAML::Value;
        to_yaml(out, config.device);

        if (config.core) {
            out << YAML::Key << "core";
            out << YAML::Value;
            to_yaml(out, config.core.value());
        }

        out << YAML::Key << "patterns";
        out << YAML::Value;
        out << YAML::BeginSeq;
        for (const auto& pattern : config.patterns) {
            to_yaml(out, pattern);
        }
        out << YAML::EndSeq;

        out << YAML::EndMap;
    }

    static std::string to_string(tt::tt_fabric::Topology topology) {
        return detail::topology_mapper.to_string(topology, "Topology");
    }

    static void to_yaml(YAML::Emitter& out, const TestFabricSetup& config) {
        out << YAML::BeginMap;
        out << YAML::Key << "topology";
        out << YAML::Value << to_string(config.topology);
        out << YAML::EndMap;
    }

    static void to_yaml(YAML::Emitter& out, const TestConfig& config) {
        out << YAML::BeginMap;
        out << YAML::Key << "name";
        out << YAML::Value << config.name;
        out << YAML::Key << "seed";
        out << YAML::Value << config.seed;

        out << YAML::Key << "fabric_setup";
        out << YAML::Value;
        to_yaml(out, config.fabric_setup);

        // We only dump concrete senders, not the high-level patterns or randomization policies
        // as they have already been resolved into the sender list.
        out << YAML::Key << "senders";
        out << YAML::Value;
        out << YAML::BeginSeq;
        for (const auto& sender : config.senders) {
            to_yaml(out, sender);
        }
        out << YAML::EndSeq;

        out << YAML::EndMap;
    }

    static std::string to_string(CoreAllocationPolicy policy) {
        return detail::core_allocation_policy_mapper.to_string(policy, "CoreAllocationPolicy");
    }

    static void to_yaml(YAML::Emitter& out, const CoreAllocationConfig& config) {
        out << YAML::BeginMap;
        out << YAML::Key << "policy";
        out << YAML::Value << to_string(config.policy);
        out << YAML::Key << "max_workers_per_core";
        out << YAML::Value << config.max_workers_per_core;
        if (config.default_payload_chunk_size.has_value()) {
            out << YAML::Key << "default_payload_chunk_size";
            out << YAML::Value << config.default_payload_chunk_size.value();
        }
        out << YAML::Key << "initial_pool_size";
        out << YAML::Value << config.initial_pool_size;
        out << YAML::Key << "pool_refill_size";
        out << YAML::Value << config.pool_refill_size;
        out << YAML::EndMap;
    }

    static void to_yaml(YAML::Emitter& out, const AllocatorPolicies& policies) {
        out << YAML::BeginMap;
        out << YAML::Key << "sender";
        out << YAML::Value;
        to_yaml(out, policies.sender_config);
        out << YAML::Key << "receiver";
        out << YAML::Value;
        to_yaml(out, policies.receiver_config);
        out << YAML::EndMap;
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
