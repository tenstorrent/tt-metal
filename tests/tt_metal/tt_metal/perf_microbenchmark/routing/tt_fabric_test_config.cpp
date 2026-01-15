// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_config.hpp"

namespace tt::tt_fabric::fabric_tests {

// Helper function to get supported pattern names from the mapper
std::vector<std::string> get_supported_high_level_patterns() {
    std::vector<std::string> patterns;
    patterns.reserve(detail::high_level_traffic_pattern_mapper.to_enum.size());
    for (const auto& [pattern_name, _] : detail::high_level_traffic_pattern_mapper.to_enum) {
        patterns.push_back(pattern_name);
    }
    return patterns;
}

ParsedYamlConfig YamlConfigParser::parse_file(const std::string& yaml_config_path) {
    std::ifstream yaml_config(yaml_config_path);
    TT_FATAL(not yaml_config.fail(), "Failed to open file: {}", yaml_config_path);

    YAML::Node yaml = YAML::LoadFile(yaml_config_path);

    ParsedYamlConfig result;

    if (yaml["physical_mesh"]) {
        result.physical_mesh_config = parse_physical_mesh_config(yaml["physical_mesh"]);
    }

    if (yaml["allocation_policies"]) {
        result.allocation_policies = parse_allocator_policies(yaml["allocation_policies"]);
    }

    const auto& tests_yaml = yaml["Tests"];
    TT_FATAL(tests_yaml.IsSequence(), "Expected 'Tests' to be a sequence at the root of the YAML file.");
    result.test_configs.reserve(tests_yaml.size());
    for (const auto& test_yaml : tests_yaml) {
        TT_FATAL(test_yaml.IsMap(), "Expected each test in Tests to be a map");

        result.test_configs.emplace_back(parse_test_config(test_yaml));
    }
    return result;
}

DeviceIdentifier YamlConfigParser::parse_device_identifier(const YAML::Node& node) {
    if (node.IsScalar()) {
        ChipId chip_id = parse_scalar<ChipId>(node);
        return chip_id;
    }
    if (node.IsSequence() && node.size() == 2) {
        MeshId mesh_id = parse_mesh_id(node[0]);
        if (node[1].IsScalar()) {
            // Format: [mesh_id, chip_id]
            ChipId chip_id = parse_scalar<ChipId>(node[1]);
            return std::make_pair(mesh_id, chip_id);
        }
        if (node[1].IsSequence()) {
            // Format: [mesh_id, [row, col]]
            MeshCoordinate mesh_coord = parse_mesh_coord(node[1]);
            return std::make_pair(mesh_id, mesh_coord);
        }
    }
    TT_THROW(
        "Unsupported device identifier format. Expected scalar chip_id, sequence [mesh_id, chip_id], or sequence "
        "[mesh_id, [row, col]].");
}

ParsedDestinationConfig YamlConfigParser::parse_destination_config(const YAML::Node& dest_yaml) {
    ParsedDestinationConfig config;
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
    if (dest_yaml["atomic_inc_address"]) {
        config.atomic_inc_address = parse_scalar<uint32_t>(dest_yaml["atomic_inc_address"]);
    }
    return config;
}

ParsedTrafficPatternConfig YamlConfigParser::parse_traffic_pattern_config(const YAML::Node& pattern_yaml) {
    TT_FATAL(pattern_yaml.IsMap(), "Expected pattern to be a map");

    ParsedTrafficPatternConfig config;
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
    if (pattern_yaml["atomic_inc_val"]) {
        config.atomic_inc_val = parse_scalar<uint32_t>(pattern_yaml["atomic_inc_val"]);
    }
    if (pattern_yaml["mcast_start_hops"]) {
        config.mcast_start_hops = parse_scalar<uint32_t>(pattern_yaml["mcast_start_hops"]);
    }

    return config;
}

ParsedSenderConfig YamlConfigParser::parse_sender_config(
    const YAML::Node& sender_yaml, const ParsedTrafficPatternConfig& defaults) {
    TT_FATAL(sender_yaml.IsMap(), "Expected sender to be a map");
    TT_FATAL(sender_yaml["device"] && sender_yaml["patterns"], "Sender config missing required keys");

    ParsedSenderConfig config;
    config.device = parse_device_identifier(sender_yaml["device"]);
    if (sender_yaml["core"]) {
        config.core = parse_core_coord(sender_yaml["core"]);
    }
    if (sender_yaml["link_id"]) {
        config.link_id = parse_scalar<uint32_t>(sender_yaml["link_id"]);
    }

    const auto& patterns_yaml = sender_yaml["patterns"];
    TT_FATAL(patterns_yaml.IsSequence(), "Expected patterns to be a sequence");
    config.patterns.reserve(patterns_yaml.size());
    for (const auto& pattern_node : patterns_yaml) {
        ParsedTrafficPatternConfig specific_pattern = parse_traffic_pattern_config(pattern_node);
        config.patterns.push_back(merge_patterns(defaults, specific_pattern));
    }
    return config;
}

static void validate_latency_test_config(const ParsedTestConfig& test_config) {
    TT_FATAL(
        !test_config.patterns.has_value() || test_config.patterns.value().empty(),
        "Test '{}': latency_test_mode does not support high-level patterns",
        test_config.name);
    TT_FATAL(
        test_config.senders.size() == 1,
        "Test '{}': latency_test_mode requires exactly one sender, got {}",
        test_config.name,
        test_config.senders.size());
    TT_FATAL(
        test_config.senders[0].patterns.size() == 1,
        "Test '{}': latency_test_mode requires exactly one pattern per sender, got {}",
        test_config.name,
        test_config.senders[0].patterns.size());
    TT_FATAL(
        test_config.senders[0].patterns[0].ftype == ChipSendType::CHIP_UNICAST,
        "Test '{}': latency_test_mode only supports unicast",
        test_config.name);
}

ParsedTestConfig YamlConfigParser::parse_test_config(const YAML::Node& test_yaml) {
    ParsedTestConfig test_config;

    test_config.name = parse_scalar<std::string>(test_yaml["name"]);
    log_info(tt::LogTest, "Parsing test: {}", test_config.name);

    TT_FATAL(test_yaml["fabric_setup"], "No fabric setup specified for test: {}", test_config.name);
    test_config.fabric_setup = parse_fabric_setup(test_yaml["fabric_setup"]);

    if (test_yaml["top_level_iterations"]) {
        test_config.num_top_level_iterations = parse_scalar<uint32_t>(test_yaml["top_level_iterations"]);
        if (test_config.num_top_level_iterations == 0) {
            TT_THROW("top_level_iterations must be greater than 0");
        }
    }

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
        const auto& patterns_yaml = test_yaml["patterns"];
        TT_FATAL(patterns_yaml.IsSequence(), "Expected 'patterns' to be a sequence.");
        std::vector<HighLevelPatternConfig> high_level_patterns;
        high_level_patterns.reserve(patterns_yaml.size());
        for (const auto& pattern_node : patterns_yaml) {
            high_level_patterns.push_back(parse_high_level_pattern_config(pattern_node));
        }
        test_config.patterns = high_level_patterns;
    }

    if (test_yaml["skip"]) {
        const auto& skip_yaml = test_yaml["skip"];
        TT_FATAL(skip_yaml.IsSequence(), "Expected 'skip' to be a sequence of platform strings.");
        std::vector<std::string> skips;
        for (const auto& s : skip_yaml) {
            skips.push_back(parse_scalar<std::string>(s));
        }
        test_config.skip = std::move(skips);
    }

    if (test_yaml["senders"]) {
        const auto& senders_yaml = test_yaml["senders"];
        TT_FATAL(senders_yaml.IsSequence(), "Expected senders to be a sequence");
        test_config.senders.reserve(senders_yaml.size());
        for (const auto& sender_node : senders_yaml) {
            test_config.senders.push_back(
                parse_sender_config(sender_node, test_config.defaults.value_or(ParsedTrafficPatternConfig{})));
        }
    }

    if (test_yaml["bw_calc_func"]) {
        test_config.bw_calc_func = parse_scalar<std::string>(test_yaml["bw_calc_func"]);
    }

    // Parse performance test mode (replaces benchmark_mode and latency_test_mode)
    if (test_yaml["benchmark_mode"]) {
        bool benchmark_mode = parse_scalar<bool>(test_yaml["benchmark_mode"]);
        if (benchmark_mode) {
            test_config.performance_test_mode = PerformanceTestMode::BANDWIDTH;
        }
    }

    if (test_yaml["latency_test_mode"]) {
        bool latency_test_mode = parse_scalar<bool>(test_yaml["latency_test_mode"]);
        if (latency_test_mode) {
            TT_FATAL(
                test_config.performance_test_mode == PerformanceTestMode::NONE,
                "Test '{}': benchmark_mode and latency_test_mode are mutually exclusive",
                test_config.name);
            test_config.performance_test_mode = PerformanceTestMode::LATENCY;
        }
    }

    // Validate latency test mode requirements
    if (test_config.performance_test_mode == PerformanceTestMode::LATENCY) {
        validate_latency_test_config(test_config);
    }

    if (test_yaml["sync"]) {
        test_config.global_sync = parse_scalar<bool>(test_yaml["sync"]);
    }

    if (test_yaml["enable_flow_control"]) {
        test_config.enable_flow_control = parse_scalar<bool>(test_yaml["enable_flow_control"]);
    }

    if (test_yaml["skip_packet_validation"]) {
        test_config.skip_packet_validation = parse_scalar<bool>(test_yaml["skip_packet_validation"]);
    }

    return test_config;
}

AllocatorPolicies YamlConfigParser::parse_allocator_policies(const YAML::Node& policies_yaml) {
    TT_FATAL(policies_yaml.IsMap(), "Expected 'allocation_policies' to be a map.");

    std::optional<CoreAllocationConfig> sender_config;
    if (policies_yaml["sender"]) {
        sender_config = parse_core_allocation_config(
            policies_yaml["sender"], CoreAllocationConfig::get_default_sender_allocation_config());
    }

    std::optional<CoreAllocationConfig> receiver_config;
    if (policies_yaml["receiver"]) {
        receiver_config = parse_core_allocation_config(
            policies_yaml["receiver"], CoreAllocationConfig::get_default_receiver_allocation_config());
    }

    std::optional<uint32_t> default_payload_chunk_size;
    if (policies_yaml["default_payload_chunk_size"]) {
        default_payload_chunk_size = parse_scalar<uint32_t>(policies_yaml["default_payload_chunk_size"]);
    }

    return AllocatorPolicies(sender_config, receiver_config, default_payload_chunk_size);
}

CoreAllocationConfig YamlConfigParser::parse_core_allocation_config(
    const YAML::Node& config_yaml, CoreAllocationConfig base_config) {
    CoreAllocationConfig config = base_config;
    if (config_yaml["policy"]) {
        config.policy = detail::core_allocation_policy_mapper.from_string(
            parse_scalar<std::string>(config_yaml["policy"]), "CoreAllocationPolicy");
    }
    if (config_yaml["max_configs_per_core"]) {
        config.max_configs_per_core = parse_scalar<uint32_t>(config_yaml["max_configs_per_core"]);
    }
    if (config_yaml["initial_pool_size"]) {
        config.initial_pool_size = parse_scalar<uint32_t>(config_yaml["initial_pool_size"]);
    }
    if (config_yaml["pool_refill_size"]) {
        config.pool_refill_size = parse_scalar<uint32_t>(config_yaml["pool_refill_size"]);
    }
    return config;
}

TestFabricSetup YamlConfigParser::parse_fabric_setup(const YAML::Node& fabric_setup_yaml) {
    TT_FATAL(fabric_setup_yaml.IsMap(), "Expected fabric setup to be a map");
    TT_FATAL(fabric_setup_yaml["topology"], "Fabric setup missing Topolgy key");
    TestFabricSetup fabric_setup;
    auto topology_str = parse_scalar<std::string>(fabric_setup_yaml["topology"]);
    fabric_setup.topology = detail::topology_mapper.from_string(topology_str, "Topology");

    if (fabric_setup_yaml["fabric_tensix_config"]) {
        auto fabric_type_str = parse_scalar<std::string>(fabric_setup_yaml["fabric_tensix_config"]);
        fabric_setup.fabric_tensix_config =
            detail::fabric_tensix_type_mapper.from_string(fabric_type_str, "FabricTensixConfig");
    } else {
        log_info(tt::LogTest, "No fabric tensix config specified, defaulting to DISABLED");
        fabric_setup.fabric_tensix_config = FabricTensixConfig::DISABLED;
    }

    if (fabric_setup_yaml["fabric_reliability_mode"]) {
        auto reliability_mode_str = parse_scalar<std::string>(fabric_setup_yaml["fabric_reliability_mode"]);
        fabric_setup.fabric_reliability_mode =
            detail::fabric_reliability_mode_mapper.from_string(reliability_mode_str, "FabricReliabilityMode");
    } else {
        log_info(tt::LogTest, "No fabric reliability mode specified, defaulting to STRICT_SYSTEM_HEALTH_SETUP_MODE");
        fabric_setup.fabric_reliability_mode = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    }

    if (fabric_setup_yaml["num_links"]) {
        fabric_setup.num_links = parse_scalar<uint32_t>(fabric_setup_yaml["num_links"]);
    } else {
        fabric_setup.num_links = 1;
    }

    if (fabric_setup_yaml["max_packet_size"]) {
        fabric_setup.max_packet_size = parse_scalar<uint32_t>(fabric_setup_yaml["max_packet_size"]);
    }

    // Handle torus_config for Torus topology
    if (fabric_setup.topology == Topology::Torus) {
        if (fabric_setup_yaml["torus_config"]) {
            fabric_setup.torus_config = parse_scalar<std::string>(fabric_setup_yaml["torus_config"]);
        } else {
            // Default to "XY" when topology is Torus but no torus_config is specified
            fabric_setup.torus_config = "XY";
        }

        // Validate torus_config value
        const auto& config = fabric_setup.torus_config.value();
        TT_FATAL(
            config == "X" || config == "Y" || config == "XY",
            "Invalid torus_config '{}'. Supported values are: 'X', 'Y', 'XY'",
            config);
    }

    return fabric_setup;
}

PhysicalMeshConfig YamlConfigParser::parse_physical_mesh_config(const YAML::Node& physical_mesh_yaml) {
    TT_FATAL(physical_mesh_yaml.IsMap(), "Expected physical mesh config to be a map");
    TT_FATAL(
        physical_mesh_yaml["mesh_descriptor_path"].IsDefined() && physical_mesh_yaml["eth_coord_mapping"].IsDefined(),
        "physical_mesh config must contain both 'mesh_descriptor_path' and 'eth_coord_mapping'");

    PhysicalMeshConfig physical_mesh_config;
    physical_mesh_config.mesh_descriptor_path = parse_scalar<std::string>(physical_mesh_yaml["mesh_descriptor_path"]);
    physical_mesh_config.eth_coord_mapping = parse_2d_array<EthCoord>(physical_mesh_yaml["eth_coord_mapping"]);

    return physical_mesh_config;
}

// CmdlineParser methods
CmdlineParser::CmdlineParser(const std::vector<std::string>& input_args) : input_args_(input_args) {
    if (test_args::has_command_option(input_args_, "--filter")) {
        auto filter = test_args::get_command_option(input_args_, "--filter", "");
        auto splitter = filter.find('.');
        filter_type = filter.substr(0, splitter);
        filter_value = filter.substr(splitter + 1);
    }
}

std::optional<std::string> CmdlineParser::get_yaml_config_path() {
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

bool CmdlineParser::check_filter(ParsedTestConfig& test_config, bool fine_grained) {
    if (filter_type.has_value()) {
        if (filter_type.value() == "name" || filter_type.value() == "Name") {
            return test_config.name == filter_value;
        }
        if (filter_type.value() == "topology" || filter_type.value() == "Topology") {
            auto topo = tt::tt_fabric::Topology::Linear;  // Default value
            if (filter_value == "Ring") {
                topo = tt::tt_fabric::Topology::Ring;
            } else if (filter_value == "Linear") {
                topo = tt::tt_fabric::Topology::Linear;
            } else if (filter_value == "Mesh") {
                topo = tt::tt_fabric::Topology::Mesh;
            } else if (filter_value == "Torus") {
                topo = tt::tt_fabric::Topology::Torus;
            } else {
                log_info(
                    tt::LogTest,
                    "Unsupported topology filter value: '{}'. Supported values are: Ring, Linear, Mesh, Torus",
                    filter_value);
                return false;
            }
            return test_config.fabric_setup.topology == topo;
        }
        if (filter_type.value() == "benchmark_mode" || filter_type.value() == "Benchmark_Mode") {
            if (filter_value == "true") {
                return test_config.performance_test_mode == PerformanceTestMode::BANDWIDTH;
            }
            if (filter_value == "false") {
                return test_config.performance_test_mode != PerformanceTestMode::BANDWIDTH;
            }
            log_info(
                tt::LogTest,
                "Unsupported benchmark filter value: '{}'. Supported values are: true, false",
                filter_value);
            return false;
        }
        if (filter_type.value() == "sync" || filter_type.value() == "Sync") {
            if (filter_value == "true") {
                return test_config.global_sync;
            }
            if (filter_value == "false") {
                return !test_config.global_sync;
            }
            log_info(
                tt::LogTest, "Unsupported sync filter value: '{}'. Supported values are: true, false", filter_value);
            return false;
        }
        if (filter_type.value() == "num_links" || filter_type.value() == "Num_Links") {
            if (fine_grained) {
                if (test_config.parametrization_params.has_value() &&
                    !test_config.parametrization_params.value().empty()) {
                    auto& params = test_config.parametrization_params.value();
                    auto it = params.find("num_links");
                    if (it != params.end() && std::holds_alternative<std::vector<uint32_t>>(it->second)) {
                        const auto& num_links_vec = std::get<std::vector<uint32_t>>(it->second);
                        for (const auto& num_links : num_links_vec) {
                            if (num_links == stoi(filter_value.value())) {
                                return true;
                            }
                        }
                    }
                }
            }
            return test_config.fabric_setup.num_links == stoi(filter_value.value());
        }
        if (filter_type.value() == "ntype") {
            if (fine_grained) {
                if (test_config.parametrization_params.has_value() &&
                    !test_config.parametrization_params.value().empty()) {
                    auto& params = test_config.parametrization_params.value();
                    auto it = params.find("ntype");
                    if (it != params.end() && std::holds_alternative<std::vector<std::string>>(it->second)) {
                        const auto& ntype_vec = std::get<std::vector<std::string>>(it->second);
                        for (const auto& ntype : ntype_vec) {
                            if (ntype == filter_value.value()) {
                                return true;
                            }
                        }
                    }
                }
            }
            // soft filter
            std::optional<tt::tt_fabric::NocSendType> ntype;
            ntype = detail::noc_send_type_mapper.from_string(filter_value.value(), "ntype");
            bool checker = false;
            for (const auto& sender : test_config.senders) {
                for (const auto& pattern : sender.patterns) {
                    if (pattern.ntype.has_value()) {
                        if (pattern.ntype.value() == ntype.value()) {
                            checker = true;
                            break;
                        }
                    }
                }
            }
            if (checker) {
                for (auto& sender : test_config.senders) {
                    sender.patterns.erase(
                        std::remove_if(
                            sender.patterns.begin(),
                            sender.patterns.end(),
                            [&](const auto& pattern) {
                                return pattern.ntype.has_value() && pattern.ntype.value() != ntype;
                            }),
                        sender.patterns.end());
                }
            }
            if (!checker && test_config.defaults.has_value() && test_config.defaults.value().ntype.has_value()) {
                checker = test_config.defaults.value().ntype.value() == ntype.value();
            }
            return checker;
        }
        if (filter_type.value() == "ftype") {
            // soft filter
            if (fine_grained) {
                if (test_config.parametrization_params.has_value() &&
                    !test_config.parametrization_params.value().empty()) {
                    auto& params = test_config.parametrization_params.value();
                    auto it = params.find("ftype");
                    if (it != params.end() && std::holds_alternative<std::vector<std::string>>(it->second)) {
                        const auto& ftype_vec = std::get<std::vector<std::string>>(it->second);
                        for (const auto& ftype : ftype_vec) {
                            if (ftype == filter_value.value()) {
                                return true;
                            }
                        }
                    }
                }
            }
            std::optional<tt::tt_fabric::ChipSendType> ftype;
            ftype = detail::chip_send_type_mapper.from_string(filter_value.value(), "ftype");
            bool checker = false;
            for (const auto& sender : test_config.senders) {
                for (const auto& pattern : sender.patterns) {
                    if (pattern.ftype.has_value()) {
                        if (pattern.ftype.value() == ftype.value()) {
                            checker = true;
                            break;
                        }
                    }
                }
            }
            if (checker) {
                for (auto& sender : test_config.senders) {
                    sender.patterns.erase(
                        std::remove_if(
                            sender.patterns.begin(),
                            sender.patterns.end(),
                            [&](const auto& pattern) {
                                return pattern.ftype.has_value() && pattern.ftype.value() != ftype.value();
                            }),
                        sender.patterns.end());
                }
            }
            if (!checker && test_config.defaults.has_value() && test_config.defaults.value().ftype.has_value()) {
                checker = test_config.defaults.value().ftype.value() == ftype.value();
            }
            return checker;
        }
        if (filter_type.value() == "num_packets") {
            if (fine_grained) {
                if (test_config.parametrization_params.has_value() &&
                    !test_config.parametrization_params.value().empty()) {
                    auto& params = test_config.parametrization_params.value();
                    auto it = params.find("num_packets");
                    if (it != params.end() && std::holds_alternative<std::vector<uint32_t>>(it->second)) {
                        const auto& num_packets_vec = std::get<std::vector<uint32_t>>(it->second);
                        for (const auto& num_packets : num_packets_vec) {
                            if (num_packets == stoi(filter_value.value())) {
                                return true;
                            }
                        }
                    }
                }
            }
            // soft filter
            uint32_t num_packets = stoi(filter_value.value());
            bool checker = false;
            for (const auto& sender : test_config.senders) {
                for (const auto& pattern : sender.patterns) {
                    if (pattern.num_packets.has_value()) {
                        if (pattern.num_packets.value() == num_packets) {
                            checker = true;
                            break;
                        }
                    }
                }
            }
            if (checker) {
                for (auto& sender : test_config.senders) {
                    sender.patterns.erase(
                        std::remove_if(
                            sender.patterns.begin(),
                            sender.patterns.end(),
                            [&](const auto& pattern) {
                                return pattern.num_packets.has_value() && pattern.num_packets.value() != num_packets;
                            }),
                        sender.patterns.end());
                }
            }
            if (!checker && test_config.defaults.has_value() && test_config.defaults.value().num_packets.has_value()) {
                checker = test_config.defaults.value().num_packets.value() == num_packets;
            }
            return checker;
        }
        if (filter_type.value() == "size") {
            if (fine_grained) {
                if (test_config.parametrization_params.has_value() &&
                    !test_config.parametrization_params.value().empty()) {
                    auto& params = test_config.parametrization_params.value();
                    auto it = params.find("size");
                    if (it != params.end() && std::holds_alternative<std::vector<uint32_t>>(it->second)) {
                        const auto& size_vec = std::get<std::vector<uint32_t>>(it->second);
                        for (const auto& size : size_vec) {
                            if (size == stoi(filter_value.value())) {
                                return true;
                            }
                        }
                    }
                }
            }
            uint32_t size = stoi(filter_value.value());
            bool checker = false;
            for (const auto& sender : test_config.senders) {
                for (const auto& pattern : sender.patterns) {
                    if (pattern.size.has_value()) {
                        if (pattern.size.value() == size) {
                            checker = true;
                            break;
                        }
                    }
                }
            }

            if (checker) {
                for (auto& sender : test_config.senders) {
                    sender.patterns.erase(
                        std::remove_if(
                            sender.patterns.begin(),
                            sender.patterns.end(),
                            [&](const auto& pattern) {
                                return pattern.size.has_value() && pattern.size.value() != size;
                            }),
                        sender.patterns.end());
                }
            }
            if (!checker && test_config.defaults.has_value() && test_config.defaults.value().size.has_value()) {
                checker = test_config.defaults.value().size.value() == size;
            }
            return checker;
        }
        if (filter_type.value() == "pattern") {
            bool checker = false;
            if (test_config.patterns.has_value()) {
                for (auto& high_level_pattern : test_config.patterns.value()) {
                    if (high_level_pattern.type == filter_value.value()) {
                        checker = true;
                        break;
                    }
                }
            }
            return checker;
        }
        log_info(
            tt::LogTest,
            "Unsupported filter type: '{}'. Supported types are: name, topology, benchmark_mode, "
            "sync, num_links, ntype, ftype, num_packets, size, pattern",
            filter_type.value());
        return false;
    }
    return true;
}

void CmdlineParser::apply_overrides(std::vector<ParsedTestConfig>& test_configs) {
    bool is_default_test = (test_configs.size() == 1 && test_configs[0].name == "DefaultCommandLineTest");

    if (!is_default_test) {
        if (test_args::has_command_option(input_args_, "--pattern") ||
            test_args::has_command_option(input_args_, "--chip-ids") ||
            test_args::has_command_option(input_args_, "--iterations") ||
            test_args::has_command_option(input_args_, "--src-device") ||
            test_args::has_command_option(input_args_, "--dst-device") ||
            test_args::has_command_option(input_args_, "--topology")) {
            log_warning(
                LogTest,
                "Ignoring structural command-line arguments (--pattern, --chip-ids, --topology, etc.) as a YAML "
                "configuration is being used. Only value overrides (--num-packets, --payload-size, etc.) are applied.");
        }
    }

    if (test_args::has_command_option(input_args_, "--num-packets")) {
        uint32_t num_packets_override = test_args::get_command_option_uint32(input_args_, "--num-packets", 1);
        log_info(LogTest, "Overriding num_packets for all patterns with: {}", num_packets_override);
        for (auto& config : test_configs) {
            if (!config.defaults.has_value()) {
                config.defaults = ParsedTrafficPatternConfig{};
            }
            config.defaults->num_packets = num_packets_override;
        }
    }

    if (test_args::has_command_option(input_args_, "--payload-size")) {
        uint32_t payload_size_override = test_args::get_command_option_uint32(input_args_, "--payload-size", 64);
        log_info(LogTest, "Overriding payload_size for all patterns with: {}", payload_size_override);
        for (auto& config : test_configs) {
            if (!config.defaults.has_value()) {
                config.defaults = ParsedTrafficPatternConfig{};
            }
            config.defaults->size = payload_size_override;
        }
    }
}

std::optional<uint32_t> CmdlineParser::get_master_seed() {
    if (test_args::has_command_option(input_args_, "--master-seed")) {
        uint32_t master_seed = test_args::get_command_option_uint32(input_args_, "--master-seed", 0);
        log_info(tt::LogTest, "Using master seed from command line: {}", master_seed);
        return std::make_optional(master_seed);
    }

    log_info(LogTest, "No master seed provided. Use --master-seed to reproduce.");
    return std::nullopt;
}

bool CmdlineParser::dump_built_tests() { return test_args::has_command_option(input_args_, "--dump-built-tests"); }

std::string CmdlineParser::get_built_tests_dump_file_name(const std::string& default_file_name) {
    auto dump_file = test_args::get_command_option(input_args_, "--built-tests-dump-file", default_file_name);
    return dump_file;
}

bool CmdlineParser::has_help_option() { return test_args::has_command_option(input_args_, "--help"); }

void CmdlineParser::print_help() {
    log_info(LogTest, "Usage: test_tt_fabric --test_config <path> [options]");
    log_info(LogTest, "[This test needs a yaml configuration file to run, see test_features.yaml for examples]");
    log_info(LogTest, "");
    log_info(LogTest, "General Options:");
    log_info(LogTest, "  --help                                       Print this help message.");
    log_info(
        LogTest,
        "  --master-seed <seed>                         Master seed for all random operations to ensure "
        "reproducibility.");
    log_info(LogTest, "");
    log_info(LogTest, "Value Overrides:");
    log_info(
        LogTest,
        "  --num-packets <N>                            Override the number of packets for all traffic "
        "patterns.");
    log_info(
        LogTest,
        "  --payload-size <bytes>                       Override the payload size in bytes for all "
        "traffic patterns.");
    log_info(LogTest, "");
    log_info(LogTest, "Debugging and Output Options:");
    log_info(
        LogTest,
        "  --dump-built-tests                           Dump the fully-expanded test configurations to a YAML file.");
    log_info(
        LogTest,
        "  --built-tests-dump-file <filename>           Specify the filename for the dumped tests. Default: "
        "built_tests.yaml.");
    log_info(LogTest, "  --filter <testname>           Specify a filter for the test suite");
    log_info(LogTest, "");
    log_info(LogTest, "Progress Monitoring Options:");
    log_info(LogTest, "  --show-progress                              Enable real-time progress monitoring.");
    log_info(LogTest, "  --progress-interval <seconds>                Poll interval (default: 2).");
    log_info(LogTest, "  --hung-threshold <seconds>                   Hung detection threshold (default: 30).");
}

// Progress monitoring methods
bool CmdlineParser::show_progress() { return test_args::has_command_option(input_args_, "--show-progress"); }

uint32_t CmdlineParser::get_progress_interval() {
    return test_args::get_command_option_uint32(input_args_, "--progress-interval", 2);
}

uint32_t CmdlineParser::get_hung_threshold() {
    return test_args::get_command_option_uint32(input_args_, "--hung-threshold", 30);
}

// YamlConfigParser private helpers
CoreCoord YamlConfigParser::parse_core_coord(const YAML::Node& node) {
    TT_FATAL(node.IsSequence() && node.size() == 2, "Expected core coordinates to be a sequence of [x, y]");
    return CoreCoord(parse_scalar<size_t>(node[0]), parse_scalar<size_t>(node[1]));
}

MeshCoordinate YamlConfigParser::parse_mesh_coord(const YAML::Node& node) {
    TT_FATAL(node.IsSequence() && node.size() == 2, "Expected mesh coordinates to be a sequence of [row, col]");
    std::vector<uint32_t> coords = {parse_scalar<uint32_t>(node[0]), parse_scalar<uint32_t>(node[1])};
    return MeshCoordinate(coords);
}

MeshId YamlConfigParser::parse_mesh_id(const YAML::Node& yaml_node) {
    uint32_t mesh_id = yaml_node.as<uint32_t>();
    return MeshId{mesh_id};
}

ParametrizationOptionsMap YamlConfigParser::parse_parametrization_params(const YAML::Node& params_yaml) {
    ParametrizationOptionsMap options;
    TT_FATAL(params_yaml.IsMap(), "Expected 'parametrization_params' to be a map.");

    for (const auto& it : params_yaml) {
        std::string key = parse_scalar<std::string>(it.first);
        const auto& node = it.second;
        TT_FATAL(node.IsSequence(), "Parametrization option '{}' must be a sequence of values.", key);

        if (key == "ftype" || key == "ntype") {
            options[key] = parse_scalar_sequence<std::string>(node);
        } else if (key == "size" || key == "num_packets" || key == "num_links") {
            options[key] = parse_scalar_sequence<uint32_t>(node);
        } else {
            TT_THROW("Unsupported parametrization parameter: {}", key);
        }
    }
    return options;
}

HighLevelPatternConfig YamlConfigParser::parse_high_level_pattern_config(const YAML::Node& pattern_yaml) {
    HighLevelPatternConfig config;
    TT_FATAL(pattern_yaml["type"], "High-level pattern must have a 'type' key.");
    config.type = parse_scalar<std::string>(pattern_yaml["type"]);

    TT_FATAL(
        detail::high_level_traffic_pattern_mapper.to_enum.contains(config.type),
        "Unsupported pattern type: '{}'. Supported types are: {}",
        config.type,
        get_supported_high_level_patterns());

    if (pattern_yaml["iterations"]) {
        config.iterations = parse_scalar<uint32_t>(pattern_yaml["iterations"]);
    }
    return config;
}

// TestConfigBuilder methods
std::vector<TestConfig> TestConfigBuilder::build_tests(
    const std::vector<ParsedTestConfig>& raw_configs, CmdlineParser& cmdline_parser) {
    std::vector<TestConfig> built_tests;

    for (const auto& raw_config : raw_configs) {
        std::vector<ParsedTestConfig> parametrized_configs = this->expand_parametrizations(raw_config);

        // For each newly generated parametrized config, expand its high-level patterns
        for (auto& p_config : parametrized_configs) {
            if (!cmdline_parser.check_filter(p_config, false)) {
                log_info(LogTest, "Skipping part of test '{}' due to filter criteria.", p_config.name);
                continue;
            }
            auto expanded_tests = this->expand_high_level_patterns(p_config);
            built_tests.insert(
                built_tests.end(),
                std::make_move_iterator(expanded_tests.begin()),
                std::make_move_iterator(expanded_tests.end()));
        }
    }

    return built_tests;
}

bool TestConfigBuilder::should_skip_test_on_platform(const ParsedTestConfig& test_config) const {
    // Skip if the test declares platforms to skip and this platform matches
    if (test_config.skip.has_value()) {
        // Determine current platform identifiers
        auto arch_name = tt::tt_metal::hal::get_arch_name();
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        std::string cluster_name = std::string(enchantum::to_string(cluster_type));
        for (const auto& token : test_config.skip.value()) {
            if (token == arch_name || token == cluster_name) {
                log_info(LogTest, "Skipping test '{}' on architecture or platform '{}'", test_config.name, token);
                return true;
            }
        }
    }
    return false;
}

bool TestConfigBuilder::should_skip_test_on_topology(const ParsedTestConfig& test_config) const {
    if (test_config.fabric_setup.topology == Topology::Ring) {
        uint32_t num_devices = device_info_provider_.get_local_node_ids().size();
        if (num_devices < MIN_RING_TOPOLOGY_DEVICES) {
            log_info(
                LogTest,
                "Skipping test '{}' - Ring topology requires at least {} devices, but only {} devices available",
                test_config.name,
                MIN_RING_TOPOLOGY_DEVICES,
                num_devices);
            return true;
        }
    }
    return false;
}

TestConfig TestConfigBuilder::resolve_test_config(const ParsedTestConfig& parsed_test, uint32_t iteration_number) {
    TestConfig resolved_test;
    resolved_test.name = parsed_test.name;
    resolved_test.parametrized_name = parsed_test.parametrized_name;
    resolved_test.iteration_number = iteration_number;
    resolved_test.fabric_setup = parsed_test.fabric_setup;
    resolved_test.on_missing_param_policy = parsed_test.on_missing_param_policy;
    resolved_test.parametrization_params = parsed_test.parametrization_params;
    resolved_test.patterns = parsed_test.patterns;
    resolved_test.bw_calc_func = parsed_test.bw_calc_func;
    resolved_test.seed = parsed_test.seed;
    resolved_test.sync_configs = parsed_test.sync_configs;
    resolved_test.performance_test_mode = parsed_test.performance_test_mode;
    resolved_test.global_sync = parsed_test.global_sync;
    resolved_test.enable_flow_control = parsed_test.enable_flow_control;
    resolved_test.skip_packet_validation = parsed_test.skip_packet_validation;

    // Resolve defaults
    if (parsed_test.defaults.has_value()) {
        resolved_test.defaults = resolve_traffic_pattern(parsed_test.defaults.value());
    }

    // Resolve senders
    resolved_test.senders.reserve(parsed_test.senders.size());
    for (const auto& parsed_sender : parsed_test.senders) {
        resolved_test.senders.push_back(resolve_sender_config(parsed_sender));
    }

    return resolved_test;
}

SenderConfig TestConfigBuilder::resolve_sender_config(const ParsedSenderConfig& parsed_sender) {
    SenderConfig resolved_sender;
    resolved_sender.device = resolve_device_identifier(parsed_sender.device, device_info_provider_);
    resolved_sender.core = parsed_sender.core;
    resolved_sender.link_id = parsed_sender.link_id.value_or(0);  // Default to link 0 if not specified

    resolved_sender.patterns.reserve(parsed_sender.patterns.size());
    for (const auto& parsed_pattern : parsed_sender.patterns) {
        resolved_sender.patterns.push_back(resolve_traffic_pattern(parsed_pattern));
    }

    return resolved_sender;
}

TrafficPatternConfig TestConfigBuilder::resolve_traffic_pattern(const ParsedTrafficPatternConfig& parsed_pattern) {
    TrafficPatternConfig resolved_pattern;
    resolved_pattern.ftype = parsed_pattern.ftype;
    resolved_pattern.ntype = parsed_pattern.ntype;
    resolved_pattern.size = parsed_pattern.size;
    resolved_pattern.num_packets = parsed_pattern.num_packets;
    resolved_pattern.atomic_inc_val = parsed_pattern.atomic_inc_val;
    resolved_pattern.mcast_start_hops = parsed_pattern.mcast_start_hops;

    if (parsed_pattern.destination.has_value()) {
        resolved_pattern.destination = resolve_destination_config(parsed_pattern.destination.value());
    }

    // Credit info fields (will be populated by GlobalAllocator during resource allocation)
    resolved_pattern.sender_credit_info = std::nullopt;
    resolved_pattern.credit_return_batch_size = std::nullopt;

    return resolved_pattern;
}

DestinationConfig TestConfigBuilder::resolve_destination_config(const ParsedDestinationConfig& parsed_dest) {
    DestinationConfig resolved_dest;
    if (parsed_dest.device.has_value()) {
        resolved_dest.device = resolve_device_identifier(parsed_dest.device.value(), device_info_provider_);
    }
    resolved_dest.core = parsed_dest.core;
    resolved_dest.hops = parsed_dest.hops;
    resolved_dest.target_address = parsed_dest.target_address;
    resolved_dest.atomic_inc_address = parsed_dest.atomic_inc_address;

    return resolved_dest;
}

std::vector<TestConfig> TestConfigBuilder::expand_high_level_patterns(ParsedTestConfig& p_config) {
    std::vector<TestConfig> expanded_tests;

    p_config.parametrization_params.reset();  // Clear now-used params before final expansion

    uint32_t max_iterations = 1;
    if (p_config.patterns) {
        for (const auto& p : p_config.patterns.value()) {
            if (p.iterations.has_value()) {
                max_iterations = std::max(max_iterations, p.iterations.value());
                // Edge Case: If both iterations and all_to_one are supplied, iterations will override the number of
                // iterations set by all_to_one
                if (p.type == "all_to_one") {
                    log_warning(
                        tt::LogTest,
                        "'iterations' specified alongside 'all_to_one' test, `iterations` will be followed instead "
                        "of auto-generating iterations based on number of devices");
                }
            } else if (p.type == "all_to_one") {
                // Dynamically calculate iterations for all_to_one patterns based on number of devices
                uint32_t num_devices = static_cast<uint32_t>(device_info_provider_.get_global_node_ids().size());
                max_iterations = std::max(max_iterations, num_devices);
                log_info(
                    LogTest,
                    "Auto-detected {} iterations for all_to_one pattern in test '{}'",
                    num_devices,
                    p_config.name);
            } else if (p.type == "sequential_all_to_all") {
                // Dynamically calculate iterations for sequential_all_to_all patterns based on all device pairs
                auto all_pairs = this->route_manager_.get_all_to_all_unicast_pairs();
                uint32_t num_pairs = static_cast<uint32_t>(all_pairs.size());
                max_iterations = std::max(max_iterations, num_pairs);
                log_info(
                    LogTest,
                    "Auto-detected {} iterations for sequential_all_to_all pattern in test '{}'",
                    num_pairs,
                    p_config.name);
            }
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

    expanded_tests.reserve(max_iterations);
    for (uint32_t i = 0; i < max_iterations; ++i) {
        ParsedTestConfig iteration_test = p_config;
        iteration_test.patterns.reset();  // Will be expanded into concrete senders.

        // Initialize parametrized_name with original name if empty
        if (iteration_test.parametrized_name.empty()) {
            iteration_test.parametrized_name = iteration_test.name;
        }
        if (max_iterations > 1) {
            // Use optimized string concatenation utility for parametrized name
            detail::append_with_separator(iteration_test.parametrized_name, "_", "iter", i);
        }

        iteration_test.seed = std::uniform_int_distribution<uint32_t>()(this->gen_);

        // Add line sync pattern expansion if enabled
        if (iteration_test.global_sync) {
            expand_sync_patterns(iteration_test);
        }

        if (p_config.patterns.has_value()) {
            if (!p_config.senders.empty()) {
                TT_FATAL(
                    false,
                    "Test '{}' has both concrete 'senders' and high-level 'patterns' specified. This is ambiguous. "
                    "Please specify one or the other.",
                    p_config.name);
            }
            expand_patterns_into_test(iteration_test, p_config.patterns.value(), i);
        } else if (p_config.defaults.has_value()) {
            // if we have concrete senders, we still need to apply the defaults to them
            for (auto& sender : iteration_test.senders) {
                for (auto& pattern : sender.patterns) {
                    pattern = merge_patterns(p_config.defaults.value(), pattern);
                }
            }
        }

        // After patterns are expanded, duplicate senders for different links if specified
        if (!expand_link_duplicates(iteration_test)) {
            // Test was skipped due to insufficient routing planes, continue to next iteration
            continue;
        }

        // After patterns are expanded, resolve any missing params based on policy
        resolve_missing_params(iteration_test);

        // After expansion and resolution, apply universal transformations like mcast splitting.
        split_all_unicast_or_multicast_patterns(iteration_test);

        // Convert to resolved TestConfig
        TestConfig resolved_test = resolve_test_config(iteration_test, i);

        validate_test(resolved_test);
        expanded_tests.push_back(resolved_test);
    }
    return expanded_tests;
}

std::vector<ParsedTestConfig> TestConfigBuilder::expand_parametrizations(const ParsedTestConfig& raw_config) {
    std::vector<ParsedTestConfig> parametrized_configs;
    parametrized_configs.push_back(raw_config);

    if (raw_config.parametrization_params.has_value()) {
        for (const auto& [param_name, values_variant] : raw_config.parametrization_params.value()) {
            std::vector<ParsedTestConfig> next_level_configs;

            // Pre-calculate total size to avoid reallocations
            size_t total_new_configs = 0;
            if (std::holds_alternative<std::vector<std::string>>(values_variant)) {
                const auto& values = std::get<std::vector<std::string>>(values_variant);
                total_new_configs = parametrized_configs.size() * values.size();
            } else if (std::holds_alternative<std::vector<uint32_t>>(values_variant)) {
                const auto& values = std::get<std::vector<uint32_t>>(values_variant);
                total_new_configs = parametrized_configs.size() * values.size();
            }
            next_level_configs.reserve(total_new_configs);

            for (const auto& current_config : parametrized_configs) {
                // Handle string-based parameters
                if (std::holds_alternative<std::vector<std::string>>(values_variant)) {
                    const auto& values = std::get<std::vector<std::string>>(values_variant);
                    for (const auto& value : values) {
                        next_level_configs.emplace_back(current_config);
                        auto& next_config = next_level_configs.back();
                        // Explicitly preserve performance_test_mode
                        next_config.performance_test_mode = current_config.performance_test_mode;

                        // Initialize parametrized_name with original name if empty
                        if (next_config.parametrized_name.empty()) {
                            next_config.parametrized_name = next_config.name;
                        }
                        // Update parametrized name to include parameter name and value
                        detail::append_with_separator(next_config.parametrized_name, "_", param_name, value);

                        ParsedTrafficPatternConfig param_default;
                        if (param_name == "ftype") {
                            param_default.ftype = detail::chip_send_type_mapper.from_string(value, "ftype");
                        } else if (param_name == "ntype") {
                            param_default.ntype = detail::noc_send_type_mapper.from_string(value, "ntype");
                        }
                        next_config.defaults = merge_patterns(
                            current_config.defaults.value_or(ParsedTrafficPatternConfig{}), param_default);
                    }
                }
                // Handle integer-based parameters
                else if (std::holds_alternative<std::vector<uint32_t>>(values_variant)) {
                    const auto& values = std::get<std::vector<uint32_t>>(values_variant);
                    for (const auto& value : values) {
                        next_level_configs.emplace_back(current_config);
                        auto& next_config = next_level_configs.back();
                        // Explicitly preserve performance_test_mode
                        next_config.performance_test_mode = current_config.performance_test_mode;

                        // Initialize parametrized_name with original name if empty
                        if (next_config.parametrized_name.empty()) {
                            next_config.parametrized_name = next_config.name;
                        }
                        // Update parametrized name to include parameter name and value
                        detail::append_with_separator(
                            next_config.parametrized_name, "_", param_name, std::to_string(value));

                        if (param_name == "num_links") {
                            // num_links is part of fabric_setup, not traffic pattern defaults
                            next_config.fabric_setup.num_links = value;
                        } else {
                            ParsedTrafficPatternConfig param_default;
                            if (param_name == "size") {
                                param_default.size = value;
                            } else if (param_name == "num_packets") {
                                param_default.num_packets = value;
                            }
                            next_config.defaults = merge_patterns(
                                current_config.defaults.value_or(ParsedTrafficPatternConfig{}), param_default);
                        }
                    }
                }
            }
            // Move the newly generated configs to be the input for the next parameter loop.
            parametrized_configs = std::move(next_level_configs);
        }
    }
    return parametrized_configs;
}

void TestConfigBuilder::validate_pattern(const TrafficPatternConfig& pattern, const TestConfig& test) const {
    // 1. Validate destination ambiguity
    TT_FATAL(
        pattern.destination.has_value(),
        "Test '{}': Pattern is missing a destination. This should have been resolved by the builder.",
        test.name);
    const auto& dest = pattern.destination.value();
    TT_FATAL(
        !(dest.device.has_value() && dest.hops.has_value()),
        "Test '{}': A pattern's destination cannot have both 'device' and 'hops' specified.",
        test.name);
    TT_FATAL(
        dest.device.has_value() || dest.hops.has_value(),
        "Test '{}': A pattern's destination must specify either a 'device' or 'hops'.",
        test.name);

    // 2. Validate atomic-related fields
    if (pattern.ntype.has_value() && pattern.ntype.value() == NocSendType::NOC_UNICAST_WRITE) {
        TT_FATAL(
            !pattern.atomic_inc_val.has_value(),
            "Test '{}': 'atomic_inc_val' should not be specified for 'unicast_write' ntype.",
            test.name);
    }

    // 3. Validate payload size
    if (pattern.size.has_value()) {
        const uint32_t max_payload_size = this->device_info_provider_.get_max_payload_size_bytes();
        TT_FATAL(
            pattern.size.value() <= max_payload_size,
            "Test '{}': Payload size {} exceeds the maximum of {} bytes",
            test.name,
            pattern.size.value(),
            max_payload_size);
    }
}

void TestConfigBuilder::validate_chip_unicast(
    const TrafficPatternConfig& pattern, const SenderConfig& sender, const TestConfig& test) const {
    TT_FATAL(
        pattern.destination.has_value() &&
            (pattern.destination->device.has_value() || pattern.destination->hops.has_value()),
        "Test '{}': Unicast pattern for sender on device {} is missing a destination device or hops.",
        test.name,
        sender.device);

    if (pattern.destination->device.has_value()) {
        TT_FATAL(
            sender.device != pattern.destination->device.value(),
            "Test '{}': Sender on device {} cannot have itself as a destination.",
            test.name,
            sender.device);
    }

    TT_FATAL(
        !pattern.mcast_start_hops.has_value(),
        "Test '{}': 'mcast_start_hops' cannot be specified for a 'unicast' ftype pattern.",
        test.name);
}

void TestConfigBuilder::validate_chip_multicast(
    const TrafficPatternConfig& pattern, const SenderConfig& sender, const TestConfig& test) const {
    TT_FATAL(
        pattern.destination.has_value() && pattern.destination->hops.has_value(),
        "Test '{}': Multicast pattern for sender on device {} must have a destination specified by 'hops'.",
        test.name,
        sender.device);
    TT_FATAL(
        test.fabric_setup.topology != tt::tt_fabric::Topology::NeighborExchange,
        "Test '{}': Multicast pattern is not supported for NeighborExchange topology.",
        test.name);
}

void TestConfigBuilder::validate_sync_pattern(
    const TrafficPatternConfig& pattern, const SenderConfig& sender, const TestConfig& test) const {
    // The NeighborExchange topology uses unicast sync patterns, so we perform a different check
    if (test.fabric_setup.topology == tt::tt_fabric::Topology::NeighborExchange) {
        TT_FATAL(
            pattern.ftype.has_value() && pattern.ftype.value() == ChipSendType::CHIP_UNICAST,
            "Test '{}': Line sync pattern for sender on device {} must use CHIP_UNICAST for NeighborExchange "
            "topology.",
            test.name,
            sender.device);
    } else {
        TT_FATAL(
            pattern.ftype.has_value() && pattern.ftype.value() == ChipSendType::CHIP_MULTICAST,
            "Test '{}': Line sync pattern for sender on device {} must use CHIP_MULTICAST.",
            test.name,
            sender.device);
    }

    TT_FATAL(
        pattern.destination.has_value() && pattern.destination->hops.has_value(),
        "Test '{}': Line sync pattern for sender on device {} must have destination specified by 'hops'.",
        test.name,
        sender.device);

    TT_FATAL(
        pattern.ntype.has_value() && pattern.ntype.value() == NocSendType::NOC_UNICAST_ATOMIC_INC,
        "Test '{}': Line sync pattern for sender on device {} must use NOC_UNICAST_ATOMIC_INC.",
        test.name,
        sender.device);

    TT_FATAL(
        pattern.size.has_value() && pattern.size.value() == 0,
        "Test '{}': Line sync pattern for sender on device {} must have size 0 (no payload).",
        test.name,
        sender.device);

    TT_FATAL(
        pattern.num_packets.has_value() && pattern.num_packets.value() == 1,
        "Test '{}': Line sync pattern for sender on device {} must have num_packets 1.",
        test.name,
        sender.device);
}

void TestConfigBuilder::validate_test(const TestConfig& test) const {
    for (const auto& sender : test.senders) {
        for (const auto& pattern : sender.patterns) {
            validate_pattern(pattern, test);

            if (pattern.ftype.value() == ChipSendType::CHIP_UNICAST) {
                validate_chip_unicast(pattern, sender, test);
            } else if (pattern.ftype.value() == ChipSendType::CHIP_MULTICAST) {
                validate_chip_multicast(pattern, sender, test);
            }
        }
    }

    // Validate line sync patterns if present
    if (test.global_sync) {
        for (const auto& sync_config : test.sync_configs) {
            const auto& sync_sender = sync_config.sender_config;
            for (const auto& sync_pattern : sync_sender.patterns) {
                validate_sync_pattern(sync_pattern, sync_sender, test);
            }
        }
    }

    if (test.fabric_setup.topology == tt::tt_fabric::Topology::Linear) {
        for (const auto& sender : test.senders) {
            for (const auto& pattern : sender.patterns) {
                if (pattern.destination->device.has_value()) {
                    TT_FATAL(
                        this->route_manager_.are_devices_linear({sender.device, pattern.destination->device.value()}),
                        "For a 'Linear' topology, all specified devices must be in the same row or column. Test: "
                        "{}",
                        test.name);
                }
            }
        }
    }
}

void TestConfigBuilder::expand_patterns_into_test(
    ParsedTestConfig& test, const std::vector<HighLevelPatternConfig>& patterns, uint32_t iteration_idx) {
    const auto& defaults = test.defaults.value_or(ParsedTrafficPatternConfig{});

    for (const auto& pattern : patterns) {
        if (pattern.iterations.has_value() && iteration_idx >= pattern.iterations.value()) {
            continue;
        }

        if (pattern.type == "all_to_all") {
            if (defaults.ftype == ChipSendType::CHIP_UNICAST) {
                expand_one_or_all_to_all_unicast(test, defaults, HighLevelTrafficPattern::AllToAll);
            } else {
                expand_one_or_all_to_all_multicast(test, defaults, HighLevelTrafficPattern::AllToAll);
            }
        } else if (pattern.type == "one_to_all") {
            if (defaults.ftype == ChipSendType::CHIP_UNICAST) {
                expand_one_or_all_to_all_unicast(test, defaults, HighLevelTrafficPattern::OneToAll);
            } else {
                expand_one_or_all_to_all_multicast(test, defaults, HighLevelTrafficPattern::OneToAll);
            }
        } else if (pattern.type == "all_to_one") {
            expand_all_to_one_unicast(test, defaults, iteration_idx);
        } else if (pattern.type == "all_to_one_random") {
            expand_all_to_one_random_unicast(test, defaults);
        } else if (pattern.type == "full_device_random_pairing") {
            expand_full_device_random_pairing(test, defaults);
        } else if (pattern.type == "unidirectional_linear") {
            expand_unidirectional_linear_unicast_or_multicast(test, defaults);
        } else if (pattern.type == "full_ring" || pattern.type == "half_ring") {
            HighLevelTrafficPattern pattern_type =
                detail::high_level_traffic_pattern_mapper.from_string(pattern.type, "HighLevelTrafficPattern");
            expand_full_or_half_ring_unicast_or_multicast(test, defaults, pattern_type);
        } else if (pattern.type == "all_devices_uniform_pattern") {
            expand_all_devices_uniform_pattern(test, defaults);
        } else if (pattern.type == "neighbor_exchange") {
            expand_neighbor_exchange(test, defaults);
        } else if (pattern.type == "sequential_all_to_all") {
            expand_sequential_all_to_all_unicast(test, defaults, iteration_idx);
        } else {
            TT_THROW("Unsupported pattern type: {}", pattern.type);
        }
    }
}

void TestConfigBuilder::expand_one_or_all_to_all_unicast(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, HighLevelTrafficPattern pattern_type) {
    log_debug(
        LogTest,
        "Expanding {}_unicast pattern for test: {}",
        (pattern_type == HighLevelTrafficPattern::OneToAll) ? "one_to_all" : "all_to_all",
        test.name);
    std::vector<std::pair<FabricNodeId, FabricNodeId>> all_pairs = this->route_manager_.get_all_to_all_unicast_pairs();

    if (pattern_type == HighLevelTrafficPattern::OneToAll) {
        TT_FATAL(!all_pairs.empty(), "Cannot expand one_to_all_unicast because no device pairs were found.");

        // Get the first device as the single sender
        FabricNodeId first_device = all_pairs[0].first;

        // Filter pairs to only include those with the first device as sender
        std::vector<std::pair<FabricNodeId, FabricNodeId>> filtered_pairs;
        for (const auto& pair : all_pairs) {
            if (pair.first == first_device) {
                filtered_pairs.push_back(pair);
            }
        }
        add_senders_from_pairs(test, filtered_pairs, base_pattern);
    } else if (device_info_provider_.is_multi_mesh()) {
        const auto mesh_adjacency_map = device_info_provider_.get_mesh_adjacency_map();

        std::vector<std::pair<FabricNodeId, FabricNodeId>> filtered_pairs;
        for (const auto& pair : all_pairs) {
            MeshId src_mesh_id = pair.first.mesh_id;
            MeshId dst_mesh_id = pair.second.mesh_id;
            bool same_mesh = (src_mesh_id == dst_mesh_id);
            bool dst_is_adjacent = false;
            auto it = mesh_adjacency_map.find(src_mesh_id);
            if (it != mesh_adjacency_map.end()) {
                dst_is_adjacent = it->second.contains(dst_mesh_id);
            }
            if (same_mesh || dst_is_adjacent) {
                filtered_pairs.push_back(pair);
            }
        }

        log_info(
            LogTest,
            "Multi-mesh all_to_all: filtered {} pairs to {} pairs with adjacent mesh destinations",
            all_pairs.size(),
            filtered_pairs.size());

        add_senders_from_pairs(test, filtered_pairs, base_pattern);
    } else {
        add_senders_from_pairs(test, all_pairs, base_pattern);
    }
}

void TestConfigBuilder::expand_all_to_one_unicast(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, uint32_t iteration_idx) {
    log_debug(LogTest, "Expanding all_to_one_unicast pattern for test: {} (iteration {})", test.name, iteration_idx);
    auto filtered_pairs = this->route_manager_.get_all_to_one_unicast_pairs(iteration_idx);
    if (!filtered_pairs.empty()) {
        add_senders_from_pairs(test, filtered_pairs, base_pattern);
    }
}

void TestConfigBuilder::expand_all_to_one_random_unicast(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern) {
    log_debug(LogTest, "Expanding all_to_one_unicast pattern for test: {}", test.name);
    uint32_t index = get_random_in_range(0, device_info_provider_.get_global_node_ids().size() - 1);
    auto filtered_pairs = this->route_manager_.get_all_to_one_unicast_pairs(index);
    add_senders_from_pairs(test, filtered_pairs, base_pattern);
}

void TestConfigBuilder::expand_full_device_random_pairing(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern) {
    log_debug(LogTest, "Expanding full_device_random_pairing pattern for test: {}", test.name);
    auto random_pairs = this->route_manager_.get_full_device_random_pairs(this->gen_);
    add_senders_from_pairs(test, random_pairs, base_pattern);
}

void TestConfigBuilder::expand_sequential_all_to_all_unicast(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, uint32_t iteration_idx) {
    log_debug(
        LogTest,
        "Expanding sequential_all_to_all_unicast pattern for test: {} (iteration {})",
        test.name,
        iteration_idx);

    auto all_pairs = this->route_manager_.get_all_to_all_unicast_pairs();

    if (all_pairs.empty()) {
        log_warning(LogTest, "No valid pairs found for sequential_all_to_all pattern");
        return;
    }

    // Select only the pair for this iteration
    if (iteration_idx < all_pairs.size()) {
        std::vector<std::pair<FabricNodeId, FabricNodeId>> single_pair = {all_pairs[iteration_idx]};
        add_senders_from_pairs(test, single_pair, base_pattern);
    } else {
        TT_THROW(
            "Iteration index {} exceeds number of available device pairs {} for sequential_all_to_all pattern",
            iteration_idx,
            all_pairs.size());
    }
}

void TestConfigBuilder::expand_all_devices_uniform_pattern(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern) {
    log_debug(LogTest, "Expanding all_devices_uniform_pattern for test: {}", test.name);
    std::vector<FabricNodeId> devices = device_info_provider_.get_global_node_ids();
    TT_FATAL(!devices.empty(), "Cannot expand all_devices_uniform_pattern because no devices were found.");

    for (const auto& src_node : devices) {
        // Apply the base pattern (from defaults) to each device
        test.senders.emplace_back(ParsedSenderConfig{.device = src_node, .patterns = {base_pattern}});
    }
}

void TestConfigBuilder::expand_one_or_all_to_all_multicast(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, HighLevelTrafficPattern pattern_type) {
    const char* pattern_name = (pattern_type == HighLevelTrafficPattern::OneToAll) ? "one_to_all" : "all_to_all";
    log_debug(LogTest, "Expanding {}_multicast pattern for test: {}", pattern_name, test.name);
    std::vector<FabricNodeId> devices = device_info_provider_.get_global_node_ids();
    TT_FATAL(!devices.empty(), "Cannot expand {}_multicast because no devices were found.", pattern_name);

    // Determine which devices should be senders
    std::vector<FabricNodeId> sender_devices;
    if (pattern_type == HighLevelTrafficPattern::OneToAll) {
        sender_devices = {devices[0]};  // Only first device
    } else {
        sender_devices = devices;  // All devices
    }

    for (const auto& src_node : sender_devices) {
        auto hops = this->route_manager_.get_full_mcast_hops(src_node);

        ParsedTrafficPatternConfig specific_pattern;
        specific_pattern.destination = ParsedDestinationConfig{.hops = hops};
        specific_pattern.ftype = ChipSendType::CHIP_MULTICAST;

        auto merged_pattern = merge_patterns(base_pattern, specific_pattern);

        auto it = std::find_if(test.senders.begin(), test.senders.end(), [&](const ParsedSenderConfig& s) {
            // Compare FabricNodeId with DeviceIdentifier
            if (std::holds_alternative<FabricNodeId>(s.device)) {
                return std::get<FabricNodeId>(s.device) == src_node;
            }
            return false;
        });

        if (it != test.senders.end()) {
            it->patterns.emplace_back(std::move(merged_pattern));
        } else {
            test.senders.emplace_back(ParsedSenderConfig{.device = src_node, .patterns = {merged_pattern}});
        }
    }
}

void TestConfigBuilder::expand_unidirectional_linear_unicast_or_multicast(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern) {
    log_debug(LogTest, "Expanding unidirectional_linear pattern for test: {}", test.name);
    std::vector<FabricNodeId> devices = device_info_provider_.get_local_node_ids();
    TT_FATAL(!devices.empty(), "Cannot expand unidirectional_linear because no devices were found.");

    for (const auto& src_node : devices) {
        // instantiate N/S E/W traffic on separate senders to avoid bottlenecking on sender.
        for (uint32_t dim = 0; dim < this->route_manager_.get_num_mesh_dims(); ++dim) {
            // Skip dimensions with only one device
            if (this->route_manager_.get_mesh_shape()[dim] < 2) {
                continue;
            }

            auto hops = this->route_manager_.get_unidirectional_linear_mcast_hops(src_node, dim);

            ParsedTrafficPatternConfig specific_pattern;
            specific_pattern.destination = ParsedDestinationConfig{.hops = hops};

            auto merged_pattern = merge_patterns(base_pattern, specific_pattern);
            test.senders.push_back(ParsedSenderConfig{.device = src_node, .patterns = {merged_pattern}});
        }
    }
}

void TestConfigBuilder::expand_neighbor_exchange(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern) {
    log_debug(LogTest, "Expanding neighbor_exchange pattern for test: {}", test.name);
    auto neighbor_pairs = this->route_manager_.get_neighbor_exchange_pairs();
    if (!neighbor_pairs.empty()) {
        add_senders_from_pairs(test, neighbor_pairs, base_pattern);
    }
}

void TestConfigBuilder::expand_full_or_half_ring_unicast_or_multicast(
    ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, HighLevelTrafficPattern pattern_type) {
    log_debug(LogTest, "Expanding full_or_half_ring pattern for test: {}", test.name);
    std::vector<FabricNodeId> devices = device_info_provider_.get_local_node_ids();
    TT_FATAL(!devices.empty(), "Cannot expand full_or_half_ring because no devices were found.");

    bool wrap_around_mesh = this->route_manager_.wrap_around_mesh(devices.front());

    std::unordered_map<RoutingDirection, uint32_t> hops;
    for (const auto& src_node : devices) {
        if (wrap_around_mesh) {
            // Get ring neighbors - returns nullopt for non-perimeter devices
            auto ring_neighbors = this->route_manager_.get_wrap_around_mesh_ring_neighbors(src_node, devices);

            // Check if the result is valid (has value)
            if (!ring_neighbors.has_value()) {
                // Skip this device as it's not on the perimeter and can't participate in ring multicast
                log_debug(LogTest, "Skipping device {} as it's not on the perimeter ring", src_node.chip_id);
                continue;
            }

            // Extract the valid ring neighbors
            auto [dst_node_forward, dst_node_backward] = ring_neighbors.value();

            hops = this->route_manager_.get_wrap_around_mesh_full_or_half_ring_mcast_hops(
                src_node, dst_node_forward, dst_node_backward, pattern_type);

            ParsedTrafficPatternConfig specific_pattern;
            specific_pattern.destination = ParsedDestinationConfig{.hops = hops};

            auto merged_pattern = merge_patterns(base_pattern, specific_pattern);

            auto it = std::find_if(test.senders.begin(), test.senders.end(), [&](const ParsedSenderConfig& s) {
                // Compare FabricNodeId with DeviceIdentifier
                if (std::holds_alternative<FabricNodeId>(s.device)) {
                    return std::get<FabricNodeId>(s.device) == src_node;
                }
                return false;
            });

            if (it != test.senders.end()) {
                it->patterns.push_back(merged_pattern);
            } else {
                test.senders.push_back(ParsedSenderConfig{.device = src_node, .patterns = {merged_pattern}});
            }
        } else {
            for (uint32_t dim = 0; dim < this->route_manager_.get_num_mesh_dims(); ++dim) {
                // Skip dimensions with only one device
                if (this->route_manager_.get_mesh_shape()[dim] < 2) {
                    continue;
                }

                hops = this->route_manager_.get_full_or_half_ring_mcast_hops(src_node, pattern_type, dim);

                ParsedTrafficPatternConfig specific_pattern;
                specific_pattern.destination = ParsedDestinationConfig{.hops = hops};

                auto merged_pattern = merge_patterns(base_pattern, specific_pattern);
                test.senders.push_back(ParsedSenderConfig{.device = src_node, .patterns = {merged_pattern}});
            }
        }
    }
}

void TestConfigBuilder::expand_sync_patterns(ParsedTestConfig& test) {
    log_debug(
        LogTest,
        "Expanding line sync patterns for test: {} with topology: {}",
        test.name,
        static_cast<int>(test.fabric_setup.topology));

    std::vector<FabricNodeId> all_devices = device_info_provider_.get_global_node_ids();
    TT_FATAL(!all_devices.empty(), "Cannot expand line sync patterns because no devices were found.");

    // Create sync patterns based on topology - returns multiple patterns per device for mcast
    for (const auto& src_device : all_devices) {
        const auto& sync_patterns_and_sync_val_pair =
            create_sync_patterns_for_topology(src_device, all_devices, test.fabric_setup.topology);

        const auto& sync_patterns = sync_patterns_and_sync_val_pair.first;
        const auto& sync_val = sync_patterns_and_sync_val_pair.second;

        log_debug(
            LogTest,
            "Generated {} sync patterns for device {}, with sync value {}",
            sync_patterns.size(),
            src_device.chip_id,
            sync_val);

        // Create sender config with all split sync patterns
        // Sync always uses link 0 (no override allowed)
        SenderConfig sync_sender = {.device = src_device, .patterns = sync_patterns, .link_id = 0};

        // Generate a SyncConfig for this device
        SyncConfig sync_config = {.sync_val = sync_val, .sender_config = std::move(sync_sender)};
        test.sync_configs.push_back(std::move(sync_config));
    }

    log_debug(LogTest, "Generated {} line sync configurations", test.sync_configs.size());
}

std::pair<std::vector<TrafficPatternConfig>, uint32_t> TestConfigBuilder::create_sync_patterns_for_topology(
    const FabricNodeId& src_device, const std::vector<FabricNodeId>& devices, tt::tt_fabric::Topology topology) {
    std::vector<TrafficPatternConfig> sync_patterns;
    uint32_t sync_val = 0;

    // Common sync pattern characteristics
    TrafficPatternConfig base_sync_pattern;
    // Edge case: NeighborExchange topology does not support multicast sync patterns, so we create unicast sync
    // patterns separately
    base_sync_pattern.ftype = topology == tt::tt_fabric::Topology::NeighborExchange
                                  ? ChipSendType::CHIP_UNICAST
                                  : ChipSendType::CHIP_MULTICAST;   // Global sync across devices
    base_sync_pattern.ntype = NocSendType::NOC_UNICAST_ATOMIC_INC;  // Sync signal via atomic increment
    base_sync_pattern.size = 0;                                     // No payload, just sync signal
    base_sync_pattern.num_packets = 1;                              // Single sync signal
    base_sync_pattern.atomic_inc_val = 1;                           // Increment by 1

    // Start by calculating multi-directional hops
    auto [multi_directional_hops, multi_directional_sync_val] =
        this->route_manager_.get_sync_hops_and_val(src_device, devices);

    sync_val = multi_directional_sync_val;

    // Split multi-directional hops into single-directional patterns
    auto split_hops_vec = this->route_manager_.split_multicast_hops(multi_directional_hops);

    log_debug(
        LogTest,
        "Splitting sync pattern for device {} from 1 multi-directional to {} single-directional patterns",
        src_device.chip_id,
        split_hops_vec.size());
    // Create separate sync pattern for each mcast direction. This is required since test infra only handle
    // mcast for one direction. Ex, mcast to E/W will split into EAST and WEST patterns.
    sync_patterns.reserve(split_hops_vec.size());
    for (const auto& single_direction_hops : split_hops_vec) {
        TrafficPatternConfig sync_pattern = base_sync_pattern;
        sync_pattern.destination = DestinationConfig{.hops = single_direction_hops};
        sync_patterns.push_back(std::move(sync_pattern));
    }
    return {sync_patterns, sync_val};
}

void TestConfigBuilder::add_senders_from_pairs(
    ParsedTestConfig& test,
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const ParsedTrafficPatternConfig& base_pattern) {
    std::map<FabricNodeId, std::vector<ParsedTrafficPatternConfig>> generated_senders;

    for (const auto& pair : pairs) {
        const auto& src_node = pair.first;
        const auto& dst_node = pair.second;

        ParsedTrafficPatternConfig specific_pattern;
        specific_pattern.destination = ParsedDestinationConfig{.device = dst_node};
        specific_pattern.ftype = ChipSendType::CHIP_UNICAST;

        // Use try_emplace to avoid creating empty vectors unnecessarily
        auto [it, inserted] = generated_senders.try_emplace(src_node);
        it->second.emplace_back(merge_patterns(base_pattern, specific_pattern));
    }

    test.senders.reserve(test.senders.size() + generated_senders.size());
    for (const auto& [src_node, patterns] : generated_senders) {
        test.senders.emplace_back(ParsedSenderConfig{.device = src_node, .patterns = patterns});
    }
}

void TestConfigBuilder::split_all_unicast_or_multicast_patterns(ParsedTestConfig& test) {
    // This function iterates through all sender patterns and splits any multi-direction
    // multicast hops.
    for (auto& sender : test.senders) {
        std::vector<ParsedTrafficPatternConfig> new_patterns;
        bool sender_was_modified = false;

        for (size_t i = 0; i < sender.patterns.size(); ++i) {
            const auto& pattern = sender.patterns[i];

            // Determine if this specific pattern needs to be split.
            bool needs_split = false;
            std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_hops_vec;
            if (pattern.destination.has_value() && pattern.destination.value().hops.has_value()) {
                const auto& hops = pattern.destination.value().hops.value();
                split_hops_vec = this->route_manager_.split_multicast_hops(hops);
                if (split_hops_vec.size() > 1) {
                    needs_split = true;
                }
            }

            if (needs_split) {
                if (!sender_was_modified) {
                    sender_was_modified = true;
                    // This is the first split for this sender.
                    // Lazily allocate and copy the patterns processed so far.
                    new_patterns.reserve(sender.patterns.size() + split_hops_vec.size() - 1);
                    new_patterns.insert(new_patterns.end(), sender.patterns.begin(), sender.patterns.begin() + i);
                }
                // Add the newly split patterns.
                for (const auto& split_hop : split_hops_vec) {
                    ParsedTrafficPatternConfig new_pattern = pattern;
                    new_pattern.destination->hops = split_hop;
                    new_patterns.emplace_back(std::move(new_pattern));
                }
            } else if (sender_was_modified) {
                // We are in copy-mode because a previous pattern was split.
                new_patterns.emplace_back(pattern);
            }
        }

        if (sender_was_modified) {
            sender.patterns = std::move(new_patterns);
        }
    }
}

bool TestConfigBuilder::expand_link_duplicates(ParsedTestConfig& test) {
    // If num_links is 1, no duplication needed
    if (test.fabric_setup.num_links <= 1) {
        return true;  // Success - no expansion needed
    }

    uint32_t num_links = test.fabric_setup.num_links;
    log_debug(LogTest, "Expanding link duplicates for test '{}' with {} links", test.name, num_links);

    // Validate that num_links doesn't exceed available routing planes for any device
    if (!route_manager_.validate_num_links_supported(num_links)) {
        return false;  // Indicate test should be skipped
    }

    std::vector<ParsedSenderConfig> new_senders;
    new_senders.reserve(test.senders.size() * num_links);

    for (const auto& sender : test.senders) {
        for (uint32_t link_id = 0; link_id < num_links; ++link_id) {
            ParsedSenderConfig duplicated_sender = sender;
            duplicated_sender.link_id = link_id;  // Assign link ID
            new_senders.push_back(duplicated_sender);
        }
    }

    test.senders = std::move(new_senders);
    return true;  // Success
}

void TestConfigBuilder::resolve_missing_params(ParsedTestConfig& test) {
    if (test.on_missing_param_policy.has_value() && test.on_missing_param_policy.value() == "randomize") {
        for (auto& sender : test.senders) {
            for (auto& pattern : sender.patterns) {
                if (!pattern.ftype.has_value()) {
                    pattern.ftype =
                        get_random_choice<ChipSendType>({ChipSendType::CHIP_UNICAST, ChipSendType::CHIP_MULTICAST});
                }
                if (!pattern.ntype.has_value()) {
                    pattern.ntype = get_random_choice<NocSendType>(
                        {NocSendType::NOC_UNICAST_WRITE, NocSendType::NOC_UNICAST_ATOMIC_INC});
                }
                if (!pattern.size.has_value()) {
                    pattern.size = get_random_in_range(64, 2048);
                }
                if (!pattern.num_packets.has_value()) {
                    pattern.num_packets = get_random_in_range(10, 1000);
                }

                if (!pattern.destination.has_value()) {
                    if (pattern.ftype.value() == ChipSendType::CHIP_UNICAST) {
                        // Need to resolve sender.device to FabricNodeId for route manager
                        FabricNodeId sender_node = resolve_device_identifier(sender.device, device_info_provider_);
                        FabricNodeId dst_node =
                            this->route_manager_.get_random_unicast_destination(sender_node, this->gen_);
                        pattern.destination = ParsedDestinationConfig{.device = dst_node};
                    } else if (pattern.ftype.value() == ChipSendType::CHIP_MULTICAST) {
                        // For multicast, the random default is an mcast to all devices.
                        FabricNodeId sender_node = resolve_device_identifier(sender.device, device_info_provider_);
                        auto hops = this->route_manager_.get_full_mcast_hops(sender_node);
                        pattern.destination = ParsedDestinationConfig{.hops = hops};
                    }
                }
            }
        }
    } else {
        // Not 'randomize', so fill with sane defaults.
        for (auto& sender : test.senders) {
            for (auto& pattern : sender.patterns) {
                if (!pattern.ftype.has_value()) {
                    pattern.ftype = ChipSendType::CHIP_UNICAST;
                }
                if (!pattern.ntype.has_value()) {
                    pattern.ntype = NocSendType::NOC_UNICAST_WRITE;
                }
                if (!pattern.size.has_value()) {
                    pattern.size = 1024;  // Default from cmdline parser
                }
                if (!pattern.num_packets.has_value()) {
                    pattern.num_packets = 10;  // A reasonable default
                }
            }
        }
    }
}

uint32_t TestConfigBuilder::get_random_in_range(uint32_t min, uint32_t max) {
    if (min > max) {
        std::swap(min, max);
    }
    std::uniform_int_distribution<uint32_t> distrib(min, max);
    return distrib(this->gen_);
}

// YamlTestConfigSerializer methods
void YamlTestConfigSerializer::dump(const PhysicalMeshConfig& physical_mesh_config, std::ofstream& fout) {
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "physical_mesh";
    out << YAML::Value;
    to_yaml(out, physical_mesh_config);

    out << YAML::EndMap;

    fout << out.c_str() << std::endl;
}

void YamlTestConfigSerializer::dump(const AllocatorPolicies& policies, std::ofstream& fout) {
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "allocation_policies";
    out << YAML::Value;
    to_yaml(out, policies);

    out << YAML::EndMap;

    fout << out.c_str() << std::endl;
}

void YamlTestConfigSerializer::dump(const std::vector<TestConfig>& test_configs, std::ofstream& fout) {
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "Test";
    out << YAML::Value;

    out << YAML::BeginSeq;
    for (const auto& test_config : test_configs) {
        to_yaml(out, test_config);
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;

    fout << out.c_str() << std::endl;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const FabricNodeId& id) {
    out << YAML::Flow;
    out << YAML::BeginSeq << *id.mesh_id << id.chip_id << YAML::EndSeq;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const CoreCoord& core) {
    out << YAML::Flow;
    out << YAML::BeginSeq << core.x << core.y << YAML::EndSeq;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const DestinationConfig& config) {
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
    if (config.atomic_inc_address) {
        out << YAML::Key << "atomic_inc_address";
        out << YAML::Value << config.atomic_inc_address.value();
    }
    out << YAML::EndMap;
}

std::string YamlTestConfigSerializer::to_string(ChipSendType ftype) {
    return detail::chip_send_type_mapper.to_string(ftype, "ChipSendType");
}

std::string YamlTestConfigSerializer::to_string(NocSendType ntype) {
    return detail::noc_send_type_mapper.to_string(ntype, "NocSendType");
}

std::string YamlTestConfigSerializer::to_string(RoutingDirection dir) {
    return detail::routing_direction_mapper.to_string(dir, "RoutingDirection");
}

std::string YamlTestConfigSerializer::to_string(FabricTensixConfig ftype) {
    return detail::fabric_tensix_type_mapper.to_string(ftype, "FabricTensixConfig");
}

std::string YamlTestConfigSerializer::to_string(FabricReliabilityMode mode) {
    return detail::fabric_reliability_mode_mapper.to_string(mode, "FabricReliabilityMode");
}

std::string YamlTestConfigSerializer::to_string(tt::tt_fabric::Topology topology) {
    return detail::topology_mapper.to_string(topology, "Topology");
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const TrafficPatternConfig& config) {
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
    if (config.atomic_inc_val) {
        out << YAML::Key << "atomic_inc_val";
        out << YAML::Value << config.atomic_inc_val.value();
    }
    if (config.mcast_start_hops) {
        out << YAML::Key << "mcast_start_hops";
        out << YAML::Value << config.mcast_start_hops.value();
    }

    out << YAML::EndMap;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const SenderConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "device";
    out << YAML::Value;
    to_yaml(out, config.device);

    if (config.core) {
        out << YAML::Key << "core";
        out << YAML::Value;
        to_yaml(out, config.core.value());
    }

    out << YAML::Key << "link_id";
    out << YAML::Value << config.link_id;

    out << YAML::Key << "patterns";
    out << YAML::Value;
    out << YAML::BeginSeq;
    for (const auto& pattern : config.patterns) {
        to_yaml(out, pattern);
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const TestConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "name";
    out << YAML::Value << config.parametrized_name;  // Use parametrized name for readability

    // Optionally include original base name as metadata if different
    if (!config.name.empty() && config.name != config.parametrized_name) {
        out << YAML::Key << "base_name";
        out << YAML::Value << config.name;  // Original name for reference
    }

    if (config.seed != 0) {
        out << YAML::Key << "seed";
        out << YAML::Value << config.seed;
    }

    if (config.performance_test_mode == PerformanceTestMode::BANDWIDTH) {
        out << YAML::Key << "benchmark_mode";
        out << YAML::Value << true;
    } else if (config.performance_test_mode == PerformanceTestMode::LATENCY) {
        out << YAML::Key << "latency_test_mode";
        out << YAML::Value << true;
    }

    if (config.global_sync) {
        out << YAML::Key << "sync";
        out << YAML::Value << config.global_sync;
    }

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

std::string YamlTestConfigSerializer::to_string(CoreAllocationPolicy policy) {
    return detail::core_allocation_policy_mapper.to_string(policy, "CoreAllocationPolicy");
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const CoreAllocationConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "policy";
    out << YAML::Value << to_string(config.policy);
    out << YAML::Key << "max_configs_per_core";
    out << YAML::Value << config.max_configs_per_core;
    out << YAML::Key << "initial_pool_size";
    out << YAML::Value << config.initial_pool_size;
    out << YAML::Key << "pool_refill_size";
    out << YAML::Value << config.pool_refill_size;
    out << YAML::EndMap;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const PhysicalMeshConfig& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "mesh_descriptor_path";
    out << YAML::Value << config.mesh_descriptor_path;
    out << YAML::Key << "eth_coord_mapping";
    out << YAML::Value;
    to_yaml(out, config.eth_coord_mapping);
    out << YAML::EndMap;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const std::vector<std::vector<EthCoord>>& mapping) {
    out << YAML::BeginSeq;
    for (const auto& row : mapping) {
        out << YAML::BeginSeq;
        for (const auto& coord : row) {
            out << YAML::Flow << YAML::BeginSeq << coord.cluster_id << coord.x << coord.y << coord.rack << coord.shelf
                << YAML::EndSeq;
        }
        out << YAML::EndSeq;
    }
    out << YAML::EndSeq;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const AllocatorPolicies& policies) {
    out << YAML::BeginMap;
    out << YAML::Key << "sender";
    out << YAML::Value;
    to_yaml(out, policies.sender_config);
    out << YAML::Key << "receiver";
    out << YAML::Value;
    to_yaml(out, policies.receiver_config);
    out << YAML::Key << "default_payload_chunk_size";
    out << YAML::Value << policies.default_payload_chunk_size;
    out << YAML::EndMap;
}

void YamlTestConfigSerializer::to_yaml(YAML::Emitter& out, const TestFabricSetup& config) {
    out << YAML::BeginMap;
    out << YAML::Key << "topology";
    out << YAML::Value << to_string(config.topology);
    if (config.fabric_tensix_config.has_value()) {
        out << YAML::Key << "fabric_tensix_config";
        out << YAML::Value << to_string(config.fabric_tensix_config.value());
    }
    if (config.fabric_reliability_mode.has_value()) {
        out << YAML::Key << "fabric_reliability_mode";
        out << YAML::Value << to_string(config.fabric_reliability_mode.value());
    }
    if (config.topology == Topology::Torus && config.torus_config.has_value()) {
        out << YAML::Key << "torus_config";
        out << YAML::Value << config.torus_config.value();
    }
    out << YAML::Key << "num_links";
    out << YAML::Value << config.num_links;
    out << YAML::EndMap;
}

}  // namespace tt::tt_fabric::fabric_tests
