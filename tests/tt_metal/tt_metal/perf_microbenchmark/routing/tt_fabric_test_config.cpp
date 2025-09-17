// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_config.hpp"

namespace tt::tt_fabric::fabric_tests {

CmdlineParser::CmdlineParser(const std::vector<std::string>& input_args) : input_args_(input_args) {
    if (test_args::has_command_option(input_args_, "--filter")) {
        auto filter = test_args::get_command_option(input_args_, "--filter", "");
        auto splitter = filter.find('.');
        filter_type = filter.substr(0, splitter);
        filter_value = filter.substr(splitter + 1);
    }
}


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
        chip_id_t chip_id = parse_scalar<chip_id_t>(node);
        return chip_id;
    } else if (node.IsSequence() && node.size() == 2) {
        MeshId mesh_id = parse_mesh_id(node[0]);
        if (node[1].IsScalar()) {
            // Format: [mesh_id, chip_id]
            chip_id_t chip_id = parse_scalar<chip_id_t>(node[1]);
            return std::make_pair(mesh_id, chip_id);
        } else if (node[1].IsSequence()) {
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
        config.atomic_inc_val = parse_scalar<uint16_t>(pattern_yaml["atomic_inc_val"]);
    }
    if (pattern_yaml["atomic_inc_wrap"]) {
        config.atomic_inc_wrap = parse_scalar<uint16_t>(pattern_yaml["atomic_inc_wrap"]);
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

    const auto& patterns_yaml = sender_yaml["patterns"];
    TT_FATAL(patterns_yaml.IsSequence(), "Expected patterns to be a sequence");
    config.patterns.reserve(patterns_yaml.size());
    for (const auto& pattern_node : patterns_yaml) {
        ParsedTrafficPatternConfig specific_pattern = parse_traffic_pattern_config(pattern_node);
        config.patterns.push_back(merge_patterns(defaults, specific_pattern));
    }
    return config;
}

ParsedTestConfig YamlConfigParser::parse_test_config(const YAML::Node& test_yaml) {
    ParsedTestConfig test_config;

    test_config.name = parse_scalar<std::string>(test_yaml["name"]);
    log_info(tt::LogTest, "name: {}", test_config.name);

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

    if (test_yaml["benchmark_mode"]) {
        test_config.benchmark_mode = parse_scalar<bool>(test_yaml["benchmark_mode"]);
    }

    if (test_yaml["sync"]) {
        test_config.global_sync = parse_scalar<bool>(test_yaml["sync"]);
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

    if (fabric_setup_yaml["routing_type"]) {
        auto routing_type_str = parse_scalar<std::string>(fabric_setup_yaml["routing_type"]);
        fabric_setup.routing_type = detail::routing_type_mapper.from_string(routing_type_str, "RoutingType");
    } else {
        log_info(tt::LogTest, "No routing type specified, defaulting to LowLatency");
        fabric_setup.routing_type = RoutingType::LowLatency;
    }

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
    physical_mesh_config.eth_coord_mapping = parse_2d_array<eth_coord_t>(physical_mesh_yaml["eth_coord_mapping"]);

    return physical_mesh_config;
}

// CmdlineParser methods
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
        } else if (filter_type.value() == "topology" || filter_type.value() == "Topology") {
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
        } else if (filter_type.value() == "routing_type" || filter_type.value() == "Routing_Type") {
            auto r_type = tt::tt_fabric::fabric_tests::RoutingType::LowLatency;  // Default value
            if (filter_value == "LowLatency") {
                r_type = tt::tt_fabric::fabric_tests::RoutingType::LowLatency;
            } else if (filter_value == "Dynamic") {
                r_type = tt::tt_fabric::fabric_tests::RoutingType::Dynamic;
            } else {
                log_info(
                    tt::LogTest,
                    "Unsupported routing type filter value: '{}'. Supported values are: LowLatency, Dynamic",
                    filter_value);
                return false;
            }
            return test_config.fabric_setup.routing_type == r_type;
        } else if (filter_type.value() == "benchmark_mode" || filter_type.value() == "Benchmark_Mode") {
            if (filter_value == "true") {
                return test_config.benchmark_mode;
            } else if (filter_value == "false") {
                return !test_config.benchmark_mode;
            } else {
                log_info(
                    tt::LogTest,
                    "Unsupported benchmark filter value: '{}'. Supported values are: true, false",
                    filter_value);
                return false;
            }
        } else if (filter_type.value() == "sync" || filter_type.value() == "Sync") {
            if (filter_value == "true") {
                return test_config.global_sync;
            } else if (filter_value == "false") {
                return !test_config.global_sync;
            } else {
                log_info(
                    tt::LogTest,
                    "Unsupported sync filter value: '{}'. Supported values are: true, false",
                    filter_value);
                return false;
            }
        } else if (filter_type.value() == "num_links" || filter_type.value() == "Num_Links") {
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
        } else if (filter_type.value() == "ntype") {
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
        } else if (filter_type.value() == "ftype") {
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
        } else if (filter_type.value() == "num_packets") {
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
        } else if (filter_type.value() == "size") {
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
        } else if (filter_type.value() == "pattern") {
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
        } else {
            log_info(
                tt::LogTest,
                "Unsupported filter type: '{}'. Supported types are: name, topology, routing_type, benchmark_mode, "
                "sync, num_links, ntype, ftype, num_packets, size, pattern",
                filter_type.value());
            return false;
        }
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

std::vector<ParsedTestConfig> CmdlineParser::generate_default_configs() {
    log_info(LogTest, "No YAML config provided. Generating a default test configuration from command-line args.");

    TestFabricSetup fabric_setup;
    if (test_args::has_command_option(input_args_, "--topology")) {
        std::string topology_str = test_args::get_command_option(input_args_, "--topology", "Linear");
        fabric_setup.topology = detail::topology_mapper.from_string(topology_str, "Topology");
    } else {
        fabric_setup.topology = Topology::Linear;
        log_info(LogTest, "No topology specified via --topology, defaulting to Linear.");
    }

    ParsedTestConfig default_test;

    if (test_args::has_command_option(input_args_, "--pattern")) {
        log_info(LogTest, "Generating a high-level pattern test from command line.");
        std::string pattern_type = test_args::get_command_option(input_args_, "--pattern", "");
        TT_FATAL(
            detail::high_level_traffic_pattern_mapper.to_enum.find(pattern_type) !=
                detail::high_level_traffic_pattern_mapper.to_enum.end(),
            "Unsupported pattern type from command line: '{}'. Supported types are: {}",
            pattern_type,
            get_supported_high_level_patterns());

        HighLevelPatternConfig hlp_config;
        hlp_config.type = pattern_type;
        hlp_config.iterations = test_args::get_command_option_uint32(input_args_, "--iterations", 1);

        default_test.patterns = std::vector<HighLevelPatternConfig>{hlp_config};
    } else {
        log_info(LogTest, "Generating a simple unicast test from command line.");
        std::string src_device_str = test_args::get_command_option(input_args_, "--src-device", "0");
        std::string dst_device_str = test_args::get_command_option(input_args_, "--dst-device", "1");

        chip_id_t src_device_id = std::stoul(src_device_str);
        chip_id_t dst_device_id = std::stoul(dst_device_str);

        ParsedTrafficPatternConfig pattern = {.destination = ParsedDestinationConfig{.device = dst_device_id}};
        ParsedSenderConfig sender = {.device = src_device_id, .patterns = {pattern}};
        default_test.senders = {sender};
    }

    default_test.name = "DefaultCommandLineTest";
    default_test.fabric_setup = fabric_setup;
    default_test.defaults = ParsedTrafficPatternConfig{};
    default_test.defaults->ftype = ChipSendType::CHIP_UNICAST;
    default_test.defaults->ntype = NocSendType::NOC_UNICAST_WRITE;

    return {default_test};
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

bool CmdlineParser::dump_built_tests() {
    return test_args::has_command_option(input_args_, "--dump-built-tests");
}

std::string CmdlineParser::get_built_tests_dump_file_name(const std::string& default_file_name) {
    auto dump_file = test_args::get_command_option(input_args_, "--built-tests-dump-file", default_file_name);
    return dump_file;
}

bool CmdlineParser::has_help_option() { return test_args::has_command_option(input_args_, "--help"); }

void CmdlineParser::print_help() {
    log_info(LogTest, "Usage: test_tt_fabric [options]");
    log_info(LogTest, "This test can be run in two modes:");
    log_info(
        LogTest,
        "1. With a YAML configuration file (--test_config), which provides detailed control over traffic patterns.");
    log_info(LogTest, "2. With command-line arguments for simpler, predefined test cases.");
    log_info(LogTest, "");
    log_info(LogTest, "General Options:");
    log_info(LogTest, "  --help                                       Print this help message.");
    log_info(
        LogTest,
        "  --test_config <path>                         Path to the YAML test configuration file. See "
        "test_features.yaml for examples.");
    log_info(
        LogTest,
        "  --master-seed <seed>                         Master seed for all random operations to ensure "
        "reproducibility.");
    log_info(LogTest, "");
    log_info(LogTest, "Options for command-line mode (when --test_config is NOT used):");
    log_info(LogTest, "  --topology <Linear|Ring|Mesh|Torus>          Specify the fabric topology. Default: Linear.");
    log_info(
        LogTest,
        "  --pattern <type>                             Specify a high-level traffic pattern. If not provided, a "
        "simple unicast test is run.");
    log_info(
        LogTest,
        "                                               Supported types: {}",
        get_supported_high_level_patterns());
    log_info(
        LogTest,
        "  --src-device <id>                            Source device for simple unicast test. "
        "Default: 0.");
    log_info(
        LogTest,
        "  --dst-device <id>                            Destination device for simple unicast test. "
        "Default: 1.");
    log_info(
        LogTest,
        "  --iterations <N>                             Number of iterations for high-level patterns (e.g., for "
        "different random pairings). Default: 1.");
    log_info(LogTest, "");
    log_info(LogTest, "Value Overrides (can be used with either mode):");
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
        detail::high_level_traffic_pattern_mapper.to_enum.find(config.type) !=
            detail::high_level_traffic_pattern_mapper.to_enum.end(),
        "Unsupported pattern type: '{}'. Supported types are: {}",
        config.type,
        get_supported_high_level_patterns());

    if (pattern_yaml["iterations"]) {
        config.iterations = parse_scalar<uint32_t>(pattern_yaml["iterations"]);
    }
    return config;
}

}  // namespace tt::tt_fabric::fabric_tests
