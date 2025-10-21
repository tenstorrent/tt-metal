// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"

#include "tests/tt_metal/test_utils/test_common.hpp"

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/routing_table_generator.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common_types.hpp"
#include <tt-metalium/hal.hpp>

namespace tt::tt_fabric {
namespace fabric_tests {

// Helper template for static_assert in visitor - must be defined before use
template <class>
inline constexpr bool always_false_v = false;

// Helper functions and mappings for converting between string representations in YAML
// and their corresponding enum types.
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
    {"unicast_scatter_write", NocSendType::NOC_UNICAST_SCATTER_WRITE},
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
    {"Torus", Topology::Torus},
});

static const StringEnumMapper<RoutingType> routing_type_mapper({
    {"LowLatency", RoutingType::LowLatency},
    {"Dynamic", RoutingType::Dynamic},
});

static const StringEnumMapper<FabricTensixConfig> fabric_tensix_type_mapper({
    {"Default", FabricTensixConfig::DISABLED},
    {"Mux", FabricTensixConfig::MUX},
});

static const StringEnumMapper<FabricReliabilityMode> fabric_reliability_mode_mapper({
    {"STRICT_SYSTEM_HEALTH_SETUP_MODE", FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE},
    {"RELAXED_SYSTEM_HEALTH_SETUP_MODE", FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE},
    {"DYNAMIC_RECONFIGURATION_SETUP_MODE", FabricReliabilityMode::DYNAMIC_RECONFIGURATION_SETUP_MODE},
});

static const StringEnumMapper<CoreAllocationPolicy> core_allocation_policy_mapper({
    {"RoundRobin", CoreAllocationPolicy::RoundRobin},
    {"ExhaustFirst", CoreAllocationPolicy::ExhaustFirst},
});

static const StringEnumMapper<HighLevelTrafficPattern> high_level_traffic_pattern_mapper({
    {"all_to_all", HighLevelTrafficPattern::AllToAll},
    {"one_to_all", HighLevelTrafficPattern::OneToAll},
    {"all_to_one", HighLevelTrafficPattern::AllToOne},
    {"all_to_one_random", HighLevelTrafficPattern::AllToOneRandom},
    {"full_device_random_pairing", HighLevelTrafficPattern::FullDeviceRandomPairing},
    {"unidirectional_linear", HighLevelTrafficPattern::UnidirectionalLinear},
    {"perimeter_linear", HighLevelTrafficPattern::PerimeterLinear},
    {"neighbor_exchange", HighLevelTrafficPattern::NeighborExchange},
    {"full_ring", HighLevelTrafficPattern::FullRing},
    {"half_ring", HighLevelTrafficPattern::HalfRing},
    {"all_devices_uniform_pattern", HighLevelTrafficPattern::AllDevicesUniformPattern},
});
// Optimized string concatenation utility to avoid multiple allocations
template <typename... Args>
void append_with_separator(std::string& target, std::string_view separator, const Args&... args) {
    // Calculate total size needed
    size_t total_size = target.size();
    auto add_size = [&total_size, &separator](const auto& arg) {
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(arg)>>) {
            // For numeric types, estimate string length (conservative estimate)
            total_size += separator.size() + 20;  // 20 chars should handle most numeric types
        } else {
            // For string types
            total_size += separator.size() + std::string(arg).size();
        }
    };

    // fold expression: calls add_size for each argument to calculate total size
    // For args (a, b, c), this expands to: add_size(a), add_size(b), add_size(c)
    (add_size(args), ...);

    // Reserve space to avoid reallocations
    target.reserve(total_size);

    // Append each argument with separator
    auto append_arg = [&target, &separator](const auto& arg) {
        target += separator;
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(arg)>>) {
            target += std::to_string(arg);
        } else {
            target += std::string(arg);
        }
    };

    // fold expression: calls append_arg for each argument in sequence
    // For args (a, b, c), this expands to: append_arg(a), append_arg(b), append_arg(c)
    (append_arg(args), ...);
}

}  // namespace detail

// Helper function to resolve DeviceIdentifier to FabricNodeId
inline FabricNodeId resolve_device_identifier(const DeviceIdentifier& device_id, const IDeviceInfoProvider& provider) {
    return std::visit(
        [&provider](const auto& id) -> FabricNodeId {
            using T = std::decay_t<decltype(id)>;
            if constexpr (std::is_same_v<T, FabricNodeId>) {
                return id;  // Already resolved
            } else if constexpr (std::is_same_v<T, ChipId>) {
                return provider.get_fabric_node_id(id);
            } else if constexpr (std::is_same_v<T, std::pair<MeshId, ChipId>>) {
                return FabricNodeId{id.first, id.second};
            } else if constexpr (std::is_same_v<T, std::pair<MeshId, MeshCoordinate>>) {
                return provider.get_fabric_node_id(id.first, id.second);
            } else {
                static_assert(always_false_v<T>, "Unsupported DeviceIdentifier type");
            }
        },
        device_id);
}

struct ParsedYamlConfig {
    std::vector<ParsedTestConfig> test_configs;
    std::optional<AllocatorPolicies> allocation_policies;
    std::optional<PhysicalMeshConfig> physical_mesh_config;
};

template <typename TrafficPatternType>
inline TrafficPatternType merge_patterns(const TrafficPatternType& base, const TrafficPatternType& specific) {
    TrafficPatternType merged;

    merged.ftype = specific.ftype.has_value() ? specific.ftype : base.ftype;
    merged.ntype = specific.ntype.has_value() ? specific.ntype : base.ntype;
    merged.size = specific.size.has_value() ? specific.size : base.size;
    merged.num_packets = specific.num_packets.has_value() ? specific.num_packets : base.num_packets;
    merged.atomic_inc_val = specific.atomic_inc_val.has_value() ? specific.atomic_inc_val : base.atomic_inc_val;
    merged.mcast_start_hops = specific.mcast_start_hops.has_value() ? specific.mcast_start_hops : base.mcast_start_hops;

    // Special handling for nested destination
    if (specific.destination.has_value()) {
        if (base.destination.has_value()) {
            // Both have destinations, merge them.
            auto merged_dest = base.destination.value();  // Start with base as default
            const auto& spec_dest = specific.destination.value();

            // Override with specific values if they exist
            if (spec_dest.device.has_value()) {
                merged_dest.device = spec_dest.device;
            }
            if (spec_dest.core.has_value()) {
                merged_dest.core = spec_dest.core;
            }
            if (spec_dest.hops.has_value()) {
                merged_dest.hops = spec_dest.hops;
            }
            if (spec_dest.target_address.has_value()) {
                merged_dest.target_address = spec_dest.target_address;
            }
            if (spec_dest.atomic_inc_address.has_value()) {
                merged_dest.atomic_inc_address = spec_dest.atomic_inc_address;
            }

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
    YamlConfigParser() = default;

    ParsedYamlConfig parse_file(const std::string& yaml_config_path);

private:
    DeviceIdentifier parse_device_identifier(const YAML::Node& node);
    ParsedDestinationConfig parse_destination_config(const YAML::Node& dest_yaml);
    ParsedTrafficPatternConfig parse_traffic_pattern_config(const YAML::Node& pattern_yaml);
    ParsedSenderConfig parse_sender_config(const YAML::Node& sender_yaml, const ParsedTrafficPatternConfig& defaults);
    TestFabricSetup parse_fabric_setup(const YAML::Node& fabric_setup_yaml);
    ParsedTestConfig parse_test_config(const YAML::Node& test_yaml);
    AllocatorPolicies parse_allocator_policies(const YAML::Node& policies_yaml);
    CoreAllocationConfig parse_core_allocation_config(const YAML::Node& config_yaml, CoreAllocationConfig base_config);
    PhysicalMeshConfig parse_physical_mesh_config(const YAML::Node& physical_mesh_yaml);

    // Parsing helpers
    CoreCoord parse_core_coord(const YAML::Node& node);
    MeshCoordinate parse_mesh_coord(const YAML::Node& node);
    MeshId parse_mesh_id(const YAML::Node& yaml_node);
    template <typename T>
    T parse_scalar(const YAML::Node& yaml_node);
    template <typename T>
    std::vector<T> parse_scalar_sequence(const YAML::Node& yaml_node);
    template <typename T>
    std::vector<std::vector<T>> parse_2d_array(const YAML::Node& yaml_node);
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
    CmdlineParser(const std::vector<std::string>& input_args);

    std::optional<std::string> get_yaml_config_path();
    bool check_filter(ParsedTestConfig& test_config, bool fine_grained);
    void apply_overrides(std::vector<ParsedTestConfig>& test_configs);
    std::optional<uint32_t> get_master_seed();
    bool dump_built_tests();
    std::string get_built_tests_dump_file_name(const std::string& default_file_name);
    bool has_help_option();
    void print_help();

    // Progress monitoring options
    bool show_progress();
    uint32_t get_progress_interval();
    uint32_t get_hung_threshold();

private:
    // Helpers for check_filter
    bool check_filter_name(const ParsedTestConfig& test_config) const;
    bool check_filter_topology(const ParsedTestConfig& test_config) const;
    bool check_filter_routing_type(const ParsedTestConfig& test_config) const;
    bool check_filter_benchmark_mode(const ParsedTestConfig& test_config) const;
    bool check_filter_sync(const ParsedTestConfig& test_config) const;
    bool check_filter_num_links(const ParsedTestConfig& test_config, bool fine_grained) const;
    bool check_filter_ntype(ParsedTestConfig& test_config, bool fine_grained);
    bool check_filter_ftype(ParsedTestConfig& test_config, bool fine_grained);
    bool check_filter_num_packets(ParsedTestConfig& test_config, bool fine_grained);
    bool check_filter_size(ParsedTestConfig& test_config, bool fine_grained);
    bool check_filter_pattern(const ParsedTestConfig& test_config) const;

    const std::vector<std::string>& input_args_;
    std::optional<std::string> filter_type;
    std::optional<std::string> filter_value;
};

const std::string no_default_test_yaml_config;

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
    sequence.reserve(yaml_node.size());
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
    pair_sequence.reserve(yaml_node.size());
    for (const auto& entry : yaml_node) {
        TT_FATAL(entry.IsSequence(), "Expected each entry to be sequence");
        pair_sequence.push_back(parse_pair<T1, T2>(entry));
    }

    return pair_sequence;
}

template <typename T>
inline std::vector<std::vector<T>> YamlConfigParser::parse_2d_array(const YAML::Node& yaml_node) {
    std::vector<std::vector<T>> array;
    TT_FATAL(yaml_node.IsSequence(), "Expected a sequence for 2D array");

    for (const auto& row : yaml_node) {
        TT_FATAL(row.IsSequence(), "Expected each row to be a sequence");
        std::vector<T> row_vector;
        row_vector.reserve(row.size());
        for (const auto& entry : row) {
            // only deals with ethernet core case
            if constexpr (std::is_same_v<T, EthCoord>) {
                TT_FATAL(entry.size() == 5, "Expected ethernet core coordinates to be a sequence of 5 elements");
                row_vector.push_back(EthCoord{
                    parse_scalar<uint32_t>(entry[0]),
                    parse_scalar<uint32_t>(entry[1]),
                    parse_scalar<uint32_t>(entry[2]),
                    parse_scalar<uint32_t>(entry[3]),
                    parse_scalar<uint32_t>(entry[4])});
            } else {
                TT_THROW("Unsupported entry type in 2D array for type: {}", entry.Type());
            }
        }
        array.push_back(std::move(row_vector));
    }

    return array;
}

template <typename T>
inline std::vector<T> YamlConfigParser::get_elements_in_range(T start, T end) {
    std::vector<T> range(end - start + 1);
    std::iota(range.begin(), range.end(), start);
    return range;
}

class TestConfigBuilder {
public:
    TestConfigBuilder(IDeviceInfoProvider& device_info_provider, IRouteManager& route_manager, std::mt19937& gen) :
        device_info_provider_(device_info_provider), route_manager_(route_manager), gen_(gen) {}

    std::vector<TestConfig> build_tests(
        const std::vector<ParsedTestConfig>& raw_configs, CmdlineParser& cmdline_parser);

private:
    static constexpr uint32_t MIN_RING_TOPOLOGY_DEVICES = 4;

    // Randomization helpers
    template <typename T>
    T get_random_choice(const std::vector<T>& choices) {
        if (choices.empty()) {
            TT_THROW("Cannot make a random choice from an empty list.");
        }
        std::uniform_int_distribution<> distrib(0, choices.size() - 1);
        return choices[distrib(this->gen_)];
    }

    uint32_t get_random_in_range(uint32_t min, uint32_t max);

    // Helper function to check if a test should be skipped based on:
    // 1. topology and device count
    // 2. architecture or cluster type
    bool should_skip_test(const ParsedTestConfig& test_config) const;

    // Convert ParsedTestConfig to TestConfig by resolving device identifiers
    TestConfig resolve_test_config(const ParsedTestConfig& parsed_test, uint32_t iteration_number);

    SenderConfig resolve_sender_config(const ParsedSenderConfig& parsed_sender);

    TrafficPatternConfig resolve_traffic_pattern(const ParsedTrafficPatternConfig& parsed_pattern);

    DestinationConfig resolve_destination_config(const ParsedDestinationConfig& parsed_dest);

    std::vector<TestConfig> expand_high_level_patterns(ParsedTestConfig& p_config);

    std::vector<ParsedTestConfig> expand_parametrizations(const ParsedTestConfig& raw_config);

    void validate_pattern(const TrafficPatternConfig& pattern, const TestConfig& test) const;

    void validate_chip_unicast(
        const TrafficPatternConfig& pattern, const SenderConfig& sender, const TestConfig& test) const;

    void validate_chip_multicast(
        const TrafficPatternConfig& pattern, const SenderConfig& sender, const TestConfig& test) const;

    void validate_sync_pattern(
        const TrafficPatternConfig& pattern, const SenderConfig& sender, const TestConfig& test) const;

    void validate_test(const TestConfig& test) const;

    void expand_patterns_into_test(
        ParsedTestConfig& test, const std::vector<HighLevelPatternConfig>& patterns, uint32_t iteration_idx);

    void expand_one_or_all_to_all_unicast(
        ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, HighLevelTrafficPattern pattern_type);

    void expand_all_to_one_unicast(
        ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, uint32_t iteration_idx);

    void expand_all_to_one_random_unicast(ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern);

    void expand_full_device_random_pairing(ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern);

    void expand_all_devices_uniform_pattern(ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern);

    void expand_one_or_all_to_all_multicast(
        ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, HighLevelTrafficPattern pattern_type);

    void expand_unidirectional_linear_unicast_or_multicast(
        ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern);

    void expand_perimeter_linear_unicast_or_multicast(
        ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern);

    void expand_full_or_half_ring_unicast_or_multicast(
        ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern, HighLevelTrafficPattern pattern_type);

    void expand_neighbor_exchange_unicast_or_multicast(
        ParsedTestConfig& test, const ParsedTrafficPatternConfig& base_pattern);

    void expand_sync_patterns(ParsedTestConfig& test);

    std::pair<std::vector<TrafficPatternConfig>, uint32_t> create_sync_patterns_for_topology(
        const FabricNodeId& src_device, const std::vector<FabricNodeId>& devices);

    void add_senders_from_pairs(
        ParsedTestConfig& test,
        const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
        const ParsedTrafficPatternConfig& base_pattern);

    void split_all_unicast_or_multicast_patterns(ParsedTestConfig& test);

    bool expand_link_duplicates(ParsedTestConfig& test);

    void resolve_missing_params(ParsedTestConfig& test);

    IDeviceInfoProvider& device_info_provider_;
    IRouteManager& route_manager_;
    std::mt19937& gen_;
};

// ======================================================================================
// Serialization to YAML
// ======================================================================================

class YamlTestConfigSerializer {
public:
    static void dump(const PhysicalMeshConfig& physical_mesh_config, std::ofstream& fout) {
        YAML::Emitter out;
        out << YAML::BeginMap;

        out << YAML::Key << "physical_mesh";
        out << YAML::Value;
        to_yaml(out, physical_mesh_config);

        out << YAML::EndMap;

        fout << out.c_str() << std::endl;
    }

    static void dump(const AllocatorPolicies& policies, std::ofstream& fout) {
        YAML::Emitter out;
        out << YAML::BeginMap;

        out << YAML::Key << "allocation_policies";
        out << YAML::Value;
        to_yaml(out, policies);

        out << YAML::EndMap;

        fout << out.c_str() << std::endl;
    }

    static void dump(const std::vector<TestConfig>& test_configs, std::ofstream& fout) {
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
        if (config.atomic_inc_address) {
            out << YAML::Key << "atomic_inc_address";
            out << YAML::Value << config.atomic_inc_address.value();
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

    static std::string to_string(RoutingType rtype) {
        return detail::routing_type_mapper.to_string(rtype, "RoutingType");
    }

    static std::string to_string(FabricTensixConfig ftype) {
        return detail::fabric_tensix_type_mapper.to_string(ftype, "FabricTensixConfig");
    }
    static std::string to_string(FabricReliabilityMode mode) {
        return detail::fabric_reliability_mode_mapper.to_string(mode, "FabricReliabilityMode");
    }

    static std::string to_string(tt::tt_fabric::Topology topology) {
        return detail::topology_mapper.to_string(topology, "Topology");
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

    static void to_yaml(YAML::Emitter& out, const TestConfig& config) {
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

        if (config.benchmark_mode) {
            out << YAML::Key << "benchmark_mode";
            out << YAML::Value << config.benchmark_mode;
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

    static std::string to_string(CoreAllocationPolicy policy) {
        return detail::core_allocation_policy_mapper.to_string(policy, "CoreAllocationPolicy");
    }

    static void to_yaml(YAML::Emitter& out, const CoreAllocationConfig& config) {
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

    static void to_yaml(YAML::Emitter& out, const PhysicalMeshConfig& config) {
        out << YAML::BeginMap;
        out << YAML::Key << "mesh_descriptor_path";
        out << YAML::Value << config.mesh_descriptor_path;
        out << YAML::Key << "eth_coord_mapping";
        out << YAML::Value;
        to_yaml(out, config.eth_coord_mapping);
        out << YAML::EndMap;
    }

    static void to_yaml(YAML::Emitter& out, const std::vector<std::vector<EthCoord>>& mapping) {
        out << YAML::BeginSeq;
        for (const auto& row : mapping) {
            out << YAML::BeginSeq;
            for (const auto& coord : row) {
                out << YAML::Flow << YAML::BeginSeq << coord.cluster_id << coord.x << coord.y << coord.rack
                    << coord.shelf << YAML::EndSeq;
            }
            out << YAML::EndSeq;
        }
        out << YAML::EndSeq;
    }

    static void to_yaml(YAML::Emitter& out, const AllocatorPolicies& policies) {
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

    static void to_yaml(YAML::Emitter& out, const TestFabricSetup& config) {
        out << YAML::BeginMap;
        out << YAML::Key << "topology";
        out << YAML::Value << to_string(config.topology);
        if (config.routing_type.has_value()) {
            out << YAML::Key << "routing_type";
            out << YAML::Value << to_string(config.routing_type.value());
        }
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
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
