// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <optional>
#include <memory>

#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_allocator.hpp"

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using TrafficParameters = tt::tt_fabric::fabric_tests::TrafficParameters;
using TestTrafficConfig = tt::tt_fabric::fabric_tests::TestTrafficConfig;
using TestTrafficSenderConfig = tt::tt_fabric::fabric_tests::TestTrafficSenderConfig;
using TestTrafficReceiverConfig = tt::tt_fabric::fabric_tests::TestTrafficReceiverConfig;
using TestWorkerType = tt::tt_fabric::fabric_tests::TestWorkerType;

using ChipSendType = tt::tt_fabric::ChipSendType;
using NocSendType = tt::tt_fabric::NocSendType;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using RoutingDirection = tt::tt_fabric::RoutingDirection;

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

using TestConfigBuilder = tt::tt_fabric::fabric_tests::TestConfigBuilder;
using YamlConfigParser = tt::tt_fabric::fabric_tests::YamlConfigParser;
using CmdlineParser = tt::tt_fabric::fabric_tests::CmdlineParser;
using YamlTestConfigSerializer = tt::tt_fabric::fabric_tests::YamlTestConfigSerializer;

using Topology = tt::tt_fabric::Topology;
using FabricConfig = tt::tt_metal::FabricConfig;
using RoutingType = tt::tt_fabric::fabric_tests::RoutingType;

const std::unordered_map<std::pair<Topology, RoutingType>, FabricConfig, tt::tt_fabric::fabric_tests::pair_hash>
    TestFixture::topology_to_fabric_config_map = {
        {{Topology::Linear, RoutingType::LowLatency}, FabricConfig::FABRIC_1D},
        {{Topology::Ring, RoutingType::LowLatency}, FabricConfig::FABRIC_1D_RING},
        {{Topology::Mesh, RoutingType::LowLatency}, FabricConfig::FABRIC_2D},
        {{Topology::Mesh, RoutingType::Dynamic}, FabricConfig::FABRIC_2D_DYNAMIC},
};

const std::string output_dir = "generated/fabric";
const std::string default_built_tests_dump_file = "built_tests.yaml";

class TestContext {
public:
    void init(std::shared_ptr<TestFixture> fixture, const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies);
    void setup_devices();
    void reset_devices();
    void process_traffic_config(TestConfig& config);
    void process_line_sync_config(TestConfig& config);
    void open_devices(Topology topology, RoutingType routing_type);
    void compile_programs();
    void launch_programs();
    void wait_for_prorgams();
    void close_devices();
    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }
    void set_line_sync(bool line_sync) { line_sync_ = line_sync; }

private:
    void add_traffic_config(const TestTrafficConfig& traffic_config);

    // Track master senders for each device to avoid adding sync to all senders
    std::unordered_map<FabricNodeId, CoreCoord> device_master_senders_;

    std::shared_ptr<TestFixture> fixture_;
    std::unordered_map<MeshCoordinate, TestDevice> test_devices_;
    std::unique_ptr<tt::tt_fabric::fabric_tests::GlobalAllocator> allocator_;
    bool benchmark_mode_ = false;  // Benchmark mode for current test
    bool line_sync_ = false;       // Line sync for current test
};

void TestContext::add_traffic_config(const TestTrafficConfig& traffic_config) {
    // This function now assumes all allocation has been done by the GlobalAllocator.
    // It is responsible for taking the planned config and setting up the TestDevice objects.
    const auto& src_node_id = traffic_config.src_node_id;
    const auto& src_coord = this->fixture_->get_device_coord(src_node_id);
    auto& src_test_device = this->test_devices_.at(src_coord);

    CoreCoord src_logical_core = traffic_config.src_logical_core.value();
    CoreCoord dst_logical_core = traffic_config.dst_logical_core.value();
    uint32_t target_address = traffic_config.target_address.value_or(0);
    uint32_t atomic_inc_address = traffic_config.atomic_inc_address.value_or(0);

    std::vector<FabricNodeId> dst_node_ids;
    std::unordered_map<RoutingDirection, uint32_t> hops;

    if (traffic_config.hops.has_value()) {
        hops = traffic_config.hops.value();
        dst_node_ids = this->fixture_->get_dst_node_ids_from_hops(
            traffic_config.src_node_id, hops, traffic_config.parameters.chip_send_type);
    } else {
        dst_node_ids = traffic_config.dst_node_ids.value();
        hops = this->fixture_->get_hops_to_chip(src_node_id, dst_node_ids[0]);
    }

    const auto& dst_rep_coord = this->fixture_->get_device_coord(dst_node_ids[0]);
    uint32_t dst_noc_encoding = this->fixture_->get_worker_noc_encoding(dst_rep_coord, dst_logical_core);
    uint32_t sender_id = fixture_->get_worker_id(traffic_config.src_node_id, src_logical_core);

    TestTrafficSenderConfig sender_config = {
        .parameters = traffic_config.parameters,
        .src_node_id = traffic_config.src_node_id,
        .dst_node_ids = dst_node_ids,
        .hops = hops,
        .dst_logical_core = dst_logical_core,
        .target_address = target_address,
        .atomic_inc_address = atomic_inc_address,
        .dst_noc_encoding = dst_noc_encoding};

    TestTrafficReceiverConfig receiver_config = {
        .parameters = traffic_config.parameters,
        .sender_id = sender_id,
        .target_address = target_address,
        .atomic_inc_address = atomic_inc_address};

    src_test_device.add_sender_traffic_config(src_logical_core, std::move(sender_config));
    for (const auto& dst_node_id : dst_node_ids) {
        const auto& dst_coord = this->fixture_->get_device_coord(dst_node_id);
        this->test_devices_.at(dst_coord).add_receiver_traffic_config(dst_logical_core, receiver_config);
    }
}

void TestContext::init(
    std::shared_ptr<TestFixture> fixture, const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies) {
    fixture_ = std::move(fixture);
    this->allocator_ =
        std::make_unique<tt::tt_fabric::fabric_tests::GlobalAllocator>(*this->fixture_, *this->fixture_, policies);
}

void TestContext::setup_devices() {
    const auto& available_coords = this->fixture_->get_available_device_coordinates();
    for (const auto& coord : available_coords) {
        test_devices_.emplace(coord, TestDevice(coord, this->fixture_, this->fixture_));
    }
}

void TestContext::reset_devices() {
    test_devices_.clear();
    this->allocator_->reset();
}

void TestContext::open_devices(Topology topology, RoutingType routing_type) {
    fixture_->open_devices(topology, routing_type);
}

void TestContext::compile_programs() {
    fixture_->setup_workload();
    // TODO: should we be taking const ref?
    for (auto& [coord, test_device] : test_devices_) {
        test_device.set_benchmark_mode(benchmark_mode_);
        test_device.set_line_sync(line_sync_);
        test_device.set_master_sender(device_master_senders_[test_device.get_node_id()]);
        test_device.create_kernels();
        auto& program_handle = test_device.get_program_handle();
        if (program_handle.num_kernels()) {
            fixture_->enqueue_program(coord, std::move(program_handle));
        }
    }
}

void TestContext::launch_programs() { fixture_->run_programs(); }

void TestContext::wait_for_prorgams() { fixture_->wait_for_programs(); }

void TestContext::close_devices() { fixture_->close_devices(); }

void TestContext::process_traffic_config(TestConfig& config) {
    this->allocator_->allocate_resources(config);
    log_info(tt::LogTest, "Resource allocation complete");

    for (const auto& sender : config.senders) {
        for (const auto& pattern : sender.patterns) {
            // The allocator has already filled in all the necessary details.
            // We just need to construct the TrafficConfig and pass it to add_traffic_config.
            const auto& dest = pattern.destination.value();

            TrafficParameters traffic_parameters = {
                .chip_send_type = pattern.ftype.value(),
                .noc_send_type = pattern.ntype.value(),
                .payload_size_bytes = pattern.size.value(),
                .num_packets = pattern.num_packets.value(),
                .atomic_inc_val = pattern.atomic_inc_val,
                .atomic_inc_wrap = pattern.atomic_inc_wrap,
                .mcast_start_hops = pattern.mcast_start_hops,
                .seed = config.seed,
                .topology = config.fabric_setup.topology,
                .mesh_shape = this->fixture_->get_mesh_shape(),
            };

            TestTrafficConfig traffic_config = {
                .parameters = traffic_parameters,
                .src_node_id = sender.device,
                .src_logical_core = sender.core,
                .dst_logical_core = dest.core,
                .target_address = dest.target_address,
                .atomic_inc_address = dest.atomic_inc_address,
            };

            if (dest.device.has_value()) {
                traffic_config.dst_node_ids = {dest.device.value()};
            }
            if (dest.hops.has_value()) {
                traffic_config.hops = dest.hops;
            }

            this->add_traffic_config(traffic_config);
        }
    }
}

void TestContext::process_line_sync_config(TestConfig& config) {
    if (!config.line_sync || config.global_line_sync_configs.empty()) {
        return;  // No line sync configs to process
    }

    log_info(tt::LogTest, "Processing line sync configurations");

    // Find the master sender for each device (first sender in the device)
    for (const auto& coord : this->fixture_->get_available_device_coordinates()) {
        auto& test_device = this->test_devices_.at(coord);
        auto& device_id = test_device.get_node_id();

        // Find the first sender core for this device (will be the master)
        for (const auto& sender : config.senders) {
            if (sender.device == device_id && sender.core.has_value()) {
                device_master_senders_[device_id] = sender.core.value();
                break;
            }
        }
    }

    // Process each sync sender config
    for (const auto& sync_sender : config.global_line_sync_configs) {
        // Only process sync for devices that have a master sender
        if (device_master_senders_.find(sync_sender.device) == device_master_senders_.end()) {
            log_warning(
                tt::LogTest, "Skipping sync config for device {} - no master sender found", sync_sender.device.chip_id);
            continue;
        }

        CoreCoord master_core = device_master_senders_[sync_sender.device];
        const auto& device_coord = this->fixture_->get_device_coord(sync_sender.device);

        // Process each sync pattern for this device
        for (const auto& sync_pattern : sync_sender.patterns) {
            // Convert sync pattern to TestTrafficSenderConfig format
            const auto& dest = sync_pattern.destination.value();

            // Split multi-directional sync hops into single-direction hops
            const auto& original_hops = dest.hops.value();
            auto split_hops_vec = this->fixture_->split_multicast_hops(original_hops);

            log_info(
                tt::LogTest,
                "Splitting sync pattern for device {} from {} to {} directional configs",
                sync_sender.device.chip_id,
                1,
                split_hops_vec.size());

            // Create separate sync config for each direction
            for (const auto& split_hops : split_hops_vec) {
                TrafficParameters sync_traffic_parameters = {
                    .chip_send_type = sync_pattern.ftype.value(),
                    .noc_send_type = sync_pattern.ntype.value(),
                    .payload_size_bytes = sync_pattern.size.value(),
                    .num_packets = sync_pattern.num_packets.value(),
                    .atomic_inc_val = sync_pattern.atomic_inc_val,
                    .atomic_inc_wrap = sync_pattern.atomic_inc_wrap,
                    .mcast_start_hops = sync_pattern.mcast_start_hops,
                    .seed = config.seed,
                    .topology = config.fabric_setup.topology,
                    .mesh_shape = this->fixture_->get_mesh_shape(),
                };

                // For sync patterns, we use a dummy destination core and address
                // The actual sync will be handled by atomic operations
                CoreCoord dummy_dst_core = {0, 0};  // Sync doesn't need specific dst core
                uint32_t sync_address = 0x50000;    // Fixed sync address
                uint32_t dst_noc_encoding = 0;      // Will be filled by device setup

                TestTrafficSenderConfig sync_config = {
                    .parameters = sync_traffic_parameters,
                    .src_node_id = sync_sender.device,
                    .dst_node_ids = {},  // Empty for multicast sync
                    .hops = split_hops,  // Use single-direction hops
                    .dst_logical_core = dummy_dst_core,
                    .target_address = sync_address,
                    .atomic_inc_address = sync_address,
                    .dst_noc_encoding = dst_noc_encoding};

                // Add sync config to the master sender on this device
                this->test_devices_.at(device_coord).add_sender_sync_config(master_core, std::move(sync_config));
            }
        }
    }

    log_info(tt::LogTest, "Line sync configuration processing complete");
}

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    auto fixture = std::make_shared<TestFixture>();
    fixture->init();

    // fixture is passed to both the parsers since it implements the device interface
    CmdlineParser cmdline_parser(input_args, *fixture);

    if (cmdline_parser.has_help_option()) {
        cmdline_parser.print_help();
        return 0;
    }

    std::vector<TestConfig> raw_test_configs;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies;

    if (auto yaml_path = cmdline_parser.get_yaml_config_path()) {
        YamlConfigParser yaml_parser(*fixture);
        auto parsed_yaml = yaml_parser.parse_file(yaml_path.value());
        raw_test_configs = std::move(parsed_yaml.test_configs);
        if (parsed_yaml.allocation_policies.has_value()) {
            allocation_policies = parsed_yaml.allocation_policies.value();
        }
    } else {
        raw_test_configs = cmdline_parser.generate_default_configs();
    }

    TestContext test_context;
    test_context.init(fixture, allocation_policies);

    cmdline_parser.apply_overrides(raw_test_configs);

    if (raw_test_configs.empty()) {
        log_fatal(tt::LogTest, "No test configurations loaded or generated. Exiting.");
        return 1;
    }

    std::optional<uint32_t> master_seed = cmdline_parser.get_master_seed();
    if (!master_seed.has_value()) {
        master_seed = std::random_device()();
        log_info(tt::LogTest, "No master seed provided. Using randomly generated seed: {}", master_seed.value());
    }
    std::mt19937 gen(master_seed.value());

    // fixture is passed twice since it implements both interfaces
    // the builder object does the initial processing of the tests parsed from yaml/cmd line and tries to fill
    // any gaps/optionals/missing values
    TestConfigBuilder builder(*fixture, *fixture, gen);

    std::ofstream output_stream;
    bool dump_built_tests = cmdline_parser.dump_built_tests();
    if (dump_built_tests) {
        std::filesystem::path dump_file_dir =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / output_dir;
        if (!std::filesystem::exists(dump_file_dir)) {
            std::filesystem::create_directory(dump_file_dir);
        }

        std::string dump_file = cmdline_parser.get_built_tests_dump_file_name(default_built_tests_dump_file);
        std::filesystem::path dump_file_path = dump_file_dir / dump_file;
        output_stream.open(dump_file_path, std::ios::out | std::ios::trunc);

        // dump allocation policies first
        YamlTestConfigSerializer::dump(allocation_policies, output_stream);
    }

    for (auto& test_config : raw_test_configs) {
        log_info(tt::LogTest, "Running Test Group: {}", test_config.name);

        test_context.open_devices(test_config.fabric_setup.topology, test_config.fabric_setup.routing_type.value());

        // Set benchmark mode and line sync for this test group
        test_context.set_benchmark_mode(test_config.benchmark_mode);
        test_context.set_line_sync(test_config.line_sync);

        auto built_tests = builder.build_tests({test_config});

        for (auto& built_test : built_tests) {
            log_info(tt::LogTest, "Running Test: {}", built_test.name);

            test_context.setup_devices();
            log_info(tt::LogTest, "Device setup complete");

            test_context.process_traffic_config(built_test);
            log_info(tt::LogTest, "Traffic config processed");

            test_context.process_line_sync_config(built_test);
            log_info(tt::LogTest, "Line sync config processed");

            if (dump_built_tests) {
                YamlTestConfigSerializer::dump({built_test}, output_stream);
            }

            log_info(tt::LogTest, "Compiling programs");
            test_context.compile_programs();

            log_info(tt::LogTest, "Launching programs");
            test_context.launch_programs();

            test_context.wait_for_prorgams();
            log_info(tt::LogTest, "Test {} Finished.", built_test.name);

            test_context.reset_devices();
        }
    }

    test_context.close_devices();

    return 0;
}
