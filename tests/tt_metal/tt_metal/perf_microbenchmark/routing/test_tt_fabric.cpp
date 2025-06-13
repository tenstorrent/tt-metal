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

#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using TestTrafficDataConfig = tt::tt_fabric::fabric_tests::TestTrafficDataConfig;
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

const std::string default_sender_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_sender.cpp";
const std::string default_receiver_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_receiver.cpp";

// TODO: move this to generated directory
const std::string default_built_tests_dump_file = "built_tests.yaml";

class TestContext {
public:
    void init(TestFixture& fixture);
    void setup_devices();
    void reset_devices();
    void process_traffic_config(const TestConfig& config);
    void open_devices(tt::tt_metal::FabricConfig fabric_config);
    void compile_programs();
    void launch_programs();
    void wait_for_prorgams();
    void close_devices();

private:
    CoreCoord find_receiver_core(const std::vector<FabricNodeId>& phys_chip_ids);
    void add_traffic_config(const TestTrafficConfig& traffic_config);

    TestFixture* fixture_;
    std::unordered_map<MeshCoordinate, TestDevice> test_devices_;
};

// TODO: this mapping should come from a global allocator that is multi-host aware
CoreCoord TestContext::find_receiver_core(const std::vector<FabricNodeId>& node_ids) {
    std::unordered_map<CoreCoord, uint32_t> available_cores_histogram;
    for (const auto& node_id : node_ids) {
        const auto& coord = this->fixture_->get_device_coord(node_id);
        const auto& available_cores = this->test_devices_.at(coord).get_available_worker_cores();
        for (const auto& core : available_cores) {
            available_cores_histogram[core]++;
        }
    }

    std::optional<CoreCoord> available_core;
    for (const auto& [core, count] : available_cores_histogram) {
        // for now return the first core that is avaialble across all the chips
        if (count == node_ids.size()) {
            available_core = core;
            break;
        }
    }

    if (!available_core.has_value()) {
        log_fatal(tt::LogTest, "Unable to find a common recv core from chips");
        throw std::runtime_error("No common cores found for allocation");
    }

    return available_core.value();
}

void TestContext::add_traffic_config(const TestTrafficConfig& traffic_config) {
    // by this point assume that the traffic config is well formed/massaged
    // for now assume that the cores are not shared
    // later we can grab the available address/core pairs

    const auto& src_node_id = traffic_config.src_node_id;
    const auto& src_coord = this->fixture_->get_device_coord(src_node_id);
    auto& src_test_device = this->test_devices_.at(src_coord);

    std::vector<FabricNodeId> dst_node_ids;
    std::unordered_map<RoutingDirection, uint32_t> hops;

    if (traffic_config.hops.has_value()) {
        hops = traffic_config.hops.value();
        dst_node_ids = this->fixture_->get_dst_node_ids_from_hops(
            traffic_config.src_node_id, hops, traffic_config.data_config.chip_send_type);
    } else {
        dst_node_ids = traffic_config.dst_node_ids.value();
        // TODO: this shouldnt be needed here since the config should be well-formed by now
        TT_FATAL(dst_node_ids.size() == 1, "Expected only a single dst node id when hops are not specified");
        hops = this->fixture_->get_hops_to_chip(src_node_id, dst_node_ids[0]);
    }

    // TODO: any handling of the src/dst should be host local
    // Add logic to skip if a srd/dst doesnt belong to one of the local chips

    // this is only a representative of the dst devices used for common operations for dst devices
    const auto& dst_rep_coord = this->fixture_->get_device_coord(dst_node_ids[0]);
    auto& dst_rep_test_device = this->test_devices_.at(dst_rep_coord);

    CoreCoord src_logical_core;
    if (traffic_config.src_logical_core.has_value()) {
        src_logical_core = traffic_config.src_logical_core.value();
    } else {
        // TODO: this allocation should move to a global allocator
        src_logical_core = src_test_device.allocate_worker_core();
    }
    src_test_device.reserve_core_for_worker(TestWorkerType::SENDER, src_logical_core, traffic_config.sender_kernel_src);

    CoreCoord dst_logical_core;
    if (traffic_config.dst_logical_core.has_value()) {
        dst_logical_core = traffic_config.dst_logical_core.value();
    } else {
        dst_logical_core = this->find_receiver_core(dst_node_ids);
    }
    for (const auto& dst_node_id : dst_node_ids) {
        // TODO: this allocation should move to a global allocator
        const auto& dst_coord = this->fixture_->get_device_coord(dst_node_id);
        this->test_devices_.at(dst_coord).reserve_core_for_worker(
            TestWorkerType::RECEIVER, dst_logical_core, traffic_config.receiver_kernel_src);
    }

    // TODO: this allocation should move to a global allocator
    size_t target_address = dst_rep_test_device.allocate_address_for_sender(dst_logical_core);
    uint32_t dst_noc_encoding = this->fixture_->get_worker_noc_encoding(dst_rep_coord, dst_logical_core);
    uint32_t sender_id = fixture_->get_worker_id(traffic_config.src_node_id, src_logical_core);

    TestTrafficSenderConfig sender_config = {
        .data_config = traffic_config.data_config,
        .dst_node_ids = dst_node_ids,
        .hops = hops,
        .dst_logical_core = dst_logical_core,
        .target_address = target_address,
        .dst_noc_encoding = dst_noc_encoding};

    TestTrafficReceiverConfig receiver_config = {
        .data_config = traffic_config.data_config, .sender_id = sender_id, .target_address = target_address};

    // TODO: make this host aware
    src_test_device.add_sender_traffic_config(src_logical_core, sender_config);
    for (const auto& dst_node_id : dst_node_ids) {
        const auto& dst_coord = this->fixture_->get_device_coord(dst_node_id);
        this->test_devices_.at(dst_coord).add_receiver_traffic_config(dst_logical_core, receiver_config);
    }
}

void TestContext::init(TestFixture& fixture) { fixture_ = &fixture; }

void TestContext::setup_devices() {
    const auto& available_coords = this->fixture_->get_available_device_coordinates();
    for (const auto& coord : available_coords) {
        test_devices_.emplace(coord, TestDevice(coord, *this->fixture_, *this->fixture_));
    }
}

void TestContext::reset_devices() { test_devices_.clear(); }

void TestContext::open_devices(tt::tt_metal::FabricConfig fabric_config) { fixture_->open_devices(fabric_config); }

void TestContext::compile_programs() {
    // TODO: should we be taking const ref?
    for (auto& [coord, test_device] : test_devices_) {
        test_device.create_kernels();
        auto& program_handle = test_device.get_program_handle();
        if (program_handle.num_kernels()) {
            fixture_->enqueue_program(coord, program_handle);
        }
    }
}

void TestContext::launch_programs() { fixture_->run_programs(); }

void TestContext::wait_for_prorgams() { fixture_->wait_for_programs(); }

void TestContext::close_devices() { fixture_->close_devices(); }

void TestContext::process_traffic_config(const TestConfig& config) {
    for (const auto& sender : config.senders) {
        for (const auto& pattern : sender.patterns) {
            // After merging, these fields must have a value.
            TT_FATAL(
                pattern.ftype.has_value(), "Missing 'ftype' in traffic pattern for sender on device {}", sender.device);
            TT_FATAL(
                pattern.ntype.has_value(), "Missing 'ntype' in traffic pattern for sender on device {}", sender.device);
            TT_FATAL(
                pattern.size.has_value(), "Missing 'size' in traffic pattern for sender on device {}", sender.device);
            TT_FATAL(
                pattern.destination.has_value(),
                "Missing 'destination' in traffic pattern for sender on device {}",
                sender.device);
            TT_FATAL(
                pattern.destination->device.has_value() || pattern.destination->hops.has_value(),
                "Missing 'device' or 'hops' in destination for sender on device {}",
                sender.device);
            TT_FATAL(
                !pattern.destination->device.has_value() || !pattern.destination->hops.has_value(),
                "Only one of 'device' and 'hops' should be provided as destination");
            TT_FATAL(
                pattern.num_packets.has_value(),
                "Missing 'num_packets' in traffic pattern for sender on device {}",
                sender.device);

            TestTrafficDataConfig data_config = {
                .chip_send_type = pattern.ftype.value(),
                .noc_send_type = pattern.ntype.value(),
                .seed = config.seed,
                .num_packets = pattern.num_packets.value(),
                .payload_size_bytes = pattern.size.value(),
            };

            TestTrafficConfig traffic_config = {
                .data_config = data_config,
                .src_node_id = sender.device,
                .src_logical_core = sender.core,
                // TODO: take this as input?
                .sender_kernel_src = default_sender_kernel_src,
                .receiver_kernel_src = default_receiver_kernel_src};

            if (pattern.destination.has_value()) {
                if (pattern.destination->device.has_value()) {
                    traffic_config.dst_node_ids = {pattern.destination->device.value()};
                }
                if (pattern.destination->core.has_value()) {
                    traffic_config.dst_logical_core = pattern.destination->core;
                }
                if (pattern.destination->hops.has_value()) {
                    traffic_config.hops = pattern.destination->hops;
                }
            }

            this->add_traffic_config(traffic_config);
        }
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    TestFixture fixture;
    fixture.init();

    TestContext test_context;
    test_context.init(fixture);

    // fixture is passed to both the parsers since it implements the device interface

    CmdlineParser cmdline_parser(input_args, fixture);
    std::vector<TestConfig> raw_test_configs;

    if (auto yaml_path = cmdline_parser.get_yaml_config_path()) {
        YamlConfigParser yaml_parser(fixture);
        raw_test_configs = yaml_parser.parse_file(yaml_path.value());
    } else {
        raw_test_configs = cmdline_parser.generate_default_configs();
    }

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
    TestConfigBuilder builder(fixture, fixture, gen);
    auto built_tests = builder.build_tests(raw_test_configs);

    if (cmdline_parser.dump_built_tests()) {
        auto dump_path = cmdline_parser.get_built_tests_dump_file_path(default_built_tests_dump_file);
        YamlTestConfigSerializer::dump(built_tests, dump_path);
    }

    // TODO: for now assume we are working with the same fabric config, later we need to close and re-open with
    // different config
    test_context.open_devices(tt::tt_metal::FabricConfig::FABRIC_1D);

    for (const auto& test_config : built_tests) {
        log_info(tt::LogTest, "Running Test: {}", test_config.name);

        test_context.setup_devices();

        test_context.process_traffic_config(test_config);

        test_context.compile_programs();

        test_context.launch_programs();

        test_context.wait_for_prorgams();

        log_info(tt::LogTest, "Test {} Finished.", test_config.name);

        test_context.reset_devices();
    }

    test_context.close_devices();

    return 0;
}
