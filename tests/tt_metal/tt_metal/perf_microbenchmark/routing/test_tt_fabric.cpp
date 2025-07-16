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
#include "tt_fabric_test_memory_map.hpp"

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
using ParsedTestConfig = tt::tt_fabric::fabric_tests::ParsedTestConfig;

using Topology = tt::tt_fabric::Topology;
using FabricConfig = tt::tt_fabric::FabricConfig;
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
    void open_devices(Topology topology, RoutingType routing_type);
    void initialize_sync_memory();
    void compile_programs();
    void launch_programs();
    void wait_for_prorgams();
    void validate_results();
    void close_devices();
    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }
    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }
    void set_global_sync_val(uint32_t val) { global_sync_val_ = val; }

private:
    void add_traffic_config(const TestTrafficConfig& traffic_config);
    void initialize_memory_maps();

    // Track sync cores for each device
    std::unordered_map<FabricNodeId, CoreCoord> device_global_sync_cores_;
    std::unordered_map<FabricNodeId, std::vector<CoreCoord>> device_local_sync_cores_;

    std::shared_ptr<TestFixture> fixture_;
    std::unordered_map<MeshCoordinate, TestDevice> test_devices_;
    std::unique_ptr<tt::tt_fabric::fabric_tests::GlobalAllocator> allocator_;

    // Uniform memory maps shared across all devices
    tt::tt_fabric::fabric_tests::SenderMemoryMap sender_memory_map_;
    tt::tt_fabric::fabric_tests::ReceiverMemoryMap receiver_memory_map_;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies_;
    bool benchmark_mode_ = false;  // Benchmark mode for current test
    bool global_sync_ = false;     // Line sync for current test
    uint32_t global_sync_val_ = 0;

    void reset_local_variables() {
        benchmark_mode_ = false;
        global_sync_ = false;
        global_sync_val_ = 0;
    }
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

    // Get payload buffer size from receiver memory map (cached during initialization)
    uint32_t payload_buffer_size = receiver_memory_map_.get_payload_chunk_size();

    TestTrafficSenderConfig sender_config = {
        .parameters = traffic_config.parameters,
        .src_node_id = traffic_config.src_node_id,
        .dst_node_ids = dst_node_ids,
        .hops = hops,
        .dst_logical_core = dst_logical_core,
        .target_address = target_address,
        .atomic_inc_address = atomic_inc_address,
        .dst_noc_encoding = dst_noc_encoding,
        .payload_buffer_size = payload_buffer_size};

    TestTrafficReceiverConfig receiver_config = {
        .parameters = traffic_config.parameters,
        .sender_id = sender_id,
        .target_address = target_address,
        .atomic_inc_address = atomic_inc_address,
        .payload_buffer_size = payload_buffer_size};

    src_test_device.add_sender_traffic_config(src_logical_core, std::move(sender_config));
    for (const auto& dst_node_id : dst_node_ids) {
        const auto& dst_coord = this->fixture_->get_device_coord(dst_node_id);
        this->test_devices_.at(dst_coord).add_receiver_traffic_config(dst_logical_core, receiver_config);
    }
}

void TestContext::init(
    std::shared_ptr<TestFixture> fixture, const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies) {
    fixture_ = std::move(fixture);
    allocation_policies_ = policies;

    // Initialize memory maps for all available devices
    initialize_memory_maps();

    // Create allocator with memory maps
    this->allocator_ = std::make_unique<tt::tt_fabric::fabric_tests::GlobalAllocator>(
        *this->fixture_, *this->fixture_, policies, sender_memory_map_, receiver_memory_map_);
}

void TestContext::initialize_memory_maps() {
    // Get uniform L1 memory layout (same across all devices)
    uint32_t l1_unreserved_base = this->fixture_->get_l1_unreserved_base();
    uint32_t l1_unreserved_size = this->fixture_->get_l1_unreserved_size();
    uint32_t l1_alignment = this->fixture_->get_l1_alignment();
    uint32_t default_payload_chunk_size = allocation_policies_.default_payload_chunk_size.value_or(
        tt::tt_fabric::fabric_tests::detail::DEFAULT_PAYLOAD_CHUNK_SIZE_BYTES);
    uint32_t max_configs_per_core = std::max(
        allocation_policies_.sender_config.max_configs_per_core,
        allocation_policies_.receiver_config.max_configs_per_core);

    // Create memory maps directly using constructors
    sender_memory_map_ = tt::tt_fabric::fabric_tests::SenderMemoryMap(
        l1_unreserved_base, l1_unreserved_size, l1_alignment, max_configs_per_core);

    receiver_memory_map_ = tt::tt_fabric::fabric_tests::ReceiverMemoryMap(
        l1_unreserved_base, l1_unreserved_size, l1_alignment, default_payload_chunk_size, max_configs_per_core);

    // Validate memory maps
    if (!sender_memory_map_.is_valid() || !receiver_memory_map_.is_valid()) {
        TT_THROW("Invalid memory map configuration");
    }
}

void TestContext::setup_devices() {
    const auto& available_coords = this->fixture_->get_available_device_coordinates();
    for (const auto& coord : available_coords) {
        // Create TestDevice with access to memory maps
        test_devices_.emplace(
            coord, TestDevice(coord, this->fixture_, this->fixture_, &sender_memory_map_, &receiver_memory_map_));
    }
}

void TestContext::reset_devices() {
    test_devices_.clear();
    device_global_sync_cores_.clear();
    device_local_sync_cores_.clear();
    this->allocator_->reset();
    reset_local_variables();
}

void TestContext::open_devices(Topology topology, RoutingType routing_type) {
    fixture_->open_devices(topology, routing_type);
}

void TestContext::initialize_sync_memory() {
    if (!global_sync_) {
        return;  // Only initialize sync memory if line sync is enabled
    }

    log_info(tt::LogTest, "Initializing sync memory for line sync");

    // Initialize sync memory location with 16 bytes of zeros on all devices
    uint32_t global_sync_address = this->sender_memory_map_.get_global_sync_address();
    uint32_t global_sync_memory_size = this->sender_memory_map_.get_global_sync_region_size();
    uint32_t local_sync_address = this->sender_memory_map_.get_local_sync_address();
    uint32_t local_sync_memory_size = this->sender_memory_map_.get_local_sync_region_size();

    // clear the global sync cores in device_global_sync_cores_ using zero_out_buffer_on_cores
    for (const auto& [device_id, global_sync_core] : device_global_sync_cores_) {
        const auto& device_coord = fixture_->get_device_coord(device_id);
        std::vector<CoreCoord> cores = {global_sync_core};
        // zero out the global sync address for global sync core
        fixture_->zero_out_buffer_on_cores(device_coord, cores, global_sync_address, global_sync_memory_size);
        // also need to zero out the local sync address for global sync core
        fixture_->zero_out_buffer_on_cores(device_coord, cores, local_sync_address, global_sync_memory_size);
    }

    // clear the local sync cores in device_local_sync_cores_ using zero_out_buffer_on_cores
    for (const auto& [device_id, local_sync_cores] : device_local_sync_cores_) {
        const auto& device_coord = fixture_->get_device_coord(device_id);
        fixture_->zero_out_buffer_on_cores(device_coord, local_sync_cores, local_sync_address, local_sync_memory_size);
    }

    log_info(
        tt::LogTest,
        "Sync memory initialization complete at address: {} and address: {}",
        global_sync_address,
        local_sync_address);
}

void TestContext::compile_programs() {
    fixture_->setup_workload();
    // TODO: should we be taking const ref?
    for (auto& [coord, test_device] : test_devices_) {
        test_device.set_benchmark_mode(benchmark_mode_);
        test_device.set_global_sync(global_sync_);
        test_device.set_global_sync_val(global_sync_val_);

        auto device_id = test_device.get_node_id();
        test_device.set_sync_core(device_global_sync_cores_[device_id]);

        test_device.create_kernels();
        auto& program_handle = test_device.get_program_handle();
        if (program_handle.num_kernels()) {
            fixture_->enqueue_program(coord, std::move(program_handle));
        }
    }
}

void TestContext::launch_programs() { fixture_->run_programs(); }

void TestContext::wait_for_prorgams() { fixture_->wait_for_programs(); }

void TestContext::validate_results() {
    for (const auto& [_, test_device] : test_devices_) {
        test_device.validate_results();
    }
}

void TestContext::close_devices() { fixture_->close_devices(); }

void TestContext::process_traffic_config(TestConfig& config) {
    this->allocator_->allocate_resources(config);
    log_info(tt::LogTest, "Resource allocation complete");

    if (config.global_sync) {
        // set it only after the test_config is built since it needs set the sync value during expand the high-level
        // patterns.
        this->set_global_sync(config.global_sync);
        this->set_global_sync_val(config.global_sync_val);

        log_info(tt::LogTest, "Enabled sync, global sync value: {}, ", global_sync_val_);

        for (const auto& sync_sender : config.global_sync_configs) {
            CoreCoord sync_core = sync_sender.core.value();
            const auto& device_coord = this->fixture_->get_device_coord(sync_sender.device);

            // Track global sync core for this device
            device_global_sync_cores_[sync_sender.device] = sync_core;

            // Process each already-split sync pattern for this device
            for (const auto& sync_pattern : sync_sender.patterns) {
                // Convert sync pattern to TestTrafficSenderConfig format
                const auto& dest = sync_pattern.destination.value();

                // Patterns are now already split into single-direction hops
                const auto& single_direction_hops = dest.hops.value();

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

                // For sync patterns, we use a dummy destination core and fixed sync address
                // The actual sync will be handled by atomic operations
                CoreCoord dummy_dst_core = {0, 0};                        // Sync doesn't need specific dst core
                uint32_t sync_address = this->sender_memory_map_.get_global_sync_address();  // Hard-coded sync address
                uint32_t dst_noc_encoding = this->fixture_->get_worker_noc_encoding(
                    sync_sender.device, sync_core);  // populate the master coord

                TestTrafficSenderConfig sync_config = {
                    .parameters = sync_traffic_parameters,
                    .src_node_id = sync_sender.device,
                    .dst_node_ids = {},             // Empty for multicast sync
                    .hops = single_direction_hops,  // Use already single-direction hops
                    .dst_logical_core = dummy_dst_core,
                    .target_address = sync_address,
                    .atomic_inc_address = sync_address,
                    .dst_noc_encoding = dst_noc_encoding};

                // Add sync config to the master sender on this device
                this->test_devices_.at(device_coord).add_sender_sync_config(sync_core, std::move(sync_config));
            }
        }

        // Validate that all sync cores have the same coordinate
        if (!device_global_sync_cores_.empty()) {
            CoreCoord reference_sync_core = device_global_sync_cores_.begin()->second;
            for (const auto& [device_id, sync_core] : device_global_sync_cores_) {
                if (sync_core.x != reference_sync_core.x || sync_core.y != reference_sync_core.y) {
                    TT_THROW(
                        "Global sync requires all devices to use the same sync core coordinate. "
                        "Device {} uses sync core ({}, {}) but expected ({}, {}) based on first device.",
                        device_id.chip_id,
                        sync_core.x,
                        sync_core.y,
                        reference_sync_core.x,
                        reference_sync_core.y);
                }
            }
            log_info(
                tt::LogTest,
                "Validated sync core consistency: all {} devices use sync core ({}, {})",
                device_global_sync_cores_.size(),
                reference_sync_core.x,
                reference_sync_core.y);
        }
    }

    for (const auto& sender : config.senders) {
        for (const auto& pattern : sender.patterns) {
            // Track local sync core for this device
            device_local_sync_cores_[sender.device].push_back(sender.core.value());

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

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    auto fixture = std::make_shared<TestFixture>();
    fixture->init();

    // Parse command line and YAML configurations
    CmdlineParser cmdline_parser(input_args);

    if (cmdline_parser.has_help_option()) {
        cmdline_parser.print_help();
        return 0;
    }

    std::vector<ParsedTestConfig> raw_test_configs;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies;

    if (auto yaml_path = cmdline_parser.get_yaml_config_path()) {
        YamlConfigParser yaml_parser;
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

        const auto& topology = test_config.fabric_setup.topology;
        const auto& routing_type = test_config.fabric_setup.routing_type.value();
        log_info(tt::LogTest, "Opening devices with topology: {} and routing type: {}", topology, routing_type);
        test_context.open_devices(topology, routing_type);

        log_info(tt::LogTest, "Building tests");
        auto built_tests = builder.build_tests({test_config});

        // Set benchmark mode and line sync for this test group
        test_context.set_benchmark_mode(test_config.benchmark_mode);

        for (auto& built_test : built_tests) {
            log_info(tt::LogTest, "Running Test: {}", built_test.name);

            test_context.setup_devices();
            log_info(tt::LogTest, "Device setup complete");

            test_context.process_traffic_config(built_test);
            log_info(tt::LogTest, "Traffic config processed");

            // Initialize sync memory if line sync is enabled
            test_context.initialize_sync_memory();

            if (dump_built_tests) {
                YamlTestConfigSerializer::dump({built_test}, output_stream);
            }

            log_info(tt::LogTest, "Compiling programs");
            test_context.compile_programs();

            log_info(tt::LogTest, "Launching programs");
            test_context.launch_programs();

            test_context.wait_for_prorgams();
            log_info(tt::LogTest, "Test {} Finished.", built_test.name);

            test_context.validate_results();
            log_info(tt::LogTest, "Test {} Results validated.", built_test.name);

            test_context.reset_devices();
        }
    }

    test_context.close_devices();

    return 0;
}
