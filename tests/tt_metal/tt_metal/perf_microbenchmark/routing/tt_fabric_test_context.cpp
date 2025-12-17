// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_context.hpp"

#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

void TestContext::add_sync_traffic_to_devices(const TestConfig& config) {
    for (const auto& sync_config : config.sync_configs) {
        // currently initializing our sync configs to be on senders local to the current host
        const auto& sync_sender = sync_config.sender_config;
        if (fixture_->is_local_fabric_node_id(sync_sender.device)) {
            CoreCoord sync_core = sync_sender.core.value();
            const auto& device_coord = this->fixture_->get_device_coord(sync_sender.device);

            // Track global sync core for this device
            device_global_sync_cores_[sync_sender.device] = sync_core;

            // Process each already-split sync pattern for this device
            for (const auto& sync_pattern : sync_sender.patterns) {
                // Convert sync pattern to TestTrafficSenderConfig format
                const auto& dest = sync_pattern.destination.value();

                TrafficParameters sync_traffic_parameters = {
                    .chip_send_type = sync_pattern.ftype.value(),
                    .noc_send_type = sync_pattern.ntype.value(),
                    .payload_size_bytes = sync_pattern.size.value(),
                    .num_packets = sync_pattern.num_packets.value(),
                    .atomic_inc_val = sync_pattern.atomic_inc_val,
                    .mcast_start_hops = sync_pattern.mcast_start_hops,
                    .seed = config.seed,
                    .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
                    .mesh_shape = this->fixture_->get_mesh_shape(),
                    .topology = this->fixture_->get_topology()};

                // For sync patterns, we use a dummy destination core and fixed sync address
                // The actual sync will be handled by atomic operations
                CoreCoord dummy_dst_core = {0, 0};  // Sync doesn't need specific dst core
                uint32_t sync_address = this->sender_memory_map_.get_global_sync_address();  // Hard-coded sync address
                uint32_t dst_noc_encoding =
                    this->fixture_->get_worker_noc_encoding(sync_core);  // populate the master coord

                TestTrafficSenderConfig sync_traffic_sender_config = {
                    .parameters = sync_traffic_parameters,
                    .src_node_id = sync_sender.device,
                    .dst_logical_core = dummy_dst_core,
                    .target_address = sync_address,
                    .atomic_inc_address = sync_address,
                    .dst_noc_encoding = dst_noc_encoding,
                    .link_id = sync_sender.link_id};  // Derive from SenderConfig (always 0 for sync)

                // Determine destination node IDs
                auto single_direction_hops = dest.hops.value();
                sync_traffic_sender_config.hops = single_direction_hops;
                // for 2d mcast case
                sync_traffic_sender_config.dst_node_ids = this->fixture_->get_dst_node_ids_from_hops(
                    sync_sender.device, single_direction_hops, sync_traffic_parameters.chip_send_type);
                // for 2d, we need to specify the mcast start node id
                if (fixture_->is_2D_routing_enabled() &&
                    sync_traffic_parameters.chip_send_type == ChipSendType::CHIP_MULTICAST) {
                    sync_traffic_sender_config.mcast_start_node_id =
                        fixture_->get_mcast_start_node_id(sync_sender.device, single_direction_hops);
                } else {
                    sync_traffic_sender_config.mcast_start_node_id = std::nullopt;
                }

                // Add sync config to the master sender on this device
                TestTrafficSyncConfig sync_traffic_sync_config = {
                    .sync_val = sync_config.sync_val, .sender_config = std::move(sync_traffic_sender_config)};

                this->test_devices_.at(device_coord)
                    .add_sender_sync_config(sync_core, std::move(sync_traffic_sync_config));
            }
        }
    }
}

void TestContext::wait_for_programs_with_progress() {
    if (!progress_config_.enabled) {
        fixture_->wait_for_programs();
        return;
    }

    // Create progress monitor (but don't start polling thread yet)
    TestProgressMonitor monitor(this, progress_config_);

    // Poll and check for completion in this thread
    log_info(
        tt::LogTest,
        "Progress monitoring started (poll interval: {}s, hung threshold: {}s)",
        progress_config_.poll_interval_seconds,
        progress_config_.hung_threshold_seconds);

    monitor.poll_until_complete();
    log_info(tt::LogTest, "Progress monitoring complete, waiting for programs to finish...");

    // Now call wait_for_programs() to ensure proper cleanup
    fixture_->wait_for_programs();
}

void TestContext::read_telemetry() {
    auto& telemetry_manager = get_telemetry_manager();
    telemetry_manager.read_telemetry();
}

void TestContext::clear_telemetry() {
    if (telemetry_manager_) {
        telemetry_manager_->clear_telemetry();
        telemetry_manager_->reset();
    }
    telemetry_entries_.clear();
}

void TestContext::clear_code_profiling_buffers() { get_code_profiler().clear_code_profiling_buffers(); }

void TestContext::read_code_profiling_results() { get_code_profiler().read_code_profiling_results(); }

void TestContext::report_code_profiling_results() { get_code_profiler().report_code_profiling_results(); }

void TestContext::process_telemetry_for_golden() {
    auto& telemetry_manager = get_telemetry_manager();
    telemetry_manager.process_telemetry_for_golden();
}

void TestContext::dump_raw_telemetry_csv(const TestConfig& config) {
    get_telemetry_manager().dump_raw_telemetry_csv(config);
}

void TestContext::collect_latency_results() {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    latency_test_manager_->collect_latency_results(test_devices_);
}

void TestContext::report_latency_results(const TestConfig& config) {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    latency_test_manager_->report_latency_results(config, test_devices_);
}

// Setup latency test workers with latency-specific configurations
void TestContext::setup_latency_test_workers(TestConfig& config) {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    latency_test_manager_->setup_latency_test_workers(config, test_devices_);
}

// Create latency kernels for a device based on its role (sender, responder, or neither)
void TestContext::create_latency_kernels_for_device(TestDevice& test_device) {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    latency_test_manager_->create_latency_kernels_for_device(test_device, test_devices_);
}

// Configures latency test mode - validates config and sets performance_test_mode flag
void TestContext::setup_latency_test_mode(const TestConfig& config) {
    TT_FATAL(
        config.performance_test_mode == PerformanceTestMode::LATENCY,
        "setup_latency_test_mode called when latency test mode is not enabled");

    this->set_performance_test_mode(PerformanceTestMode::LATENCY);
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    latency_test_manager_->setup_latency_test_mode(config);
}

LatencyTestManager::LatencyWorkerLocation TestContext::get_latency_sender_location() {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    return latency_test_manager_->get_latency_sender_location(test_devices_);
}

LatencyTestManager::LatencyWorkerLocation TestContext::get_latency_receiver_location() {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    return latency_test_manager_->get_latency_receiver_location(test_devices_);
}

void TestContext::initialize_latency_results_csv_file() {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    latency_test_manager_->initialize_latency_results_csv_file();
}

void TestContext::generate_latency_summary() {
    TT_FATAL(latency_test_manager_, "Latency manager not initialized");
    latency_test_manager_->generate_latency_summary();
    if (latency_test_manager_->has_failures()) {
        has_test_failures_ = true;
    }
}

std::vector<std::string> TestContext::get_all_failed_tests() const {
    std::vector<std::string> combined;
    combined.insert(combined.end(), all_failed_bandwidth_tests_.begin(), all_failed_bandwidth_tests_.end());
    if (latency_test_manager_) {
        const auto failed = latency_test_manager_->get_failed_tests();
        combined.insert(combined.end(), failed.begin(), failed.end());
    }
    return combined;
}

void TestContext::init(
    std::shared_ptr<TestFixture> fixture,
    const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies,
    bool use_dynamic_policies) {
    fixture_ = std::move(fixture);
    allocation_policies_ = policies;
    use_dynamic_policies_ = use_dynamic_policies;

    initialize_memory_maps();

    if (use_dynamic_policies_) {
        policy_manager_ =
            std::make_unique<tt::tt_fabric::fabric_tests::DynamicPolicyManager>(*this->fixture_, *this->fixture_);
    }

    this->allocator_ = std::make_unique<tt::tt_fabric::fabric_tests::GlobalAllocator>(
        *this->fixture_, *this->fixture_, policies, sender_memory_map_, receiver_memory_map_);

    bandwidth_profiler_ = std::make_unique<BandwidthProfiler>(*fixture_, *fixture_, *fixture_);
    bandwidth_results_manager_ = std::make_unique<BandwidthResultsManager>();
    latency_test_manager_ = std::make_unique<LatencyTestManager>(*fixture_, sender_memory_map_);
}

void TestContext::prepare_for_test(const TestConfig& config) {
    if (!use_dynamic_policies_) {
        return;
    }

    auto new_policy = policy_manager_->get_new_policy_for_test(config);

    if (new_policy.has_value()) {
        update_memory_maps(new_policy.value());

        allocator_.reset();
        allocator_ = std::make_unique<tt::tt_fabric::fabric_tests::GlobalAllocator>(
            *fixture_, *fixture_, new_policy.value(), sender_memory_map_, receiver_memory_map_);
    }

    const auto& policy_to_validate = new_policy.has_value() ? new_policy.value() : policy_manager_->get_cached_policy();
    validate_packet_sizes_for_policy(config, policy_to_validate.default_payload_chunk_size);
}

void TestContext::setup_devices() {
    const auto& available_coords = this->fixture_->get_host_local_device_coordinates();
    for (const auto& coord : available_coords) {
        test_devices_.emplace(
            coord, TestDevice(coord, this->fixture_, this->fixture_, &sender_memory_map_, &receiver_memory_map_));
    }
}

void TestContext::reset_devices() {
    test_devices_.clear();
    device_global_sync_cores_.clear();
    device_local_sync_cores_.clear();
    this->allocator_->reset();

    code_profiler_.reset();
    telemetry_manager_.reset();
    eth_readback_.reset();

    reset_local_variables();
}

void TestContext::reset_local_variables() {
    performance_test_mode_ = PerformanceTestMode::NONE;
    skip_packet_validation_ = false;
    global_sync_ = false;
}

void TestContext::profile_results(const TestConfig& config) {
    TT_FATAL(bandwidth_profiler_ && bandwidth_results_manager_, "Bandwidth managers not initialized");

    bandwidth_profiler_->profile_results(config, test_devices_, sender_memory_map_);

    if (telemetry_enabled_) {
        auto& telemetry_manager = get_telemetry_manager();
        bandwidth_profiler_->set_telemetry_bandwidth(
            telemetry_manager.get_measured_bw_min(),
            telemetry_manager.get_measured_bw_avg(),
            telemetry_manager.get_measured_bw_max());
    }

    const auto& latest_results = bandwidth_profiler_->get_latest_results();
    for (const auto& result : latest_results) {
        bandwidth_results_manager_->add_result(config, result);
    }
    bandwidth_results_manager_->add_summary(config, bandwidth_profiler_->get_latest_summary());

    if (!latest_results.empty()) {
        bandwidth_results_manager_->append_to_csv(config, latest_results.back());
    }
}

void TestContext::generate_bandwidth_summary() {
    TT_FATAL(bandwidth_results_manager_, "Bandwidth results manager not initialized");
    bandwidth_results_manager_->load_golden_csv();
    bandwidth_results_manager_->generate_summary();
    bandwidth_results_manager_->validate_against_golden();
    if (bandwidth_results_manager_->has_failures()) {
        const auto failed = bandwidth_results_manager_->get_failed_tests();
        all_failed_bandwidth_tests_.insert(all_failed_bandwidth_tests_.end(), failed.begin(), failed.end());
        has_test_failures_ = true;
    }
}

void TestContext::initialize_bandwidth_results_csv_file() {
    TT_FATAL(bandwidth_results_manager_, "Bandwidth results manager not initialized");
    bandwidth_results_manager_->initialize_bandwidth_csv_file(this->telemetry_enabled_);
}

void TestContext::compile_programs() {
    fixture_->setup_workload();
    for (auto& [coord, test_device] : test_devices_) {
        bool enable_kernel_benchmark =
            skip_packet_validation_ || (performance_test_mode_ == PerformanceTestMode::BANDWIDTH);
        test_device.set_benchmark_mode(enable_kernel_benchmark);
        test_device.set_global_sync(global_sync_);
        test_device.set_progress_monitoring_enabled(progress_config_.enabled);

        auto device_id = test_device.get_node_id();
        test_device.set_sync_core(device_global_sync_cores_[device_id]);

        // Create kernels (latency or normal)
        if (performance_test_mode_ == PerformanceTestMode::LATENCY) {
            create_latency_kernels_for_device(test_device);
        } else {
            // Normal mode: create standard kernels for all devices
            test_device.create_kernels();
        }
    }

    // Enqueue all programs
    for (auto& [coord, test_device] : test_devices_) {
        auto& program_handle = test_device.get_program_handle();
        if (program_handle.impl().num_kernels()) {
            fixture_->enqueue_program(coord, std::move(program_handle));
        }
    }
}

void TestContext::enable_progress_monitoring(const ProgressMonitorConfig& config) {
    progress_config_ = config;
    progress_config_.enabled = true;
}

void TestContext::process_telemetry_data(TestConfig& built_test_config) {
    if (this->get_telemetry_enabled() && performance_test_mode_ != PerformanceTestMode::LATENCY) {
        this->read_telemetry();
        this->process_telemetry_for_golden();
        this->dump_raw_telemetry_csv(built_test_config);
    }
}

void TestContext::validate_results() {
    if (performance_test_mode_ != PerformanceTestMode::NONE) {
        log_info(
            tt::LogTest,
            "Skipping validation (performance_test_mode: {})",
            enchantum::to_string(performance_test_mode_));
        return;
    }

    constexpr uint32_t MAX_CONCURRENT_DEVICES = 16;

    std::vector<std::pair<MeshCoordinate, const TestDevice*>> devices;
    devices.reserve(test_devices_.size());
    for (const auto& [coord, device] : test_devices_) {
        devices.push_back({coord, &device});
    }

    for (size_t i = 0; i < devices.size(); i += MAX_CONCURRENT_DEVICES) {
        size_t group_end = std::min(i + MAX_CONCURRENT_DEVICES, devices.size());

        std::vector<TestDevice::ValidationReadOps> read_ops;
        read_ops.reserve(group_end - i);
        for (size_t j = i; j < group_end; ++j) {
            read_ops.push_back(devices[j].second->initiate_results_readback());
        }

        fixture_->barrier_reads();

        for (size_t j = i; j < group_end; ++j) {
            devices[j].second->validate_results_after_readback(read_ops[j - i]);
        }
    }
}

void TestContext::set_code_profiling_enabled(bool enabled) {
    code_profiling_enabled_ = enabled;
    if (code_profiler_) {
        code_profiler_->set_enabled(enabled);
    }
}

void TestContext::setup_ci_artifacts() {
    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path ci_artifacts_path = tt_metal_home / std::string(CI_ARTIFACTS_DIR);
    if (!std::filesystem::exists(ci_artifacts_path)) {
        try {
            std::filesystem::create_directories(ci_artifacts_path);
        } catch (const std::filesystem::filesystem_error& e) {
            log_error(
                tt::LogTest, "Failed to create CI artifacts directory, skipping CI artifacts creation: {}", e.what());
            return;
        }
    }

    if (bandwidth_results_manager_) {
        bandwidth_results_manager_->setup_ci_artifacts();
    }
    if (latency_test_manager_) {
        latency_test_manager_->setup_ci_artifacts();
    }
    log_trace(tt::LogTest, "Copied CSV files to CI artifacts directory: {}", ci_artifacts_path.string());
}

void TestContext::initialize_memory_maps() { update_memory_maps(allocation_policies_); }

void TestContext::update_memory_maps(const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies) {
    auto l1_unreserved_base = fixture_->get_l1_unreserved_base();
    auto l1_unreserved_size = fixture_->get_l1_unreserved_size();
    auto l1_alignment = fixture_->get_l1_alignment();

    sender_memory_map_ =
        tt::tt_fabric::fabric_tests::SenderMemoryMap(l1_unreserved_base, l1_unreserved_size, l1_alignment);

    receiver_memory_map_ = tt::tt_fabric::fabric_tests::ReceiverMemoryMap(
        l1_unreserved_base,
        l1_unreserved_size,
        l1_alignment,
        policies.default_payload_chunk_size,
        policies.receiver_config.max_configs_per_core);

    if (!sender_memory_map_.is_valid() || !receiver_memory_map_.is_valid()) {
        TT_THROW("Invalid memory map configuration");
    }
}

void TestContext::validate_packet_sizes_for_policy(const TestConfig& config, uint32_t payload_chunk_size) {
    uint32_t max_packet_size = 0;
    for (const auto& sender : config.senders) {
        for (const auto& pattern : sender.patterns) {
            if (pattern.size.has_value()) {
                max_packet_size = std::max(max_packet_size, pattern.size.value());
            }
        }
    }

    if (max_packet_size > payload_chunk_size) {
        TT_FATAL(
            false,
            "Test '{}' configuration is INVALID!\n"
            "  Max packet size: {} bytes\n"
            "  Computed buffer size: {} bytes\n"
            "  The packet size exceeds buffer capacity.\n"
            "  Fix: Reduce packet size to <= {} bytes or adjust parametrization.",
            config.parametrized_name,
            max_packet_size,
            payload_chunk_size,
            payload_chunk_size);
    }
}

void TestContext::process_traffic_config(TestConfig& config) {
    // Latency test mode: manually populate senders_ and receivers_ maps
    // with latency-specific kernels and configurations
    if (config.performance_test_mode == PerformanceTestMode::LATENCY) {
        setup_latency_test_workers(config);
        return;
    }

    log_debug(tt::LogTest, "Allocating resources for test config");
    this->allocator_->allocate_resources(config);
    log_debug(tt::LogTest, "Resource allocation complete");

    // Use unified connection manager when BOTH sync AND flow control are enabled
    // - This ensures sync and credit returns use the same link tracking for correct mux detection
    // - When only sync is enabled (no flow control), separate managers avoid mux overhead
    if (config.enable_flow_control && config.global_sync) {
        for (auto& [_, device] : test_devices_) {
            device.set_use_unified_connection_manager(true);
        }
    }

    // Transfer pristine cores from allocator to each device
    for (auto& [coord, device] : test_devices_) {
        auto node_id = device.get_node_id();
        auto pristine_cores = allocator_->get_pristine_cores_for_device(node_id);
        device.set_pristine_cores(std::move(pristine_cores));
    }

    if (config.global_sync) {
        // set it only after the test_config is built since it needs set the sync value during expand the high-level
        // patterns.
        this->set_global_sync(config.global_sync);
        this->set_performance_test_mode(config.performance_test_mode);
        this->set_skip_packet_validation(config.skip_packet_validation);

        // Convert sync configs to traffic configs recognized by the test devuces
        this->add_sync_traffic_to_devices(config);

        log_debug(tt::LogTest, "Enabled sync and created sync configs");
        log_debug(tt::LogTest, "Set performance test mode to: {}", performance_test_mode_);

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
            log_debug(
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
                .mcast_start_hops = pattern.mcast_start_hops,
                .enable_flow_control = config.enable_flow_control,
                .seed = config.seed,
                .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
                .mesh_shape = this->fixture_->get_mesh_shape(),
                .topology = this->fixture_->get_topology()};

            TestTrafficConfig traffic_config = {
                .parameters = traffic_parameters,
                .src_node_id = sender.device,
                .src_logical_core = sender.core,
                .dst_logical_core = dest.core,
                .target_address = dest.target_address,
                .atomic_inc_address = dest.atomic_inc_address,
                .link_id = sender.link_id,
                .sender_credit_info = pattern.sender_credit_info,
                .credit_return_batch_size = pattern.credit_return_batch_size,
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

void TestContext::add_traffic_config(const TestTrafficConfig& traffic_config) {
    const auto& src_node_id = traffic_config.src_node_id;

    CoreCoord src_logical_core = traffic_config.src_logical_core.value();
    CoreCoord dst_logical_core = traffic_config.dst_logical_core.value();
    uint32_t target_address = traffic_config.target_address.value_or(0);
    uint32_t atomic_inc_address = traffic_config.atomic_inc_address.value_or(0);

    std::vector<FabricNodeId> dst_node_ids;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops = std::nullopt;

    if (traffic_config.hops.has_value()) {
        hops = traffic_config.hops;
        dst_node_ids = this->fixture_->get_dst_node_ids_from_hops(
            traffic_config.src_node_id, hops.value(), traffic_config.parameters.chip_send_type);
    } else {
        dst_node_ids = traffic_config.dst_node_ids.value();

        if (src_node_id.mesh_id == dst_node_ids[0].mesh_id) {
            hops = this->fixture_->get_hops_to_chip(src_node_id, dst_node_ids[0]);
        }
    }

    std::optional<FabricNodeId> mcast_start_node_id = std::nullopt;
    if (fixture_->is_2D_routing_enabled() && traffic_config.parameters.chip_send_type == ChipSendType::CHIP_MULTICAST) {
        mcast_start_node_id = fixture_->get_mcast_start_node_id(src_node_id, hops.value());
    }

    uint32_t dst_noc_encoding = this->fixture_->get_worker_noc_encoding(dst_logical_core);
    uint32_t sender_id = fixture_->get_worker_id(traffic_config.src_node_id, src_logical_core);

    uint32_t payload_buffer_size = receiver_memory_map_.get_payload_chunk_size();

    TestTrafficSenderConfig sender_config = {
        .parameters = traffic_config.parameters,
        .src_node_id = traffic_config.src_node_id,
        .dst_node_ids = dst_node_ids,
        .hops = hops,
        .mcast_start_node_id = mcast_start_node_id,
        .dst_logical_core = dst_logical_core,
        .target_address = target_address,
        .atomic_inc_address = atomic_inc_address,
        .dst_noc_encoding = dst_noc_encoding,
        .payload_buffer_size = payload_buffer_size,
        .link_id = traffic_config.link_id};

    TestTrafficReceiverConfig receiver_config = {
        .parameters = traffic_config.parameters,
        .sender_id = sender_id,
        .target_address = target_address,
        .atomic_inc_address = atomic_inc_address,
        .payload_buffer_size = payload_buffer_size,
        .link_id = traffic_config.link_id};

    if (traffic_config.parameters.enable_flow_control) {
        TT_FATAL(
            traffic_config.sender_credit_info.has_value(),
            "Sender credit info not allocated for sender {} with flow control enabled",
            traffic_config.src_node_id);

        sender_config.sender_credit_info = traffic_config.sender_credit_info.value();

        TT_FATAL(
            traffic_config.credit_return_batch_size.has_value(),
            "Credit batch size not calculated for sender {} with flow control enabled",
            traffic_config.src_node_id);
        uint32_t credit_return_batch_size = traffic_config.credit_return_batch_size.value();

        receiver_config.receiver_credit_info = ReceiverCreditInfo{
            .receiver_node_id = FabricNodeId(MeshId{0}, 0),
            .sender_node_id = traffic_config.src_node_id,
            .sender_logical_core = src_logical_core,
            .sender_noc_encoding = fixture_->get_worker_noc_encoding(src_logical_core),
            .credit_return_address = 0,
            .credit_return_batch_size = credit_return_batch_size,
            .hops = std::nullopt};
    } else {
        sender_config.sender_credit_info = std::nullopt;
        receiver_config.receiver_credit_info = std::nullopt;
    }

    uint32_t receiver_idx = 0;
    for (const auto& dst_node_id : dst_node_ids) {
        if (fixture_->is_local_fabric_node_id(dst_node_id)) {
            const auto& dst_coord = this->fixture_->get_device_coord(dst_node_id);
            TestTrafficReceiverConfig per_receiver_config = receiver_config;

            if (traffic_config.parameters.enable_flow_control) {
                TT_FATAL(
                    per_receiver_config.receiver_credit_info.has_value(),
                    "Receiver credit info not allocated for receiver with flow control enabled");

                uint32_t credit_chunk_base = sender_config.sender_credit_info->credit_reception_address_base;
                uint32_t credit_return_address =
                    SenderMemoryMap::get_receiver_credit_address(credit_chunk_base, receiver_idx);

                per_receiver_config.receiver_credit_info->receiver_node_id = dst_node_id;
                per_receiver_config.receiver_credit_info->credit_return_address = credit_return_address;

                std::optional<std::unordered_map<RoutingDirection, uint32_t>> reverse_hops = std::nullopt;
                if (src_node_id.mesh_id == dst_node_id.mesh_id) {
                    reverse_hops = fixture_->get_hops_to_chip(dst_node_id, src_node_id);
                }
                per_receiver_config.receiver_credit_info->hops = reverse_hops;
            }

            this->test_devices_.at(dst_coord).add_receiver_traffic_config(dst_logical_core, per_receiver_config);
        }

        receiver_idx++;
    }

    if (fixture_->is_local_fabric_node_id(src_node_id)) {
        const auto& src_coord = this->fixture_->get_device_coord(src_node_id);
        auto& src_test_device = this->test_devices_.at(src_coord);
        src_test_device.add_sender_traffic_config(src_logical_core, std::move(sender_config));
    }
}
