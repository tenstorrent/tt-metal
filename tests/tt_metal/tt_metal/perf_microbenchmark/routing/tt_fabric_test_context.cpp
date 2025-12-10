// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_context.hpp"

#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

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
    auto& tm = get_telemetry_manager();
    tm.read_telemetry();
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
    auto& tm = get_telemetry_manager();
    tm.process_telemetry_for_golden();
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

void TestContext::profile_results(const TestConfig& config) {
    TT_FATAL(bandwidth_profiler_ && bandwidth_results_manager_, "Bandwidth managers not initialized");

    bandwidth_profiler_->profile_results(config, test_devices_, sender_memory_map_);

    if (telemetry_enabled_) {
        auto& telemetry = get_telemetry_manager();
        bandwidth_profiler_->set_telemetry_bandwidth(
            telemetry.get_measured_bw_min(), telemetry.get_measured_bw_avg(), telemetry.get_measured_bw_max());
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
