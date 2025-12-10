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
