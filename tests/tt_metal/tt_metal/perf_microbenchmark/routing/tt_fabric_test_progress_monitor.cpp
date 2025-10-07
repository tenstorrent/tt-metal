// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_progress_monitor.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

#include "tt_fabric_test_context.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>

namespace tt::tt_fabric::fabric_tests {

TestProgressMonitor::TestProgressMonitor(::TestContext* ctx, const ProgressMonitorConfig& config) :
    ctx_(ctx), config_(config), hung_threshold_(config.hung_threshold_seconds) {}

TestProgressMonitor::~TestProgressMonitor() { stop(); }

void TestProgressMonitor::start() {
    should_stop_ = false;
    polling_thread_ = std::thread(&TestProgressMonitor::polling_loop, this);
    log_info(
        tt::LogTest,
        "Progress monitoring started (poll interval: {}s, hung threshold: {}s)",
        config_.poll_interval_seconds,
        config_.hung_threshold_seconds);
}

void TestProgressMonitor::stop() {
    if (polling_thread_.joinable()) {
        should_stop_ = true;
        polling_thread_.join();

        // Print newline to move cursor to next line after progress bar
        if (!first_display_) {  // Only if we actually displayed progress
            std::cout << std::endl;
        }

        log_info(tt::LogTest, "Progress monitoring stopped");
    }
}

void TestProgressMonitor::poll_until_complete() {
    start_time_ = std::chrono::steady_clock::now();
    last_poll_time_ = start_time_;

    bool programs_complete = false;
    int poll_count = 0;

    while (!programs_complete) {
        poll_count++;
        auto poll_start = std::chrono::steady_clock::now();

        log_info(tt::LogTest, "Starting poll #{}", poll_count);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_poll_time_);

        // Poll all devices
        auto progress = poll_devices();

        auto poll_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - poll_start);
        log_info(tt::LogTest, "Poll #{} completed in {}ms", poll_count, poll_duration.count());

        check_for_hung_devices(progress);
        display_progress(progress, elapsed);

        // Check if all devices are complete
        programs_complete = true;
        for (const auto& [device_id, prog] : progress) {
            if (prog.current_packets < prog.total_packets) {
                programs_complete = false;
                log_debug(
                    tt::LogTest, "Device {} not complete: {}/{}", device_id, prog.current_packets, prog.total_packets);
                break;
            }
        }

        if (programs_complete) {
            log_info(tt::LogTest, "All devices complete after {} polls", poll_count);
        }

        last_poll_time_ = now;

        // If not complete, sleep before next poll
        if (!programs_complete) {
            log_info(tt::LogTest, "Sleeping {}s before next poll", config_.poll_interval_seconds);
            std::this_thread::sleep_for(std::chrono::seconds(config_.poll_interval_seconds));
        }
    }

    // Print final newline after progress bar
    if (!first_display_) {
        std::cout << std::endl;
    }
}

void TestProgressMonitor::polling_loop() {
    start_time_ = std::chrono::steady_clock::now();
    last_poll_time_ = start_time_;

    // Poll immediately on start (don't wait for first interval)
    {
        log_info(tt::LogTest, "Initial poll starting...");
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(0);  // First poll, no elapsed time

        auto progress = poll_devices();
        log_info(tt::LogTest, "Initial poll found {} devices", progress.size());
        check_for_hung_devices(progress);
        display_progress(progress, elapsed);

        last_poll_time_ = now;
    }

    while (!should_stop_) {
        log_info(tt::LogTest, "Polling thread sleeping for {}s...", config_.poll_interval_seconds);
        std::this_thread::sleep_for(std::chrono::seconds(config_.poll_interval_seconds));

        // Check again after sleep - might have been stopped during sleep
        if (should_stop_) {
            log_info(tt::LogTest, "Polling thread woke up, should_stop=true, exiting");
            break;
        }

        log_info(tt::LogTest, "Polling thread woke up, polling devices...");
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_poll_time_);

        auto progress = poll_devices();
        check_for_hung_devices(progress);
        display_progress(progress, elapsed);

        last_poll_time_ = now;
    }
}

std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress> TestProgressMonitor::poll_devices() {
    auto poll_start = std::chrono::steady_clock::now();
    std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress> device_progress;

    auto* device_info = ctx_->get_device_info_provider();

    size_t total_devices = ctx_->get_test_devices().size();
    size_t local_devices = 0;

    for (const auto& [coord, test_device] : ctx_->get_test_devices()) {
        tt::tt_fabric::FabricNodeId device_id = test_device.get_node_id();

        // Skip non-local devices in multi-host setups
        if (!device_info->is_local_fabric_node_id(device_id)) {
            continue;
        }

        local_devices++;
        auto device_start = std::chrono::steady_clock::now();
        auto progress = poll_device_senders(coord, test_device);
        auto device_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - device_start);

        // Warn if device polling is slow (>100ms suggests blocking behavior)
        if (device_elapsed.count() > 100) {
            log_warning(
                tt::LogTest,
                "  Device {} polling took {}ms (SLOW - may be blocking)",
                device_id,
                device_elapsed.count());
        } else {
            log_debug(tt::LogTest, "  Device {} polling took {}ms", device_id, device_elapsed.count());
        }

        device_progress[progress.device_id] = progress;
    }

    // Debug: log if no local devices found
    if (local_devices == 0 && total_devices > 0) {
        log_warning(tt::LogTest, "Progress monitor: No local devices found (total devices: {})", total_devices);
    }

    auto poll_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - poll_start);
    log_info(tt::LogTest, "Total poll_devices() took {}ms for {} devices", poll_elapsed.count(), local_devices);

    return device_progress;
}

DeviceProgress TestProgressMonitor::poll_device_senders(const MeshCoordinate& coord, const TestDevice& test_device) {
    DeviceProgress progress;
    progress.device_id = test_device.get_node_id();

    // Get cluster and control plane for reading L1 memory
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Convert FabricNodeId to physical chip ID for cluster API
    chip_id_t physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(progress.device_id);

    auto* device_info = ctx_->get_device_info_provider();

    // Get result buffer address (uniform across all senders)
    uint32_t result_addr = ctx_->get_sender_memory_map().get_result_buffer_address();

    // Poll all senders on this device
    for (const auto& [core, sender] : test_device.get_senders()) {
        CoreCoord logical_core = sender.get_core();

        // Convert logical to virtual coordinates
        CoreCoord virtual_core = device_info->get_virtual_core_from_logical_core(logical_core);

        // Read result buffer LIVE using Cluster::read_core API
        uint32_t result_size = 4 * sizeof(uint32_t);
        auto result_data = cluster.read_core<uint32_t>(physical_chip_id, virtual_core, result_addr, result_size);

        // Defensive bounds check
        if (result_data.size() < 4) {
            log_error(
                tt::LogTest,
                "Device {} core ({},{}) returned incomplete result buffer: expected 4 words, got {}. Skipping this "
                "core.",
                progress.device_id,
                logical_core.x,
                logical_core.y,
                result_data.size());
            continue;
        }

        // Extract packet count from result buffer
        uint32_t packets_low = result_data[TT_FABRIC_WORD_CNT_INDEX];
        uint32_t packets_high = result_data[TT_FABRIC_WORD_CNT_INDEX + 1];
        uint64_t packets_sent = (static_cast<uint64_t>(packets_high) << 32) | packets_low;

        uint64_t total_packets = sender.get_total_packets();

        progress.current_packets += packets_sent;
        progress.total_packets += total_packets;
        progress.num_senders++;
    }

    return progress;
}

bool TestProgressMonitor::is_device_hung(tt::tt_fabric::FabricNodeId device_id, uint64_t current_packets) {
    auto& state = device_states_[device_id];
    auto now = std::chrono::steady_clock::now();

    // First time seeing this device - use timepoint to detect initialization
    if (state.last_progress_time.time_since_epoch().count() == 0) {
        state.last_packet_count = current_packets;
        state.last_progress_time = now;
        return false;
    }

    // Check if progress was made
    if (current_packets > state.last_packet_count) {
        state.last_packet_count = current_packets;
        state.last_progress_time = now;
        state.warned = false;
        return false;
    }

    // No progress - check if hung
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - state.last_progress_time);

    return elapsed >= hung_threshold_;
}

void TestProgressMonitor::check_for_hung_devices(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress) {
    for (const auto& [device_id, prog] : progress) {
        if (is_device_hung(device_id, prog.current_packets)) {
            auto& state = device_states_[device_id];

            // Only warn once per device
            if (!state.warned) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - state.last_progress_time);

                log_warning(
                    tt::LogTest,
                    "⚠️  Device {} may be HUNG: no progress for {} seconds (packets: {})",
                    device_id,
                    elapsed.count(),
                    prog.current_packets);

                state.warned = true;
            }
        }
    }
}

void TestProgressMonitor::display_progress(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
    std::chrono::duration<double> elapsed) {
    if (config_.verbose) {
        display_verbose_progress(progress, elapsed);
    } else {
        display_summary_progress(progress, elapsed);
    }
}

void TestProgressMonitor::display_summary_progress(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
    std::chrono::duration<double> elapsed) {
    // Aggregate across all devices
    uint64_t total_current = 0, total_target = 0;
    for (const auto& [_, prog] : progress) {
        total_current += prog.current_packets;
        total_target += prog.total_packets;
    }

    double overall_pct = total_target > 0 ? 100.0 * total_current / total_target : 0.0;

    std::stringstream ss;
    ss << "\rProgress: " << std::fixed << std::setprecision(1) << overall_pct << "% "
       << "(" << format_count(total_current) << "/" << format_count(total_target) << ") | ";

    // Throughput and ETA (based on delta since last poll)
    if (elapsed.count() > 0 && total_current > last_total_packets_) {
        double throughput = (total_current - last_total_packets_) / elapsed.count();
        ss << format_throughput(throughput);

        auto eta = estimate_eta(total_current, total_target, throughput);
        if (eta.has_value()) {
            ss << " | ETA: " << format_duration(*eta);
        }

        last_total_packets_ = total_current;
    }

    // Progress bar
    ss << " " << format_progress_bar(overall_pct, 20);

    std::cout << ss.str() << std::flush;

    // Debug: Also log to verify polling is happening
    log_info(tt::LogTest, "Progress poll: {:.1f}% ({}/{})", overall_pct, total_current, total_target);
}

void TestProgressMonitor::display_verbose_progress(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
    std::chrono::duration<double> elapsed) {
    // Move cursor up to overwrite previous display (after first update)
    if (!first_display_) {
        std::cout << "\033[" << (progress.size() + 2) << "A";  // Move up N+2 lines (devices + summary line)
    }
    first_display_ = false;

    // Display each device with progress bar
    for (const auto& [device_id, prog] : progress) {
        double pct = prog.total_packets > 0 ? 100.0 * prog.current_packets / prog.total_packets : 0.0;

        std::cout << "Device " << std::setw(2) << device_id << ": " << format_progress_bar(pct, 40) << " " << std::fixed
                  << std::setprecision(1) << pct << "% "
                  << "(" << format_count(prog.current_packets) << "/" << format_count(prog.total_packets) << ")";

        // Check if this device is hung
        auto it = device_states_.find(device_id);
        if (it != device_states_.end() && it->second.warned) {
            std::cout << " ⚠️  HUNG";
        }

        std::cout << "\n";
    }

    // Overall summary line
    uint64_t total_current = 0, total_target = 0;
    for (const auto& [_, prog] : progress) {
        total_current += prog.current_packets;
        total_target += prog.total_packets;
    }

    double overall_pct = total_target > 0 ? 100.0 * total_current / total_target : 0.0;

    std::cout << "Overall:   " << format_progress_bar(overall_pct, 40) << " " << std::fixed << std::setprecision(1)
              << overall_pct << "% | ";

    if (elapsed.count() > 0 && total_current > last_total_packets_) {
        double throughput = (total_current - last_total_packets_) / elapsed.count();
        std::cout << format_throughput(throughput) << " | ";

        auto eta = estimate_eta(total_current, total_target, throughput);
        if (eta.has_value()) {
            std::cout << "ETA: " << format_duration(*eta);
        }

        last_total_packets_ = total_current;
    }

    std::cout << "   \n" << std::flush;
}

std::string TestProgressMonitor::format_count(uint64_t count) const {
    if (count >= 1000000) {
        return std::to_string(count / 1000000) + "M";
    } else if (count >= 1000) {
        return std::to_string(count / 1000) + "K";
    }
    return std::to_string(count);
}

std::string TestProgressMonitor::format_throughput(double packets_per_second) const {
    std::stringstream ss;
    if (packets_per_second >= 1000000) {
        ss << std::fixed << std::setprecision(1) << (packets_per_second / 1000000.0) << "M/s";
    } else if (packets_per_second >= 1000) {
        ss << std::fixed << std::setprecision(1) << (packets_per_second / 1000.0) << "K/s";
    } else {
        ss << std::fixed << std::setprecision(0) << packets_per_second << "/s";
    }
    return ss.str();
}

std::string TestProgressMonitor::format_duration(double seconds) const {
    if (seconds >= 3600) {
        uint32_t hours = static_cast<uint32_t>(seconds / 3600);
        uint32_t minutes = static_cast<uint32_t>((seconds - hours * 3600) / 60);
        return std::to_string(hours) + "h" + std::to_string(minutes) + "m";
    } else if (seconds >= 60) {
        uint32_t minutes = static_cast<uint32_t>(seconds / 60);
        uint32_t secs = static_cast<uint32_t>(seconds - minutes * 60);
        return std::to_string(minutes) + "m" + std::to_string(secs) + "s";
    } else {
        return std::to_string(static_cast<uint32_t>(seconds)) + "s";
    }
}

std::string TestProgressMonitor::format_progress_bar(double percentage, uint32_t width) const {
    uint32_t filled = static_cast<uint32_t>((percentage / 100.0) * width);
    uint32_t empty = width - filled;

    std::string bar = "[";
    for (uint32_t i = 0; i < filled; ++i) {
        bar += "=";
    }
    if (filled < width && percentage < 100.0) {
        bar += ">";
        empty--;
    }
    for (uint32_t i = 0; i < empty; ++i) {
        bar += " ";
    }
    bar += "]";

    return bar;
}

std::optional<double> TestProgressMonitor::estimate_eta(
    uint64_t current_total, uint64_t target_total, double throughput) const {
    if (throughput <= 0 || current_total >= target_total) {
        return std::nullopt;
    }

    uint64_t remaining = target_total - current_total;
    return remaining / throughput;
}

}  // namespace tt::tt_fabric::fabric_tests
