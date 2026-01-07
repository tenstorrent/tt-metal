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
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_fabric::fabric_tests {

TestProgressMonitor::TestProgressMonitor(::TestContext* ctx, const ProgressMonitorConfig& config) :
    ctx_(ctx), config_(config), hung_threshold_(config.hung_threshold_seconds) {}

TestProgressMonitor::~TestProgressMonitor() = default;

void TestProgressMonitor::poll_until_complete() {
    start_time_ = std::chrono::steady_clock::now();
    last_poll_time_ = start_time_;

    bool programs_complete = false;

    while (!programs_complete) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_poll_time_);

        auto progress = poll_devices();
        check_for_hung_devices(progress);
        display_progress(progress, elapsed);

        programs_complete = true;
        for (const auto& [device_id, prog] : progress) {
            if (prog.current_packets < prog.total_packets) {
                programs_complete = false;
                break;
            }
        }

        last_poll_time_ = now;

        if (!programs_complete) {
            std::this_thread::sleep_for(std::chrono::seconds(config_.poll_interval_seconds));
        }
    }

    // Always print newline after final progress update
    std::cout << std::endl;
}

std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress> TestProgressMonitor::poll_devices() {
    std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress> device_progress;

    auto* device_info = ctx_->get_device_info_provider();

    for (const auto& [coord, test_device] : ctx_->get_test_devices()) {
        tt::tt_fabric::FabricNodeId device_id = test_device.get_node_id();

        if (!device_info->is_local_fabric_node_id(device_id)) {
            continue;
        }

        auto progress = poll_device_senders(coord, test_device);
        device_progress[progress.device_id] = progress;
    }

    return device_progress;
}

DeviceProgress TestProgressMonitor::poll_device_senders(const MeshCoordinate& coord, const TestDevice& test_device) {
    DeviceProgress progress;
    progress.device_id = test_device.get_node_id();

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(progress.device_id);

    auto* device_info = ctx_->get_device_info_provider();
    uint32_t result_addr = ctx_->get_sender_memory_map().get_result_buffer_address();

    for (const auto& [core, sender] : test_device.get_senders()) {
        CoreCoord logical_core = sender.get_core();
        CoreCoord virtual_core = device_info->get_virtual_core_from_logical_core(logical_core);

        uint32_t result_size = 4 * sizeof(uint32_t);
        auto result_data = cluster.read_core<uint32_t>(physical_chip_id, virtual_core, result_addr, result_size);

        if (result_data.size() < 4) {
            continue;
        }

        uint32_t packets_low = result_data[TT_FABRIC_WORD_CNT_INDEX];
        uint32_t packets_high = result_data[TT_FABRIC_WORD_CNT_INDEX + 1];
        uint64_t packets_sent = (static_cast<uint64_t>(packets_high) << 32) | packets_low;

        progress.current_packets += packets_sent;
        progress.total_packets += sender.get_total_packets();
        progress.num_senders++;
    }

    return progress;
}

bool TestProgressMonitor::is_device_hung(tt::tt_fabric::FabricNodeId device_id, uint64_t current_packets) {
    auto& state = device_states_[device_id];
    auto now = std::chrono::steady_clock::now();

    if (state.last_progress_time.time_since_epoch().count() == 0) {
        state.last_packet_count = current_packets;
        state.last_progress_time = now;
        return false;
    }

    if (current_packets > state.last_packet_count) {
        state.last_packet_count = current_packets;
        state.last_progress_time = now;
        state.warned = false;
        return false;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - state.last_progress_time);
    return elapsed >= hung_threshold_;
}

void TestProgressMonitor::check_for_hung_devices(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress) {
    for (const auto& [device_id, prog] : progress) {
        // Skip devices that have already completed
        if (prog.current_packets >= prog.total_packets) {
            continue;
        }

        if (is_device_hung(device_id, prog.current_packets)) {
            auto& state = device_states_[device_id];

            // Only warn once per device
            if (!state.warned) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - state.last_progress_time);

                log_warning(
                    tt::LogTest,
                    "⚠️  Device {} may be HUNG: no progress for {} seconds (packets: {}/{})",
                    device_id,
                    elapsed.count(),
                    prog.current_packets,
                    prog.total_packets);

                state.warned = true;
            }
        }
    }
}

void TestProgressMonitor::display_progress(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
    std::chrono::duration<double> elapsed) {
    uint64_t total_current = 0, total_target = 0;
    for (const auto& [_, prog] : progress) {
        total_current += prog.current_packets;
        total_target += prog.total_packets;
    }

    double overall_pct = total_target > 0 ? 100.0 * total_current / total_target : 0.0;

    std::stringstream ss;
    ss << "\rProgress: " << std::fixed << std::setprecision(1) << overall_pct << "% "
       << "(" << format_count(total_current) << "/" << format_count(total_target) << ")";

    // Throughput and ETA (based on delta since last poll)
    // Skip first poll (last_total_packets_ == 0) and require at least 0.5s elapsed
    if (elapsed.count() >= 0.5 && last_total_packets_ > 0 && total_current > last_total_packets_) {
        double throughput = (total_current - last_total_packets_) / elapsed.count();
        ss << " | " << format_throughput(throughput);

        auto eta = estimate_eta(total_current, total_target, throughput);
        if (eta.has_value()) {
            ss << " | ETA: " << format_duration(*eta);
        }
    }

    // Always update last_total_packets for next iteration
    if (total_current > 0) {
        last_total_packets_ = total_current;
    }

    // Pad with spaces to clear any leftover text from previous longer updates
    ss << "          ";

    std::cout << ss.str() << std::flush;
}

std::string TestProgressMonitor::format_count(uint64_t count) const {
    if (count >= 1000000) {
        return std::to_string(count / 1000000) + "M";
    }
    if (count >= 1000) {
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
        uint32_t minutes = static_cast<uint32_t>((seconds - (hours * 3600)) / 60);
        return std::to_string(hours) + "h" + std::to_string(minutes) + "m";
    }
    if (seconds >= 60) {
        uint32_t minutes = static_cast<uint32_t>(seconds / 60);
        uint32_t secs = static_cast<uint32_t>(seconds - (minutes * 60));
        return std::to_string(minutes) + "m" + std::to_string(secs) + "s";
    }
    return std::to_string(static_cast<uint32_t>(seconds)) + "s";
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
