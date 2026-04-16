// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "tt_fabric_test_progress_monitor.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "tt_fabric_test_constants.hpp"
#include "tt_fabric_test_context.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

namespace tt::tt_fabric::fabric_tests {

static std::filesystem::path resolve_report_path(const std::string& filename) {
    std::filesystem::path root =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path dir = root / std::string(OUTPUT_DIR);
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
    return dir / filename;
}

TestProgressMonitor::TestProgressMonitor(::TestContext* ctx, const ProgressMonitorConfig& config) :
    ctx_(ctx),
    config_(config),
    hung_threshold_(config.hung_threshold_seconds),
    summary_report_path_(resolve_report_path(config.summary_file)),
    detail_report_path_(resolve_report_path(config.detail_file)) {
    auto* device_info = ctx_->get_device_info_provider();

    for (const auto& [coord, test_device] : ctx_->get_test_devices()) {
        FabricNodeId node_id = test_device.get_node_id();

        if (!device_info->is_local_fabric_node_id(node_id)) {
            continue;
        }

        if (!test_device.get_senders().empty()) {
            total_active_devices_++;
        }

        if (!config_.granular) {
            continue;
        }

        for (const auto& [core, sender] : test_device.get_senders()) {
            for (uint16_t ci = 0; ci < sender.configs_.size(); ci++) {
                const auto& [cfg, conn_key] = sender.configs_[ci];
                EndpointId eid{EndpointRole::Sender, node_id, core, ci};
                EndpointProgressState eps{
                    .flow_uid = cfg.flow_uid, .endpoint_id = eid, .packets_expected = cfg.parameters.num_packets};
                endpoint_states_.insert_or_assign(eid, eps);
                total_endpoints_++;
            }
        }

        for (const auto& [core, receiver] : test_device.get_receivers()) {
            for (uint16_t ci = 0; ci < receiver.configs_.size(); ci++) {
                const auto& [cfg, opt_key] = receiver.configs_[ci];
                EndpointId eid{EndpointRole::Receiver, node_id, core, ci};
                EndpointProgressState eps{
                    .flow_uid = cfg.flow_uid, .endpoint_id = eid, .packets_expected = cfg.parameters.num_packets};
                endpoint_states_.insert_or_assign(eid, eps);
                total_endpoints_++;
            }
        }
    }
}

TestProgressMonitor::~TestProgressMonitor() = default;

// =====================================================================
// Non-granular polling (legacy path)
// =====================================================================

bool TestProgressMonitor::poll_until_complete() {
    start_time_ = std::chrono::steady_clock::now();
    last_poll_time_ = start_time_;

    bool programs_complete = false;

    while (!programs_complete) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_poll_time_);

        auto progress = poll_devices();

        bool all_hung = check_for_hung_devices(progress);

        programs_complete = true;
        for (const auto& [device_id, prog] : progress) {
            if (prog.num_senders == 0) {
                continue;
            }
            if (prog.current_packets >= prog.total_packets && prog.total_packets > 0) {
                if (!completed_devices_.contains(device_id)) {
                    completed_devices_.insert(device_id);
                    if (config_.show_workers) {
                        std::cout << std::endl;
                        log_info(
                            tt::LogTest,
                            "Device {} completed ({} packets) [{}/{} done]",
                            format_device_label(device_id),
                            format_count(prog.total_packets),
                            completed_devices_.size(),
                            total_active_devices_);
                    }
                }
            } else {
                programs_complete = false;
            }
        }

        display_progress(progress, elapsed);
        last_poll_time_ = now;

        if (all_hung && !programs_complete) {
            std::cout << std::endl;
            generate_hung_report(progress);
            return false;
        }

        if (!programs_complete) {
            std::this_thread::sleep_for(std::chrono::seconds(config_.poll_interval_seconds));
        }
    }

    std::cout << std::endl;
    return true;
}

// =====================================================================
// Granular polling (Phase 3 path)
// =====================================================================

MonitorResult TestProgressMonitor::poll_until_complete_or_hung() {
    start_time_ = std::chrono::steady_clock::now();
    last_poll_time_ = start_time_;
    auto last_log_time = start_time_;
    uint32_t poll_count = 0;

    while (!all_endpoints_resolved()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_poll_time_);

        poll_endpoints();
        check_for_hung_endpoints();
        display_granular_progress(elapsed);
        poll_count++;

        auto since_last_log = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time);
        if (since_last_log.count() >= 10) {
            uint64_t total_current = 0, total_target = 0;
            for (const auto& [eid, eps] : endpoint_states_) {
                total_current += eps.packets_processed;
                total_target += eps.packets_expected;
            }
            double pct = total_target > 0 ? 100.0 * total_current / total_target : 0.0;
            auto total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_);
            log_info(
                tt::LogTest,
                "Granular monitor: {:.1f}% ({}/{} packets) | {}/{} endpoints done | "
                "{} hung | {:.0f}s elapsed | {} polls",
                pct,
                total_current,
                total_target,
                completed_endpoints_,
                total_endpoints_,
                confirmed_hung_endpoints_,
                total_elapsed.count(),
                poll_count);
            last_log_time = now;
        }

        last_poll_time_ = now;

        if (!all_endpoints_resolved()) {
            std::this_thread::sleep_for(std::chrono::seconds(config_.poll_interval_seconds));
        }
    }

    std::cout << std::endl;

    if (!local_hung_records_.empty()) {
        return MonitorResult::HUNG_DETECTED;
    }
    return MonitorResult::ALL_COMPLETE;
}

void TestProgressMonitor::poll_endpoints() {
    auto* device_info = ctx_->get_device_info_provider();

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    uint32_t result_addr = ctx_->get_sender_memory_map().get_result_buffer_address();
    uint32_t result_buf_size = ctx_->get_sender_memory_map().get_result_buffer_size();

    for (const auto& [coord, test_device] : ctx_->get_test_devices()) {
        FabricNodeId node_id = test_device.get_node_id();
        if (!device_info->is_local_fabric_node_id(node_id)) {
            continue;
        }

        const auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(node_id);

        auto read_core_results = [&](CoreCoord logical_core) -> std::vector<uint32_t> {
            CoreCoord virtual_core = device_info->get_virtual_core_from_logical_core(logical_core);
            return cluster.read_core<uint32_t>(physical_chip_id, virtual_core, result_addr, result_buf_size);
        };

        std::unordered_map<CoreCoord, std::vector<uint32_t>> sender_data;
        for (const auto& [core, _] : test_device.get_senders()) {
            sender_data[core] = read_core_results(core);
        }
        process_sender_read_results(node_id, test_device, sender_data);

        std::unordered_map<CoreCoord, std::vector<uint32_t>> receiver_data;
        for (const auto& [core, _] : test_device.get_receivers()) {
            receiver_data[core] = read_core_results(core);
        }
        process_receiver_read_results(node_id, test_device, receiver_data);
    }
}

void TestProgressMonitor::process_sender_read_results(
    FabricNodeId node_id,
    const TestDevice& test_device,
    const std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_data) {
    for (const auto& [core, sender] : test_device.get_senders()) {
        auto it = core_data.find(core);
        if (it == core_data.end()) {
            continue;
        }
        uint8_t num_configs = static_cast<uint8_t>(sender.configs_.size());
        auto parsed = parse_per_config_results(it->second, num_configs);

        for (uint16_t ci = 0; ci < num_configs; ci++) {
            EndpointId eid{EndpointRole::Sender, node_id, core, ci};
            auto state_it = endpoint_states_.find(eid);
            if (state_it != endpoint_states_.end()) {
                state_it->second.packets_processed = parsed[ci].packets_processed;
            }
        }
    }
}

void TestProgressMonitor::process_receiver_read_results(
    FabricNodeId node_id,
    const TestDevice& test_device,
    const std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_data) {
    for (const auto& [core, receiver] : test_device.get_receivers()) {
        auto it = core_data.find(core);
        if (it == core_data.end()) {
            continue;
        }
        uint8_t num_configs = static_cast<uint8_t>(receiver.configs_.size());
        auto parsed = parse_per_config_results(it->second, num_configs);

        for (uint16_t ci = 0; ci < num_configs; ci++) {
            EndpointId eid{EndpointRole::Receiver, node_id, core, ci};
            auto state_it = endpoint_states_.find(eid);
            if (state_it != endpoint_states_.end()) {
                state_it->second.packets_processed = parsed[ci].packets_processed;
            }
        }
    }
}

void TestProgressMonitor::check_for_hung_endpoints() {
    auto now = std::chrono::steady_clock::now();

    for (auto& [eid, eps] : endpoint_states_) {
        if (eps.hung.confirmed_hung) {
            continue;
        }

        // Skip completed endpoints
        if (eps.packets_expected > 0 && eps.packets_processed >= eps.packets_expected) {
            continue;
        }

        auto& hung = eps.hung;

        // Initialize on first check
        if (hung.last_progress_time.time_since_epoch().count() == 0) {
            hung.last_packet_count = eps.packets_processed;
            hung.last_progress_time = now;
            continue;
        }

        if (eps.packets_processed > hung.last_packet_count) {
            hung.last_packet_count = eps.packets_processed;
            hung.last_progress_time = now;
            hung.consecutive_stall_rounds = 0;
            continue;
        }

        // No progress — check if we've exceeded the time threshold
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - hung.last_progress_time);
        if (elapsed < hung_threshold_) {
            continue;
        }

        hung.consecutive_stall_rounds++;

        if (!hung.emitted) {
            const char* role_str = (eid.role == EndpointRole::Sender) ? "Sender" : "Receiver";
            log_warning(
                tt::LogTest,
                "Endpoint stall detected: {} on {} core ({},{}) config#{} flow_uid={} — "
                "no progress for {}s (round {}/{}, packets: {}/{})",
                role_str,
                format_device_label(eid.node_id),
                eid.logical_core.x,
                eid.logical_core.y,
                eid.config_idx,
                eps.flow_uid,
                elapsed.count(),
                hung.consecutive_stall_rounds,
                config_.hung_confirmation_rounds,
                eps.packets_processed,
                eps.packets_expected);
            hung.emitted = true;
        }

        if (hung.consecutive_stall_rounds >= config_.hung_confirmation_rounds) {
            hung.confirmed_hung = true;
            confirmed_hung_endpoints_++;

            const char* role_str = (eid.role == EndpointRole::Sender) ? "Sender" : "Receiver";
            log_warning(
                tt::LogTest,
                "CONFIRMED HUNG: {} on {} core ({},{}) config#{} flow_uid={} — "
                "stalled for {} consecutive rounds (packets: {}/{})",
                role_str,
                format_device_label(eid.node_id),
                eid.logical_core.x,
                eid.logical_core.y,
                eid.config_idx,
                eps.flow_uid,
                hung.consecutive_stall_rounds,
                eps.packets_processed,
                eps.packets_expected);

            local_hung_records_.push_back(HungEndpointRecord{
                .flow_uid = eps.flow_uid,
                .endpoint_id = eid,
                .packets_processed = eps.packets_processed,
                .packets_expected = eps.packets_expected,
                .stall_seconds = static_cast<uint32_t>(elapsed.count()),
                .confirmation_rounds = hung.consecutive_stall_rounds});
        }
    }
}

void TestProgressMonitor::display_granular_progress(std::chrono::duration<double> elapsed) {
    uint64_t total_current = 0, total_target = 0;
    completed_endpoints_ = 0;

    for (const auto& [eid, eps] : endpoint_states_) {
        total_current += eps.packets_processed;
        total_target += eps.packets_expected;
        if (eps.packets_expected > 0 && eps.packets_processed >= eps.packets_expected) {
            completed_endpoints_++;
        }
    }

    double overall_pct = total_target > 0 ? 100.0 * total_current / total_target : 0.0;

    std::stringstream ss;
    ss << "\rProgress: " << std::fixed << std::setprecision(1) << overall_pct << "% "
       << "(" << format_count(total_current) << "/" << format_count(total_target) << ")";

    if (elapsed.count() >= 0.5 && last_total_packets_ > 0 && total_current > last_total_packets_) {
        double throughput = (total_current - last_total_packets_) / elapsed.count();
        ss << " | " << format_throughput(throughput);

        auto eta = estimate_eta(total_current, total_target, throughput);
        if (eta.has_value()) {
            ss << " | ETA: " << format_duration(*eta);
        }
    }

    if (total_current > 0) {
        last_total_packets_ = total_current;
    }

    ss << " | Endpoints: " << completed_endpoints_ << "/" << total_endpoints_ << " done";

    if (confirmed_hung_endpoints_ > 0) {
        ss << " | HUNG: " << confirmed_hung_endpoints_;
    }

    ss << "          ";
    std::cout << ss.str() << std::flush;
}

bool TestProgressMonitor::all_endpoints_resolved() const {
    for (const auto& [eid, eps] : endpoint_states_) {
        if (eps.hung.confirmed_hung) {
            continue;
        }
        if (eps.packets_expected > 0 && eps.packets_processed >= eps.packets_expected) {
            continue;
        }
        return false;
    }
    return true;
}

// =====================================================================
// Device-level helpers (unchanged from pre-Phase 3)
// =====================================================================

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

DeviceProgress TestProgressMonitor::poll_device_senders(
    const MeshCoordinate& /*coord*/, const TestDevice& test_device) {
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

bool TestProgressMonitor::check_for_hung_devices(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress) {
    uint32_t incomplete_count = 0;
    uint32_t hung_count = 0;

    for (const auto& [device_id, prog] : progress) {
        if (prog.current_packets >= prog.total_packets) {
            continue;
        }

        incomplete_count++;

        if (is_device_hung(device_id, prog.current_packets)) {
            hung_count++;
            auto& state = device_states_[device_id];

            if (!state.warned) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - state.last_progress_time);

                log_warning(
                    tt::LogTest,
                    "Device {} may be HUNG: no progress for {} seconds (packets: {}/{})",
                    format_device_label(device_id),
                    elapsed.count(),
                    prog.current_packets,
                    prog.total_packets);

                state.warned = true;
            }
        }
    }

    return incomplete_count > 0 && hung_count == incomplete_count;
}

void TestProgressMonitor::generate_hung_report(
    const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress) {
    const auto total_elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

    log_error(tt::LogTest, "============ FABRIC TEST HUNG - ABORTING ============");
    log_error(tt::LogTest, "Test ran for {} seconds before all remaining devices stalled.", total_elapsed.count());

    for (const auto& [device_id, prog] : progress) {
        if (prog.num_senders == 0) {
            continue;
        }

        const bool is_complete = prog.current_packets >= prog.total_packets && prog.total_packets > 0;
        const bool is_hung = device_states_.contains(device_id) && device_states_.at(device_id).warned;

        const char* status = "IN PROGRESS";
        if (is_complete) {
            status = "COMPLETED";
        } else if (is_hung) {
            status = "HUNG";
        }
        const double pct =
            prog.total_packets > 0 ? 100.0 * static_cast<double>(prog.current_packets) / prog.total_packets : 0.0;

        log_error(
            tt::LogTest,
            "  {} | {} | {}/{} packets ({:.1f}%) | {} sender(s)",
            format_device_label(device_id),
            status,
            format_count(prog.current_packets),
            format_count(prog.total_packets),
            pct,
            prog.num_senders);
    }

    log_error(tt::LogTest, "Devices completed: {}/{}", completed_devices_.size(), total_active_devices_);
    log_error(tt::LogTest, "=====================================================");
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

    if (elapsed.count() >= 0.5 && last_total_packets_ > 0 && total_current > last_total_packets_) {
        double throughput = (total_current - last_total_packets_) / elapsed.count();
        ss << " | " << format_throughput(throughput);

        auto eta = estimate_eta(total_current, total_target, throughput);
        if (eta.has_value()) {
            ss << " | ETA: " << format_duration(*eta);
        }
    }

    if (total_current > 0) {
        last_total_packets_ = total_current;
    }

    if (total_active_devices_ > 0) {
        ss << " | Devices: " << completed_devices_.size() << "/" << total_active_devices_ << " done";
    }

    ss << "          ";

    std::cout << ss.str() << std::flush;
}

// =====================================================================
// Formatting helpers
// =====================================================================

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

// =====================================================================
// Wire record conversion
// =====================================================================

HungEndpointWireRecord to_wire_record(
    const HungEndpointRecord& rec, uint32_t rank, const std::vector<FlowDescriptor>& flow_descriptors) {
    HungEndpointWireRecord wire{};
    wire.flow_uid = rec.flow_uid;
    wire.role = static_cast<uint8_t>(rec.endpoint_id.role);
    wire.mesh_id = *rec.endpoint_id.node_id.mesh_id;
    wire.chip_id = rec.endpoint_id.node_id.chip_id;
    wire.core_x = rec.endpoint_id.logical_core.x;
    wire.core_y = rec.endpoint_id.logical_core.y;
    wire.config_idx = rec.endpoint_id.config_idx;
    wire.packets_processed = rec.packets_processed;
    wire.packets_expected = rec.packets_expected;
    wire.stall_seconds = rec.stall_seconds;
    wire.confirmation_rounds = rec.confirmation_rounds;
    wire.host_rank = rank;

    if (rec.flow_uid < flow_descriptors.size()) {
        const auto& fd = flow_descriptors[rec.flow_uid];
        wire.src_mesh_id = *fd.src_node_id.mesh_id;
        wire.src_chip_id = fd.src_node_id.chip_id;
        if (!fd.dst_node_ids.empty()) {
            wire.dst_mesh_id = *fd.dst_node_ids[0].mesh_id;
            wire.dst_chip_id = fd.dst_node_ids[0].chip_id;
        }
    }
    return wire;
}

HungEndpointRecord from_wire_record(const HungEndpointWireRecord& wire) {
    HungEndpointRecord rec;
    rec.flow_uid = wire.flow_uid;
    rec.endpoint_id = EndpointId{
        static_cast<EndpointRole>(wire.role),
        FabricNodeId(MeshId{wire.mesh_id}, wire.chip_id),
        CoreCoord(wire.core_x, wire.core_y),
        wire.config_idx};
    rec.packets_processed = wire.packets_processed;
    rec.packets_expected = wire.packets_expected;
    rec.stall_seconds = wire.stall_seconds;
    rec.confirmation_rounds = wire.confirmation_rounds;
    return rec;
}

// =====================================================================
// Phase 4: MPI exchange of hung records
// =====================================================================

std::vector<HungEndpointWireRecord> TestProgressMonitor::exchange_hung_records(
    const std::vector<FlowDescriptor>& flow_descriptors) {
    using namespace tt::tt_metal::distributed::multihost;
    const auto& ctx = DistributedContext::get_current_world();
    int my_rank = *ctx->rank();
    int world_size = *ctx->size();

    uint32_t local_count = static_cast<uint32_t>(local_hung_records_.size());

    // Each rank converts its local records using its own flow_descriptors (flow_uid is rank-local)
    std::vector<HungEndpointWireRecord> local_wire;
    local_wire.reserve(local_count);
    for (const auto& rec : local_hung_records_) {
        local_wire.push_back(to_wire_record(rec, static_cast<uint32_t>(my_rank), flow_descriptors));
    }

    if (world_size <= 1) {
        return local_wire;
    }

    // Phase 1: gather per-rank counts to rank 0 (uniform uint32_t per rank)
    std::vector<uint32_t> all_counts(world_size, 0);
    ctx->gather(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&local_count), sizeof(local_count)),
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(all_counts.data()), all_counts.size() * sizeof(uint32_t)),
        Rank{0});

    constexpr int kWireRecordTag = 42;

    if (my_rank == 0) {
        // Start with local records
        std::vector<HungEndpointWireRecord> all_records = std::move(local_wire);

        // Phase 2: receive variable-length records from non-empty ranks
        for (int r = 1; r < world_size; r++) {
            if (all_counts[r] == 0) {
                continue;
            }
            std::vector<HungEndpointWireRecord> remote(all_counts[r]);
            ctx->recv(
                ttsl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(remote.data()), remote.size() * sizeof(HungEndpointWireRecord)),
                Rank{r},
                Tag{kWireRecordTag});
            all_records.insert(all_records.end(), remote.begin(), remote.end());
        }
        return all_records;
    } else {
        // Phase 2: send local records to rank 0 if non-empty
        if (!local_wire.empty()) {
            ctx->send(
                ttsl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(local_wire.data()),
                    local_wire.size() * sizeof(HungEndpointWireRecord)),
                Rank{0},
                Tag{kWireRecordTag});
        }
        return {};
    }
}

// =====================================================================
// Phase 4: Summary report
// =====================================================================

static std::string get_timestamp_string() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

void TestProgressMonitor::write_summary_report(
    const std::vector<HungEndpointWireRecord>& all_records, const std::vector<FlowDescriptor>& flow_descriptors) {
    std::ofstream ofs(summary_report_path_);
    if (!ofs.is_open()) {
        log_warning(tt::LogTest, "Failed to open summary report file: {}", summary_report_path_.string());
        return;
    }

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& psd = control_plane.get_physical_system_descriptor();

    std::string timestamp = get_timestamp_string();
    uint32_t total_flows = static_cast<uint32_t>(flow_descriptors.size());

    // Group by flow_uid
    std::map<uint32_t, std::vector<const HungEndpointWireRecord*>> by_flow;
    for (const auto& rec : all_records) {
        by_flow[rec.flow_uid].push_back(&rec);
    }
    uint32_t hung_flow_count = static_cast<uint32_t>(by_flow.size());

    ofs << "================================================================\n";
    ofs << " PAIRWISE VALIDATION — LINK HEALTH SUMMARY\n";
    ofs << " Timestamp: " << timestamp << "\n";
    if (all_records.empty()) {
        ofs << " Result: PASS — No hung endpoints detected across " << total_flows << " flows\n";
    } else {
        ofs << " Result: FAIL — " << hung_flow_count << " hung flow(s) out of " << total_flows << "\n";
    }
    ofs << "================================================================\n\n";

    if (!all_records.empty()) {
        ofs << "HUNG FLOWS (" << hung_flow_count << "):\n\n";

        uint32_t flow_idx = 1;
        for (const auto& [flow_uid, records] : by_flow) {
            ofs << "  [" << flow_idx++ << "] Flow UID: " << flow_uid << "\n";

            const auto* first_rec = records.front();
            FabricNodeId src_node_id(MeshId{first_rec->src_mesh_id}, first_rec->src_chip_id);
            FabricNodeId dst_node_id(MeshId{first_rec->dst_mesh_id}, first_rec->dst_chip_id);
            ofs << "      Source:      " << format_device_label(src_node_id) << "\n";
            ofs << "      Destination: " << format_device_label(dst_node_id) << "\n";

            auto src_asic_id = control_plane.get_asic_id_from_fabric_node_id(src_node_id);
            auto src_host = psd.get_host_name_for_asic(src_asic_id);
            auto src_tray = psd.get_tray_id(src_asic_id);
            auto src_loc = psd.get_asic_location(src_asic_id);
            ofs << "      Src Physical: host=" << src_host << " tray=" << *src_tray << " asic=" << *src_loc << "\n";

            for (const auto* rec : records) {
                const char* role_str =
                    (rec->role == static_cast<uint8_t>(EndpointRole::Sender)) ? "Sender" : "Receiver";
                FabricNodeId node_id(MeshId{rec->mesh_id}, rec->chip_id);
                ofs << "      " << role_str << ": " << rec->packets_processed << "/" << rec->packets_expected
                    << " packets | no progress for " << rec->stall_seconds << "s"
                    << " [" << format_device_label(node_id) << " rank " << rec->host_rank << "]\n";
            }
            ofs << "\n";
        }
    }

    ofs << "CLUSTER HEALTH: " << hung_flow_count << "/" << total_flows << " flows have hung endpoints\n";
    ofs << "================================================================\n";
    ofs.close();
    log_info(tt::LogTest, "Summary report written to: {}", summary_report_path_.string());
}

// =====================================================================
// Phase 4: Detailed report
// =====================================================================

void TestProgressMonitor::write_detailed_report(
    const std::vector<HungEndpointWireRecord>& all_records, const std::vector<FlowDescriptor>& flow_descriptors) {
    std::ofstream ofs(detail_report_path_);
    if (!ofs.is_open()) {
        log_warning(tt::LogTest, "Failed to open detailed report file: {}", detail_report_path_.string());
        return;
    }

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& psd = control_plane.get_physical_system_descriptor();

    std::string timestamp = get_timestamp_string();

    ofs << "================================================================\n";
    ofs << " PAIRWISE VALIDATION — DETAILED DIAGNOSTIC REPORT\n";
    ofs << " Timestamp: " << timestamp << "\n";
    ofs << " Configuration: hung_threshold=" << config_.hung_threshold_seconds
        << "s, confirmation_rounds=" << config_.hung_confirmation_rounds
        << ", poll_interval=" << config_.poll_interval_seconds << "s\n";
    ofs << "================================================================\n\n";

    // Group by host rank
    std::map<uint32_t, std::vector<const HungEndpointWireRecord*>> by_rank;
    for (const auto& rec : all_records) {
        by_rank[rec.host_rank].push_back(&rec);
    }

    uint32_t total_hung_senders = 0;
    uint32_t total_hung_receivers = 0;
    for (const auto& rec : all_records) {
        if (rec.role == static_cast<uint8_t>(EndpointRole::Sender)) {
            total_hung_senders++;
        } else {
            total_hung_receivers++;
        }
    }

    for (const auto& [rank, records] : by_rank) {
        std::string hostname = psd.get_hostname_for_rank(rank);

        ofs << "--- Host: " << hostname << " (Rank " << rank << ") ---\n\n";
        ofs << "HUNG ENDPOINTS (" << records.size() << "):\n\n";

        uint32_t entry_idx = 1;
        for (const auto* rec : records) {
            const char* role_str = (rec->role == static_cast<uint8_t>(EndpointRole::Sender)) ? "Sender" : "Receiver";
            FabricNodeId node_id(MeshId{rec->mesh_id}, rec->chip_id);
            FabricNodeId src_node_id(MeshId{rec->src_mesh_id}, rec->src_chip_id);
            FabricNodeId dst_node_id(MeshId{rec->dst_mesh_id}, rec->dst_chip_id);

            ofs << "  [" << entry_idx++ << "] " << role_str << " endpoint\n";
            ofs << "      flow_uid: " << rec->flow_uid << "\n";
            ofs << "      Configured: " << format_device_label(src_node_id) << " -> "
                << format_device_label(dst_node_id) << "\n";

            auto fwd_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
            if (!fwd_chans.empty()) {
                ofs << "      Eth Chans (src->dst): ";
                bool first = true;
                for (const auto& chan : fwd_chans) {
                    if (!first) {
                        ofs << ", ";
                    }
                    ofs << "ch" << static_cast<unsigned>(chan);
                    first = false;
                }
                ofs << "\n";
            }

            ofs << "      Core: (" << rec->core_x << "," << rec->core_y << ") | config_idx: " << rec->config_idx
                << "\n";
            ofs << "      Packets: " << rec->packets_processed << "/" << rec->packets_expected
                << " | Stalled for: " << rec->stall_seconds << "s | Confirmed: " << rec->confirmation_rounds
                << " rounds\n";

            auto asic_id = control_plane.get_asic_id_from_fabric_node_id(node_id);
            auto tray_id = psd.get_tray_id(asic_id);
            auto asic_loc = psd.get_asic_location(asic_id);
            auto asic_host = psd.get_host_name_for_asic(asic_id);
            ofs << "      Physical: host=" << asic_host << " tray=" << *tray_id << " asic=" << *asic_loc << "\n";
            ofs << "\n";
        }
    }

    ofs << "================================================================\n";
    ofs << " GLOBAL SUMMARY\n";
    ofs << " Total hung sender endpoints:   " << total_hung_senders << "\n";
    ofs << " Total hung receiver endpoints: " << total_hung_receivers << "\n";
    ofs << " Total hung endpoints:          " << all_records.size() << "\n";
    ofs << " Total flows:                   " << flow_descriptors.size() << "\n";
    ofs << "================================================================\n";
    ofs.close();
    log_info(tt::LogTest, "Detailed report written to: {}", detail_report_path_.string());
}

}  // namespace tt::tt_fabric::fabric_tests
