// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "tt_fabric_test_progress_monitor.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <thread>

#include "tt_fabric_test_constants.hpp"
#include "tt_fabric_test_context.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <board/board.hpp>
#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

namespace tt::tt_fabric::fabric_tests {

namespace {

// Cache Boards by board_type so we only call create_board() once per unique board.
// Provides per-channel port labels and per-link labels that show the *first-hop*
// physical wire: src chip's port on one end, immediate neighbor chip's port on the
// other end. Note: for multi-hop fabric routes the "neighbor" here is NOT the
// configured destination chip — it's whichever chip the wire physically reaches.
//
// Neighbor chip identity is derived only from PSD asic descriptors (host/tray/
// asic_location); we deliberately avoid ControlPlane::get_fabric_node_id_from_asic_id
// because it is local-host only and TT_FATALs (with noisy critical logs) on
// cross-host asic ids.
class PortLabelResolver {
public:
    explicit PortLabelResolver(const tt::tt_metal::PhysicalSystemDescriptor& psd) : psd_(psd) {}

    // Returns "QSFP_DD#5" if the chan maps to an external port on the board, or
    // an empty string if it's an internal/trace channel or the board is unknown.
    std::string port_label(tt::tt_metal::AsicID asic_id, uint8_t chan) {
        const auto& descriptors = psd_.get_asic_descriptors();
        auto desc_it = descriptors.find(asic_id);
        if (desc_it == descriptors.end()) {
            return "";
        }
        const auto& desc = desc_it->second;

        const auto* board = get_or_create_board(desc.board_type);
        if (board == nullptr) {
            return "";
        }

        try {
            const auto& port = board->get_port_for_asic_channel(
                tt::scaleout_tools::AsicChannel{*(desc.asic_location), tt::scaleout_tools::ChanId{chan}});
            return fmt::format("{}#{}", enchantum::to_string(port.port_type), *port.port_id);
        } catch (const std::exception&) {
            return "";
        }
    }

    // Returns a one-line description of the physical first-hop link for a given
    // (src_asic, src_chan), e.g.
    //     "ch0[QSFP_DD#6] -> bh-glx-c02u08_2/T4/N4 ch0[QSFP_DD#6]"
    // The neighbor's host prefix is omitted only when the neighbor sits on the
    // same host as the src chip, so "no prefix" always means intra-host.
    // If the connection lookup fails (e.g. cross-host info not available locally)
    // we fall back to just the src side: "ch0[QSFP_DD#6]".
    std::string link_label(tt::tt_metal::AsicID src_asic, uint8_t src_chan) {
        std::string src_port = port_label(src_asic, src_chan);
        std::string src_part = src_port.empty() ? fmt::format("ch{}", static_cast<unsigned>(src_chan))
                                                : fmt::format("ch{}[{}]", static_cast<unsigned>(src_chan), src_port);

        const auto& descriptors = psd_.get_asic_descriptors();
        auto src_desc_it = descriptors.find(src_asic);
        const std::string src_host = (src_desc_it != descriptors.end()) ? src_desc_it->second.host_name : std::string{};

        try {
            auto [dst_asic, dst_chan] = psd_.get_connected_asic_and_channel(src_asic, src_chan);
            std::string dst_port = port_label(dst_asic, dst_chan);
            std::string dst_chip_label = format_neighbor_chip(dst_asic, src_host);
            std::string dst_chan_part = dst_port.empty()
                                            ? fmt::format("ch{}", static_cast<unsigned>(dst_chan))
                                            : fmt::format("ch{}[{}]", static_cast<unsigned>(dst_chan), dst_port);
            return fmt::format("{} -> {} {}", src_part, dst_chip_label, dst_chan_part);
        } catch (const std::exception&) {
            return src_part;
        }
    }

private:
    const tt::scaleout_tools::Board* get_or_create_board(BoardType board_type) {
        auto it = boards_.find(board_type);
        if (it != boards_.end()) {
            return &it->second;
        }
        try {
            auto inserted = boards_.emplace(board_type, tt::scaleout_tools::create_board(board_type));
            return &inserted.first->second;
        } catch (const std::exception&) {
            return nullptr;
        }
    }

    // Compact identifier for a neighbor chip. Omits the host prefix only when
    // the neighbor lives on `reference_host` (i.e. same host as the src chip
    // for this link), so an unprefixed neighbor always means intra-host.
    std::string format_neighbor_chip(tt::tt_metal::AsicID asic_id, const std::string& reference_host) {
        const auto& descriptors = psd_.get_asic_descriptors();
        auto desc_it = descriptors.find(asic_id);
        if (desc_it == descriptors.end()) {
            return "(neighbor)";
        }
        const auto& d = desc_it->second;
        if (!reference_host.empty() && d.host_name == reference_host) {
            return fmt::format("T{}/N{}", *d.tray_id, *d.asic_location);
        }
        return fmt::format("{}/T{}/N{}", d.host_name, *d.tray_id, *d.asic_location);
    }

    const tt::tt_metal::PhysicalSystemDescriptor& psd_;
    std::unordered_map<BoardType, tt::scaleout_tools::Board> boards_;
};

}  // namespace

static std::filesystem::path resolve_report_path(const std::string& filename) {
    std::filesystem::path root =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path path = root / std::string(OUTPUT_DIR) / filename;
    // Cover both the default output dir and any extra subdirectories the user
    // passed in via filename (e.g. "subdir/report.log").
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            log_warning(tt::LogTest, "Failed to create report directory '{}': {}", parent.string(), ec.message());
        }
    }
    return path;
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
            for (size_t ci = 0; ci < sender.configs_.size(); ci++) {
                const auto& [cfg, conn_key] = sender.configs_[ci];
                EndpointId eid{EndpointRole::Sender, node_id, core, static_cast<uint16_t>(ci)};
                EndpointProgressState eps{
                    .flow_uid = cfg.flow_uid, .endpoint_id = eid, .packets_expected = cfg.parameters.num_packets};
                endpoint_states_.insert_or_assign(eid, eps);
                total_endpoints_++;
            }
        }

        for (const auto& [core, receiver] : test_device.get_receivers()) {
            for (size_t ci = 0; ci < receiver.configs_.size(); ci++) {
                const auto& [cfg, opt_key] = receiver.configs_[ci];
                EndpointId eid{EndpointRole::Receiver, node_id, core, static_cast<uint16_t>(ci)};
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

        // Skip completed endpoints (including zero-expected configs, which are trivially complete)
        if (eps.packets_processed >= eps.packets_expected) {
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
            log_debug(
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
            log_debug(
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
        if (eps.packets_processed >= eps.packets_expected) {
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
        if (eps.packets_processed >= eps.packets_expected) {
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
    }

    // Non-zero ranks: Phase 2 — send local records to rank 0 if non-empty.
    if (!local_wire.empty()) {
        ctx->send(
            ttsl::Span<std::byte>(
                reinterpret_cast<std::byte*>(local_wire.data()), local_wire.size() * sizeof(HungEndpointWireRecord)),
            Rank{0},
            Tag{kWireRecordTag});
    }
    return {};
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
    PortLabelResolver port_resolver(psd);

    std::string timestamp = get_timestamp_string();
    uint32_t total_flows = static_cast<uint32_t>(flow_descriptors.size());

    // Aggregate per device-pair (src_node -> dst_node), then group those pairs by source host rank.
    struct PairAggregate {
        FabricNodeId src_node{MeshId{0}, 0};
        FabricNodeId dst_node{MeshId{0}, 0};
        uint32_t dst_host_rank = 0;
        std::string dst_host;
        uint32_t sender_hung = 0;
        uint32_t receiver_hung = 0;
        uint64_t min_packets_processed = std::numeric_limits<uint64_t>::max();
        uint64_t packets_expected = 0;
        uint32_t max_stall_seconds = 0;
        std::set<uint32_t> flow_uids;  // distinct flow_uids contributing to this pair
    };

    using PairKey = std::pair<FabricNodeId, FabricNodeId>;
    std::map<uint32_t, std::map<PairKey, PairAggregate>> pairs_by_src_rank;
    std::set<uint32_t> distinct_flow_uids;

    for (const auto& rec : all_records) {
        FabricNodeId src(MeshId{rec.src_mesh_id}, rec.src_chip_id);
        FabricNodeId dst(MeshId{rec.dst_mesh_id}, rec.dst_chip_id);

        auto src_asic = control_plane.get_asic_id_from_fabric_node_id(src);
        auto dst_asic = control_plane.get_asic_id_from_fabric_node_id(dst);
        auto src_host = psd.get_host_name_for_asic(src_asic);
        auto dst_host = psd.get_host_name_for_asic(dst_asic);
        uint32_t src_rank = psd.get_rank_for_hostname(src_host);
        uint32_t dst_rank = psd.get_rank_for_hostname(dst_host);

        auto& agg = pairs_by_src_rank[src_rank][PairKey{src, dst}];
        agg.src_node = src;
        agg.dst_node = dst;
        agg.dst_host_rank = dst_rank;
        agg.dst_host = dst_host;
        if (rec.role == static_cast<uint8_t>(EndpointRole::Sender)) {
            agg.sender_hung++;
        } else {
            agg.receiver_hung++;
        }
        agg.min_packets_processed = std::min(agg.min_packets_processed, rec.packets_processed);
        agg.packets_expected = rec.packets_expected;
        agg.max_stall_seconds = std::max(agg.max_stall_seconds, rec.stall_seconds);
        agg.flow_uids.insert(rec.flow_uid);
        distinct_flow_uids.insert(rec.flow_uid);
    }

    uint32_t hung_flow_count = static_cast<uint32_t>(distinct_flow_uids.size());
    uint32_t hung_pair_count = 0;
    for (const auto& [rank, pairs] : pairs_by_src_rank) {
        (void)rank;
        hung_pair_count += static_cast<uint32_t>(pairs.size());
    }

    ofs << "================================================================\n";
    ofs << " PAIRWISE VALIDATION — LINK HEALTH SUMMARY\n";
    ofs << " Timestamp: " << timestamp << "\n";
    if (all_records.empty()) {
        ofs << " Result: PASS — No hung endpoints detected across " << total_flows << " flows\n";
    } else {
        ofs << " Result: FAIL — " << hung_flow_count << " hung flow(s) out of " << total_flows << " ("
            << hung_pair_count << " unique device pair(s))\n";
    }
    ofs << "================================================================\n";

    if (!all_records.empty()) {
        for (const auto& [src_rank, pairs] : pairs_by_src_rank) {
            std::string hostname = psd.get_hostname_for_rank(src_rank);

            // Tally per-host stats
            uint32_t host_sender_hung = 0;
            uint32_t host_receiver_hung = 0;
            std::set<uint32_t> host_flow_uids;
            for (const auto& [pair_key, agg] : pairs) {
                (void)pair_key;
                host_sender_hung += agg.sender_hung;
                host_receiver_hung += agg.receiver_hung;
                host_flow_uids.insert(agg.flow_uids.begin(), agg.flow_uids.end());
            }

            ofs << "\n--- Source Host: " << hostname << " (Rank " << src_rank << ") ---\n";
            ofs << "    " << pairs.size() << " hung device pair(s) | " << host_flow_uids.size()
                << " flow(s) | senders=" << host_sender_hung << " receivers=" << host_receiver_hung << "\n\n";

            uint32_t pair_idx = 1;
            for (const auto& [pair_key, agg] : pairs) {
                (void)pair_key;
                ofs << "  [" << pair_idx++ << "] " << format_device_label(agg.src_node) << "  ->  "
                    << format_device_label(agg.dst_node) << "\n";
                ofs << "      Dst Host: " << agg.dst_host << " (Rank " << agg.dst_host_rank << ")\n";

                auto fwd_chans = control_plane.get_forwarding_eth_chans_to_chip(agg.src_node, agg.dst_node);
                auto src_asic_id = control_plane.get_asic_id_from_fabric_node_id(agg.src_node);
                if (fwd_chans.empty()) {
                    ofs << "      Eth Links (first hop from src): (none reported)\n";
                } else {
                    ofs << "      Eth Links (first hop from src):\n";
                    for (const auto& chan : fwd_chans) {
                        ofs << "        " << port_resolver.link_label(src_asic_id, chan) << "\n";
                    }
                }

                uint64_t shown_packets =
                    (agg.min_packets_processed == std::numeric_limits<uint64_t>::max()) ? 0 : agg.min_packets_processed;
                ofs << "      Hung endpoints: senders=" << agg.sender_hung << " receivers=" << agg.receiver_hung
                    << " | flows=" << agg.flow_uids.size() << " | packets=" << shown_packets << "/"
                    << agg.packets_expected << " | stalled " << agg.max_stall_seconds << "s\n\n";
            }
        }
    }

    ofs << "CLUSTER HEALTH: " << hung_flow_count << "/" << total_flows << " flows have hung endpoints across "
        << hung_pair_count << " device pair(s)\n";
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
    PortLabelResolver port_resolver(psd);

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
                auto src_asic_id = control_plane.get_asic_id_from_fabric_node_id(src_node_id);
                ofs << "      Eth Links (first hop from src):\n";
                for (const auto& chan : fwd_chans) {
                    ofs << "        " << port_resolver.link_label(src_asic_id, chan) << "\n";
                }
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
