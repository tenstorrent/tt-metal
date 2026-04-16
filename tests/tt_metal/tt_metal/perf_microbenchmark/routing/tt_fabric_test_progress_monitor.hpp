// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include "tt_fabric_test_constants.hpp"
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/distributed_context.hpp>

class TestContext;

namespace tt::tt_fabric::fabric_tests {

struct TestDevice;

// Result of the polling loop
enum class MonitorResult { ALL_COMPLETE, HUNG_DETECTED };

// Progress monitoring configuration
struct ProgressMonitorConfig {
    bool enabled = false;
    bool show_workers = false;
    bool granular = false;  // Per-endpoint detail mode (--show-progress-detail)
    uint32_t poll_interval_seconds = 2;
    uint32_t hung_threshold_seconds = 30;
    uint32_t hung_confirmation_rounds = 3;
    bool wait_on_hang = false;
    std::string summary_file = DEFAULT_VALIDATION_SUMMARY_FILE;
    std::string detail_file = DEFAULT_VALIDATION_DETAIL_FILE;
};

// --- Device-level types (used when granular == false) ---

struct DeviceProgress {
    tt::tt_fabric::FabricNodeId device_id{tt::tt_fabric::MeshId{0}, 0};
    uint64_t current_packets = 0;
    uint64_t total_packets = 0;
    uint32_t num_senders = 0;
    uint32_t num_receivers = 0;
};

struct DeviceState {
    uint64_t last_packet_count = 0;
    std::chrono::steady_clock::time_point last_progress_time;
    bool warned = false;
};

// --- Endpoint-level types (used when granular == true) ---

enum class EndpointRole : uint8_t { Sender, Receiver };

struct EndpointId {
    EndpointRole role = EndpointRole::Sender;
    FabricNodeId node_id{MeshId{0}, 0};
    CoreCoord logical_core;
    uint16_t config_idx = 0;

    bool operator==(const EndpointId&) const = default;
};

struct EndpointIdHash {
    std::size_t operator()(const EndpointId& id) const {
        std::size_t h = std::hash<uint8_t>()(static_cast<uint8_t>(id.role));
        h ^= std::hash<FabricNodeId>()(id.node_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(id.logical_core.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(id.logical_core.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint16_t>()(id.config_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct EndpointHungState {
    uint64_t last_packet_count = 0;
    std::chrono::steady_clock::time_point last_progress_time{};
    uint32_t consecutive_stall_rounds = 0;
    bool confirmed_hung = false;
    bool emitted = false;
};

struct EndpointProgressState {
    FlowUid flow_uid = 0;
    EndpointId endpoint_id;

    uint64_t packets_processed = 0;
    uint64_t packets_expected = 0;

    EndpointHungState hung;
};

struct HungEndpointRecord {
    FlowUid flow_uid;
    EndpointId endpoint_id;

    uint64_t packets_processed = 0;
    uint64_t packets_expected = 0;
    uint32_t stall_seconds = 0;
    uint32_t confirmation_rounds = 0;
};

struct ParsedConfigProgress {
    uint64_t packets_processed = 0;
};

// Flat POD wire struct for MPI serialization of hung endpoint records.
// All fields are fixed-size primitives so the struct can be sent as raw bytes.
struct HungEndpointWireRecord {
    uint32_t flow_uid;
    uint8_t role;  // EndpointRole as uint8_t
    uint32_t mesh_id;
    uint32_t chip_id;
    uint32_t core_x;
    uint32_t core_y;
    uint16_t config_idx;
    uint64_t packets_processed;
    uint64_t packets_expected;
    uint32_t stall_seconds;
    uint32_t confirmation_rounds;
    uint32_t host_rank;
    uint32_t src_mesh_id;
    uint32_t src_chip_id;
    uint32_t dst_mesh_id;
    uint32_t dst_chip_id;
};
static_assert(
    std::is_trivially_copyable_v<HungEndpointWireRecord>,
    "HungEndpointWireRecord must be trivially copyable for MPI serialization");

HungEndpointWireRecord to_wire_record(
    const HungEndpointRecord& rec, uint32_t rank, const std::vector<FlowDescriptor>& flow_descriptors);
HungEndpointRecord from_wire_record(const HungEndpointWireRecord& wire);

// Parse per-config results from a result buffer readback
inline std::vector<ParsedConfigProgress> parse_per_config_results(
    const std::vector<uint32_t>& result_data, uint8_t num_configs) {
    std::vector<ParsedConfigProgress> results(num_configs);
    if (result_data.size() < PER_CONFIG_RESULT_BASE_WORD_INDEX + num_configs * 2) {
        return results;
    }
    auto* per_config = reinterpret_cast<const PerConfigResult*>(result_data.data() + PER_CONFIG_RESULT_BASE_WORD_INDEX);
    for (uint8_t i = 0; i < num_configs; i++) {
        results[i].packets_processed =
            static_cast<uint64_t>(per_config[i].packets_high) << 32 | per_config[i].packets_low;
    }
    return results;
}

// Progress monitor - polls devices and displays progress during test execution
class TestProgressMonitor {
public:
    TestProgressMonitor(::TestContext* ctx, const ProgressMonitorConfig& config);
    ~TestProgressMonitor();

    TestProgressMonitor(const TestProgressMonitor&) = delete;
    TestProgressMonitor& operator=(const TestProgressMonitor&) = delete;
    TestProgressMonitor(TestProgressMonitor&&) = delete;
    TestProgressMonitor& operator=(TestProgressMonitor&&) = delete;

    // Non-granular mode: poll until all devices complete or all remaining are hung.
    // Returns true if all completed, false if all remaining are hung.
    bool poll_until_complete();

    // Granular mode: poll until all endpoints complete or hung is confirmed
    MonitorResult poll_until_complete_or_hung();

    // Access hung records for Phase 4 reporting
    const std::vector<HungEndpointRecord>& get_hung_records() const { return local_hung_records_; }

    // Phase 4: MPI exchange and report generation
    std::vector<HungEndpointWireRecord> exchange_hung_records(const std::vector<FlowDescriptor>& flow_descriptors);
    void write_summary_report(
        const std::vector<HungEndpointWireRecord>& all_records, const std::vector<FlowDescriptor>& flow_descriptors);
    void write_detailed_report(
        const std::vector<HungEndpointWireRecord>& all_records, const std::vector<FlowDescriptor>& flow_descriptors);

    const std::filesystem::path& get_summary_report_path() const { return summary_report_path_; }
    const std::filesystem::path& get_detail_report_path() const { return detail_report_path_; }

private:
    // --- Device-level polling (non-granular) ---
    std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress> poll_devices();
    DeviceProgress poll_device_senders(const MeshCoordinate& coord, const TestDevice& test_device);
    bool check_for_hung_devices(const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress);
    void generate_hung_report(const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress);
    void display_progress(
        const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
        std::chrono::duration<double> elapsed_since_last_poll);
    bool is_device_hung(tt::tt_fabric::FabricNodeId device_id, uint64_t current_packets);

    // --- Endpoint-level polling (granular, uses batched reads) ---
    void poll_endpoints();
    void process_sender_read_results(
        FabricNodeId node_id,
        const TestDevice& test_device,
        const std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_data);
    void process_receiver_read_results(
        FabricNodeId node_id,
        const TestDevice& test_device,
        const std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_data);
    void check_for_hung_endpoints();
    void display_granular_progress(std::chrono::duration<double> elapsed);
    bool all_endpoints_resolved() const;

    // Formatting helpers
    std::string format_count(uint64_t count) const;
    std::string format_throughput(double packets_per_second) const;
    std::string format_duration(double seconds) const;

    // ETA calculation
    std::optional<double> estimate_eta(uint64_t current_total, uint64_t target_total, double throughput) const;

    ::TestContext* ctx_;
    ProgressMonitorConfig config_;

    // Timing
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_poll_time_;

    // Throughput state
    uint64_t last_total_packets_ = 0;

    // Device-level state (non-granular mode)
    std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceState> device_states_;
    std::chrono::seconds hung_threshold_;
    std::unordered_set<tt::tt_fabric::FabricNodeId> completed_devices_;
    uint32_t total_active_devices_ = 0;

    // Endpoint-level state (granular mode)
    std::unordered_map<EndpointId, EndpointProgressState, EndpointIdHash> endpoint_states_;
    std::vector<HungEndpointRecord> local_hung_records_;
    uint32_t total_endpoints_ = 0;
    uint32_t completed_endpoints_ = 0;
    uint32_t confirmed_hung_endpoints_ = 0;

    // Resolved report file paths (computed once at construction)
    std::filesystem::path summary_report_path_;
    std::filesystem::path detail_report_path_;
};

}  // namespace tt::tt_fabric::fabric_tests
