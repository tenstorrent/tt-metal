// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "control_plane.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <device.hpp>
#include <mesh_device.hpp>
#include <mesh_device_view.hpp>
#include <hal.hpp>
#include <tt_metal.hpp>

#include <cstdint>
#include <optional>
#include <sstream>
#include <vector>

struct ReportedBandwidthTelemetry {
    uint64_t total_bytes;
    uint64_t total_packets;
    uint64_t total_cycles;
    double bandwidth_GB_s;
    double packets_per_second;
};

struct TelemetryEntry {
    ReportedBandwidthTelemetry results;
    tt::tt_metal::distributed::MeshCoordinate mesh_coord;
    uint32_t eth_channel;
    tt::tt_metal::distributed::MeshCoordinate connected_mesh_coord;
    uint32_t connected_eth_channel;
};
struct RiscTimestamp {
    union {
        uint64_t full;
        struct {
            uint32_t lo;
            uint32_t hi;
        };
    };
};
struct LowResolutionBandwidthTelemetry {
    RiscTimestamp timestamp_start;
    RiscTimestamp timestamp_end;
    uint32_t num_words_sent;
    uint32_t num_packets_sent;
};

struct SingleSampleBandwidthTelemetryReader {
    static constexpr size_t telemetry_struct_size_bytes = 24;
    SingleSampleBandwidthTelemetryReader();

    std::optional<ReportedBandwidthTelemetry> read_bw_telemetry_data(
        tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& eth_logical_core);

    void clear_bw_telemetry_data(tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& eth_logical_core);

    std::vector<uint32_t> telemetry_raw_data;
    std::vector<uint32_t> clearing_data_vec;
};

std::string get_connectivity_table_as_csv_string(
    const tt::tt_metal::distributed::MeshDeviceView& mesh_device, const std::vector<TelemetryEntry>& telemetry_entries);
std::string get_bandwidth_table_as_csv_string(const std::vector<TelemetryEntry>& telemetry_entries);
void print_results_to_csv(
    const tt::tt_metal::distributed::MeshDeviceView& mesh_device, const std::vector<TelemetryEntry>& telemetry_entries);

// Reads back and report the bandwidth telemetry data from active Ethernet core L1.
std::vector<TelemetryEntry> read_fabric_telemetry_data(const tt::tt_metal::distributed::MeshDeviceView& mesh_device);
