// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_fabric_telemetry_reader.hpp"

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

SingleSampleBandwidthTelemetryReader::SingleSampleBandwidthTelemetryReader() {
    telemetry_raw_data.resize(telemetry_struct_size_bytes / sizeof(uint32_t));
    std::fill(telemetry_raw_data.begin(), telemetry_raw_data.end(), 0);

    clearing_data_vec.resize(telemetry_struct_size_bytes / sizeof(uint32_t));
    std::fill(clearing_data_vec.begin(), clearing_data_vec.end(), 0);
}

std::optional<ReportedBandwidthTelemetry> SingleSampleBandwidthTelemetryReader::read_bw_telemetry_data(
    tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& eth_logical_core) {
    const auto telemetry_buffer_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    log_trace(tt::LogMetal, "Reading telemetry data for device {}, ethernet core {}", device->id(), eth_logical_core);
    // Read the telemetry struct from L1

    tt::tt_metal::detail::ReadFromDeviceL1(
        device,
        eth_logical_core,
        telemetry_buffer_address,
        telemetry_struct_size_bytes,
        telemetry_raw_data,
        CoreType::ETH);

    ReportedBandwidthTelemetry processed_telemetry_data;
    LowResolutionBandwidthTelemetry telemetry_data;
    std::memcpy(&telemetry_data, telemetry_raw_data.data(), telemetry_struct_size_bytes);

    // Calculate bandwidth
    // Calculate bandwidth
    uint32_t total_eth_words = telemetry_data.num_words_sent;
    processed_telemetry_data.total_bytes = total_eth_words << 4;

    processed_telemetry_data.total_packets = telemetry_data.num_packets_sent;
    // I stored the cycles elapsed directly from device side since for much of the time, the fabric could just
    // be idle, and that shouldn't count towards the bandwidth.
    processed_telemetry_data.total_cycles = telemetry_data.timestamp_start.full;

    if (total_eth_words == 0 || processed_telemetry_data.total_packets == 0 ||
        processed_telemetry_data.total_cycles == 0) {
        log_warning(
            tt::LogTest, "Invalid telemetry data for device {}, ethernet channel {}", device->id(), eth_logical_core);
        return std::nullopt;
    }
    if (telemetry_data.timestamp_end.full != 0) {
        log_warning(
            tt::LogMetal,
            "Telemetry data for device {}, ethernet channel {} is corrupted. Ignoring data",
            device->id(),
            eth_logical_core);
        return std::nullopt;
    }

    return processed_telemetry_data;
}

void SingleSampleBandwidthTelemetryReader::clear_bw_telemetry_data(
    tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& eth_logical_core) {
    const auto telemetry_buffer_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    tt::tt_metal::detail::WriteToDeviceL1(
        device, eth_logical_core, telemetry_buffer_address, clearing_data_vec, CoreType::ETH);
}

std::string get_connectivity_table_as_csv_string(
    const tt::tt_metal::distributed::MeshDeviceView& mesh_device_view,
    const std::vector<TelemetryEntry>& telemetry_entries) {
    // Generate CSV tables from stored telemetry data
    const size_t num_rows = mesh_device_view.num_rows();
    const size_t num_cols = mesh_device_view.num_cols();

    // Build CSV tables as single strings
    std::stringstream connectivity_table;
    connectivity_table << "=== CHIP CONNECTIVITY TABLE (CSV) ===\n";
    connectivity_table << "MeshRow,MeshCol,ConnectedMeshRow,ConnectedMeshCol,EthernetChannel\n";

    // Enumerate all ethernet connections from mesh device
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (size_t row = 0; row < num_rows; row++) {
        for (size_t col = 0; col < num_cols; col++) {
            tt::tt_metal::distributed::MeshCoordinate current_coord(row, col);

            // Get the physical chip ID for this mesh coordinate
            tt::tt_metal::IDevice* device = mesh_device_view.get_device(current_coord);
            chip_id_t chip_id = device->id();

            // Get all ethernet connections for this chip
            const auto& connected_chips_and_cores = cluster.get_ethernet_cores_grouped_by_connected_chips(chip_id);
            const auto& soc_desc = cluster.get_soc_desc(chip_id);

            // Add entry for each ethernet connection
            for (const auto& [connected_chip_id, local_eth_cores] : connected_chips_and_cores) {
                // Find the mesh coordinate of the connected chip
                tt::tt_metal::distributed::MeshCoordinate connected_coord =
                    mesh_device_view.find_device(connected_chip_id);

                for (const auto& eth_core : local_eth_cores) {
                    // Get the ethernet channel for this core
                    auto eth_channel_iter = soc_desc.logical_eth_core_to_chan_map.find(eth_core);
                    TT_FATAL(
                        eth_channel_iter != soc_desc.logical_eth_core_to_chan_map.end(),
                        "Ethernet channel not found for core {}",
                        eth_core);
                    uint32_t eth_channel = eth_channel_iter->second;
                    connectivity_table << fmt::format(
                        "{},{},{},{},{}\n", row, col, connected_coord[0], connected_coord[1], eth_channel);
                }
            }
        }
    }

    return connectivity_table.str();
}

std::string get_bandwidth_table_as_csv_string(const std::vector<TelemetryEntry>& telemetry_entries) {
    std::stringstream bandwidth_table;
    bandwidth_table << "=== BANDWIDTH TABLE (CSV) ===\n";
    bandwidth_table << "MeshRow,MeshCol,EthernetChannel,BandwidthGB_s,PacketsPerSecond\n";

    // Add entries for each telemetry data point
    for (const auto& entry : telemetry_entries) {
        bandwidth_table << fmt::format(
            "{},{},{},{:.2f},{:.2f}\n",
            entry.mesh_coord[0],
            entry.mesh_coord[1],
            entry.eth_channel,
            entry.results.bandwidth_GB_s,
            entry.results.packets_per_second);
    }

    return bandwidth_table.str();
}

void print_results_to_csv(
    const tt::tt_metal::distributed::MeshDeviceView& mesh_device_view,
    const std::vector<TelemetryEntry>& telemetry_entries) {
    // Output complete tables as single log messages
    log_info(tt::LogMetal, "");
    log_info(tt::LogMetal, "{}", get_connectivity_table_as_csv_string(mesh_device_view, telemetry_entries));
    log_info(tt::LogMetal, "{}", get_bandwidth_table_as_csv_string(telemetry_entries));
}

// Reads back and report the bandwidth telemetry data from active Ethernet core L1.
std::vector<TelemetryEntry> read_fabric_telemetry_data(
    const tt::tt_metal::distributed::MeshDeviceView& mesh_device_view) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Store telemetry data for CSV generation
    std::vector<TelemetryEntry> telemetry_entries;

    SingleSampleBandwidthTelemetryReader single_sample_bw_reader;

    for (const auto& device : mesh_device_view.get_devices()) {
        log_info(tt::LogMetal, "Reading telemetry data for device {}", device->id());
        auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
        auto device_mesh_coord = mesh_device_view.find_device(device->id());

        for (const auto& eth_logical_core : control_plane.get_active_ethernet_cores(device->id())) {
            auto bw_telemetry = single_sample_bw_reader.read_bw_telemetry_data(device, eth_logical_core);
            if (!bw_telemetry) {
                log_warning(
                    tt::LogMetal, "No telemetry data for device {}, ethernet core {}", device->id(), eth_logical_core);
                continue;
            }

            single_sample_bw_reader.clear_bw_telemetry_data(device, eth_logical_core);

            log_info(
                tt::LogMetal,
                "Fabric Telemetry - Device: {}, Ethernet Channel: {}, "
                "Total Words: {}, Total Packets: {}, Total Cycles: {}, Bandwidth: {:.2f} GB/s, Packets/s: {:.2f}",
                device->id(),
                eth_logical_core,
                bw_telemetry->total_bytes,
                bw_telemetry->total_packets,
                bw_telemetry->total_cycles,
                bw_telemetry->bandwidth_GB_s,
                bw_telemetry->packets_per_second);

            // Store telemetry data for CSV generation
            // Find ethernet channel for this core
            uint32_t eth_channel = 0;
            if (soc_desc.logical_eth_core_to_chan_map.find(eth_logical_core) !=
                soc_desc.logical_eth_core_to_chan_map.end()) {
                eth_channel = soc_desc.logical_eth_core_to_chan_map.at(eth_logical_core);
            }

            // Get connectivity info
            auto [connected_chip_id, connected_eth_core] = device->get_connected_ethernet_core(eth_logical_core);
            auto connected_mesh_coord = mesh_device_view.find_device(connected_chip_id);

            auto connected_soc_desc =
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(connected_chip_id);
            uint32_t connected_eth_channel = 0;
            if (connected_soc_desc.logical_eth_core_to_chan_map.find(connected_eth_core) !=
                connected_soc_desc.logical_eth_core_to_chan_map.end()) {
                connected_eth_channel = connected_soc_desc.logical_eth_core_to_chan_map.at(connected_eth_core);
            }

            telemetry_entries.push_back(
                {*bw_telemetry, device_mesh_coord, eth_channel, connected_mesh_coord, connected_eth_channel});
        }
    }

    return telemetry_entries;
}
