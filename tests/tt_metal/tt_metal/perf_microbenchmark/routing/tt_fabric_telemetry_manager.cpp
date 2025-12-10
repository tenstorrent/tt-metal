// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_telemetry_manager.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <fstream>
#include <filesystem>

#include "impl/context/metal_context.hpp"

namespace {
// Calculate bandwidth in bytes/cycle given total ETH words and cycles.
double calc_bw_bytes_per_cycle(uint32_t total_words, uint64_t cycles) {
    constexpr uint32_t bytes_per_eth_word = 16;
    return (total_words * bytes_per_eth_word) / static_cast<double>(cycles);
}
}  // namespace

TelemetryManager::TelemetryManager(TestFixture& fixture, EthCoreBufferReadback& eth_readback) :
    fixture_(fixture), eth_readback_(eth_readback) {}

void TelemetryManager::read_telemetry() {
    telemetry_entries_.clear();

    // Get telemetry buffer address and size
    const auto telemetry_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t telemetry_buffer_size = sizeof(LowResolutionBandwidthTelemetryResult);

    // Read buffer data from all active ethernet cores (including intermediate forwarding hops)
    auto results = eth_readback_.read_buffer(telemetry_addr, telemetry_buffer_size, false);

    // Process telemetry results
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    for (const auto& result : results) {
        const auto& core_data = result.buffer_data;
        auto device_id = result.fabric_node_id;
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto freq_mhz = fixture_.get_device_frequency_mhz(device_id);
        double freq_ghz = double(freq_mhz) / 1000.0;

        LowResolutionBandwidthTelemetryResult tel{};
        if (reinterpret_cast<uintptr_t>(core_data.data()) % alignof(LowResolutionBandwidthTelemetryResult) == 0) {
            constexpr size_t NUM_ELEMENTS =
                tt::align(sizeof(LowResolutionBandwidthTelemetryResult), sizeof(uint32_t)) / sizeof(uint32_t);
            const std::array<uint32_t, NUM_ELEMENTS>& data_array =
                *reinterpret_cast<const std::array<uint32_t, NUM_ELEMENTS>*>(core_data.data());
            tel = std::bit_cast<LowResolutionBandwidthTelemetryResult>(data_array);
        } else {
            // Fall back to memcpy approach
            std::array<std::byte, sizeof(LowResolutionBandwidthTelemetryResult)> staging_buf{};
            memcpy(staging_buf.data(), core_data.data(), sizeof(LowResolutionBandwidthTelemetryResult));
            tel = std::bit_cast<LowResolutionBandwidthTelemetryResult>(staging_buf);
        }

        uint64_t cycles = tel.duration.full;
        double bytes_per_cycle = calc_bw_bytes_per_cycle(tel.num_words_sent, cycles);
        double bw_GB_s = bytes_per_cycle * double(freq_ghz);
        double time_s = static_cast<double>(cycles) / (freq_mhz * 1e6);
        double pps = static_cast<double>(tel.num_packets_sent) / time_s;
        log_info(
            tt::LogTest,
            "Telemetry from {} core {}: BW (GB/s)={:.6f}, pps={:.6f}, cycles={:d}, eth_words_sent={:d}, "
            "packets_sent={:d}",
            result.coord,
            result.eth_core.str(),
            bw_GB_s,
            pps,
            cycles,
            tel.num_words_sent,
            tel.num_packets_sent);
        auto [connected_physical_id, connected_eth_core] =
            cluster.get_connected_ethernet_core({physical_chip_id, result.eth_core});
        auto connected_device_id = control_plane.get_fabric_node_id_from_physical_chip_id(connected_physical_id);
        ::tt::tt_metal::distributed::MeshCoordinate connected_coord = fixture_.get_device_coord(connected_device_id);
        uint32_t connected_eth_channel =
            cluster.get_soc_desc(connected_physical_id).logical_eth_core_to_chan_map.at(connected_eth_core);
        telemetry_entries_.push_back(
            {result.coord, result.eth_channel, bw_GB_s, pps, connected_coord, connected_eth_channel});
    }
}

void TelemetryManager::clear_telemetry() {
    telemetry_entries_.clear();

    // Get telemetry buffer address and size
    const auto telemetry_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t telemetry_buffer_size = sizeof(LowResolutionBandwidthTelemetryResult);

    // Note: clear_buffer already clears all active ethernet cores regardless of registration
    eth_readback_.clear_buffer(telemetry_addr, telemetry_buffer_size);
}

void TelemetryManager::process_telemetry_for_golden() {
    std::sort(
        telemetry_entries_.begin(), telemetry_entries_.end(), [](const TelemetryEntry& a, const TelemetryEntry& b) {
            return a.bw_gbps > b.bw_gbps;
        });

    if (telemetry_entries_.empty()) {
        measured_bw_min_ = std::numeric_limits<double>::max();
        measured_bw_avg_ = 0.0;
        measured_bw_max_ = 0.0;
        return;
    }
    measured_bw_min_ = std::numeric_limits<double>::max();
    measured_bw_avg_ = 0.0;
    measured_bw_max_ = 0.0;

    auto max_entry = telemetry_entries_.front();
    auto max_coord = max_entry.coord;
    auto max_connected = max_entry.connected_coord;

    double minimal_accepted_BW_GB_s = 0.08;
    for (const auto& entry : telemetry_entries_) {
        if (entry.bw_gbps < minimal_accepted_BW_GB_s) {
            continue;
        }

        measured_bw_min_ = std::min(measured_bw_min_, entry.bw_gbps);
        measured_bw_max_ = std::max(measured_bw_max_, entry.bw_gbps);
        measured_bw_avg_ += entry.bw_gbps;
    }
    measured_bw_avg_ /= telemetry_entries_.size();

    auto is_on_edge = [&](const TelemetryEntry& e) {
        return (e.coord == max_coord && e.connected_coord == max_connected) ||
               (e.coord == max_connected && e.connected_coord == max_coord);
    };

    std::map<int, std::vector<double>> plane_bws;
    for (const auto& entry : telemetry_entries_) {
        if (is_on_edge(entry)) {
            int plane = entry.eth_channel / 4;
            plane_bws[plane].push_back(entry.bw_gbps);
        }
    }

    double min_of_mins = std::numeric_limits<double>::max();
    for (const auto& [plane, bws] : plane_bws) {
        if (!bws.empty()) {
            double min_bw = *std::min_element(bws.begin(), bws.end());
            min_of_mins = std::min(min_of_mins, min_bw);
        }
    }

    log_info(
        tt::LogTest,
        "Measured BW (min/avg/max): {:.6f} GB/s, {:.6f} GB/s, {:.6f} GB/s",
        measured_bw_min_,
        measured_bw_avg_,
        measured_bw_max_);
}

void TelemetryManager::dump_raw_telemetry_csv(const TestConfig& config) {
    std::filesystem::path raw_telemetry_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        (std::string(OUTPUT_DIR) + std::string("/") + config.name + "_raw_telemetry.csv");

    if (!std::filesystem::exists(raw_telemetry_path)) {
        std::ofstream header_stream(raw_telemetry_path, std::ios::out | std::ios::trunc);
        if (header_stream.is_open()) {
            header_stream << "coord,eth_channel,bw_gbps,pps,connected_coord,connected_eth_channel\n";
            header_stream.close();
        }
    }

    std::ofstream data_stream(raw_telemetry_path, std::ios::out | std::ios::app);
    if (data_stream.is_open()) {
        for (const auto& entry : telemetry_entries_) {
            data_stream << "\"" << entry.coord[0] << "," << entry.coord[1] << "\"," << entry.eth_channel << ","
                        << entry.bw_gbps << "," << entry.pps << "," << entry.connected_coord << ","
                        << entry.connected_eth_channel << "\n";
        }
        data_stream.close();
    }
    log_info(tt::LogTest, "Dumped raw telemetry to: {}", raw_telemetry_path.string());
}

void TelemetryManager::reset() {
    telemetry_entries_.clear();
    measured_bw_min_ = 0.0;
    measured_bw_avg_ = 0.0;
    measured_bw_max_ = 0.0;
}
