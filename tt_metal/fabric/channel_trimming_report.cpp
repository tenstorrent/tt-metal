// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_trimming_report.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <yaml-cpp/yaml.h>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/channel_trimming_io.hpp"

namespace tt::tt_fabric {

namespace {

// Returns {expected_sender_channels, expected_receiver_channels} for a router
// given the fabric topology, VC1 presence, and the router's direction string from the YAML.
//
// Channel counts from builder_config:
//   Z direction (always has VC1):  9 sender, 2 receiver
//   2D with VC1:                   8 sender, 2 receiver
//   2D without VC1 (VC0 only):    4 sender, 1 receiver
//   1D:                            2 sender, 1 receiver
std::pair<uint32_t, uint32_t> get_expected_channels(
    Topology topology, const std::string& direction, bool has_vc1) {
    if (direction == "Z") {
        // Z routers always have both VCs
        return {builder_config::num_sender_channels_z_router, builder_config::num_receiver_channels_z_router};
    }
    if (is_2D_topology(topology)) {
        if (has_vc1) {
            return {builder_config::num_sender_channels_2d, builder_config::num_receiver_channels_2d};
        }
        // VC0 only: Worker + 3 mesh directions = 4 sender, 1 receiver
        return {builder_config::num_sender_channels_2d_mesh, builder_config::num_receiver_channels_1d};
    }
    // 1D topologies: Linear, Ring, NeighborExchange
    return {builder_config::num_sender_channels_1d, builder_config::num_receiver_channels_1d};
}

// Count set bits in the lowest `n_bits` of a bitfield.
uint32_t count_used_bits(uint16_t bitfield, uint32_t n_bits) {
    uint16_t mask = (n_bits >= 16) ? 0xFFFF : static_cast<uint16_t>((1u << n_bits) - 1);
    return static_cast<uint32_t>(std::popcount(static_cast<uint16_t>(bitfield & mask)));
}

// Helper to log a single histogram with full-block bar characters.
void log_histogram(const char* label, const char* x_axis_label, const std::vector<uint32_t>& hist) {
    log_debug(tt::LogFabric, "  {} ({} -> router count):", label, x_axis_label);
    for (size_t i = 0; i < hist.size(); i++) {
        if (hist[i] == 0) {
            continue;
        }
        std::string bars;
        for (uint32_t b = 0; b < hist[i]; b++) {
            bars += "\u2588";  // U+2588 FULL BLOCK
        }
        log_debug(tt::LogFabric, "    {:>2}: {}  ({} routers)", i, bars, hist[i]);
    }
}

}  // namespace

void generate_and_log_channel_trimming_report(const std::string& yaml_path, Topology topology, bool has_vc1) {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["channel_trimming_capture"]) {
        log_warning(tt::LogFabric, "Channel trimming report: YAML missing 'channel_trimming_capture' key");
        return;
    }

    const auto& capture = root["channel_trimming_capture"];
    std::vector<RouterTrimmingStats> stats;

    auto chip_keys = collect_map_keys(capture);
    for (const auto& chip_key : chip_keys) {
        ChipId chip_id = parse_chip_key(chip_key);
        const auto& channels = capture[chip_key];

        auto chan_keys = collect_map_keys(channels);
        for (const auto& chan_key : chan_keys) {
            chan_id_t eth_chan = parse_eth_channel_key(chan_key);
            const auto& chan_data = channels[chan_key];

            std::string direction = chan_data["direction"] ? chan_data["direction"].as<std::string>() : "UNKNOWN";

            auto [expected_sender, expected_receiver] = get_expected_channels(topology, direction, has_vc1);

            uint16_t sender_bitfield = 0;
            if (chan_data["sender_channel_used_bitfield"]) {
                sender_bitfield = parse_hex_bitfield(chan_data["sender_channel_used_bitfield"].as<std::string>());
            }

            uint16_t receiver_bitfield = 0;
            if (chan_data["receiver_channel_data_forwarded_bitfield"]) {
                receiver_bitfield =
                    parse_hex_bitfield(chan_data["receiver_channel_data_forwarded_bitfield"].as<std::string>());
            }

            stats.push_back(RouterTrimmingStats{
                .chip_id = chip_id,
                .eth_chan = eth_chan,
                .total_sender_channels = expected_sender,
                .used_sender_channels = count_used_bits(sender_bitfield, expected_sender),
                .total_receiver_channels = expected_receiver,
                .used_receiver_channels = count_used_bits(receiver_bitfield, expected_receiver),
            });
        }
    }

    if (stats.empty()) {
        log_info(tt::LogFabric, "Channel Trimming Report: no routers found in profile");
        return;
    }

    // Compute summary stats
    uint32_t total_sender_removed = 0, total_sender_available = 0;
    uint32_t total_receiver_removed = 0, total_receiver_available = 0;
    uint32_t min_sender_removed = UINT32_MAX, max_sender_removed = 0;
    uint32_t min_receiver_removed = UINT32_MAX, max_receiver_removed = 0;
    uint32_t min_aggregate_removed = UINT32_MAX, max_aggregate_removed = 0;
    uint32_t total_aggregate_removed = 0, total_aggregate_available = 0;

    // Removed histograms
    std::vector<uint32_t> sender_removed_hist;
    std::vector<uint32_t> receiver_removed_hist;
    std::vector<uint32_t> aggregate_removed_hist;

    // Instantiated (used) histograms
    std::vector<uint32_t> sender_instantiated_hist;
    std::vector<uint32_t> receiver_instantiated_hist;
    std::vector<uint32_t> aggregate_instantiated_hist;

    auto grow_and_increment = [](std::vector<uint32_t>& hist, uint32_t bucket) {
        if (bucket >= hist.size()) {
            hist.resize(bucket + 1, 0);
        }
        hist[bucket]++;
    };

    for (const auto& s : stats) {
        uint32_t sr = s.sender_channels_removed();
        uint32_t rr = s.receiver_channels_removed();
        uint32_t ar = s.total_channels_removed();

        total_sender_removed += sr;
        total_sender_available += s.total_sender_channels;
        total_receiver_removed += rr;
        total_receiver_available += s.total_receiver_channels;
        total_aggregate_removed += ar;
        total_aggregate_available += s.total_sender_channels + s.total_receiver_channels;

        min_sender_removed = std::min(min_sender_removed, sr);
        max_sender_removed = std::max(max_sender_removed, sr);
        min_receiver_removed = std::min(min_receiver_removed, rr);
        max_receiver_removed = std::max(max_receiver_removed, rr);
        min_aggregate_removed = std::min(min_aggregate_removed, ar);
        max_aggregate_removed = std::max(max_aggregate_removed, ar);

        grow_and_increment(sender_removed_hist, sr);
        grow_and_increment(receiver_removed_hist, rr);
        grow_and_increment(aggregate_removed_hist, ar);

        grow_and_increment(sender_instantiated_hist, s.used_sender_channels);
        grow_and_increment(receiver_instantiated_hist, s.used_receiver_channels);
        uint32_t total_instantiated = s.used_sender_channels + s.used_receiver_channels;
        grow_and_increment(aggregate_instantiated_hist, total_instantiated);
    }

    size_t n = stats.size();
    double avg_sender = static_cast<double>(total_sender_removed) / n;
    double avg_receiver = static_cast<double>(total_receiver_removed) / n;
    double avg_aggregate = static_cast<double>(total_aggregate_removed) / n;

    // Log the report
    log_info(
        tt::LogFabric,
        "Channel Trimming Report ({} routers profiled, topology={}, vc1={}):",
        n,
        topology,
        has_vc1 ? "enabled" : "disabled");
    log_info(
        tt::LogFabric,
        "  Sender channels:    removed min={}  max={}  avg={:.1f}  total={}/{}",
        min_sender_removed,
        max_sender_removed,
        avg_sender,
        total_sender_removed,
        total_sender_available);
    log_info(
        tt::LogFabric,
        "  Receiver channels:  removed min={}  max={}  avg={:.1f}  total={}/{}",
        min_receiver_removed,
        max_receiver_removed,
        avg_receiver,
        total_receiver_removed,
        total_receiver_available);
    log_info(
        tt::LogFabric,
        "  Aggregate:          removed min={}  max={}  avg={:.1f}  total={}/{}",
        min_aggregate_removed,
        max_aggregate_removed,
        avg_aggregate,
        total_aggregate_removed,
        total_aggregate_available);

    // Channels removed histograms
    log_histogram("Sender removed", "channels removed", sender_removed_hist);
    log_histogram("Receiver removed", "channels removed", receiver_removed_hist);
    log_histogram("Aggregate removed", "channels removed", aggregate_removed_hist);

    // Channels instantiated histograms
    log_histogram("Sender instantiated", "channels kept", sender_instantiated_hist);
    log_histogram("Receiver instantiated", "channels kept", receiver_instantiated_hist);
    log_histogram("Aggregate instantiated", "channels kept", aggregate_instantiated_hist);
}

}  // namespace tt::tt_fabric
