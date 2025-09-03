// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// 
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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

struct TelemetryEntry {
    ::tt::tt_metal::distributed::MeshCoordinate coord;
    uint32_t eth_channel;
    double bw_gbps;
    double pps;
    ::tt::tt_metal::distributed::MeshCoordinate connected_coord;
    uint32_t connected_eth_channel;
};

double calc_bw_gbps(uint32_t total_words, uint64_t cycles) {
    return (total_words * 16.0) / static_cast<double>(cycles);
}

double calc_pps(uint32_t num_packets, uint64_t cycles) {
    // Assuming 1GHz clock
    double time_s = static_cast<double>(cycles) / 1e9;
    return num_packets / time_s;
}

const uint32_t telemetry_addr = ::tt::tt_metal::hal::get_erisc_l1_unreserved_base();

// TODO: Define enum for filtering
enum class EthCoreFilter {
    All,
    IgnoreTunnelerRouter
};
