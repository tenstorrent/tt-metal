// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>
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

struct LowResolutionBandwidthTelemetryResult {
    RiscTimestamp duration{};
    uint64_t reserved{};
    uint64_t num_words_sent{};
    uint64_t num_packets_sent{};
};

struct TelemetryEntry {
    ::tt::tt_metal::distributed::MeshCoordinate coord;
    uint32_t eth_channel{};
    double bw_gbps{};
    double pps{};
    ::tt::tt_metal::distributed::MeshCoordinate connected_coord;
    uint32_t connected_eth_channel{};
};

const uint32_t telemetry_addr = ::tt::tt_metal::hal::get_erisc_l1_unreserved_base();

// TODO: Define enum for filtering
enum class EthCoreFilter {
    All,
    IgnoreTunnelerRouter
};
