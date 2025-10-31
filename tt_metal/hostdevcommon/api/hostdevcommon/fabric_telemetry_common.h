// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Public API header for Fabric Heartbeat telemetry data structures
#include <cstdint>
#include <cstddef>
#include "hostdevcommon/fabric_common.h"

// Forward declaration to avoid including heavy host-only headers in common kernel/host API
namespace tt {
namespace tt_fabric {
enum class FabricConfig : uint32_t;
}
}  // namespace tt

enum RouterState : uint8_t { STANDBY = 0, ACTIVE = 1, PAUSED = 2, DRAINING = 3 };

// TODO: this V2 need to be deleted.
//       just for avoiding conflicts
struct RiscTimestampV2 {
    union {
        uint64_t full;
        struct {
            uint32_t lo;
            uint32_t hi;
        };
    };
};

// bitmask of dynamic stats present in DynamicInfo
enum DynamicStatistics : uint8_t {
    NONE = 0x00,
    ROUTER_STATE = 0x01,
    BANDWIDTH = 0x02,
    HEARTBEAT = 0x04,
    STAT_COUNT = 3
};

// Bandwidth telemetry values intended for software delta sampling.
// All counters are free-running 64-bit and may wrap; software should compute deltas between samples.
struct BandwidthTelemetry {
    // Accumulated active cycles across router inner loops where work was performed
    RiscTimestampV2 elapsed_active_cycles;
    // Accumulated cycles across all inner loops (active + idle)
    RiscTimestampV2 elapsed_cycles;
    uint64_t num_words_sent;
    uint64_t num_packets_sent;
};

struct DynamicInfo {
    RouterState router_state;
    // Free-running heartbeat. Software can use deltas to detect liveness
    RiscTimestampV2 heartbeat;
    BandwidthTelemetry bandwidth;
};

struct StaticInfo {
    uint16_t mesh_id;
    uint8_t device_id;
    tt::tt_fabric::eth_chan_directions direction;
    tt::tt_fabric::FabricConfig fabric_config;
    DynamicStatistics supported_stats;
};

template <size_t NUM_ERISCS>
struct FabricTelemetry {
    DynamicInfo dynamic_info[NUM_ERISCS];
    StaticInfo static_info;
};
