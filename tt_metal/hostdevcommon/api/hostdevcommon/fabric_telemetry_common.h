// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include "hostdevcommon/fabric_common.h"

namespace tt {
namespace tt_fabric {
enum class FabricConfig : uint32_t;
}
}  // namespace tt

enum RouterState : uint8_t { STANDBY = 0, ACTIVE = 1, PAUSED = 2, DRAINING = 3 };

enum class FabricArch : uint8_t { WORMHOLE_B0 = 0, BLACKHOLE = 1, QUASAR = 2, STATIC_ONLY = 3 };

template <FabricArch ARCH>
struct EriscCount;

template <>
struct EriscCount<FabricArch::STATIC_ONLY> {
    static constexpr size_t value = 0;
};

template <>
struct EriscCount<FabricArch::WORMHOLE_B0> {
    static constexpr size_t value = 1;
};
template <>
struct EriscCount<FabricArch::BLACKHOLE> {
    static constexpr size_t value = 2;
};
template <>
struct EriscCount<FabricArch::QUASAR> {
    static constexpr size_t value = 2;
};

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
    HEARTBEAT_TX = 0x04,
    HEARTBEAT_RX = 0x08,
    STAT_COUNT = 4
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

template <size_t NUM_ERISCS>
struct DynamicInfo {
    struct EriscDynamicEntry {
        RouterState router_state;
        // TX heartbeat: incremented when all sender queues are empty, or a packet was sent over Ethernet
        RiscTimestampV2 tx_heartbeat;
        // RX heartbeat: incremented when receiver queues are empty, or a packet was forwarded from receiver to
        // NoC/local
        RiscTimestampV2 rx_heartbeat;
    };

    // Per-core shared bandwidth counters
    BandwidthTelemetry tx_bandwidth;
    BandwidthTelemetry rx_bandwidth;
    // Per-ERISC dynamic entries
    EriscDynamicEntry erisc[NUM_ERISCS];
};

struct StaticInfo {
    uint16_t mesh_id;
    uint8_t device_id;
    tt::tt_fabric::eth_chan_directions direction;
    tt::tt_fabric::FabricConfig fabric_config;
    DynamicStatistics supported_stats;
};

template <FabricArch ARCH>
struct FabricTelemetry {
    StaticInfo static_info;
    DynamicInfo<EriscCount<ARCH>::value> dynamic_info;
};
