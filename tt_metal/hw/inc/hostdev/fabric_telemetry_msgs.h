// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// NOTE:
// This header is consumed by tt_metal/llrt/hal/codegen/codegen.sh to generate
// HAL struct accessors.  Keep all definitions to the limited subset of C++ that
// the generator supports (structs, enums, constants, and 1-D arrays).

#pragma once

#include <cstddef>
#include <cstdint>

enum RouterState : uint8_t { STANDBY = 0, ACTIVE = 1, PAUSED = 2, DRAINING = 3 };

enum class FabricArch : uint8_t { WORMHOLE_B0 = 0, BLACKHOLE = 1, QUASAR = 2, STATIC_ONLY = 3 };

// TODO: this V2 needs to be deleted.
//       Just for avoiding conflicts.
struct RiscTimestampV2 {
    union {
        uint64_t full;
        struct {
            uint32_t lo;
            uint32_t hi;
        };
    };
};

// Bitmask of dynamic stats present in DynamicInfo.
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
    // Accumulated active cycles across router inner loops where work was performed.
    RiscTimestampV2 elapsed_active_cycles;
    // Accumulated cycles across all inner loops (active + idle).
    RiscTimestampV2 elapsed_cycles;
    uint64_t num_words_sent;
    uint64_t num_packets_sent;
};

struct EriscDynamicEntry {
    RouterState router_state;
    // TX heartbeat: incremented when all sender queues are empty, or a packet was sent over Ethernet.
    RiscTimestampV2 tx_heartbeat;
    // RX heartbeat: incremented when receiver queues are empty, or a packet was forwarded from receiver to NoC/local.
    RiscTimestampV2 rx_heartbeat;
};

struct DynamicInfo {
    BandwidthTelemetry tx_bandwidth;
    BandwidthTelemetry rx_bandwidth;
    // max 2 ERISCs per router (BH has 2, WH has 1)
    EriscDynamicEntry erisc[2];
};

struct StaticInfo {
    uint16_t mesh_id;
    uint8_t device_id;
    uint8_t direction;
    uint32_t fabric_config;
    DynamicStatistics supported_stats;
};

struct FabricTelemetryStaticOnly {
    StaticInfo static_info;
};

struct FabricTelemetry {
    StaticInfo static_info;
    DynamicInfo dynamic_info;
    uint32_t postcode;
    uint32_t scratch[7];
};
