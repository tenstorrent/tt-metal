// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>

namespace tt::tt_fabric::test_utils {

// Worker kernel teardown protocol constants
constexpr uint32_t WORKER_KEEP_RUNNING = 0;
constexpr uint32_t WORKER_TEARDOWN = 1;

// Named constants for telemetry validation
constexpr std::chrono::milliseconds DEFAULT_TRAFFIC_SAMPLE_INTERVAL{100};
constexpr std::chrono::milliseconds DEFAULT_PAUSE_TIMEOUT{5000};
constexpr std::chrono::milliseconds DEFAULT_POLL_INTERVAL{100};

// Compile-time args structure for traffic generator kernel
struct TrafficGeneratorCompileArgs {
    uint32_t source_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t target_noc_encoding;
    uint32_t teardown_signal_address;
    uint32_t is_2d_fabric;
};

// Runtime args structure for traffic generator kernel
struct TrafficGeneratorRuntimeArgs {
    uint32_t dest_chip_id;
    uint32_t dest_mesh_id;
    uint32_t random_seed;
};

} // namespace tt::tt_fabric::test_utils
