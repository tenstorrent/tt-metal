// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_fabric::bench {

// Fixed per-sender packet count across the matrix so fan-in comparisons hold offered
// load constant. open() is outside the timed region; close() remains inside as the
// completion barrier, so P should be large enough that the send loop dominates close().
inline constexpr uint32_t kPacketsPerSender = 10000;
inline constexpr uint32_t kDefaultDrainerNumBuffers = 16;

struct MuxV2ThroughputCase {
    std::string name_suffix;
    uint32_t num_senders = 1;
    uint32_t packet_payload_size_bytes = 0;  // 0 => use fabric max payload size at runtime.
    uint8_t num_buffers_per_channel = 1;
    tt::tt_metal::NOC forwarder_noc = tt::tt_metal::NOC::RISCV_0_default;
    uint32_t num_drainer_buffers = kDefaultDrainerNumBuffers;
    uint32_t num_packets = kPacketsPerSender;
};

inline uint32_t resolve_packet_payload_size_bytes(const MuxV2ThroughputCase& benchmark_case) {
    if (benchmark_case.packet_payload_size_bytes != 0) {
        return benchmark_case.packet_payload_size_bytes;
    }
    return static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
}

class FabricMuxV2BenchmarkContext {
public:
    void initialize();
    void shutdown();

    bool can_support_case(const MuxV2ThroughputCase& benchmark_case, std::string* rejection_reason = nullptr) const;

    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& get_mesh_device() const { return mesh_device_; }
    tt::tt_metal::IDevice* get_device() const { return device_; }
    const std::vector<tt::tt_metal::CoreCoord>& get_worker_cores() const { return worker_cores_; }

    tt::tt_metal::CoreCoord get_mux_logical_core() const;
    tt::tt_metal::CoreCoord get_drainer_logical_core() const;
    std::vector<tt::tt_metal::CoreCoord> get_sender_logical_cores(const MuxV2ThroughputCase& benchmark_case) const;

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    tt::tt_metal::IDevice* device_ = nullptr;
    std::vector<tt::tt_metal::CoreCoord> worker_cores_;
};

struct StandaloneMuxV2BenchmarkRunResult {
    bool success = false;
    std::string error_message;
    uint64_t aggregate_bytes = 0;
    uint64_t max_sender_cycles = 0;
};

StandaloneMuxV2BenchmarkRunResult run_standalone_mux_v2_benchmark_once(
    const FabricMuxV2BenchmarkContext& context, const MuxV2ThroughputCase& benchmark_case);

}  // namespace tt::tt_fabric::bench
