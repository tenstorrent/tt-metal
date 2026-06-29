#pragma once

#include <algorithm>
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

inline constexpr uint64_t kDefaultTargetAggregatePayloadBytes = 256ull * 1024ull * 1024ull;
inline constexpr uint32_t kDefaultForwarderServiceBurstSize = 8;
inline constexpr uint32_t kDefaultDrainerNumBuffers = 16;
inline constexpr uint32_t kDefaultTridRingCapacity = tt::tt_fabric::FabricMuxV2Config::kDefaultTridRingCapacity;

struct MuxV2ThroughputCase {
    std::string name_suffix;
    uint32_t num_senders = 1;
    uint32_t packet_payload_size_bytes = 0;  // 0 => use fabric max payload size at runtime.
    uint8_t num_buffers_per_channel = 1;
    tt::tt_metal::NOC forwarder_noc = tt::tt_metal::NOC::RISCV_0_default;
    uint32_t service_burst_size = kDefaultForwarderServiceBurstSize;
    uint32_t trid_ring_capacity = kDefaultTridRingCapacity;
    uint32_t num_drainer_buffers = kDefaultDrainerNumBuffers;
    uint64_t target_aggregate_payload_bytes = kDefaultTargetAggregatePayloadBytes;
};

inline uint32_t resolve_packet_payload_size_bytes(const MuxV2ThroughputCase& benchmark_case) {
    if (benchmark_case.packet_payload_size_bytes != 0) {
        return benchmark_case.packet_payload_size_bytes;
    }
    return static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
}

inline uint32_t derive_num_packets(const MuxV2ThroughputCase& benchmark_case) {
    const auto packet_payload_size_bytes = resolve_packet_payload_size_bytes(benchmark_case);
    const uint64_t bytes_per_round =
        std::max<uint64_t>(1, static_cast<uint64_t>(benchmark_case.num_senders) * packet_payload_size_bytes);
    return static_cast<uint32_t>(
        std::max<uint64_t>(1, (benchmark_case.target_aggregate_payload_bytes + bytes_per_round - 1) / bytes_per_round));
}

class FabricMuxV2BenchmarkContext {
public:
    void initialize();
    void shutdown();

    bool can_support_case(const MuxV2ThroughputCase& benchmark_case, std::string* rejection_reason = nullptr) const;

    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& get_mesh_device() const { return mesh_device_; }
    tt::tt_metal::IDevice* get_device() const { return device_; }
    const std::vector<CoreCoord>& get_worker_cores() const { return worker_cores_; }

    CoreCoord get_mux_logical_core() const;
    CoreCoord get_drainer_logical_core() const;
    std::vector<CoreCoord> get_sender_logical_cores(const MuxV2ThroughputCase& benchmark_case) const;

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    tt::tt_metal::IDevice* device_ = nullptr;
    std::vector<CoreCoord> worker_cores_;
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
