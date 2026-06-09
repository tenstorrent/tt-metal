#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_fabric::bench {

enum class SaturationImplementation : uint8_t {
    V1,
    V2,
};

enum class SaturationTopology : uint8_t {
    Fabric1D,
    Fabric2D,
};

struct SaturationVariant {
    SaturationImplementation implementation = SaturationImplementation::V2;
    SaturationTopology topology = SaturationTopology::Fabric1D;
};

struct SaturationCase {
    std::string name_suffix;
    uint32_t num_clients = 1;
    uint8_t num_buffers_per_channel = 1;
    uint32_t channel_buffer_size_bytes = 0;
    uint32_t num_packets_per_sender = 0;
};

class FabricMuxSaturationBenchmarkContext {
public:
    void initialize(SaturationTopology topology);
    void shutdown();

    bool can_support_case(
        const SaturationVariant& variant,
        const SaturationCase& benchmark_case,
        std::string* rejection_reason = nullptr) const;

    SaturationTopology get_topology() const { return topology_; }
    const std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& get_devices_by_physical_id() const {
        return devices_by_physical_id_;
    }
    const std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& get_devices() const { return devices_; }

private:
    SaturationTopology topology_ = SaturationTopology::Fabric1D;
    std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_by_physical_id_;
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
};

struct SaturationRunResult {
    bool success = false;
    std::string error_message;
    uint64_t aggregate_bytes = 0;
    uint64_t max_sender_cycles = 0;
};

SaturationRunResult run_mux_saturation_once(
    const FabricMuxSaturationBenchmarkContext& context,
    const SaturationVariant& variant,
    const SaturationCase& benchmark_case);

}  // namespace tt::tt_fabric::bench
