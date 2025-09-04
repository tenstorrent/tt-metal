// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include <stdint.h>
#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "buffer.hpp"
#include "common/TracyTTDeviceData.hpp"
#include "core_coord.hpp"
#include "profiler_optional_metadata.hpp"
#include "profiler_types.hpp"
#include "tracy/TracyTTDevice.hpp"

namespace tt {
enum class ARCH;
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

using RuntimeID = uint32_t;

namespace tt {

namespace tt_metal {

template <typename T1, typename T2>
struct pair_hash {
    size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        constexpr std::size_t hash_combine_prime = 0x9e3779b9;
        return h1 ^ (h2 + hash_combine_prime + (h1 << 6) + (h1 >> 2));
    }
};

class FabricRoutingLookup;

struct SyncInfo {
    double cpu_time = 0.0;
    double device_time = 0.0;
    double frequency = 0.0;

    SyncInfo(double cpu_time, double device_time, double frequency) :
        cpu_time(cpu_time), device_time(device_time), frequency(frequency) {}

    SyncInfo() : SyncInfo(0.0, 0.0, 0.0) {}
};

struct FabricEventMarkers {
    std::vector<tracy::TTDeviceMarker> fabric_write_markers;
    tracy::TTDeviceMarker fabric_routing_fields_marker;
    tracy::TTDeviceMarker local_noc_write_marker;
    std::optional<tracy::TTDeviceMarker> fabric_mux_marker;
};

class DeviceProfiler {
private:
    // Device architecture
    tt::ARCH device_arch{tt::ARCH::Invalid};

    // Device ID
    chip_id_t device_id{};

    // Device frequency
    int device_core_frequency{};

    // Last fast dispatch read performed flag
    bool is_last_fd_read_done{};

    // Smallest timestamp
    uint64_t smallest_timestamp = (1lu << 63);

    // Output directory for device profiler logs
    std::filesystem::path output_dir;

    // Hash to zone source locations
    std::unordered_map<uint16_t, tracy::MarkerDetails> hash_to_zone_src_locations;

    // Device-Core tracy context
    std::unordered_map<std::pair<uint16_t, CoreCoord>, TracyTTCtx, pair_hash<uint16_t, CoreCoord>>
        device_tracy_contexts;

    // (cpu time, device time, frequency) for sync propagated from root device
    SyncInfo device_sync_info;

    // Per-core sync info used to make tracy context
    std::unordered_map<CoreCoord, SyncInfo> core_sync_info;

    // (Device ID, Core Coord) pairs that keep track of cores which need to have their Tracy contexts updated
    std::unordered_set<std::pair<chip_id_t, CoreCoord>, pair_hash<chip_id_t, CoreCoord>> device_cores;

    // Storage for all core's control buffers
    std::unordered_map<CoreCoord, std::vector<uint32_t>> core_control_buffers;

    // Storage for all core's L1 data buffers
    std::unordered_map<CoreCoord, std::vector<uint32_t>> core_l1_data_buffers;

    // Storage for all noc trace data
    std::vector<std::unordered_map<RuntimeID, nlohmann::json::array_t>> noc_trace_data;

    // Storage for all noc trace markers that have been converted to json to ensure that the same marker isn't processed
    // twice
    std::unordered_set<tracy::TTDeviceMarker> noc_trace_markers_processed;

    // Output directory for noc trace data
    std::filesystem::path noc_trace_data_output_dir;

    // Read all control buffers
    void readControlBuffers(IDevice* device, const std::vector<CoreCoord>& virtual_cores);

    // Read control buffer for a single core
    void readControlBufferForCore(IDevice* device, const CoreCoord& virtual_core);

    // Reset all control buffers
    void resetControlBuffers(IDevice* device, const std::vector<CoreCoord>& virtual_cores);

    // Read all L1 data buffers
    void readL1DataBuffers(IDevice* device, const std::vector<CoreCoord>& virtual_cores);

    // Read L1 data buffer for a single core
    void readL1DataBufferForCore(
        IDevice* device, const CoreCoord& virtual_core, std::vector<uint32_t>& core_l1_data_buffer);

    // Read device profiler buffer
    void readProfilerBuffer(IDevice* device);

    // Read data from profiler buffer using fast dispatch
    void issueFastDispatchReadFromProfilerBuffer(IDevice* device);

    // Read data from profiler buffer using slow dispatch
    void issueSlowDispatchReadFromProfilerBuffer(IDevice* device);

    // Read data from L1 data buffer using fast dispatch
    void issueFastDispatchReadFromL1DataBuffer(
        IDevice* device, const CoreCoord& worker_core, std::vector<uint32_t>& core_l1_data_buffer);

    // Read data from L1 data buffer using slow dispatch
    void issueSlowDispatchReadFromL1DataBuffer(
        IDevice* device, const CoreCoord& worker_core, std::vector<uint32_t>& core_l1_data_buffer);

    // Helper function for reading risc profile results
    void readRiscProfilerResults(
        IDevice* device,
        const CoreCoord& worker_core,
        ProfilerDataBufferSource data_source,
        const std::optional<ProfilerOptionalMetadata>& metadata);

    // Read marker data to be displayed
    void readDeviceMarkerData(
        std::set<tracy::TTDeviceMarker>& device_markers,
        uint32_t run_host_id,
        const std::string& op_name,
        chip_id_t device_id,
        const CoreCoord& physical_core,
        tracy::RiscType risc_type,
        uint64_t data,
        uint32_t timer_id,
        uint64_t timestamp);

    // Track the smallest timestamp read
    void updateFirstTimestamp(uint64_t timestamp);

    // Get tracy context for the core
    void updateTracyContext(std::pair<uint32_t, CoreCoord> device_core);

    // Iterate over all markers and update their data if needed
    void processDeviceMarkerData(std::set<tracy::TTDeviceMarker>& device_markers);

public:
    DeviceProfiler(const IDevice* device, bool new_logs);

    DeviceProfiler() = delete;

    ~DeviceProfiler();

    // Device-core Syncdata
    std::map<CoreCoord, SyncInfo> device_core_sync_info;

    // DRAM Vector
    std::vector<uint32_t> profile_buffer;

    // Number of bytes reserved in each DRAM bank for storing device profiling data
    uint32_t profile_buffer_bank_size_bytes{};

    // Device markers grouped by (physical core, risc type)
    std::map<CoreCoord, std::map<tracy::RiscType, std::set<tracy::TTDeviceMarker>>> device_markers_per_core_risc_map;

    std::set<tracy::TTDeviceMarker> device_sync_markers;

    std::set<tracy::TTDeviceMarker> device_sync_new_markers;

    // shift
    int64_t shift = 0;

    // frequency scale
    double freq_scale = 1.0;

    // Freshen device logs
    void freshDeviceLog();

    // Change the output dir of device profile logs
    void setOutputDir(const std::string& new_output_dir);

    // Traverse all cores on the device and read the device profile results
    void readResults(
        IDevice* device,
        const std::vector<CoreCoord>& virtual_cores,
        ProfilerReadState state = ProfilerReadState::NORMAL,
        ProfilerDataBufferSource data_source = ProfilerDataBufferSource::DRAM,
        const std::optional<ProfilerOptionalMetadata>& metadata = {});

    // Process the device profile results previously read
    void processResults(
        IDevice* device,
        const std::vector<CoreCoord>& virtual_cores,
        ProfilerReadState state = ProfilerReadState::NORMAL,
        ProfilerDataBufferSource data_source = ProfilerDataBufferSource::DRAM,
        const std::optional<ProfilerOptionalMetadata>& metadata = {});

    void dumpRoutingInfo() const;

    void dumpClusterCoordinates() const;

    // Dump device results to files
    void dumpDeviceResults() const;

    // Push device results to tracy
    void pushTracyDeviceResults(std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers_vec);

    // Update sync info for this device
    void setSyncInfo(const SyncInfo& sync_info);

    // Get marker details for the marker corresponding to the given timer id
    tracy::MarkerDetails getMarkerDetails(uint16_t timer_id) const;

    // setter and getter on last fast dispatch read
    void setLastFDReadAsDone();

    void setLastFDReadAsNotDone();

    bool isLastFDReadDone() const;
};

// Merges markers from each (physical core, risc type) group into a single sorted vector. The markers in each group
// should already be sorted.
//
// IMPORTANT: This function creates a vector of references to the TTDeviceMarker objects stored in
// device_markers_per_core_risc_map. These are direct references to the original objects, not copies of the data.
// Thread safety warning: device_markers_per_core_risc_map MUST NOT be modified (no insertions, deletions, or rehashing)
// while these references are in use, as this could invalidate the references and cause undefined behavior.
std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>> getSortedDeviceMarkersVector(
    const std::map<CoreCoord, std::map<tracy::RiscType, std::set<tracy::TTDeviceMarker>>>&
        device_markers_per_core_risc_map);

bool useFastDispatch(IDevice* device);

void writeToCoreControlBuffer(IDevice* device, const CoreCoord& virtual_core, const std::vector<uint32_t>& data);

}  // namespace tt_metal

}  // namespace tt
