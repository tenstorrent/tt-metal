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
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "buffer.hpp"
#include "program.hpp"
#include "common/TracyTTDeviceData.hpp"
#include "core_coord.hpp"
#include "hostdevcommon/profiler_common.h"
#include "profiler_optional_metadata.hpp"
#include "profiler_types.hpp"
#include "tt-metalium/program.hpp"
#include "tracy/TracyTTDevice.hpp"

namespace tt {
enum class ARCH;
namespace tt_metal {
class IDevice;
class Program;
}  // namespace tt_metal
}  // namespace tt

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

// defined locally in profiler.cpp
class FabricRoutingLookup;

struct DisptachMetaData {
    // Dispatch command queue command type
    std::string cmd_type = "";

    // Worker's runtime id
    uint32_t worker_runtime_id = 0;

    // dispatch command subtype.
    std::string cmd_subtype = "";
};

struct ZoneDetails {
    enum class ZoneNameKeyword : uint16_t {
        BRISC_FW,
        ERISC_FW,
        SYNC_ZONE,
        PROFILER,
        DISPATCH,
        PROCESS_CMD,
        RUNTIME_HOST_ID_DISPATCH,
        PACKED_DATA_DISPATCH,
        PACKED_LARGE_DATA_DISPATCH,
        COUNT
    };

    static inline std::unordered_map<std::string, ZoneNameKeyword> zone_name_keywords_map = {
        {"BRISC-FW", ZoneNameKeyword::BRISC_FW},
        {"ERISC-FW", ZoneNameKeyword::ERISC_FW},
        {"SYNC-ZONE", ZoneNameKeyword::SYNC_ZONE},
        {"PROFILER", ZoneNameKeyword::PROFILER},
        {"DISPATCH", ZoneNameKeyword::DISPATCH},
        {"process_cmd", ZoneNameKeyword::PROCESS_CMD},
        {"runtime_host_id_dispatch", ZoneNameKeyword::RUNTIME_HOST_ID_DISPATCH},
        {"packed_data_dispatch", ZoneNameKeyword::PACKED_DATA_DISPATCH},
        {"packed_large_data_dispatch", ZoneNameKeyword::PACKED_LARGE_DATA_DISPATCH},
    };

    std::string zone_name;
    std::string source_file;
    uint64_t source_line_num;
    std::array<bool, static_cast<uint16_t>(ZoneNameKeyword::COUNT)> zone_name_keyword_flags;

    ZoneDetails(const std::string& zone_name, const std::string& source_file, uint64_t source_line_num) :
        zone_name(zone_name), source_file(source_file), source_line_num(source_line_num) {
        for (const auto& [keyword_str, keyword] : zone_name_keywords_map) {
            zone_name_keyword_flags[static_cast<uint16_t>(keyword)] = zone_name.find(keyword_str) != std::string::npos;
        }
    }
};

const ZoneDetails UnidentifiedZoneDetails = ZoneDetails("", "", 0);

struct SyncInfo {
    double cpu_time = 0.0;
    double device_time = 0.0;
    double frequency = 0.0;

    SyncInfo(double cpu_time, double device_time, double frequency) :
        cpu_time(cpu_time), device_time(device_time), frequency(frequency) {}

    SyncInfo() : SyncInfo(0.0, 0.0, 0.0) {}
};

class DeviceProfiler {
private:
    // Device architecture
    tt::ARCH device_architecture;

    // Device frequency
    int device_core_frequency;

    // Smallest timestamp
    uint64_t smallest_timestamp = (1lu << 63);

    // Output Dir for device Profile Logs
    std::filesystem::path output_dir;

    // Device-Core tracy context
    std::unordered_map<std::pair<uint16_t, CoreCoord>, TracyTTCtx, pair_hash<uint16_t, CoreCoord>>
        device_tracy_contexts;

    // Hash to zone source locations
    std::unordered_map<uint16_t, ZoneDetails> hash_to_zone_src_locations;

    // Zone sourece locations
    std::unordered_set<std::string> zone_src_locations;

    // Iterator on the current zone being processed
    std::unordered_set<tracy::TTDeviceEvent>::iterator current_zone_it;

    // Holding current data collected for dispatch command queue zones
    DisptachMetaData current_dispatch_meta_data;

    // (cpu time, device time, frequency) for sync propagated from root device
    SyncInfo device_sync_info;

    // Per-core sync info used to make tracy context
    std::unordered_map<CoreCoord, SyncInfo> core_sync_info;

    // (Device ID, Core Coord) pairs that keep track of cores which need to have their Tracy contexts updated
    std::unordered_set<std::pair<chip_id_t, CoreCoord>, pair_hash<chip_id_t, CoreCoord>> device_cores;

    // Storage for all core's control buffers
    std::unordered_map<CoreCoord, std::vector<uint32_t>> core_control_buffers;

    // 32bit FNV-1a hashing
    uint32_t hash32CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261));

    // XORe'd 16-bit FNV-1a hashing functions
    uint16_t hash16CT(const std::string& str);

    void populateZoneSrcLocations(
        const std::string& new_log_name, const std::string& log_name = "", bool push_new = false);

    // Iterate through all zone source locations and generate hash
    void generateZoneSourceLocationsHashes();

    // serialize all noc trace data into per-op json trace files
    void serializeJsonNocTraces(
        const nlohmann::ordered_json& noc_trace_json_log,
        const std::filesystem::path& output_dir,
        chip_id_t device_id,
        const FabricRoutingLookup& routing_lookup);

    void emitCSVHeader(
        std::ofstream& log_file_ofs, const tt::ARCH& device_architecture, int device_core_frequency) const;

    // translates potentially-virtual coordinates recorded on Device into physical coordinates
    CoreCoord getPhysicalAddressFromVirtual(chip_id_t device_id, const CoreCoord& c) const;

    // Read all control buffers
    void readControlBuffers(
        IDevice* device, const std::vector<CoreCoord>& virtual_cores, ProfilerDumpState state);

    // Read control buffer for a single core
    void readControlBufferForCore(IDevice* device, const CoreCoord& virtual_core, ProfilerDumpState state);

    // Reset all control buffers
    void resetControlBuffers(
        IDevice* device, const std::vector<CoreCoord>& virtual_cores, ProfilerDumpState state);

    // Reset control buffer for a single core
    void resetControlBufferForCore(IDevice* device, const CoreCoord& virtual_core, ProfilerDumpState state);

    // Read all L1 data buffers
    void readL1DataBuffers(
        IDevice* device,
        const std::vector<CoreCoord>& virtual_cores,
        ProfilerDumpState state,
        std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_l1_data_buffers);

    // Read L1 data buffer for a single core
    void readL1DataBufferForCore(
        IDevice* device,
        const CoreCoord& virtual_core,
        ProfilerDumpState state,
        std::vector<uint32_t>& core_l1_data_buffer);

    // Read device profiler buffer
    void readProfilerBuffer(IDevice* device, ProfilerDumpState state);

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

    // Dumping profile result to file
    void logPacketData(
        std::ofstream& log_file_ofs,
        nlohmann::ordered_json& noc_trace_json_log,
        uint32_t runHostID,
        const std::string& opname,
        chip_id_t device_id,
        CoreCoord core,
        int risc_num,
        uint64_t stat_value,
        uint32_t timer_id,
        uint64_t timestamp);

    // logs packet data to CSV file
    void logPacketDataToCSV(
        std::ofstream& log_file_ofs,
        chip_id_t device_id,
        int core_x,
        int core_y,
        std::string_view risc_name,
        uint32_t timer_id,
        uint64_t timestamp,
        uint64_t data,
        uint32_t run_host_id,
        std::string_view zone_name,
        kernel_profiler::PacketTypes packet_type,
        uint64_t source_line,
        std::string_view source_file,
        const nlohmann::json& metaData);

    // dump noc trace related profile data to json file
    void logNocTracePacketDataToJson(
        nlohmann::ordered_json& noc_trace_json_log,
        chip_id_t device_id,
        int core_x,
        int core_y,
        std::string_view risc_name,
        uint64_t timestamp,
        uint64_t data,
        uint32_t run_host_id,
        std::string_view opname,
        std::string_view zone_name,
        kernel_profiler::PacketTypes packet_type);

    // Helper function for reading risc profile results
    void readRiscProfilerResults(
        IDevice* device,
        const CoreCoord& worker_core,
        ProfilerDumpState state,
        const std::vector<uint32_t>& data_buffer,
        ProfilerDataBufferSource data_source,
        const std::optional<ProfilerOptionalMetadata>& metadata,
        std::ofstream& log_file_ofs,
        nlohmann::ordered_json& noc_trace_json_log);

    // Track the smallest timestamp dumped to file
    void firstTimestamp(uint64_t timestamp);

    // Get tracy context for the core
    void updateTracyContext(std::pair<uint32_t, CoreCoord> device_core);

public:
    DeviceProfiler(const IDevice* device, bool new_logs);

    DeviceProfiler() = delete;

    ~DeviceProfiler();

    // Device-core Syncdata
    std::map<CoreCoord, SyncInfo> device_core_sync_info;

    // DRAM Vector
    std::vector<uint32_t> profile_buffer;

    // Number of bytes reserved in each DRAM bank for storing device profiling data
    uint32_t profile_buffer_bank_size_bytes;

    // Device events
    std::unordered_set<tracy::TTDeviceEvent> device_events;

    std::set<tracy::TTDeviceEvent> device_sync_events;

    std::set<tracy::TTDeviceEvent> device_sync_new_events;

    // shift
    int64_t shift = 0;

    // frequency scale
    double freq_scale = 1.0;

    // Freshen device logs
    void freshDeviceLog();

    // Set the device architecture
    void setDeviceArchitecture(tt::ARCH device_arch);

    // Change the output dir of device profile logs
    void setOutputDir(const std::string& new_output_dir);

    // Traverse all cores on the device and dump the device profile results
    void dumpResults(
        IDevice* device,
        const std::vector<CoreCoord>& virtual_cores,
        ProfilerDumpState state = ProfilerDumpState::NORMAL,
        ProfilerDataBufferSource data_source = ProfilerDataBufferSource::DRAM,
        const std::optional<ProfilerOptionalMetadata>& metadata = {});

    // Push device results to tracy
    void pushTracyDeviceResults();

    // Update sync info for this device
    void setSyncInfo(const SyncInfo& sync_info);

    // Get zone details for the zone corresponding to the given timer id
    ZoneDetails getZoneDetails(uint16_t timer_id) const;
};

bool useFastDispatchForControlBuffers(const IDevice* device, ProfilerDumpState state);

void writeToCoreControlBuffer(
    IDevice* device, const CoreCoord& virtual_core, ProfilerDumpState state, const std::vector<uint32_t>& data);

bool onlyProfileDispatchCores(ProfilerDumpState state);

bool isGalaxyMMIODevice(const IDevice* device);

}  // namespace tt_metal

}  // namespace tt
