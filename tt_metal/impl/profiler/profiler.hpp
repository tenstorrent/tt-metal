// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include <stdint.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "buffer.hpp"
#include "mesh_buffer.hpp"
#include "program.hpp"
#include "common/TracyTTDeviceData.hpp"
#include "core_coord.hpp"
#include "hostdevcommon/profiler_common.h"
#include "profiler_optional_metadata.hpp"
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
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

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;

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
    std::string zone_name;
    std::string source_file;
    uint64_t source_line_num;
    bool is_zone_in_brisc_or_erisc;
};

const ZoneDetails UnidentifiedZoneDetails = ZoneDetails{"", "", 0, false};

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
    std::tuple<double, double, double> device_sync_info;

    // Per-core sync info used to make tracy context
    std::map<CoreCoord, std::tuple<double, double, double>> core_sync_info;

    // 32bit FNV-1a hashing
    uint32_t hash32CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261));

    // XORe'd 16-bit FNV-1a hashing functions
    uint16_t hash16CT(const std::string& str);

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

    ZoneDetails getZoneDetails(uint16_t timer_id) const;

    // Storage for all core's control buffers
    std::unordered_map<CoreCoord, std::vector<uint32_t>> core_control_buffers;

    // Read all control buffers
    void readControlBuffers(IDevice* device, const CoreCoord& worker_core, const ProfilerDumpState state);

    // reset control buffers
    void resetControlBuffers(IDevice* device, const CoreCoord& worker_core, const ProfilerDumpState state);

    // Dumping profile result to file
    void logPacketData(
        std::ofstream& log_file_ofs,
        nlohmann::ordered_json& noc_trace_json_log,
        uint32_t runHostID,
        const std::string& opname,
        chip_id_t device_id,
        CoreCoord core,
        int core_flat,
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
        const std::string_view risc_name,
        uint32_t timer_id,
        uint64_t timestamp,
        uint64_t data,
        uint32_t run_host_id,
        const std::string_view opname,
        const std::string_view zone_name,
        kernel_profiler::PacketTypes packet_type,
        uint64_t source_line,
        const std::string_view source_file,
        const nlohmann::json& metaData);

    // dump noc trace related profile data to json file
    void logNocTracePacketDataToJson(
        nlohmann::ordered_json& noc_trace_json_log,
        chip_id_t device_id,
        int core_x,
        int core_y,
        const std::string_view risc_name,
        uint32_t timer_id,
        uint64_t timestamp,
        uint64_t data,
        uint32_t run_host_id,
        const std::string_view opname,
        const std::string_view zone_name,
        kernel_profiler::PacketTypes packet_type,
        uint64_t source_line,
        const std::string_view source_file);

    // Helper function for reading risc profile results
    void readRiscProfilerResults(
        IDevice* device,
        const CoreCoord& worker_core,
        const ProfilerDumpState state,
        const std::vector<uint32_t>& data_buffer,
        const ProfilerDataBufferSource data_source,
        const std::optional<ProfilerOptionalMetadata>& metadata,
        std::ofstream& log_file_ofs,
        nlohmann::ordered_json& noc_trace_json_log);

    // Track the smallest timestamp dumped to file
    void firstTimestamp(uint64_t timestamp);

    // Get tracy context for the core
    void updateTracyContext(std::pair<uint32_t, CoreCoord> device_core);

public:
    DeviceProfiler(const IDevice* device, const bool new_logs);

    DeviceProfiler() = delete;

    ~DeviceProfiler();

    std::shared_ptr<tt::tt_metal::Program> sync_program = nullptr;

    // Device-core Syncdata
    std::map<CoreCoord, std::tuple<double, double, double>> device_core_sync_info;

    // DRAM Vector
    std::vector<uint32_t> profile_buffer;

    // Number of bytes reserved in each DRAM bank for storing device profiling data
    uint32_t profile_buffer_bank_size_bytes;

    // (Device ID, Core Coord) pairs that keep track of cores which need to have their Tracy contexts updated
    std::unordered_set<std::pair<chip_id_t, CoreCoord>, pair_hash<chip_id_t, CoreCoord>> device_cores;

    // Device events
    std::unordered_set<tracy::TTDeviceEvent> device_events;

    std::set<tracy::TTDeviceEvent> device_sync_events;

    std::set<tracy::TTDeviceEvent> device_sync_new_events;

    // shift
    int64_t shift = 0;

    // frequency scale
    double freqScale = 1.0;

    // Freshen device logs
    void freshDeviceLog();

    // Set the device architecture
    void setDeviceArchitecture(tt::ARCH device_arch);

    // Change the output dir of device profile logs
    void setOutputDir(const std::string& new_output_dir);

    // Traverse all cores on the device and dump the device profile results
    void dumpResults(
        IDevice* device,
        const std::vector<CoreCoord>& worker_cores,
        const ProfilerDumpState state = ProfilerDumpState::NORMAL,
        const ProfilerDataBufferSource data_source = ProfilerDataBufferSource::DRAM,
        const std::optional<ProfilerOptionalMetadata>& metadata = {});

    // Push device results to tracy
    void pushTracyDeviceResults();

    // Update sync info for this device
    void setSyncInfo(const std::tuple<double, double, double>& sync_info);

    // Read data from profiler buffer using fast dispatch
    void issueFastDispatchReadFromProfilerBuffer(IDevice* device);

    // Read data from profiler buffer using slow dispatch
    void issueSlowDispatchReadFromProfilerBuffer(IDevice* device);
};

void write_control_buffer_to_core(
    IDevice* device,
    const CoreCoord& core,
    const HalProgrammableCoreType core_type,
    const ProfilerDumpState state,
    const std::vector<uint32_t>& control_buffer);

}  // namespace tt_metal

}  // namespace tt
