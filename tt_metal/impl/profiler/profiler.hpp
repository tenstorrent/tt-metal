// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include <stdint.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
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
#include "common/TracyTTDeviceData.hpp"
#include "core_coord.hpp"
#include "hostdevcommon/profiler_common.h"
#include "profiler_optional_metadata.hpp"
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "profiler_types.hpp"
#include "program_impl.hpp"
#include "system_memory_manager.hpp"
#include "tracy/TracyTTDevice.hpp"

namespace tt {
enum class ARCH;
namespace tt_metal {
class Buffer;
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

struct DisptachMetaData {
    // Dispatch command queue command type
    std::string cmd_type = "";

    // Worker's runtime id
    uint32_t worker_runtime_id = 0;

    // dispatch command subtype.
    std::string cmd_subtype = "";
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
    std::map<std::pair<uint16_t, CoreCoord>, TracyTTCtx> device_tracy_contexts;

    // Hash to zone source locations
    std::unordered_map<uint16_t, std::string> hash_to_zone_src_locations;

    // Zone sourece locations
    std::unordered_set<std::string> zone_src_locations;

    // Iterator on the current zone being processed
    std::set<tracy::TTDeviceEvent>::iterator current_zone_it;

    // Holding current data collected for dispatch command queue zones
    DisptachMetaData current_dispatch_meta_data;

    // 32bit FNV-1a hashing
    uint32_t hash32CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261));

    // XORe'd 16-bit FNV-1a hashing functions
    uint16_t hash16CT(const std::string& str);

    // Iterate through all zone source locations and generate hash
    void generateZoneSourceLocationsHashes();

    // serialize all noc trace data into per-op json trace files
    void serializeJsonNocTraces(
        const nlohmann::ordered_json& noc_trace_json_log, const std::filesystem::path& output_dir, chip_id_t device_id);

    void emitCSVHeader(
        std::ofstream& log_file_ofs, const tt::ARCH& device_architecture, int device_core_frequency) const;

    // translates potentially-virtual coordinates recorded on Device into physical coordinates
    CoreCoord getPhysicalAddressFromVirtual(chip_id_t device_id, const CoreCoord& c) const;

    // Dumping profile result to file
    void logPacketData(
        std::ofstream& log_file_ofs,
        nlohmann::ordered_json& noc_trace_json_log,
        uint32_t runID,
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
        uint32_t run_id,
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
        uint32_t run_id,
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
        const std::optional<ProfilerOptionalMetadata>& metadata,
        std::ofstream& log_file_ofs,
        nlohmann::ordered_json& noc_trace_json_log);

    // Push device results to tracy
    void pushTracyDeviceResults();

    // Track the smallest timestamp dumped to file
    void firstTimestamp(uint64_t timestamp);

public:
    DeviceProfiler(const bool new_logs);

    DeviceProfiler() = delete;

    ~DeviceProfiler();

    // DRAM buffer for device side results
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = nullptr;
    std::shared_ptr<tt::tt_metal::Program> sync_program = nullptr;

    // Device-core Syncdata
    std::map<CoreCoord, std::tuple<double, double, double>> device_core_sync_info;

    // DRAM Vector
    std::vector<uint32_t> profile_buffer;

    // Device events
    std::set<tracy::TTDeviceEvent> device_events;

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
        ProfilerDumpState state = ProfilerDumpState::NORMAL,
        const std::optional<ProfilerOptionalMetadata>& metadata = {});
};

}  // namespace tt_metal

}  // namespace tt
