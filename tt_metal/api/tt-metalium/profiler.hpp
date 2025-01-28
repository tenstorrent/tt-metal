// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <filesystem>

#include "buffer.hpp"
#include "program_impl.hpp"
#include "profiler_state.hpp"
#include "common.hpp"
#include "tracy/TracyTTDevice.hpp"
#include "common/TracyTTDeviceData.hpp"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;

namespace tt {

namespace tt_metal {

enum class ProfilerDumpState { NORMAL, CLOSE_DEVICE_SYNC, LAST_CLOSE_DEVICE };
enum class ProfilerSyncState { INIT, CLOSE_DEVICE };

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

    // 32bit FNV-1a hashing
    uint32_t hash32CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261));

    // XORe'd 16-bit FNV-1a hashing functions
    uint16_t hash16CT(const std::string& str);

    // Iterate through all zone source locations and generate hash
    void generateZoneSourceLocationsHashes();

    // Dumping profile result to file
    void dumpResultToFile(
        uint32_t runID,
        uint32_t runHostID,
        int device_id,
        CoreCoord core,
        int core_flat,
        int risc_num,
        uint64_t stat_value,
        uint32_t timer_id,
        uint64_t timestamp);

    // Helper function for reading risc profile results
    void readRiscProfilerResults(
        IDevice* device, CoreCoord& worker_core);

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

    //Device events
    std::set<tracy::TTDeviceEvent> device_events;

    std::set<tracy::TTDeviceEvent> device_sync_events;

    std::set<tracy::TTDeviceEvent> device_sync_new_events;

    //shift
    int64_t shift = 0;

    //frequency scale
    double freqScale = 1.0;

    uint32_t my_device_id = 0;

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
        ProfilerDumpState state = ProfilerDumpState::NORMAL);
};

}  // namespace tt_metal

}  // namespace tt
