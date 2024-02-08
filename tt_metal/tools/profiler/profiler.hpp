// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <filesystem>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "llrt/llrt.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "tools/profiler/common.hpp"
#include "tt_metal/third_party/tracy/public/tracy/TracyTTDevice.hpp"
#include "tt_metal/third_party/tracy/public/common/TracyTTDeviceData.hpp"

using std::chrono::steady_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

namespace tt {

namespace tt_metal {

// struct for holding start, stop and duration of a timer period in integer format
struct TimerPeriodInt {
    uint64_t start;
    uint64_t stop;
    uint64_t delta;
};

// struct for holding start, stop of a timer in steady_clock::time_point format
struct TimerPeriod {
    steady_clock::time_point start;
    steady_clock::time_point stop;
};


class HostProfiler {
    private:

        // Holds name to timers
        std::unordered_map <std::string, TimerPeriod> name_to_timer_map;

        // Recreate host side log file with header
        bool new_log;

        // Output Dir for Profile Logs
        std::filesystem::path output_dir;

        // Turn steady clock start and stop into integer start, stop and duration
        TimerPeriodInt timerToTimerInt(TimerPeriod period);

        //Traverse all timers and dump the results, appending addtional fields
        void dumpResults(
                const std::string& timer_name,
                const std::vector<std::pair<std::string,std::string>>& additional_fields = {});

    public:
        //Constructor
        HostProfiler();

        //Mark the steady_clock for the start of the asked name
        void markStart(const std::string& timer_name);

        //Mark the steady_clock time for the end of the asked name
        void markStop(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields = {});

        //Set the host side file flag
        void setNewLogFlag(bool new_log_flag);

        //Change the output dir of host profile logs
        void setOutputDir(const std::string& new_output_dir);
};

class DeviceProfiler {
    private:

        // Recreate device side log file with header
        bool new_log;

        // Device architecture
        tt::ARCH device_architecture;

        // Device frequency
        int device_core_frequency;

        //Smallest timestamp
        uint64_t smallest_timestamp = (1lu << 63);

        // Output Dir for device Profile Logs
        std::filesystem::path output_dir;

        // Global custom marker counter
        uint32_t customMarkerCount = 0;

        // Device-Core tracy context
        std::map<std::pair<uint16_t,CoreCoord>, TracyTTCtx> device_tracy_contexts;

        // Device-Core tracy context
        std::vector<tracy::TTDeviceEvent> device_events;

        // Hash to zone source locations
        std::unordered_map<uint16_t, std::string> hash_to_zone_src_locations;

        // Zone sourece locations
        std::unordered_set<std::string> zone_src_locations;

        //32bit FNV-1a hashing
        uint32_t hash32CT( const char * str, size_t n, uint32_t basis = UINT32_C( 2166136261 ) );

        // XORe'd 16-bit FNV-1a hashing functions
        uint16_t hash16CT( const std::string& str);

        // Iterate through all zone source locations and generate hash
        void generateZoneSourceLocationsHashes();

        // Dumping profile result to file
        void dumpResultToFile(
                uint32_t runID,
                int device_id,
                CoreCoord core,
                int core_flat,
                int risc_num,
                uint32_t timer_id,
                uint64_t timestamp
                );

        // Helper function for reading risc profile results
        void readRiscProfilerResults(
                int device_id,
                vector<std::uint32_t> profile_buffer,
                const CoreCoord &worker_core);

        //Push device results to tracy
        void pushTracyDeviceResults();

        //Track the smallest timestamp dumped to file
        void firstTimestamp(uint64_t timestamp);

    public:
        //Constructor
        DeviceProfiler();

        //DRAM buffer for device side results
        std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = nullptr;


        //Set the device side file flag
        void setNewLogFlag(bool new_log_flag);

        //Set the device architecture
        void setDeviceArchitecture(tt::ARCH device_arch);

        //Change the output dir of device profile logs
        void setOutputDir(const std::string& new_output_dir);

        //Traverse all cores on the device and dump the device profile results
        void dumpResults(Device *device, const vector<CoreCoord> &worker_cores);
};

}  // namespace tt_metal

}  // namespace tt
