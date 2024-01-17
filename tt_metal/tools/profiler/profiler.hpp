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
#include "tt_metal/third_party/tracy/public/tracy/TracyOpenCL.hpp"

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

        // Dumping profile result to file
        void dumpResultToFile(
                bool debug,
                uint32_t runID,
                int chip_id,
                int core_x,
                int core_y,
                int core_flat,
                int core_flat_read,
                int core_flat_read_ts,
                int risc_num,
                int risc_num_read,
                int risc_num_read_ts,
                uint32_t timer_id,
                uint64_t timestamp,
                std::map<uint32_t, std::map<tracy::TTDeviceEvent, uint64_t, tracy::TTDeviceEvent_cmp>> &device_data
                );

        // Helper function for reading risc profile results
        void readRiscProfilerResults(
                int device_id,
                vector<std::uint32_t> profile_buffer,
                const CoreCoord &worker_core,
                std::map<uint32_t, std::map<tracy::TTDeviceEvent, uint64_t, tracy::TTDeviceEvent_cmp>> &device_data
                );

        //Push device results to tracy
        void pushTracyDeviceResults(
                TracyCLCtx tracyTTCtx,
                std::map<uint32_t, std::map<tracy::TTDeviceEvent, uint64_t, tracy::TTDeviceEvent_cmp>> &device_data,
                bool isPopulated
                );

        //Track the smallest timestamp dumped to file
        void firstTimestamp(uint64_t timestamp);

    public:
        //Constructor
        DeviceProfiler();

        //DRAM buffer for device side results
        Buffer output_dram_buffer;

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
