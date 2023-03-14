#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <filesystem>

#include "tt_metal/impl/device/device.hpp"
#include "llrt/llrt.hpp"

using std::chrono::steady_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

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

class Profiler {
    private:
        std::unordered_map <std::string, TimerPeriod> name_to_timer_map;

        // Recreate host side log file with header
        bool host_new_log;

        // Recreate device side log file with header
        bool device_new_log;

        // Output Dir for Profile Logs
        std::filesystem::path output_dir;

        // Turn steady clock start and stop into integer start, stop and duration
        TimerPeriodInt timerToTimerInt(TimerPeriod period);

        // Dumping profile result to file
        void dumpDeviceResultToFile(
                int chip_id,
                int core_x,
                int core_y,
                std::string hart_name,
                uint64_t timestamp,
                uint32_t timer_id);

        // Helper function for reading risc profile results
        void readRiscProfilerResults(
                tt::tt_metal::Device *device,
                const tt_xy_pair &logical_core,
                std::string risc_name,
                int risc_print_buffer_addr);

    public:
        //Constructor
        Profiler();

        //Mark the steady_clock for the start of the asked name
        void markStart(std::string timer_name);

        //Mark the steady_clock time for the end of the asked name
        void markStop(std::string timer_name);

        //Set the host side file flag
        void setHostNewLogFlag(bool new_log_flag);

        //Set the device side file flag
        void setDeviceNewLogFlag(bool new_log_flag);

        //Change the output dir of the profile logs
        void setOutputDir(std::string new_output_dir);

        //Traverse all timers and dump the results
        void dumpHostResults(std::string name_append);

        //Traverse all cores on the device and dump the device profile results
        void dumpDeviceResults(tt::tt_metal::Device *device, const vector<tt_xy_pair> &logical_cores);
};
