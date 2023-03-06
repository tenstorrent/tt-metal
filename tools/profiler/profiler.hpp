#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <filesystem>

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

        // Turn steady clock start and stop into integer start, stop and duration
        TimerPeriodInt timerToTimerInt(TimerPeriod period);

        // First Run
        bool firstRun;

        // Output Dir for Profile Logs
        std::filesystem::path output_dir;

    public:
        //Constructor
        Profiler();

        //Mark the steady_clock for the start of the asked name
        void markStart(std::string timer_name);

        //Mark the steady_clock time for the end of the asked name
        void markStop(std::string timer_name);

        //Change the output dir of the profile logs
        void setOutputDir(std::string new_output_dir);

        //Traverse all timers and dump the results
        void dumpResults(std::string name_append, bool add_header=false);

        //Get kernel profile log filename
        std::string getKernelProfilerLogName();

        //Callback on receiving profiler data from kernels
        static void kernelProfilerCallback(
            std::ostream& stream,
            int chip_id,
            int core_x,
            int core_y,
            std::string hart_name,
            uint64_t timestamp,
            uint32_t timer_id,
            bool add_header);

};
