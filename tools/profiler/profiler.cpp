#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

#include "tools/profiler/profiler.hpp"

#define HOST_SIDE_LOG "profile_log.csv"
#define DEVICE_SIDE_LOG "profile_log_kernel.csv"

Profiler::Profiler()
{
    firstRun = true;
    output_dir = std::filesystem::path("tools/profiler");
}

void Profiler::setOutputDir(std::string new_output_dir)
 {
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
 }

TimerPeriodInt Profiler::timerToTimerInt(TimerPeriod period)
{
    TimerPeriodInt ret;

    ret.start = duration_cast<nanoseconds>(period.start.time_since_epoch()).count();
    ret.stop = duration_cast<nanoseconds>(period.stop.time_since_epoch()).count();
    ret.delta = duration_cast<nanoseconds>(period.stop - period.start).count();

    return ret;
}

void Profiler::markStart(std::string timer_namer)
{
    name_to_timer_map[timer_namer].start = steady_clock::now();
}

void Profiler::markStop(std::string timer_namer)
{
    name_to_timer_map[timer_namer].stop = steady_clock::now();
}

void Profiler::dumpResults(std::string name_append, bool add_header)
{
    const int large_width = 30;
    const int medium_width = 25;

    std::filesystem::path log_path = output_dir / HOST_SIDE_LOG;
    std::ofstream log_file;

    if (firstRun || add_header)
    {
        log_file.open(log_path);

        log_file << "Section Name" << ", ";
        log_file << "Function Name" << ", ";
        log_file << "Start timer count [ns]"  << ", ";
        log_file << "Stop timer count [ns]"  << ", ";
        log_file << "Delta timer count [ns]";
        log_file << std::endl;
        firstRun = false;
    }
    else
    {
        log_file.open(log_path,  std::ios_base::app);
    }

    for (auto timer : name_to_timer_map)
    {
        auto timer_period_ns = timerToTimerInt(timer.second);
        log_file << name_append << ", ";
        log_file << timer.first << ", ";
        log_file << timer_period_ns.start  << ", ";
        log_file << timer_period_ns.stop  << ", ";
        log_file << timer_period_ns.delta;
        log_file << std::endl;
    }

    log_file.close();

    name_to_timer_map.clear();
}

void Profiler::dumpKernelResults(
        int chip_id,
        int core_x,
        int core_y,
        std::string hart_name,
        uint64_t timestamp,
        uint32_t timer_id)
{
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    if (firstRun)
    {
        log_file.open(log_path);
        log_file << "Chip clock is at 1.2 GHz" << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        firstRun = false;
    }
    else
    {
        log_file.open(log_path,  std::ios_base::app);
    }
    constexpr int DRAM_ROW = 6;
    if (core_y > DRAM_ROW){
       core_y = core_y - 2;
    }
    else{
       core_y--;
    }
    core_x--;
    log_file << chip_id << ", " << core_x << ", " << core_y << ", " << hart_name << ", ";
    log_file << timer_id << ", ";
    log_file << timestamp;
    log_file << std::endl;
    log_file.close();
}

void Profiler::kernelProfilerCallback(
        std::ostream& stream,
        int chip_id,
        int core_x,
        int core_y,
        std::string hart_name,
        uint64_t timestamp,
        uint32_t timer_id,
        bool add_header)
{
    if (add_header)
    {
        stream << "Chip clock is at 1.2 GHz" << std::endl;
        stream << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
    }
    constexpr int DRAM_ROW = 6;
    if (core_y > DRAM_ROW){
       core_y = core_y - 2;
    }
    else{
       core_y--;
    }
    core_x--;
    stream << chip_id << ", " << core_x << ", " << core_y << ", " << hart_name << ", ";
    stream << timer_id << ", ";
    stream << timestamp;
    stream << std::endl;
}

std::string Profiler::getKernelProfilerLogName()
{
    return output_dir / DEVICE_SIDE_LOG;
}
