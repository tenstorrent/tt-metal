// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

#include "tools/profiler/profiler.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "hostdevcommon/profiler_common.h"
#include "llrt/rtoptions.hpp"

#define HOST_SIDE_LOG "profile_log_host.csv"
#define DEVICE_SIDE_LOG "profile_log_device.csv"

namespace tt {

namespace tt_metal {

TimerPeriodInt Profiler::timerToTimerInt(TimerPeriod period)
{
    TimerPeriodInt ret;

    ret.start = duration_cast<nanoseconds>(period.start.time_since_epoch()).count();
    ret.stop = duration_cast<nanoseconds>(period.stop.time_since_epoch()).count();
    ret.delta = duration_cast<nanoseconds>(period.stop - period.start).count();

    return ret;
}

void Profiler::dumpHostResults(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
    auto timer = name_to_timer_map[timer_name];

    auto timer_period_ns = timerToTimerInt(timer);
    TT_FATAL (timer_period_ns.start != 0 , "Timer start cannot be zero on : " + timer_name);
    TT_FATAL (timer_period_ns.stop != 0 , "Timer stop cannot be zero on : " + timer_name);

    std::filesystem::path log_path = output_dir / HOST_SIDE_LOG;
    std::ofstream log_file;

    if (host_new_log|| !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);

        log_file << "Name" << ", ";
        log_file << "Start timer count [ns]"  << ", ";
        log_file << "Stop timer count [ns]"  << ", ";
        log_file << "Delta timer count [ns]";

        for (auto &field: additional_fields)
        {
            log_file  << ", "<< field.first;
        }

        log_file << std::endl;
        host_new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    log_file << timer_name << ", ";
    log_file << timer_period_ns.start  << ", ";
    log_file << timer_period_ns.stop  << ", ";
    log_file << timer_period_ns.delta;

    for (auto &field: additional_fields)
    {
        log_file  << ", "<< field.second;
    }

    log_file << std::endl;

    log_file.close();
}

void Profiler::readRiscProfilerResults(
    int device_id, const CoreCoord &physical_core, std::string risc_name, int risc_print_buffer_addr) {
    vector<std::uint32_t> profile_buffer;
    uint32_t end_index;
    uint32_t dropped_marker_counter;

    profile_buffer =
        tt::llrt::read_hex_vec_from_core(device_id, physical_core, risc_print_buffer_addr, PRINT_BUFFER_SIZE);

    end_index = profile_buffer[kernel_profiler::BUFFER_END_INDEX];
    TT_ASSERT (end_index < (PRINT_BUFFER_SIZE/sizeof(uint32_t)));
    dropped_marker_counter = profile_buffer[kernel_profiler::DROPPED_MARKER_COUNTER];

    if(dropped_marker_counter > 0){
        log_debug(
            tt::LogDevice,
            "{} device markers on device {} physical core {},{} risc {} were dropped. End index {}",
            dropped_marker_counter,
            device_id,
            physical_core.x,
            physical_core.y,
            risc_name,
            end_index);
    }

    for (int i = kernel_profiler::MARKER_DATA_START; i < end_index; i+=kernel_profiler::TIMER_DATA_UINT32_SIZE) {
        dumpDeviceResultToFile(
            device_id,
            physical_core.x,
            physical_core.y,
            risc_name,
            (uint64_t(profile_buffer[i + kernel_profiler::TIMER_VAL_H]) << 32) |
                profile_buffer[i + kernel_profiler::TIMER_VAL_L],
            profile_buffer[i + kernel_profiler::TIMER_ID]);
    }
}

void Profiler::dumpDeviceResultToFile(
        int chip_id,
        int core_x,
        int core_y,
        std::string hart_name,
        uint64_t timestamp,
        uint32_t timer_id){

    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    if (device_new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "ARCH: " << get_string_lowercase(device_architecture) << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        device_new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    if (hart_name != "ERISC") {
        constexpr int DRAM_ROW = 6;
        if (core_y > DRAM_ROW) {
            core_y = core_y - 2;
        } else {
            core_y--;
        }
        core_x--;
    }

    log_file << chip_id << ", " << core_x << ", " << core_y << ", " << hart_name << ", ";
    log_file << timer_id << ", ";
    log_file << timestamp;
    log_file << std::endl;
    log_file.close();
}

Profiler::Profiler()
{
#if defined(PROFILER)
    host_new_log = true;
    device_new_log = true;
    output_dir = std::filesystem::path(string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME));
    std::filesystem::create_directories(output_dir);
#endif
}

void Profiler::markStart(const std::string& timer_name)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].start = steady_clock::now();
#endif
}

void Profiler::markStop(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].stop = steady_clock::now();
    dumpHostResults(timer_name, additional_fields);
#endif
}

void Profiler::setDeviceNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    device_new_log = new_log_flag;
#endif
}

void Profiler::setHostNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    host_new_log = new_log_flag;
#endif
}

void Profiler::setOutputDir(const std::string& new_output_dir)
{
#if defined(PROFILER)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}

void Profiler::setDeviceArchitecture(tt::ARCH device_arch)
{
#if defined(PROFILER)
    device_architecture = device_arch;
#endif
}

void Profiler::dumpTensixDeviceResults(int device_id, const vector<CoreCoord> &worker_cores) {
#if defined(PROFILER)
    device_core_frequency = tt::Cluster::instance().get_device_aiclk(device_id);
    for (const auto &worker_core : worker_cores) {
        readRiscProfilerResults(
            device_id,
            worker_core,
            "NCRISC",
            PRINT_BUFFER_NC);
        readRiscProfilerResults(
            device_id,
            worker_core,
            "BRISC",
            PRINT_BUFFER_BR);
        readRiscProfilerResults(
            device_id,
            worker_core,
            "TRISC_0",
            PRINT_BUFFER_T0);
        readRiscProfilerResults(device_id, worker_core, "TRISC_1", PRINT_BUFFER_T1);
        readRiscProfilerResults(device_id, worker_core, "TRISC_2", PRINT_BUFFER_T2);
    }
#endif
}

void Profiler::dumpEthernetDeviceResults(int device_id, const vector<CoreCoord> &eth_cores) {
#if defined(PROFILER)
    device_core_frequency = tt::Cluster::instance().get_device_aiclk(device_id);
    for (const auto &eth_core : eth_cores) {
        readRiscProfilerResults(device_id, eth_core, "ERISC", eth_l1_mem::address_map::PRINT_BUFFER_ER);
    }
#endif
}

bool getHostProfilerState ()
{
    bool profile_host = false;
#if defined(PROFILER)
    profile_host = true;
#endif
    return profile_host;
}

bool getDeviceProfilerState ()
{
    return tt::llrt::OptionsG.get_profiler_enabled();
}

}  // namespace tt_metal

}  // namespace tt
