// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>


#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tools/profiler/profiler.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "tools/profiler/common.hpp"
#include "hostdevcommon/profiler_common.h"
#include "llrt/rtoptions.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/third_party/tracy/public/tracy/TracyOpenCL.hpp"

namespace tt {

namespace tt_metal {


TimerPeriodInt HostProfiler::timerToTimerInt(TimerPeriod period)
{
    TimerPeriodInt ret;

    ret.start = duration_cast<nanoseconds>(period.start.time_since_epoch()).count();
    ret.stop = duration_cast<nanoseconds>(period.stop.time_since_epoch()).count();
    ret.delta = duration_cast<nanoseconds>(period.stop - period.start).count();

    return ret;
}

void HostProfiler::dumpResults(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
    auto timer = name_to_timer_map[timer_name];

    auto timer_period_ns = timerToTimerInt(timer);
    TT_FATAL (timer_period_ns.start != 0 , "Timer start cannot be zero on : " + timer_name);
    TT_FATAL (timer_period_ns.stop != 0 , "Timer stop cannot be zero on : " + timer_name);

    std::filesystem::path log_path = output_dir / HOST_SIDE_LOG;
    std::ofstream log_file;

    if (new_log|| !std::filesystem::exists(log_path))
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
        new_log = false;
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

HostProfiler::HostProfiler()
{
#if defined(PROFILER)
    ZoneScopedC(tracy::Color::Green);
    new_log = true;
    output_dir = std::filesystem::path(string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME));
    std::filesystem::create_directories(output_dir);
#endif
}

void HostProfiler::markStart(const std::string& timer_name)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].start = steady_clock::now();
#endif
}

void HostProfiler::markStop(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].stop = steady_clock::now();
    dumpResults(timer_name, additional_fields);
#endif
}

void HostProfiler::setNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    new_log = new_log_flag;
#endif
}
void HostProfiler::setOutputDir(const std::string& new_output_dir)
{
#if defined(PROFILER)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}

void DeviceProfiler::readRiscProfilerResults(
        int device_id,
        vector<std::uint32_t> profile_buffer,
        const CoreCoord &worker_core){

    ZoneScoped;
    uint32_t core_flat_id = get_flat_id(worker_core.x, worker_core.y);


//#define DEBUG_CORES
#ifdef DEBUG_CORES
    uint32_t dram_noc_x = (core_flat_id % 4) * 3 + 1;
    uint32_t dram_noc_y = worker_core.y > 6 ? 6 : 0;

    static size_t preRow = -1;
    if (preRow == -1)
    {
        preRow = worker_core.y;
    }
    else if (preRow != worker_core.y)
    {
        std::cout << std::endl;
        preRow = worker_core.y;
    }
    std::cout << worker_core.x << "," << worker_core.y <<  "," << core_flat_id << "," << dram_noc_x << "," << dram_noc_y << " | ";
#endif

    uint32_t startIndex = core_flat_id * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;

    vector<std::uint32_t> control_buffer;

    control_buffer = tt::llrt::read_hex_vec_from_core(
        device_id,
        worker_core,
        PROFILER_L1_BUFFER_CONTROL,
        PROFILER_L1_CONTROL_BUFFER_SIZE);

    if (control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_BR] == 0)
        return;

//#define DEBUG_PRINT_L1
#ifdef DEBUG_PRINT_L1
    if (core_flat_id < 8)
    {
    vector<std::uint32_t> profile_buffer_l1;

    profile_buffer_l1 = tt::llrt::read_hex_vec_from_core(
            device_id,
            worker_core,
            PROFILER_L1_BUFFER_BR,
            PROFILER_RISC_COUNT * PROFILER_L1_BUFFER_SIZE);

    std::cout << worker_core.x << "," << worker_core.y <<  "," << core_flat_id << "," << startIndex <<  std::endl ;
    for (int j = 0; j < PROFILER_RISC_COUNT; j++)
    {
        for (int i= 0; i < 12; i ++)
        {
            std::cout << profile_buffer_l1[j*PROFILER_L1_VECTOR_SIZE + i] << ",";
        }
        std::cout <<  std::endl;
        for (int i= 0; i < 12; i ++)
        {
            std::cout << profile_buffer[startIndex + j*PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC + i] << ",";
        }
        std::cout <<  std::endl;
        std::cout <<  std::endl;
    }
    std::cout << "Control Buffer :" << control_buffer [0] << "," << control_buffer [5] << "," << std::endl;
    std::cout << "Control Buffer :" << control_buffer [1] << "," << control_buffer [6] << "," << std::endl;
    std::cout << "Control Buffer :" << control_buffer [2] << "," << control_buffer [7] << "," << std::endl;
    std::cout << "Control Buffer :" << control_buffer [3] << "," << control_buffer [8] << "," << std::endl;
    std::cout << "Control Buffer :" << control_buffer [4] << "," << control_buffer [9] << "," << std::endl;
    }
#endif

    for (int riscNum = 0; riscNum < PROFILER_RISC_COUNT; riscNum++) {

        uint32_t bufferEndIndex = control_buffer[riscNum];
        if (bufferEndIndex > 0)
        {
            uint32_t bufferRiscShift = riscNum * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC + startIndex;
            if (bufferEndIndex > PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC)
            {
                bufferEndIndex = PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC;
            }

            uint16_t programID = 0;

            for (int index = bufferRiscShift; index < (bufferRiscShift + bufferEndIndex); index += PROFILER_L1_MARKER_UINT32_SIZE)
            {
                if (profile_buffer[index] != 0)
                {
                    uint32_t marker = (profile_buffer[index] & 0xFFFF0000) >> 16;
                    uint32_t time_H = profile_buffer[index] & 0x0000FFFF;
                    uint32_t time_L = profile_buffer[index + 1];

                    dumpResultToFile(
                            programID,
                            device_id,
                            worker_core.x,
                            worker_core.y,
                            riscNum,
                            (uint64_t(time_H) << 32) | time_L,
                            marker);
                }
                else
                {
                    index += PROFILER_L1_MARKER_UINT32_SIZE;
                    programID = profile_buffer[index] & 0x0000FFFF;
                }
            }
        }
    }

    std::vector<uint32_t> zero_buffer(kernel_profiler::CONTROL_BUFFER_SIZE, 0);
    tt::llrt::write_hex_vec_to_core(
            device_id,
            worker_core,
            zero_buffer,
            PROFILER_L1_BUFFER_CONTROL);
}

void DeviceProfiler::dumpResultToFile(
        uint16_t programID,
        int chip_id,
        int core_x,
        int core_y,
        int risc,
        uint64_t timestamp,
        uint32_t timer_id){
    ZoneScoped;
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    std::string riscName[] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};


    if (new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "ARCH: " << get_string_lowercase(device_architecture) << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, Program ID, timer_id, time[cycles since reset]" << std::endl;
        new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    constexpr int DRAM_ROW = 6;
    if (core_y > DRAM_ROW) {
        core_y = core_y - 2;
    } else {
        core_y--;
    }
    core_x--;

    uint64_t threadID = core_x*1000000+core_y*10000+risc*100;
    uint64_t eventID = timer_id + threadID;

    if (device_data.find (eventID) != device_data.end())
    {
        ZoneScopedNC("eventFound",tracy::Color::Green);
        device_data.at(eventID).push_back(timestamp);
    }
    else
    {
        ZoneScopedNC("eventNotFound",tracy::Color::Red);
        device_data.emplace(eventID,std::list<uint64_t>{timestamp});
    }

    log_file << chip_id << ", " << core_x << ", " << core_y << ", " << riscName[risc] << ", " << programID << ", ";
    log_file << timer_id << ", ";
    log_file << timestamp;
    log_file << std::endl;
    log_file.close();
}

DeviceProfiler::DeviceProfiler()
{
#if defined(PROFILER)
    ZoneScopedC(tracy::Color::Green);
    new_log = true;
    output_dir = std::filesystem::path(string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME));
    std::filesystem::create_directories(output_dir);

    tracyTTCtx = TracyCLContext();
#endif
}

DeviceProfiler::~DeviceProfiler()
{
#if defined(PROFILER)
    TracyCLDestroy(tracyTTCtx);
#endif
}

void DeviceProfiler::setNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    new_log = new_log_flag;
#endif
}


void DeviceProfiler::setOutputDir(const std::string& new_output_dir)
{
#if defined(PROFILER)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}


void DeviceProfiler::setDeviceArchitecture(tt::ARCH device_arch)
{
#if defined(PROFILER)
    device_architecture = device_arch;
#endif
}

void DeviceProfiler::dumpResults (
        Device *device,
        const vector<CoreCoord> &worker_cores){
#if defined(PROFILER)
    ZoneScoped;
    auto device_id = device->id();
    device_core_frequency = tt::Cluster::instance().get_device_aiclk(device_id);
    std::vector<uint32_t> profile_buffer(PROFILER_FULL_HOST_BUFFER_SIZE/sizeof(uint32_t), 0);

    tt_metal::detail::ReadFromBuffer(output_dram_buffer, profile_buffer);

//#define DEBUG_READ_BANKS
#ifdef DEBUG_READ_BANKS
    std::vector<uint32_t> profile_buffer_dram(PROFILER_FULL_HOST_BUFFER_SIZE_PER_DRAM_BANK/sizeof(uint32_t), 0);
    for (int j = 0; j < 8; j ++)
    {
        tt_metal::detail::ReadFromDeviceDRAMChannel(device, j, output_dram_buffer.address(), PROFILER_FULL_HOST_BUFFER_SIZE_PER_DRAM_BANK, profile_buffer_dram);

        for (int i =0; i < 36; i++)
        {
            cout << profile_buffer_dram [i] << ",";
        }
        cout << std::endl;

        for (int i =0; i < 36; i++)
        {
            cout << profile_buffer [i + j * PROFILER_FULL_HOST_BUFFER_SIZE_PER_DRAM_BANK/sizeof(uint32_t)] << ",";
        }
        cout << std::endl;
        cout << std::endl;
    }
#endif


    for (const auto &worker_core : worker_cores) {
        readRiscProfilerResults(
            device_id,
            profile_buffer,
            worker_core);
    }
#endif
}


void DeviceProfiler::pushTracyDeviceResults(int device_id)
{
#if defined(PROFILER)
    tracyTTCtx->PopulateCLContext();

    std::string riscName[] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};

    for (auto& data: device_data)
    {
        ZoneScopedNC("Marker",tracy::Color::Red);
        uint64_t threadID = 100*(data.first/100);
        uint64_t row = int(threadID / 1000000);
        uint64_t col = int((threadID-row*1000000)/10000);
        uint64_t risc = int ((threadID-row*1000000-col*10000)/100);
        uint64_t markerID = data.first - threadID;

        if (row == 0 && col == 0 && markerID == 1)
        {
            int i = 1;
            for (auto event : data.second)
            {
                switch (risc)
                {
                    case 0:
                        {
                            TracyCLZoneC(tracyTTCtx, "FW", tracy::Color::Red3,threadID);
                            {
                                TracyCLZoneC(tracyTTCtx, "KERNEL", tracy::Color::Red2,threadID);
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                            }
                            TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                        }
                        break;
                    case 1:
                        {
                            TracyCLZoneC(tracyTTCtx, "FW", tracy::Color::Green4,threadID);
                            {
                                TracyCLZoneC(tracyTTCtx, "KERNEL", tracy::Color::Green3,threadID);
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                            }
                            TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                        }
                        break;
                    case 2:
                        {
                            TracyCLZoneC(tracyTTCtx, "FW", tracy::Color::Blue4,threadID);
                            {
                                TracyCLZoneC(tracyTTCtx, "KERNEL", tracy::Color::Blue3,threadID);
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                            }
                            TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                        }
                        break;
                    case 3:
                        {
                            TracyCLZoneC(tracyTTCtx, "FW", tracy::Color::Purple3,threadID);
                            {
                                TracyCLZoneC(tracyTTCtx, "KERNEL", tracy::Color::Purple2,threadID);
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                            }
                            TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                        }
                        break;
                    case 4:
                        {
                            TracyCLZoneC(tracyTTCtx, "FW", tracy::Color::Yellow4,threadID);
                            {
                                TracyCLZoneC(tracyTTCtx, "KERNEL", tracy::Color::Yellow3,threadID);
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                            }
                            TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                        }
                        break;

                    default:
                        break;
                }
            }
        }
    }

    TracyCLCollect(tracyTTCtx, device_data);

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
