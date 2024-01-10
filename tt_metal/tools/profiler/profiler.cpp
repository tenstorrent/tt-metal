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

    uint32_t coreFlatID = get_flat_id(worker_core.x, worker_core.y);


//#define DEBUG_CORES
#ifdef DEBUG_CORES
    uint32_t dram_noc_x = (coreFlatID % 4) * 3 + 1;
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
    std::cout << worker_core.x << "," << worker_core.y <<  "," << coreFlatID << "," << dram_noc_x << "," << dram_noc_y << " | ";
#endif

    uint32_t startIndex = coreFlatID * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;

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
    if (coreFlatID < 1)
    {
    vector<std::uint32_t> profile_buffer_l1;

    profile_buffer_l1 = tt::llrt::read_hex_vec_from_core(
            device_id,
            worker_core,
            PROFILER_L1_BUFFER_BR,
            PROFILER_RISC_COUNT * PROFILER_L1_BUFFER_SIZE);

    std::cout << worker_core.x << "," << worker_core.y <<  "," << coreFlatID << "," << startIndex <<  std::endl ;
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
    //std::cout << "Control Buffer :" << control_buffer [0] << "," << control_buffer [5] << "," << std::endl;
    //std::cout << "Control Buffer :" << control_buffer [1] << "," << control_buffer [6] << "," << std::endl;
    //std::cout << "Control Buffer :" << control_buffer [2] << "," << control_buffer [7] << "," << std::endl;
    //std::cout << "Control Buffer :" << control_buffer [3] << "," << control_buffer [8] << "," << std::endl;
    //std::cout << "Control Buffer :" << control_buffer [4] << "," << control_buffer [9] << "," << std::endl;

    std::cout << "\nDRAM SIZE PER RISC :" << PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC << " L1 SIZE :" << PROFILER_L1_VECTOR_SIZE << std::endl;
    }
#endif

    for (int riscNum = 0; riscNum < PROFILER_RISC_COUNT; riscNum++) {

        uint32_t bufferEndIndex = control_buffer[riscNum];
        if (bufferEndIndex > 0)
        {
            uint32_t bufferRiscShift = riscNum * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC + startIndex;
            if (bufferEndIndex > PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC)
            {
                bufferEndIndex = PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;
            }

            uint32_t runCounter = 0;
            uint32_t riscNumRead = 0;
            uint32_t coreFlatIDRead = 0;
            uint32_t runCounterRead = 0;

            bool newRunStart = false;

            for (int index = bufferRiscShift; index < (bufferRiscShift + bufferEndIndex); index += PROFILER_L1_MARKER_UINT32_SIZE)
            {
                if (!newRunStart && profile_buffer[index] == 0 && profile_buffer[index + 1] == 0)
                {
                    newRunStart = true;
                }
                else if (newRunStart)
                {
                    //TODO(MO): Cleanup magic numbers
                    riscNumRead = profile_buffer[index] & 0x07;
                    coreFlatIDRead = (profile_buffer[index] & 0x3F8) >> 3;

                    runCounterRead = profile_buffer[index + 1];

                    newRunStart = false;
                }
                else
                {
                    uint32_t time_H = profile_buffer[index] & 0x0000FFFF;
                    if (time_H)
                    {
                        uint32_t marker = (profile_buffer[index] & 0x00070000) >> 16;
                        uint32_t riscNumReadts = (profile_buffer[index] & 0x00380000) >> 19;
                        uint32_t coreFlatIDReadts = (profile_buffer[index] & 0x7FC00000) >> 22;
                        uint32_t time_L = profile_buffer[index + 1];


                        TT_ASSERT (riscNumReadts == riscNum && riscNumRead == riscNum,
                                fmt::format("Unexpected risc id, expected {}, read ts {} and id {}. In core {},{} at run {}",
                                    riscNum,
                                    riscNumReadts,
                                    riscNumRead,
                                    worker_core.x,
                                    worker_core.y,
                                    runCounterRead)
                                );
                        TT_ASSERT (coreFlatIDReadts == coreFlatID && coreFlatIDRead == coreFlatID,
                                fmt::format("Unexpected core id, expected {}, read ts {} and id {}. In core {},{} at run {}",
                                    coreFlatID,
                                    coreFlatIDReadts,
                                    coreFlatIDRead,
                                    worker_core.x,
                                    worker_core.y,
                                    runCounterRead));


                        dumpResultToFile(
                                true,
                                runCounterRead,
                                device_id,
                                worker_core.x,
                                worker_core.y,
                                coreFlatID,
                                coreFlatIDRead,
                                coreFlatIDReadts,
                                riscNum,
                                riscNumRead,
                                riscNumReadts,
                                marker,
                                (uint64_t(time_H) << 32) | time_L);


                        if (riscNum == 0 && marker == 1)
                        {
                            TT_ASSERT (runCounterRead == runCounter, fmt::format("Unexpected run id, expected {}, read {}", runCounter, runCounterRead));
                            runCounter ++;
                        }
                    }
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

void DeviceProfiler::firstTimestamp(uint64_t timestamp)
{
    if (timestamp < smallest_timestamp)
    {
        smallest_timestamp = timestamp;
    }
}

void DeviceProfiler::dumpResultToFile(
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
        uint64_t timestamp){
    ZoneScoped;
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    //TODO(MO) : use enums here
    std::string riscName[] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};


    if (new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "ARCH: " << get_string_lowercase(device_architecture) << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
        if (!debug)
        {
            log_file << "Program ID, PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        }
        else
        {
            log_file << "Program ID, PCIe slot, core_x, core_y, core_flat, core_flat_read, core_flat_read_ts, RISC, RISC read, RISC read ts, timer_id, time[cycles since reset]" << std::endl;
        }

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

    tracy::TTDeviceEvent event = tracy::TTDeviceEvent(chip_id, core_x, core_y, risc_num, timer_id);

    if (device_data.find (event) != device_data.end())
    {
        ZoneScopedNC("eventFound",tracy::Color::Green);
        device_data.at(event).push_back(timestamp);
    }
    else
    {
        ZoneScopedNC("eventNotFound",tracy::Color::Red);
        device_data.emplace(event,std::list<uint64_t>{timestamp});
    }

    firstTimestamp(timestamp);

    log_file << runID << ", " << chip_id;
    log_file << ", " << core_x << ", " << core_y;
    if (debug)
    {
        log_file << ", " << core_flat << ", " << core_flat_read << ", " << core_flat_read_ts;
    }

    log_file << ", " << riscName[risc_num];

    if (debug)
    {
        log_file << ", " << riscName[risc_num_read] << ", " << riscName[risc_num_read_ts];
    }

    log_file  << ", " << timer_id << ", " << timestamp;
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
    tracyTTCtx->PopulateCLContext( smallest_timestamp, 1000.f / (float)device_core_frequency);

    std::string riscName[] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};

    static int BR_COUNTER = 0;
    static int NC_COUNTER = 0;
    static int T0_COUNTER = 0;
    static int T1_COUNTER = 0;
    static int T2_COUNTER = 0;

    for (auto& data: device_data)
    {
        ZoneScopedNC("Marker",tracy::Color::Red);
        uint64_t threadID = data.first.get_thread_id();
        uint64_t row = data.first.core_y;
        uint64_t col = data.first.core_x;
        uint64_t risc = data.first.risc;
        uint64_t markerID = data.first.marker;

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
                            BR_COUNTER ++;
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
                            NC_COUNTER ++;
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
                            T0_COUNTER ++;
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
                            T1_COUNTER ++;
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
                            T2_COUNTER ++;
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
