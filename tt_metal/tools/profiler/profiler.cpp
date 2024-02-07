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

void DeviceProfiler::readRiscProfilerResults(
        int device_id,
        vector<std::uint32_t> profile_buffer,
        const CoreCoord &worker_core
        ){

    ZoneScoped;

    std::pair<uint32_t, CoreCoord> deviceCore = {device_id,worker_core};
    TT_ASSERT (device_core_data.find(deviceCore) != device_core_data.end(), "Device {}, worker core {},{} not present in recorded data" , device_id, worker_core.x, worker_core.y);

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device_id);
    uint32_t coreFlatID = soc_d.physical_routing_to_profiler_flat_id.at(worker_core);
    uint32_t startIndex = coreFlatID * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;

    vector<std::uint32_t> control_buffer;

    auto ethCores = soc_d.get_physical_ethernet_cores() ;

    std::vector<uint32_t> riscEndIndices;

    if (std::find(ethCores.begin(), ethCores.end(), worker_core) == ethCores.end())
    {
        control_buffer = tt::llrt::read_hex_vec_from_core(
            device_id,
            worker_core,
            PROFILER_L1_BUFFER_CONTROL,
            PROFILER_L1_CONTROL_BUFFER_SIZE);

        riscEndIndices.push_back(kernel_profiler::HOST_BUFFER_END_INDEX_BR);
        riscEndIndices.push_back(kernel_profiler::HOST_BUFFER_END_INDEX_NC);
        riscEndIndices.push_back(kernel_profiler::HOST_BUFFER_END_INDEX_T0);
        riscEndIndices.push_back(kernel_profiler::HOST_BUFFER_END_INDEX_T1);
        riscEndIndices.push_back(kernel_profiler::HOST_BUFFER_END_INDEX_T2);

        //std::cout << worker_core.x << "," << worker_core.y  << "," << coreFlatID << "," <<  output_dram_buffer.address() << std::endl
            //<< control_buffer[kernel_profiler::NOC_X] << "," << control_buffer[kernel_profiler::NOC_Y]
            //<< "," << control_buffer[kernel_profiler::FLAT_ID]
            //<< "," << control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS]
            //<< "," << control_buffer[kernel_profiler::RUN_COUNTER]
            //<< "," << control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_BR]
            //<< "," << control_buffer[kernel_profiler::DEVICE_BUFFER_END_INDEX_BR]<< std::endl;

        //vector<std::uint32_t> profile_buffer_L1;
        //profile_buffer_L1 = tt::llrt::read_hex_vec_from_core(
                //device_id,
                //worker_core,
                //PROFILER_L1_BUFFER_BR,
                //kernel_profiler::CUSTOM_MARKERS * sizeof(uint32_t));
        //for (int i = kernel_profiler::ID_HH; i < kernel_profiler::CUSTOM_MARKERS; i++)
        //{
             //std::cout << profile_buffer_L1[i] << ",";
        //}
        //std::cout << std::endl<< std::endl;
    }
    else
    {
        control_buffer = tt::llrt::read_hex_vec_from_core(
            device_id,
            worker_core,
            eth_l1_mem::address_map::PROFILER_L1_BUFFER_CONTROL,
            PROFILER_L1_CONTROL_BUFFER_SIZE);

        riscEndIndices.push_back(kernel_profiler::HOST_BUFFER_END_INDEX_ER);
    }


    if ((control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_BR] == 0) &&
        (control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_ER] == 0))
    {
        return;
    }

    int riscNum = 0;
    for (auto riscEndIndex : riscEndIndices) {
        uint32_t bufferEndIndex = control_buffer[riscEndIndex];
        if (bufferEndIndex > 0)
        {
            uint32_t bufferRiscShift = riscNum * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC + startIndex;
            if (bufferEndIndex > PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC)
            {
                log_warning("Profiler DRAM buffers were full, markers were dropped! worker core {}, {}, Risc {},  bufferEndIndex = {}, host_size = {}",worker_core.x, worker_core.y, riscEndIndex, bufferEndIndex , PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC );
                bufferEndIndex = PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;
            }

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
                    riscNumRead = profile_buffer[index] & 0x7;
                    coreFlatIDRead = (profile_buffer[index] >> 3) & 0xFF;

                    runCounterRead = profile_buffer[index + 1];
                    if (device_core_data[deviceCore].runCounter == 0)
                    {
                        runCounterRead = 0;
                    }

                    newRunStart = false;
                }
                else
                {
                    uint32_t time_H = profile_buffer[index] & 0xFFF;
                    if (time_H)
                    {
                        uint32_t marker = (profile_buffer[index] >> 12) & 0xFF ;
                        uint32_t riscNumReadts = (profile_buffer[index] >> 20) & 0x7;
                        uint32_t coreFlatIDReadts = (profile_buffer[index] >> 23) & 0xFF;
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
                                false,
                                runCounterRead,
                                device_id,
                                worker_core,
                                coreFlatID,
                                coreFlatIDRead,
                                coreFlatIDReadts,
                                riscEndIndex,
                                riscNumRead,
                                riscNumReadts,
                                marker,
                                (uint64_t(time_H) << 32) | time_L);


                        if (riscNum == 0 && marker == 1)
                        {
                            TT_ASSERT (runCounterRead == device_core_data[deviceCore].runCounter, fmt::format("Unexpected run id, expected {}, read {}. In core {},{}", device_core_data[deviceCore].runCounter, runCounterRead, worker_core.x, worker_core.y));
                            device_core_data[deviceCore].runCounter ++;
                        }
                    }
                }
            }
        }
        riscNum ++;
    }

    std::vector<uint32_t> zero_buffer(PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
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
        int device_id,
        CoreCoord core,
        int core_flat,
        int core_flat_read,
        int core_flat_read_ts,
        int risc_num,
        int risc_num_read,
        int risc_num_read_ts,
        uint32_t timer_id,
        uint64_t timestamp
        ){
    std::pair<uint32_t, CoreCoord> deviceCore = {device_id,core};
    TT_ASSERT (device_core_data.find(deviceCore) != device_core_data.end(), "Device {}, core {}, {} not present in recorded data" , device_id, core.x, core.y);

    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    //TODO(MO) : use enums here
    std::string riscName[] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC"};

    tracy::TTDeviceEvent event = tracy::TTDeviceEvent(runID, device_id, core.x, core.y, risc_num, timer_id);
    if (timer_id > PROFILER_L1_GUARANTEED_MARKER_COUNT)
    {
        customMarkerCount ++;
        event.marker = (uint64_t(customMarkerCount) << 32 ) | timer_id;
    }

    if (device_core_data[deviceCore].data.find (runID) != device_core_data[deviceCore].data.end())
    {
        TT_ASSERT (device_core_data[deviceCore].data[runID].find (event) == device_core_data[deviceCore].data[runID].end(), "Unexpexted marker ID repeat");
        device_core_data[deviceCore].data[runID].emplace(event,timestamp);
    }
    else
    {
        std::map<tracy::TTDeviceEvent, uint64_t, tracy::TTDeviceEvent_cmp> eventMap = {{event,timestamp}};
        device_core_data[deviceCore].data.emplace(runID, eventMap);
    }

    firstTimestamp(timestamp);

    if (new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "ARCH: " << get_string_lowercase(device_architecture) << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
        if (!debug)
        {
            log_file << "Run ID, PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        }
        else
        {
            log_file << "Run ID, PCIe slot, core_x, core_y, core_flat, core_flat_read, core_flat_read_ts, RISC, RISC read, RISC read ts, timer_id, time[cycles since reset]" << std::endl;
        }

        new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }


    log_file << runID << ", " << device_id;
    log_file << ", " << core.x << ", " << core.y;
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

    TracySetCpuTime();

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
    std::vector<uint32_t> profile_buffer(output_dram_buffer.size()/sizeof(uint32_t), 0);

    tt_metal::detail::ReadFromBuffer(output_dram_buffer, profile_buffer);

    for (const auto &worker_core : worker_cores) {
        std::pair<uint32_t, CoreCoord> device_core = {device_id, worker_core};
        auto tracyCtx = TracyTTContext();
        if (device_core_data.find(device_core) == device_core_data.end())
        {
            device_core_data.emplace(
                    device_core,
                    (CoreTracyData){
                        (std::map<uint32_t, std::map<tracy::TTDeviceEvent, uint64_t, tracy::TTDeviceEvent_cmp>>){},
                        tracyCtx,
                        false
                    }
                );
        }
        readRiscProfilerResults(
            device_id,
            profile_buffer,
            worker_core);

    }

    for (const auto &worker_core : worker_cores) {
        std::pair<uint32_t, CoreCoord> device_core = {device_id, worker_core};
        if (!device_core_data[device_core].data.empty())
        {
            pushTracyDeviceResults(device_core);

            device_core_data[device_core].data.clear();
            device_core_data[device_core].runCounter = 0;

            std::string tracyTTCtxName = fmt::format("Device: {}, Core ({},{})", device_id, worker_core.x, worker_core.y);
            TracyTTContextName(device_core_data[device_core].tracyContext, tracyTTCtxName.c_str(), tracyTTCtxName.size());
        }
    }
#endif
}

void DeviceProfiler::pushTracyDeviceResults(std::pair<uint32_t,CoreCoord> device_core)
{
#if defined(PROFILER) && defined(TRACY_ENABLE)
    TT_ASSERT (device_core_data.find(device_core) != device_core_data.end(), "Device {}, core {},{} not present in recorded data" , device_core.first, device_core.second.x, device_core.second.y);
    if (!device_core_data[device_core].contextPopulated)
    {
        device_core_data[device_core].tracyContext->PopulateTTContext( smallest_timestamp, 1000.f / (float)device_core_frequency);
        device_core_data[device_core].contextPopulated = true;
    }

    std::string riscName[] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC"};
    uint32_t FWColors[] = {tracy::Color::Red4, tracy::Color::Green4, tracy::Color::Blue4, tracy::Color::Purple3, tracy::Color::Yellow4, tracy::Color::Aquamarine4};
    uint32_t KernelColors[] = {tracy::Color::Red2, tracy::Color::Green3, tracy::Color::Blue3, tracy::Color::Purple1, tracy::Color::Yellow3, tracy::Color::Aquamarine1};
    uint32_t customColors[] = {tracy::Color::Orange2, tracy::Color::Cyan3, tracy::Color::Orchid1, tracy::Color::Plum1, tracy::Color::PaleTurquoise2, tracy::Color::CadetBlue1};

    for (auto& run: device_core_data[device_core].data)
    {
        std::unordered_map<uint32_t, std::vector<tracy::TTDeviceEvent>> customMarkers;
        for (auto& data: run.second)
        {
            uint64_t threadID = data.first.get_thread_id();
            uint64_t markerID = data.first.marker;
            if (markerID > PROFILER_L1_GUARANTEED_MARKER_COUNT)
            {
                if (customMarkers.find (threadID) != customMarkers.end())
                {
                    customMarkers[threadID].push_back(data.first);
                }
                else
                {
                    customMarkers.emplace(threadID, (std::vector<tracy::TTDeviceEvent>){data.first});
                }
            }
        }
        for (auto& data: run.second)
        {
            uint64_t threadID = data.first.get_thread_id();
            uint64_t device_id = data.first.chip_id;
            uint64_t row = data.first.core_y;
            uint64_t col = data.first.core_x;
            uint64_t risc = data.first.risc;
            uint64_t markerID = data.first.marker;
            uint64_t runID = data.first.run_num;

            if (markerID == 1 )
            {
                TracyTTZoneTransient(device_core_data[device_core].tracyContext, FWScope, fmt::format("{} FW",riscName[risc]).c_str(), FWColors[risc], true, threadID);
                {
                    TracyTTZoneTransient(device_core_data[device_core].tracyContext, KernelScope, fmt::format("{} Kernel",riscName[risc]).c_str(), KernelColors[risc], true, threadID);
                    for (auto &customMarker : customMarkers[threadID])
                    {
                        uint64_t actualMarkerID = (customMarker.marker << 32) >> 32;
                        TracyTTZoneTransient(
                                device_core_data[device_core].tracyContext,
                                customMarkerScope,
                                fmt::format("{}",actualMarkerID).c_str(),
                                customColors[actualMarkerID % (sizeof(customColors) / sizeof(uint32_t))],
                                true,
                                threadID);
                        customMarkerScope.SetEvent(tracy::TTDeviceEvent(runID,device_id,col,row,risc,customMarker.marker));
                    }

                    KernelScope.SetEvent(tracy::TTDeviceEvent(runID,device_id,col,row,risc,1));
                }
                FWScope.SetEvent(tracy::TTDeviceEvent(runID,device_id,col,row,risc,0));
            }
            TracyTTCollect(device_core_data[device_core].tracyContext, device_core_data[device_core].data);
        }
    }
    device_core_data[device_core].data.clear();
    device_core_data[device_core].runCounter = 0;

#endif
}


bool getDeviceProfilerState ()
{
    return tt::llrt::OptionsG.get_profiler_enabled();
}

}  // namespace tt_metal

}  // namespace tt
