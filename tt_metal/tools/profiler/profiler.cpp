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
#include "dev_msgs.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

void DeviceProfiler::readRiscProfilerResults(
        int device_id,
        const vector<std::uint32_t> &profile_buffer,
        const CoreCoord &worker_core
        ){

    ZoneScoped;

    std::pair<uint32_t, CoreCoord> deviceCore = {device_id,worker_core};

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device_id);
    uint32_t coreFlatID = soc_d.physical_routing_to_profiler_flat_id.at(worker_core);
    uint32_t startIndex = coreFlatID * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;

    auto ethCores = soc_d.get_physical_ethernet_cores() ;

    HalProgrammableCoreType CoreType;
    profiler_msg_t *profiler_msg;
    if (std::find(ethCores.begin(), ethCores.end(), worker_core) == ethCores.end())
    {
        profiler_msg = hal.get_dev_addr<profiler_msg_t *>(HalProgrammableCoreType::TENSIX, HalMemAddrType::PROFILER);
        CoreType = HalProgrammableCoreType::TENSIX;
    }
    else
    {
        profiler_msg = hal.get_dev_addr<profiler_msg_t *>(HalProgrammableCoreType::ACTIVE_ETH, HalMemAddrType::PROFILER);
        CoreType = HalProgrammableCoreType::ACTIVE_ETH;
    }

    vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
        device_id,
        worker_core,
        reinterpret_cast<uint64_t>(profiler_msg->control_vector),
        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);

    if ((control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_BR_ER] == 0) &&
        (control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_NC] == 0))
    {
        return;
    }

    int riscNum = 0;
    for (int riscEndIndex = 0; riscEndIndex < PROFILER_RISC_COUNT; riscEndIndex ++ ) {
        uint32_t bufferEndIndex = control_buffer[riscEndIndex];
        uint32_t riscType;
        if(CoreType == HalProgrammableCoreType::TENSIX)
        {
            riscType = riscEndIndex;
        }
        else
        {
            riscType = 5;
        }
        if (bufferEndIndex > 0)
        {
            uint32_t bufferRiscShift = riscNum * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC + startIndex;
            if ((control_buffer[kernel_profiler::DROPPED_ZONES] >> riscEndIndex) & 1)
            {
                std::string warningMsg = fmt::format("Profiler DRAM buffers were full, markers were dropped! device {}, worker core {}, {}, Risc {},  bufferEndIndex = {}", device_id, worker_core.x, worker_core.y, tracy::riscName[riscEndIndex], bufferEndIndex);
                TracyMessageC(warningMsg.c_str(), warningMsg.size(), tracy::Color::Tomato3);
                log_warning(warningMsg.c_str());
            }

            uint32_t riscNumRead = 0;
            uint32_t coreFlatIDRead = 0;
            uint32_t runCounterRead = 0;
            uint32_t runHostCounterRead = 0;

            bool newRunStart = false;

            uint32_t opTime_H = 0;
            uint32_t opTime_L = 0;
            for (int index = bufferRiscShift; index < (bufferRiscShift + bufferEndIndex); index += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE)
            {
                if (!newRunStart && profile_buffer[index] == 0 && profile_buffer[index + 1] == 0)
                {
                    newRunStart = true;
                    opTime_H = 0;
                    opTime_L = 0;
                }
                else if (newRunStart)
                {
                    newRunStart = false;

                    //TODO(MO): Cleanup magic numbers
                    riscNumRead = profile_buffer[index] & 0x7;
                    coreFlatIDRead = (profile_buffer[index] >> 3) & 0xFF;
                    runCounterRead = profile_buffer[index + 1] & 0xFFFF;
                    runHostCounterRead = (profile_buffer[index + 1] >> 16 ) & 0xFFFF;

                }
                else
                {
                    uint32_t phase = (profile_buffer[index] >> 28) & 0x7;
                    if (phase < 2)
                    {
                        uint32_t time_H = profile_buffer[index] & 0xFFF;
                        uint32_t marker = (profile_buffer[index] >> 12) & 0x7FFFF ;
                        if (marker || time_H)
                        {
                            uint32_t time_L = profile_buffer[index + 1];

                            if (opTime_H == 0)
                            {
                                opTime_H = time_H;
                            }
                            if (opTime_L == 0)
                            {
                                opTime_L = time_L;
                            }

                            TT_ASSERT (riscNumRead == riscNum,
                                    "Unexpected risc id, expected {}, read {}. In core {},{} at run {}",
                                        riscNum,
                                        riscNumRead,
                                        worker_core.x,
                                        worker_core.y,
                                        runCounterRead
                                    );
                            TT_ASSERT (coreFlatIDRead == coreFlatID,
                                    "Unexpected core id, expected {}, read {}. In core {},{} at run {}",
                                        coreFlatID,
                                        coreFlatIDRead,
                                        worker_core.x,
                                        worker_core.y,
                                        runCounterRead);

                            dumpResultToFile(
                                    runCounterRead,
                                    runHostCounterRead,
                                    device_id,
                                    worker_core,
                                    coreFlatID,
                                    riscType,
                                    0,
                                    marker,
                                    (uint64_t(time_H) << 32) | time_L);
                        }
                    }
                    else if (phase == 2)
                    {
                        uint32_t marker = (profile_buffer[index] >> 12) & 0x7FFFF ;
                        uint32_t sum = profile_buffer[index + 1];

                        uint32_t time_H = opTime_H;
                        uint32_t time_L = opTime_L;
                        dumpResultToFile(
                                runCounterRead,
                                runHostCounterRead,
                                device_id,
                                worker_core,
                                coreFlatID,
                                riscType,
                                sum,
                                marker,
                                (uint64_t(time_H) << 32) | time_L);
                    }
                }
            }
        }
        riscNum ++;
    }

    std::vector<uint32_t> control_buffer_reset(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    control_buffer_reset[kernel_profiler::DRAM_PROFILER_ADDRESS] = output_dram_buffer->address();

    profiler_msg = hal.get_dev_addr<profiler_msg_t *>(HalProgrammableCoreType::TENSIX, HalMemAddrType::PROFILER);
    tt::llrt::write_hex_vec_to_core(
            device_id,
            worker_core,
            control_buffer_reset,
            reinterpret_cast<uint64_t>(profiler_msg->control_vector));
}

void DeviceProfiler::firstTimestamp(uint64_t timestamp)
{
    if (timestamp < smallest_timestamp)
    {
        smallest_timestamp = timestamp;
    }
}

void DeviceProfiler::dumpResultToFile(
        uint32_t run_id,
        uint32_t run_host_id,
        int device_id,
        CoreCoord core,
        int core_flat,
        int risc_num,
        uint64_t stat_value,
        uint32_t timer_id,
        uint64_t timestamp
        ){
    std::pair<uint32_t, CoreCoord> deviceCore = {device_id,core};
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    tracy::TTDeviceEventPhase zone_phase = tracy::TTDeviceEventPhase::begin;
    if (stat_value > 0)
    {
        zone_phase = tracy::TTDeviceEventPhase::sum;
    }
    else if (timer_id & (1<<16))
    {
        zone_phase = tracy::TTDeviceEventPhase::end;
    }

    std::string zone_name = "";
    std::string source_file = "";
    uint64_t source_line = 0;
    if (hash_to_zone_src_locations.find((uint16_t)timer_id) != hash_to_zone_src_locations.end())
    {
        std::stringstream source_info(hash_to_zone_src_locations[timer_id]);
        getline(source_info, zone_name, ',');
        getline(source_info, source_file, ',');

        std::string source_line_str;
        getline(source_info, source_line_str, ',');
        source_line = stoi(source_line_str);
    }

    tracy::TTDeviceEvent event = tracy::TTDeviceEvent(run_host_id, device_id, core.x, core.y, risc_num, timer_id, timestamp, source_line, source_file, zone_name, zone_phase);

    auto ret = device_events.insert(event);

    if (!ret.second) return;

    firstTimestamp(timestamp);

    if (new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "ARCH: " << get_string_lowercase(device_architecture) << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], stat value, run ID, run host ID,  zone name, zone phase, source line, source file" << std::endl;
        new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    //log_file << fmt::format("{:4},{:3},{:3},{:>7},{:7},{:15},{:15},{:5},{:>25},{:>6},{:6},{}",
    log_file << fmt::format("{},{},{},{},{},{},{},{},{},{},{},{},{}",
            device_id,
            core.x,
            core.y,
            tracy::riscName[risc_num],
            timer_id,
            timestamp,
            stat_value,
            run_id,
            run_host_id,
            zone_name,
            magic_enum::enum_name(zone_phase),
            source_line,
            source_file
            );
    log_file << std::endl;
    log_file.close();
}

DeviceProfiler::DeviceProfiler(const bool new_logs)
{
#if defined(TRACY_ENABLE)
    ZoneScopedC(tracy::Color::Green);
    new_log = new_logs;
    output_dir = std::filesystem::path(string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME));
    std::filesystem::create_directories(output_dir);

#endif
}

DeviceProfiler::~DeviceProfiler()
{
#if defined(TRACY_ENABLE)
    ZoneScoped;
    pushTracyDeviceResults();
    for (auto tracyCtx : device_tracy_contexts)
    {
        TracyTTDestroy(tracyCtx.second);
    }
#endif
}


void DeviceProfiler::setNewLogFlag(bool new_log_flag)
{
#if defined(TRACY_ENABLE)
    new_log = new_log_flag;
#endif
}


void DeviceProfiler::setOutputDir(const std::string& new_output_dir)
{
#if defined(TRACY_ENABLE)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}


void DeviceProfiler::setDeviceArchitecture(tt::ARCH device_arch)
{
#if defined(TRACY_ENABLE)
    device_architecture = device_arch;
#endif
}

uint32_t DeviceProfiler::hash32CT( const char * str, size_t n, uint32_t basis)
{
    return n == 0 ? basis : hash32CT( str + 1, n - 1, ( basis ^ str[ 0 ] ) * UINT32_C( 16777619 ) );
}

uint16_t DeviceProfiler::hash16CT( const std::string& str)
{
    uint32_t res = hash32CT (str.c_str(), str.length());
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

void DeviceProfiler::generateZoneSourceLocationsHashes()
{
    std::ifstream log_file (tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG);
    std::string line;
    while(std::getline(log_file, line))
    {
        std::string delimiter = "'#pragma message: ";
        int delimiter_index = line.find(delimiter) + delimiter.length();
        std::string zone_src_location = line.substr( delimiter_index, line.length() - delimiter_index - 1);


        uint16_t hash_16bit = hash16CT(zone_src_location);

        auto did_insert = zone_src_locations.insert(zone_src_location);
        if (did_insert.second && (hash_to_zone_src_locations.find(hash_16bit) != hash_to_zone_src_locations.end()))
        {
            log_warning("Source location hashes are colliding, two different locations are having the same hash");
        }
        hash_to_zone_src_locations.emplace(hash_16bit,zone_src_location);
    }
}


void DeviceProfiler::dumpResults (
        Device *device,
        const vector<CoreCoord> &worker_cores,
        bool lastDump){
#if defined(TRACY_ENABLE)
    ZoneScoped;

    auto device_id = device->id();
    device_core_frequency = tt::Cluster::instance().get_device_aiclk(device_id);

    generateZoneSourceLocationsHashes();

    if (output_dram_buffer != nullptr)
    {
        std::vector<uint32_t> profile_buffer(output_dram_buffer->size()/sizeof(uint32_t), 0);

        const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;
        if (USE_FAST_DISPATCH)
        {
            if (lastDump)
            {
                if (tt::llrt::OptionsG.get_profiler_do_dispatch_cores())
                {
                    tt_metal::detail::ReadFromBuffer(output_dram_buffer, profile_buffer);
                }
            }
            else
            {
                EnqueueReadBuffer(device->command_queue(),output_dram_buffer, profile_buffer, true);
            }
        }
        else
        {
            if (!lastDump)
            {
                tt_metal::detail::ReadFromBuffer(output_dram_buffer, profile_buffer);
            }
        }


        for (const auto &worker_core : worker_cores) {
            readRiscProfilerResults(
                device_id,
                profile_buffer,
                worker_core);

        }
    }
    else
    {
        log_warning("DRAM profiler buffer is not initialized");
    }
#endif
}

void DeviceProfiler::pushTracyDeviceResults()
{
#if defined(TRACY_ENABLE)
    ZoneScoped;
    std::set<std::pair<uint32_t, CoreCoord>> device_cores_set;
    std::vector<std::pair<uint32_t, CoreCoord>> device_cores;
    for (auto& event: device_events)
    {
        std::pair<uint32_t, CoreCoord> device_core = {event.chip_id, (CoreCoord){event.core_x,event.core_y}};
        auto ret = device_cores_set.insert(device_core);
        if (ret.second )
        {
            device_cores.push_back(device_core);
        }
    }

    double delay = 0;
    double frequency = 0;
    uint64_t cpuTime = 0;

    for (auto& device_core: device_cores)
    {
        int device_id = device_core.first;
        CoreCoord worker_core = device_core.second;

        if (device_core_sync_info.find(worker_core) != device_core_sync_info.end())
        {
            cpuTime = get<0>(device_core_sync_info.at(worker_core));
            delay = get<1>(device_core_sync_info.at(worker_core));
            frequency = get<2>(device_core_sync_info.at(worker_core));
            log_info("Device {} sync info are, frequency {} GHz,  delay {} cycles and, sync point {} seconds",
                        device_id,
                        frequency,
                        delay,
                        cpuTime);
        }
    }

    for (auto& device_core: device_cores)
    {
        int device_id = device_core.first;
        CoreCoord worker_core = device_core.second;

        if (delay == 0.0 || frequency == 0.0)
        {
            delay = smallest_timestamp;
            frequency = device_core_frequency/1000.0;
            cpuTime = TracyGetCpuTime();
            log_warning("For device {}, core {},{} default frequency was used and its zones will be out of sync", device_id, worker_core.x, worker_core.y);
        }


        if (device_tracy_contexts.find(device_core) == device_tracy_contexts.end())
        {
            auto tracyCtx = TracyTTContext();
            std::string tracyTTCtxName = fmt::format("Device: {}, Core ({},{})", device_id, worker_core.x, worker_core.y);

            TracyTTContextPopulate(tracyCtx, cpuTime, delay, frequency);

            TracyTTContextName(tracyCtx, tracyTTCtxName.c_str(), tracyTTCtxName.size());

            device_tracy_contexts.emplace(
                    device_core,
                    tracyCtx
                );
        }
    }

    for (auto& event: device_events)
    {
        std::pair<uint32_t, CoreCoord> device_core = {event.chip_id, (CoreCoord){event.core_x,event.core_y}};
        if (event.zone_phase == tracy::TTDeviceEventPhase::begin)
        {
            TracyTTPushStartZone(device_tracy_contexts[device_core], event);
        }
        else if (event.zone_phase == tracy::TTDeviceEventPhase::end)
        {
            TracyTTPushEndZone(device_tracy_contexts[device_core], event);
        }

    }
    device_events.clear();
#endif
}


bool getDeviceProfilerState ()
{
    return tt::llrt::OptionsG.get_profiler_enabled();
}

}  // namespace tt_metal

}  // namespace tt
