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
        const vector<std::uint32_t> &profile_buffer,
        const CoreCoord &worker_core
        ){

    ZoneScoped;

    std::pair<uint32_t, CoreCoord> deviceCore = {device_id,worker_core};

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
                log_warning("Profiler DRAM buffers were full, markers were dropped! device {}, worker core {}, {}, Risc {},  bufferEndIndex = {}, host_size = {}", device_id, worker_core.x, worker_core.y, tracy::riscName[riscEndIndex], bufferEndIndex , PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC );
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
                    newRunStart = false;

                    //TODO(MO): Cleanup magic numbers
                    riscNumRead = profile_buffer[index] & 0x7;
                    coreFlatIDRead = (profile_buffer[index] >> 3) & 0xFF;
                    runCounterRead = profile_buffer[index + 1];
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

                            TT_ASSERT (riscNumRead == riscNum,
                                    fmt::format("Unexpected risc id, expected {}, read {}. In core {},{} at run {}",
                                        riscNum,
                                        riscNumRead,
                                        worker_core.x,
                                        worker_core.y,
                                        runCounterRead)
                                    );
                            TT_ASSERT (coreFlatIDRead == coreFlatID,
                                    fmt::format("Unexpected core id, expected {}, read {}. In core {},{} at run {}",
                                        coreFlatID,
                                        coreFlatIDRead,
                                        worker_core.x,
                                        worker_core.y,
                                        runCounterRead));

                            dumpResultToFile(
                                    runCounterRead,
                                    device_id,
                                    worker_core,
                                    coreFlatID,
                                    riscEndIndex,
                                    0,
                                    marker,
                                    (uint64_t(time_H) << 32) | time_L);
                        }
                    }
                    else if (phase == 2)
                    {
                        uint32_t marker = (profile_buffer[index] >> 12) & 0x7FFFF ;
                        uint32_t sum = profile_buffer[index + 1];

                        uint32_t time_H = profile_buffer[bufferRiscShift + kernel_profiler::GUARANTEED_MARKER_1_H] & 0xFFF;
                        uint32_t time_L = profile_buffer[bufferRiscShift + kernel_profiler::GUARANTEED_MARKER_1_H + 1];
                        dumpResultToFile(
                                runCounterRead,
                                device_id,
                                worker_core,
                                coreFlatID,
                                riscEndIndex,
                                sum,
                                marker,
                                (uint64_t(time_H) << 32) | time_L);
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
        uint32_t run_id,
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

    tracy::TTDeviceEvent event = tracy::TTDeviceEvent(run_id, device_id, core.x, core.y, risc_num, timer_id, timestamp, source_line, source_file, zone_name, zone_phase);

    device_events.push_back(event);

    firstTimestamp(timestamp);

    if (new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "ARCH: " << get_string_lowercase(device_architecture) << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], stat value, Run ID, zone name, zone phase, source line, source file" << std::endl;
        new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    //log_file << fmt::format("{:4},{:3},{:3},{:>7},{:7},{:15},{:15},{:5},{:>25},{:>6},{:6},{}",
    log_file << fmt::format("{},{},{},{},{},{},{},{},{},{},{},{}",
            device_id,
            core.x,
            core.y,
            tracy::riscName[risc_num],
            timer_id,
            timestamp,
            stat_value,
            run_id,
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
#if defined(PROFILER)
    ZoneScopedC(tracy::Color::Green);
    new_log = new_logs;
    output_dir = std::filesystem::path(string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME));
    std::filesystem::create_directories(output_dir);

#endif
}

DeviceProfiler::~DeviceProfiler()
{
#if defined(PROFILER)
    ZoneScoped;
    for (auto tracyCtx : device_tracy_contexts)
    {
        TracyTTDestroy(tracyCtx.second);
    }
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
        const vector<CoreCoord> &worker_cores){
#if defined(PROFILER)
    ZoneScoped;
    auto device_id = device->id();
    device_core_frequency = tt::Cluster::instance().get_device_aiclk(device_id);

    generateZoneSourceLocationsHashes();

    if (output_dram_buffer != nullptr)
    {
        std::vector<uint32_t> profile_buffer(output_dram_buffer->size()/sizeof(uint32_t), 0);

        tt_metal::detail::ReadFromBuffer(output_dram_buffer, profile_buffer);

        for (const auto &worker_core : worker_cores) {
            readRiscProfilerResults(
                device_id,
                profile_buffer,
                worker_core);

        }

        for (const auto &worker_core : worker_cores) {
            std::pair<uint32_t, CoreCoord> device_core = {device_id, worker_core};
            if (device_tracy_contexts.find(device_core) == device_tracy_contexts.end())
            {
                auto tracyCtx = TracyTTContext();
                std::string tracyTTCtxName = fmt::format("Device: {}, Core ({},{})", device_id, worker_core.x, worker_core.y);
                TracyTTContextPopulate(tracyCtx, smallest_timestamp, 1000.f / (float)device_core_frequency);
                TracyTTContextName(tracyCtx, tracyTTCtxName.c_str(), tracyTTCtxName.size());

                device_tracy_contexts.emplace(
                        device_core,
                        tracyCtx
                    );
            }
        }

        std::sort (device_events.begin(), device_events.end());

        pushTracyDeviceResults();
    }
    else
    {
        log_warning("DRAM profiler buffer is not initialized");
    }
#endif
}

void DeviceProfiler::pushTracyDeviceResults()
{
#if defined(PROFILER) && defined(TRACY_ENABLE)
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
