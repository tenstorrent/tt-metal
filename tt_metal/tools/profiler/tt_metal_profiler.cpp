// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "impl/debug/dprint_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {

namespace tt_metal {

void DumpDeviceProfileResults(Device *device, const Program &program) {
    const auto &all_logical_cores = program.logical_cores();
    detail::DumpDeviceProfileResults(device, program.logical_cores());
}


namespace detail {

DeviceProfiler tt_metal_device_profiler;
HostProfiler tt_metal_host_profiler;

void InitDeviceProfiler(Device *device){
#if defined(PROFILER)
    ZoneScoped;

    tt::tt_metal::InterleavedBufferConfig dram_config{
                .device= device,
                .size = PROFILER_FULL_HOST_BUFFER_SIZE,
                .page_size = PROFILER_FULL_HOST_BUFFER_SIZE_PER_DRAM_BANK,
                .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    tt_metal_device_profiler.output_dram_buffer = tt_metal::CreateBuffer(dram_config);

    CoreCoord compute_with_storage_size = device->logical_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};

    std::vector<uint32_t> control_buffer(kernel_profiler::CONTROL_BUFFER_SIZE, 0);
    control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS] = tt_metal_device_profiler.output_dram_buffer.address();

    for (size_t x=start_core.x; x <= end_core.x; x++)
    {
        for (size_t y=start_core.y; y <= end_core.y; y++)
        {
            CoreCoord curr_core = {x, y};
            tt_metal::detail::WriteToDeviceL1(device, curr_core, PROFILER_L1_BUFFER_CONTROL, control_buffer);
        }
    }

    std::vector<uint32_t> inputs_DRAM(PROFILER_FULL_HOST_BUFFER_SIZE/sizeof(uint32_t), 0);
    tt_metal::detail::WriteToBuffer(tt_metal_device_profiler.output_dram_buffer, inputs_DRAM);

#endif
}

void DumpDeviceProfileResults(Device *device) {
#if defined(PROFILER)
    CoreCoord compute_with_storage_size = device->logical_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};

    std::unordered_map<CoreType, std::vector<CoreCoord>> logicalCores;
    for (size_t y=start_core.y; y <= end_core.y; y++)
    {
        for (size_t x=start_core.x; x <= end_core.x; x++)
        {
            CoreCoord logical_core = {x, y};
            logicalCores[CoreType::WORKER].push_back(logical_core);
        }
    }
    DumpDeviceProfileResults(device, logicalCores);
#endif
}

void DumpDeviceProfileResults(Device *device, const std::unordered_map<CoreType, std::vector<CoreCoord>> &logical_cores){
#if defined(PROFILER)
    ZoneScoped;
    TT_FATAL(DprintServerIsRunning() == false, "Debug print server is running, cannot dump device profiler data");
    if (getDeviceProfilerState())
    {
        auto device_id = device->id();
        tt_metal_device_profiler.setDeviceArchitecture(device->arch());
        if (logical_cores.find(CoreType::WORKER) != logical_cores.end()) {
            auto worker_cores_used_in_program =
                device->worker_cores_from_logical_cores(logical_cores.at(CoreType::WORKER));
            tt_metal_device_profiler.dumpResults(device, worker_cores_used_in_program);
        }
    }
#endif
}

void SetDeviceProfilerDir(std::string output_dir){
#if defined(PROFILER)
     tt_metal_device_profiler.setOutputDir(output_dir);
#endif
}

void SetHostProfilerDir(std::string output_dir){
#if defined(PROFILER)
     tt_metal_host_profiler.setOutputDir(output_dir);
#endif
}

void FreshProfilerHostLog(){
#if defined(PROFILER)
     tt_metal_host_profiler.setNewLogFlag(true);
#endif
}

void FreshProfilerDeviceLog(){
#if defined(PROFILER)
     tt_metal_device_profiler.setNewLogFlag(true);
#endif
}

ProfileTTMetalScope::ProfileTTMetalScope (const string& scopeNameArg) : scopeName(scopeNameArg){
#if defined(PROFILER)
    tt_metal_host_profiler.markStart(scopeName);
#endif
}

ProfileTTMetalScope::~ProfileTTMetalScope ()
{
#if defined(PROFILER)
    tt_metal_host_profiler.markStop(scopeName);
#endif
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
