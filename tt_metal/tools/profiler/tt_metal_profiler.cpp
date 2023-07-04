// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "impl/debug/dprint_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/TracyTTDevice.hpp"

namespace tt {

namespace tt_metal {

void DumpDeviceProfileResults(Device *device, const Program &program) {
    auto worker_cores_used_in_program =
        device->worker_cores_from_logical_cores(program.logical_cores().at(CoreType::WORKER));

    detail::DumpDeviceProfileResults(device, worker_cores_used_in_program);
}


namespace detail {

std::map <uint32_t, DeviceProfiler> tt_metal_device_profiler_map;

void InitDeviceProfiler(Device *device){
#if defined(PROFILER)
    ZoneScoped;

    TracySetCpuTime();
    auto device_id = device->id();
    if (getDeviceProfilerState())
    {
        static std::atomic<bool> firstInit = true;

        auto device_id = device->id();
        if (tt_metal_device_profiler_map.find(device_id) == tt_metal_device_profiler_map.end())
        {
            if (firstInit.exchange(false))
            {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(true));
            }
            else
            {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(false));
            }
        }
        uint32_t dramBankCount = tt::Cluster::instance().get_soc_desc(device_id).get_num_dram_channels();
        uint32_t coreCountPerDram = tt::Cluster::instance().get_soc_desc(device_id).profiler_ceiled_core_count_perf_dram_bank;

        uint32_t pageSize =
            PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * PROFILER_RISC_COUNT * coreCountPerDram;


        if (tt_metal_device_profiler_map.at(device_id).output_dram_buffer == nullptr )
        {
            tt::tt_metal::InterleavedBufferConfig dram_config{
                        .device= device,
                        .size = pageSize * dramBankCount,
                        .page_size =  pageSize,
                        .buffer_type = tt::tt_metal::BufferType::DRAM
            };
            tt_metal_device_profiler_map.at(device_id).output_dram_buffer = tt_metal::CreateBuffer(dram_config);
        }

        std::vector<uint32_t> control_buffer(PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
        control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS] = tt_metal_device_profiler_map.at(device_id).output_dram_buffer->address();

        const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device_id);
        auto ethCores = soc_d.get_physical_ethernet_cores() ;

        for (auto &core : tt::Cluster::instance().get_soc_desc(device_id).physical_routing_to_profiler_flat_id)
        {
            if (std::find(ethCores.begin(), ethCores.end(), core.first) == ethCores.end())
            {
                tt::llrt::write_hex_vec_to_core(
                        device_id,
                        core.first,
                        control_buffer,
                        PROFILER_L1_BUFFER_CONTROL);
            }
            else
            {
                tt::llrt::write_hex_vec_to_core(
                        device_id,
                        core.first,
                        control_buffer,
                        eth_l1_mem::address_map::PROFILER_L1_BUFFER_CONTROL);
            }
        }

        std::vector<uint32_t> inputs_DRAM(tt_metal_device_profiler_map.at(device_id).output_dram_buffer->size()/sizeof(uint32_t), 0);
        tt_metal::detail::WriteToBuffer(tt_metal_device_profiler_map.at(device_id).output_dram_buffer, inputs_DRAM);
    }
#endif
}

void DumpDeviceProfileResults(Device *device, bool free_buffers) {
#if defined(PROFILER)
    std::vector<CoreCoord> workerCores;
    auto device_id = device->id();
    auto device_num_hw_cqs = device->num_hw_cqs();
    for (const CoreCoord& core : tt::get_logical_compute_cores(device_id, device_num_hw_cqs)) {
        const CoreCoord curr_core = device->worker_core_from_logical_core(core);
        workerCores.push_back(curr_core);
    }
    for (const CoreCoord& core : tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs)) {
        CoreType dispatch_core_type = tt::get_dispatch_core_type(device_id, device_num_hw_cqs);
        const auto curr_core = device->physical_core_from_logical_core(core, dispatch_core_type);
        workerCores.push_back(curr_core);
    }
    for (const CoreCoord& core : tt::Cluster::instance().get_soc_desc(device_id).physical_ethernet_cores)
    {
        workerCores.push_back(core);
    }
    DumpDeviceProfileResults(device, workerCores, free_buffers);
#endif
}

void DumpDeviceProfileResults(Device *device, std::vector<CoreCoord> &worker_cores, bool free_buffers){
#if defined(PROFILER)
    ZoneScoped;
    if (getDeviceProfilerState())
    {
        const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;
        if (USE_FAST_DISPATCH)
        {
            Finish(device->command_queue());
        }
        TT_FATAL(DprintServerIsRunning() == false, "Debug print server is running, cannot dump device profiler data");
        auto device_id = device->id();
        if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end())
        {
            tt_metal_device_profiler_map.at(device_id).setDeviceArchitecture(device->arch());
            tt_metal_device_profiler_map.at(device_id).dumpResults(device, worker_cores);
            if (free_buffers)
            {
                // Process is ending, no more device dumps are coming, reset your ref on the buffer so deallocate is the last
                // owner.
                tt_metal_device_profiler_map.at(device_id).output_dram_buffer.reset();
            }
            else
            {
                InitDeviceProfiler(device);
            }
        }
    }
#endif
}

void SetDeviceProfilerDir(std::string output_dir){
#if defined(PROFILER)
    for (auto& device_id : tt_metal_device_profiler_map)
    {
        tt_metal_device_profiler_map.at(device_id.first).setOutputDir(output_dir);
    }
#endif
}

void FreshProfilerDeviceLog(){
#if defined(PROFILER)
    for (auto& device_id : tt_metal_device_profiler_map)
    {
        tt_metal_device_profiler_map.at(device_id.first).setNewLogFlag(true);
    }
#endif
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
