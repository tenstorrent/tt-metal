// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <thread>
#include <cmath>

#include "tt_metal/host_api.hpp"
#include "impl/debug/dprint_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/TracyTTDevice.hpp"
#include "tt_metal/impl/device/device_impl.hpp"

namespace tt {

namespace tt_metal {

void DumpDeviceProfileResults(Device* device, const Program& program) {
#if defined(TRACY_ENABLE)
    std::vector<CoreCoord> worker_cores_in_program;
    std::vector<CoreCoord> eth_cores_in_program;

    std::vector<std::vector<CoreCoord>>logical_cores = program.logical_cores();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        if (hal.get_core_type(index) == CoreType::WORKER) {
            worker_cores_in_program = device->worker_cores_from_logical_cores(logical_cores[index]);
        }
        if (hal.get_core_type(index) == CoreType::ETH) {
            eth_cores_in_program = device->ethernet_cores_from_logical_cores(logical_cores[index]);
        }
    }

    std::vector<CoreCoord> cores_in_program;
    cores_in_program.reserve(worker_cores_in_program.size() + eth_cores_in_program.size());
    std::copy(worker_cores_in_program.begin(), worker_cores_in_program.end(), std::back_inserter(cores_in_program));
    std::copy(eth_cores_in_program.begin(), eth_cores_in_program.end(), std::back_inserter(cores_in_program));

    detail::DumpDeviceProfileResults(device, cores_in_program);
#endif
}

namespace detail {

std::map <uint32_t, DeviceProfiler> tt_metal_device_profiler_map;

std::unordered_map <uint32_t, std::vector <std::pair<uint64_t,uint64_t>>> deviceHostTimePair;
std::unordered_map <uint32_t, uint64_t> smallestHostime;

std::mutex device_mutex;

constexpr CoreCoord SYNC_CORE = {0,0};

void setControlBuffer(uint32_t device_id, std::vector<uint32_t>& control_buffer)
{
#if defined(TRACY_ENABLE)
    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device_id);

    auto ethCores = soc_d.get_physical_ethernet_cores() ;
    for (auto &core : soc_d.physical_routing_to_profiler_flat_id)
    {
        if (std::find(ethCores.begin(), ethCores.end(), core.first) == ethCores.end())
        {
            //Tensix
            tt::llrt::write_hex_vec_to_core(
                    device_id,
                    core.first,
                    control_buffer,
                    PROFILER_L1_BUFFER_CONTROL);
        }
        else
        {
            //ETH
            tt::llrt::write_hex_vec_to_core(
                    device_id,
                    core.first,
                    control_buffer,
                    eth_l1_mem::address_map::PROFILER_L1_BUFFER_CONTROL);
        }
    }
#endif
}

void syncDeviceHost(Device *device, CoreCoord logical_core, std::shared_ptr<tt_metal::Program> &sync_program, bool doHeader)
{
    if (!tt::llrt::OptionsG.get_profiler_sync_enabled()) return;
    ZoneScopedC(tracy::Color::Tomato3);
    auto core = device->worker_core_from_logical_core(logical_core);
    auto device_id = device->id();

    deviceHostTimePair.emplace(device_id, (std::vector <std::pair<uint64_t,uint64_t>>){});
    smallestHostime.emplace(device_id, 0);

    constexpr uint16_t sampleCount = 249;
    if (sync_program == nullptr) {
        sync_program = std::make_shared<tt_metal::Program>();

        std::map<string, string> kernel_defines = {
            {"SAMPLE_COUNT", std::to_string(sampleCount)},
        };

        tt_metal::KernelHandle brisc_kernel = tt_metal::CreateKernel(
            *sync_program, "tt_metal/tools/profiler/sync/sync_kernel.cpp",
            logical_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .defines = kernel_defines}
            );
    }
    constexpr bool wait_for_all_cores_done = false;
    constexpr bool force_slow_dispatch = true;
    LaunchProgram(device, sync_program, wait_for_all_cores_done, force_slow_dispatch);

    std::filesystem::path output_dir = std::filesystem::path(string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME));
    std::filesystem::path log_path = output_dir / "sync_device_info.csv";
    std::ofstream log_file;

    int64_t writeSum = 0;

    constexpr int millisecond_wait = 4;

    const double tracyToSecRatio = TracyGetTimerMul();
    const int64_t tracyBaseTime = TracyGetBaseTime();
    const int64_t hostStartTime = TracyGetCpuTime();
    std::vector<int64_t> writeTimes(sampleCount);

    for (int i = 0; i < sampleCount; i++)
    {
        ZoneScopedC(tracy::Color::Tomato2);
        std::this_thread::sleep_for(std::chrono::milliseconds(millisecond_wait));
        int64_t writeStart = TracyGetCpuTime();
        uint32_t sinceStart = writeStart - hostStartTime;
        tt::Cluster::instance().write_reg(&sinceStart, tt_cxy_pair(device_id, core) , PROFILER_L1_BUFFER_CONTROL + kernel_profiler::FW_RESET_L * sizeof(uint32_t));
        writeTimes[i] = (TracyGetCpuTime() - writeStart);
    }

    std::unordered_set<CoreCoord> not_done_cores({core});
    llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);

    log_info ("SYNC PROGRAM FINISH IS DONE ON {}",device_id);
    if ((smallestHostime[device_id] == 0) || (smallestHostime[device_id] > hostStartTime))
    {
        smallestHostime[device_id] = hostStartTime;
    }

    for (auto writeTime : writeTimes)
    {
        writeSum += writeTime;
    }
    double writeOverhead = (double)writeSum / sampleCount;
    vector<std::uint32_t> sync_times = tt::llrt::read_hex_vec_from_core(
            device_id,
            core,
            PROFILER_L1_BUFFER_BR + kernel_profiler::CUSTOM_MARKERS * sizeof(uint32_t),
            (sampleCount + 1) * 2 * sizeof(uint32_t));

    uint32_t preDeviceTime = 0;
    uint32_t preHostTime = 0;
    bool firstSample = true;

    uint64_t deviceStartTime = (uint64_t(sync_times[0] & 0xFFF) << 32) | sync_times[1];
    uint32_t deviceStartTime_H = sync_times[0] & 0xFFF;
    uint32_t deviceStartTime_L = sync_times[1];
    preDeviceTime = deviceStartTime_L;

    uint32_t hostStartTime_H = 0;

    uint64_t preDeviceTimeLarge = 0;
    uint64_t preHostTimeLarge = 0;
    uint64_t firstDeviceTimeLarge = 0;
    uint64_t firstHostTimeLarge = 0;

    for (int i = 2; i < 2 * (sampleCount + 1); i += 2)
    {

        uint32_t deviceTime = sync_times[i];
        if (deviceTime < preDeviceTime) deviceStartTime_H ++;
        preDeviceTime = deviceTime;
        uint64_t deviceTimeLarge = (uint64_t(deviceStartTime_H) << 32) | deviceTime;

        uint32_t hostTime = sync_times[i + 1] + writeTimes[i/2 - 1];
        if (hostTime < preHostTime) hostStartTime_H ++;
        preHostTime = hostTime;
        uint64_t hostTimeLarge = hostStartTime - smallestHostime[device_id] + ((uint64_t(hostStartTime_H) << 32) | hostTime);

        deviceHostTimePair[device_id].push_back(std::pair<uint64_t,uint64_t> {deviceTimeLarge,hostTimeLarge});

        if (firstSample)
        {
            firstDeviceTimeLarge = deviceTimeLarge;
            firstHostTimeLarge = hostTimeLarge;
            firstSample = false;
        }

        preDeviceTimeLarge = deviceTimeLarge;
        preHostTimeLarge = hostTimeLarge;
    }

    double hostSum = 0;
    double deviceSum = 0;
    double hostSquaredSum = 0;
    double hostDeviceProductSum = 0;

    for (auto& deviceHostTime : deviceHostTimePair[device_id])
    {
        double deviceTime = deviceHostTime.first;
        double hostTime = deviceHostTime.second;

        deviceSum += deviceTime;
        hostSum += hostTime;
        hostSquaredSum += (hostTime * hostTime);
        hostDeviceProductSum += (hostTime * deviceTime);
    }

    uint16_t accumulateSampleCount = deviceHostTimePair[device_id].size();

    double frequencyFit = (hostDeviceProductSum * accumulateSampleCount - hostSum * deviceSum)  / ((hostSquaredSum * accumulateSampleCount - hostSum * hostSum) * tracyToSecRatio);

    double delay = (deviceSum - frequencyFit * hostSum * tracyToSecRatio) / accumulateSampleCount;

    log_file.open(log_path, std::ios_base::app);
    if (doHeader)
    {
        log_file << fmt::format("device id,core_x, core_y,device,host_tracy,host_real,write_overhead,host_start,delay,frequency,tracy_ratio,tracy_base_time") << std::endl;
    }
    int init = deviceHostTimePair[device_id].size() - sampleCount;
    for (int i = init ;i < deviceHostTimePair[device_id].size(); i++)
    {
        log_file << fmt::format(
                "{:5},{:5},{:5},{:20},{:20},{:20.2f},{:20},{:20},{:20.2f},{:20.15f},{:20.15f},{:20}",
                device_id,
                core.x,
                core.y,
                deviceHostTimePair[device_id][i].first,
                deviceHostTimePair[device_id][i].second,
                (double) deviceHostTimePair[device_id][i].second  * tracyToSecRatio,
                writeTimes[i - init],
                smallestHostime[device_id],
                delay,
                frequencyFit,
                tracyToSecRatio,
                tracyBaseTime
                )
            << std::endl;
    }

    log_info("Sync data for device: {}, c:{}, d:{}, f:{}",device_id, smallestHostime[device_id], delay, frequencyFit);

    tt_metal_device_profiler_map.at(device_id).device_core_sync_info.emplace(core, std::make_tuple(smallestHostime[device_id], delay, frequencyFit));
    tt_metal_device_profiler_map.at(device_id).device_core_sync_info[core] = std::make_tuple(smallestHostime[device_id], delay, frequencyFit);
}


void ClearProfilerControlBuffer(Device *device){
    auto device_id = device->id();
    std::vector<uint32_t> control_buffer(PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    setControlBuffer (device_id, control_buffer);
}


void InitDeviceProfiler(Device *device){
#if defined(TRACY_ENABLE)
    ZoneScoped;

    auto device_id = device->id();
    CoreCoord logical_grid_size = device->logical_grid_size();
    TracySetCpuTime (TracyGetCpuTime());

    bool doSync = false;
    if (getDeviceProfilerState())
    {
        static std::atomic<bool> firstInit = true;
        bool doHeader = firstInit;

        auto device_id = device->id();

        if (tt_metal_device_profiler_map.find(device_id) == tt_metal_device_profiler_map.end())
        {
            doSync = true;
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
        setControlBuffer (device_id, control_buffer);

        std::vector<uint32_t> inputs_DRAM(tt_metal_device_profiler_map.at(device_id).output_dram_buffer->size()/sizeof(uint32_t), 0);
        tt_metal::detail::WriteToBuffer(tt_metal_device_profiler_map.at(device_id).output_dram_buffer, inputs_DRAM);
        if (doSync)
        {
            syncDeviceHost (device, SYNC_CORE, tt_metal_device_profiler_map.at(device_id).sync_program, doHeader);
        }
    }
#endif
}

void DumpDeviceProfileResults(Device *device, bool lastDump) {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    std::vector<CoreCoord> workerCores;
    auto device_id = device->id();
    auto device_num_hw_cqs = device->num_hw_cqs();
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device_id);
    for (const CoreCoord& core : tt::get_logical_compute_cores(device_id, device_num_hw_cqs, dispatch_core_type)) {
        const CoreCoord curr_core = device->worker_core_from_logical_core(core);
        workerCores.push_back(curr_core);
    }
    for (const CoreCoord& core : device->get_active_ethernet_cores(true)){
        auto physicalCore = device->physical_core_from_logical_core(core, CoreType::ETH);
        workerCores.push_back(physicalCore);
    }
    device->push_work([device, workerCores, lastDump] () mutable {
        DumpDeviceProfileResults(device, workerCores, lastDump);
    });

#endif
}


void DumpDeviceProfileResults(Device *device, std::vector<CoreCoord> &worker_cores, bool lastDump){
#if defined(TRACY_ENABLE)
    ZoneScoped;

    std::scoped_lock<std::mutex> lock(device_mutex);
    auto dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    if (tt::llrt::OptionsG.get_profiler_do_dispatch_cores()) {
        auto device_id = device->id();
        auto device_num_hw_cqs = device->num_hw_cqs();
        for (const CoreCoord& core : tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_type)) {
            const auto curr_core = device->physical_core_from_logical_core(core, dispatch_core_type);
            worker_cores.push_back(curr_core);
        }
        for (const CoreCoord& core : tt::Cluster::instance().get_soc_desc(device_id).physical_ethernet_cores){
            worker_cores.push_back(core);
        }
    }
    if (getDeviceProfilerState())
    {
	if (!lastDump)
	{
	    const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;
	    if (USE_FAST_DISPATCH)
	    {
		Finish(device->command_queue());
	    }
	}
        else
        {
            if (tt::llrt::OptionsG.get_profiler_do_dispatch_cores())
            {
                bool waitForDispatch = true;
                uint8_t loopCount = 0;
                CoreCoord unfinishedCore = {0,0};
                constexpr uint8_t maxLoopCount = 10;
                constexpr uint32_t loopDuration_us = 10000;
                while (waitForDispatch)
                {
                    waitForDispatch = false;
                    std::this_thread::sleep_for(std::chrono::microseconds(loopDuration_us));
                    auto device_id = device->id();
                    auto device_num_hw_cqs = device->num_hw_cqs();
                    loopCount++;
                    if (loopCount > maxLoopCount)
                    {
                        std::string msg = fmt::format(
                                "Device profiling never finished on device {}, worker core {}, {}",
                                device_id, unfinishedCore.x, unfinishedCore.y);
                        TracyMessageC(msg.c_str(), msg.size(), tracy::Color::Tomato3);
                        log_warning(msg.c_str());
                        break;
                    }
                    for (const CoreCoord& core :
                         tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_type)) {
                        const auto curr_core = device->physical_core_from_logical_core(core, dispatch_core_type);
                        vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
                                device_id,
                                curr_core,
                                PROFILER_L1_BUFFER_CONTROL,
                                PROFILER_L1_CONTROL_BUFFER_SIZE);
                        if (control_buffer[kernel_profiler::PROFILER_DONE] == 0)
                        {
                            unfinishedCore = curr_core;
                            waitForDispatch = true;
                            continue;
                        }
                    }
                    if (waitForDispatch)
                    {
                        continue;
                    }
                    for (const CoreCoord& core : tt::Cluster::instance().get_soc_desc(device_id).physical_ethernet_cores)
                    {
                        vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
                                device_id,
                                core,
                                eth_l1_mem::address_map::PROFILER_L1_BUFFER_CONTROL,
                                PROFILER_L1_CONTROL_BUFFER_SIZE);
                        if (control_buffer[kernel_profiler::PROFILER_DONE] == 0)
                        {
                            unfinishedCore = core;
                            waitForDispatch = true;
                            continue;
                        }
                    }

                }
            }
        }
	TT_FATAL(DprintServerIsRunning() == false, "Debug print server is running, cannot dump device profiler data");
        auto device_id = device->id();

        if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end())
        {
            if (!lastDump)
            {
                syncDeviceHost (device, SYNC_CORE, tt_metal_device_profiler_map.at(device_id).sync_program, false);
            }
            tt_metal_device_profiler_map.at(device_id).setDeviceArchitecture(device->arch());
            tt_metal_device_profiler_map.at(device_id).dumpResults(device, worker_cores, lastDump);
            if (lastDump)
            {
                // Process is ending, no more device dumps are coming, reset your ref on the buffer so deallocate is the last
                // owner. Sync program also contains a buffer so it is safter to release it here
                tt_metal_device_profiler_map.at(device_id).output_dram_buffer.reset();
                tt_metal_device_profiler_map.at(device_id).sync_program.reset();
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
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map)
    {
        tt_metal_device_profiler_map.at(device_id.first).setOutputDir(output_dir);
    }
#endif
}

void FreshProfilerDeviceLog(){
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map)
    {
        tt_metal_device_profiler_map.at(device_id.first).setNewLogFlag(true);
    }
#endif
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
