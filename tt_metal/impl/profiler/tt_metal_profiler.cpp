// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <core_descriptor.hpp>
#include <device.hpp>
#include <device_pool.hpp>
#include <dispatch_core_common.hpp>
#include <host_api.hpp>
#include <profiler.hpp>
#include <mesh_workload.hpp>
#include <mesh_command_queue.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "assert.hpp"
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "dev_msgs.h"
#include "dprint_server.hpp"
#include "hal_types.hpp"
#include "hostdevcommon/profiler_common.h"
#include "impl/context/metal_context.hpp"
#include "kernel_types.hpp"
#include "llrt.hpp"
#include "llrt/hal.hpp"
#include "logger.hpp"
#include "metal_soc_descriptor.h"
#include "profiler_optional_metadata.hpp"
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "profiler_types.hpp"
#include "tt-metalium/program.hpp"
#include "rtoptions.hpp"
#include "tracy/Tracy.hpp"
#include "tracy/TracyTTDevice.hpp"
#include <tt-metalium/distributed.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/xy_pair.h>
#include "utils.hpp"

namespace tt {

namespace tt_metal {

void DumpDeviceProfileResults(IDevice* device, const Program& program) {
#if defined(TRACY_ENABLE)
    std::vector<CoreCoord> worker_cores_in_program;
    std::vector<CoreCoord> eth_cores_in_program;

    std::vector<std::vector<CoreCoord>> logical_cores = program.logical_cores();
    const auto& hal = MetalContext::instance().hal();
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

std::map<uint32_t, DeviceProfiler> tt_metal_device_profiler_map;

std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>> deviceHostTimePair;
std::unordered_map<chip_id_t, uint64_t> smallestHostime;

std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>>>
    deviceDeviceTimePair;
std::mutex device_mutex;

bool do_sync_on_close = true;
std::set<chip_id_t> sync_set_devices;
constexpr CoreCoord SYNC_CORE = {0, 0};

void setControlBuffer(IDevice* device, std::vector<uint32_t>& control_buffer) {
#if defined(TRACY_ENABLE)
    chip_id_t device_id = device->id();
    const metal_SocDescriptor& soc_d = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);

    control_buffer[kernel_profiler::CORE_COUNT_PER_DRAM] = soc_d.profiler_ceiled_core_count_perf_dram_bank;
    const auto& hal = MetalContext::instance().hal();
    for (auto core :
         tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_routing_to_profiler_flat_id(device_id)) {
        HalProgrammableCoreType CoreType;
        auto curr_core = core.first;
        if (tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(curr_core, device_id)) {
            CoreType = HalProgrammableCoreType::TENSIX;
        } else {
            CoreType = tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
            auto active_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_id);
            if (active_eth_cores.find(
                    tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
                        device_id, curr_core)) != active_eth_cores.end()) {
                CoreType = tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
            }
            auto idle_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_inactive_ethernet_cores(device_id);
            if (idle_eth_cores.find(
                    tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
                        device_id, curr_core)) != idle_eth_cores.end()) {
                CoreType = tt_metal::HalProgrammableCoreType::IDLE_ETH;
            }
        }
        profiler_msg_t* profiler_msg = hal.get_dev_addr<profiler_msg_t*>(CoreType, HalL1MemAddrType::PROFILER);

        control_buffer[kernel_profiler::FLAT_ID] = core.second;

        write_control_buffer_to_core(device, curr_core, CoreType, ProfilerDumpState::NORMAL, control_buffer);
    }
#endif
}

void syncDeviceHost(IDevice* device, CoreCoord logical_core, bool doHeader) {
    ZoneScopedC(tracy::Color::Tomato3);
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }
    auto device_id = device->id();
    auto core = device->worker_core_from_logical_core(logical_core);

    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
    auto phys_core = soc_desc.translate_coord_to(core, CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);

    deviceHostTimePair.emplace(device_id, (std::vector<std::pair<uint64_t, uint64_t>>){});
    smallestHostime.emplace(device_id, 0);

    constexpr uint16_t sampleCount = 249;
    // TODO(MO): Always recreate a new program until subdevice
    // allows using the first program generated by default manager
    tt_metal::Program sync_program;

    std::map<string, string> kernel_defines = {
        {"SAMPLE_COUNT", std::to_string(sampleCount)},
    };

    tt_metal::KernelHandle brisc_kernel = tt_metal::CreateKernel(
        sync_program,
        "tt_metal/tools/profiler/sync/sync_kernel.cpp",
        logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = kernel_defines});

    // Using MeshDevice APIs if the current device is managed by MeshDevice
    if (device->dispatch_firmware_active()) {
        if (auto mesh_device = device->get_mesh_device()) {
            auto device_coord = mesh_device->get_view().find_device(device_id);
            distributed::MeshWorkload workload;
            workload.add_program(distributed::MeshCoordinateRange(device_coord, device_coord), std::move(sync_program));
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        } else {
            EnqueueProgram(device->command_queue(), sync_program, false);
        }
    } else {
        tt_metal::detail::LaunchProgram(
            device, sync_program, false /* wait_until_cores_done */, /* force_slow_dispatch */ true);
    }

    std::filesystem::path output_dir = std::filesystem::path(get_profiler_logs_dir());
    std::filesystem::path log_path = output_dir / "sync_device_info.csv";
    std::ofstream log_file;

    int64_t writeSum = 0;

    constexpr int millisecond_wait = 10;

    const double tracyToSecRatio = TracyGetTimerMul();
    const int64_t tracyBaseTime = TracyGetBaseTime();
    const int64_t hostStartTime = TracyGetCpuTime();
    std::vector<int64_t> writeTimes(sampleCount);

    auto* profiler_msg = reinterpret_cast<profiler_msg_t*>(device->get_dev_addr(core, HalL1MemAddrType::PROFILER));
    uint64_t control_addr = reinterpret_cast<uint64_t>(&profiler_msg->control_vector[kernel_profiler::FW_RESET_L]);
    for (int i = 0; i < sampleCount; i++) {
        ZoneScopedC(tracy::Color::Tomato2);
        std::this_thread::sleep_for(std::chrono::milliseconds(millisecond_wait));
        int64_t writeStart = TracyGetCpuTime();
        uint32_t sinceStart = writeStart - hostStartTime;

        tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
            &sinceStart, tt_cxy_pair(device_id, core), control_addr);
        writeTimes[i] = (TracyGetCpuTime() - writeStart);
    }
    if (device->dispatch_firmware_active()) {
        if (auto mesh_device = device->get_mesh_device()) {
            mesh_device->mesh_command_queue().finish();
        } else {
            Finish(device->command_queue());
        }
    } else {
        tt_metal::detail::WaitProgramDone(device, sync_program, false);
    }

    log_info("SYNC PROGRAM FINISH IS DONE ON {}", device_id);
    if ((smallestHostime[device_id] == 0) || (smallestHostime[device_id] > hostStartTime)) {
        smallestHostime[device_id] = hostStartTime;
    }

    for (auto writeTime : writeTimes) {
        writeSum += writeTime;
    }
    double writeOverhead = (double)writeSum / sampleCount;

    constexpr uint32_t briscIndex = 0;
    uint64_t addr = reinterpret_cast<uint64_t>(&profiler_msg->buffer[briscIndex][kernel_profiler::CUSTOM_MARKERS]);

    std::vector<std::uint32_t> sync_times =
        tt::llrt::read_hex_vec_from_core(device_id, core, addr, (sampleCount + 1) * 2 * sizeof(uint32_t));

    uint32_t preDeviceTime = 0;
    uint32_t preHostTime = 0;
    bool firstSample = true;

    uint64_t deviceStartTime = (uint64_t(sync_times[0] & 0xFFF) << 32) | sync_times[1];
    uint32_t deviceStartTime_H = sync_times[0] & 0xFFF;
    uint32_t deviceStartTime_L = sync_times[1];
    preDeviceTime = deviceStartTime_L;

    uint32_t hostStartTime_H = 0;

    for (int i = 2; i < 2 * (sampleCount + 1); i += 2) {
        uint32_t deviceTime = sync_times[i];
        if (deviceTime < preDeviceTime) {
            deviceStartTime_H++;
        }
        preDeviceTime = deviceTime;
        uint64_t deviceTimeLarge = (uint64_t(deviceStartTime_H) << 32) | deviceTime;

        uint32_t hostTime = sync_times[i + 1] + writeTimes[i / 2 - 1];
        if (hostTime < preHostTime) {
            hostStartTime_H++;
        }
        preHostTime = hostTime;
        uint64_t hostTimeLarge =
            hostStartTime - smallestHostime[device_id] + ((uint64_t(hostStartTime_H) << 32) | hostTime);

        deviceHostTimePair[device_id].push_back(std::pair<uint64_t, uint64_t>{deviceTimeLarge, hostTimeLarge});

        if (firstSample) {
            firstSample = false;
        }
    }

    double hostSum = 0;
    double deviceSum = 0;
    double hostSquaredSum = 0;
    double hostDeviceProductSum = 0;

    for (auto& deviceHostTime : deviceHostTimePair[device_id]) {
        double deviceTime = deviceHostTime.first;
        double hostTime = deviceHostTime.second;

        deviceSum += deviceTime;
        hostSum += hostTime;
        hostSquaredSum += (hostTime * hostTime);
        hostDeviceProductSum += (hostTime * deviceTime);
    }

    uint16_t accumulateSampleCount = deviceHostTimePair[device_id].size();

    double frequencyFit = (hostDeviceProductSum * accumulateSampleCount - hostSum * deviceSum) /
                          ((hostSquaredSum * accumulateSampleCount - hostSum * hostSum) * tracyToSecRatio);

    double delay = (deviceSum - frequencyFit * hostSum * tracyToSecRatio) / accumulateSampleCount;

    if (doHeader) {
        log_file.open(log_path);
        log_file << fmt::format(
                        "device id,core_x, "
                        "core_y,device,host_tracy,host_real,write_overhead,host_start,delay,frequency,tracy_ratio,"
                        "tracy_base_time,device_frequency_ratio,device_shift")
                 << std::endl;
    } else {
        log_file.open(log_path, std::ios_base::app);
    }

    int init = deviceHostTimePair[device_id].size() - sampleCount;
    for (int i = init; i < deviceHostTimePair[device_id].size(); i++) {
        log_file << fmt::format(
                        "{:5},{:5},{:5},{:20},{:20},{:20.2f},{:20},{:20},{:20.2f},{:20.15f},{:20.15f},{:20},1.0,0",
                        device_id,
                        phys_core.x,
                        phys_core.y,
                        deviceHostTimePair[device_id][i].first,
                        deviceHostTimePair[device_id][i].second,
                        (double)deviceHostTimePair[device_id][i].second * tracyToSecRatio,
                        writeTimes[i - init],
                        smallestHostime[device_id],
                        delay,
                        frequencyFit,
                        tracyToSecRatio,
                        tracyBaseTime)
                 << std::endl;
    }
    log_file.close();
    log_info(
        "Host sync data for device: {}, cpu_start:{}, delay:{}, freq:{} Hz",
        device_id,
        smallestHostime[device_id],
        delay,
        frequencyFit);

    double host_timestamp = hostStartTime;
    double device_timestamp = delay + (host_timestamp - smallestHostime[device_id]) * frequencyFit * tracyToSecRatio;
    tt_metal_device_profiler_map.at(device_id).device_core_sync_info[phys_core] =
        std::make_tuple(host_timestamp, device_timestamp, frequencyFit);
}

void setShift(int device_id, int64_t shift, double scale, std::tuple<double, double, double>& root_sync_info) {
    if (std::isnan(scale)) {
        return;
    }
    log_info("Device sync data for device: {}, delay: {} ns, freq scale: {}", device_id, shift, scale);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_tracy_mid_run_push()) {
        log_warning(
            "Note that tracy mid-run push is enabled. This means device-device sync is not as accurate. "
            "Please do not use tracy mid-run push for sensitive device-device event analysis.");
    }
    if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end()) {
        tt_metal_device_profiler_map.at(device_id).shift = shift;
        tt_metal_device_profiler_map.at(device_id).freqScale = scale;
        tt_metal_device_profiler_map.at(device_id).setSyncInfo(root_sync_info);

        std::filesystem::path output_dir = std::filesystem::path(get_profiler_logs_dir());
        std::filesystem::path log_path = output_dir / "sync_device_info.csv";
        std::ofstream log_file;
        log_file.open(log_path, std::ios_base::app);
        log_file << fmt::format("{:5},,,,,,,,,,,,{:20.15f},{:20}", device_id, scale, shift) << std::endl;
        log_file.close();
    }
}

void peekDeviceData(IDevice* device, std::vector<CoreCoord>& worker_cores) {
    ZoneScoped;
    auto device_id = device->id();
    std::string zoneName = fmt::format("peek {}", device_id);
    ZoneName(zoneName.c_str(), zoneName.size());
    if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end()) {
        tt_metal_device_profiler_map.at(device_id).device_sync_new_events.clear();
        tt_metal_device_profiler_map.at(device_id).dumpResults(device, worker_cores, ProfilerDumpState::FORCE_UMD_READ);
        for (auto& event : tt_metal_device_profiler_map.at(device_id).device_events) {
            if (event.zone_name.find("SYNC-ZONE") != std::string::npos) {
                ZoneScopedN("Adding_device_sync_event");
                auto ret = tt_metal_device_profiler_map.at(device_id).device_sync_events.insert(event);
                if (ret.second) {
                    tt_metal_device_profiler_map.at(device_id).device_sync_new_events.insert(event);
                }
            }
        }
    }
}

void syncDeviceDevice(chip_id_t device_id_sender, chip_id_t device_id_receiver) {
    ZoneScopedC(tracy::Color::Tomato4);
    std::string zoneName = fmt::format("sync_device_device_{}->{}", device_id_sender, device_id_receiver);
    ZoneName(zoneName.c_str(), zoneName.size());
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }

    IDevice* device_sender = nullptr;
    IDevice* device_receiver = nullptr;

    if (tt::DevicePool::instance().is_device_active(device_id_receiver)) {
        device_receiver = tt::DevicePool::instance().get_active_device(device_id_receiver);
    }

    if (tt::DevicePool::instance().is_device_active(device_id_sender)) {
        device_sender = tt::DevicePool::instance().get_active_device(device_id_sender);
    }

    if (device_sender != nullptr and device_receiver != nullptr) {
        FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_config();
        TT_FATAL(
            fabric_config != FabricConfig::DISABLED,
            "Cannot support device to device synchronization when TT-Fabric is disabled.");
        log_info("Calling {} when TT-Fabric is enabled. This may take a while", __FUNCTION__);

        constexpr std::uint16_t sample_count = 240;
        constexpr std::uint16_t sample_size = 16;
        constexpr std::uint16_t channel_count = 1;

        const auto& active_eth_cores = device_sender->get_active_ethernet_cores(false);
        auto eth_sender_core_iter = active_eth_cores.begin();
        tt_xy_pair eth_receiver_core;
        tt_xy_pair eth_sender_core;

        chip_id_t device_id_receiver_curr = std::numeric_limits<chip_id_t>::max();
        while ((device_id_receiver != device_id_receiver_curr) and (eth_sender_core_iter != active_eth_cores.end())) {
            eth_sender_core = *eth_sender_core_iter;
            if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                    device_sender->id(), eth_sender_core)) {
                eth_sender_core_iter++;
                continue;
            }
            std::tie(device_id_receiver_curr, eth_receiver_core) =
                device_sender->get_connected_ethernet_core(eth_sender_core);
            eth_sender_core_iter++;
        }

        if (device_id_receiver != device_id_receiver_curr) {
            log_warning(
                "No eth connection could be found between device {} and {}", device_id_sender, device_id_receiver);
            return;
        }

        const std::vector<uint32_t>& ct_args = {
            channel_count, static_cast<uint32_t>(sample_count), static_cast<uint32_t>(sample_size)};

        Program program_sender;
        Program program_receiver;

        auto local_kernel = tt_metal::CreateKernel(
            program_sender,
            "tt_metal/tools/profiler/sync/sync_device_kernel_sender.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});

        auto remote_kernel = tt_metal::CreateKernel(
            program_receiver,
            "tt_metal/tools/profiler/sync/sync_device_kernel_receiver.cpp",
            eth_receiver_core,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});

        try {
            tt::tt_metal::detail::CompileProgram(device_sender, program_sender);
            tt::tt_metal::detail::CompileProgram(device_receiver, program_receiver);
        } catch (std::exception& e) {
            log_error("Failed compile: {}", e.what());
            throw e;
        }
        tt_metal::detail::LaunchProgram(
            device_sender, program_sender, false /* wait_until_cores_done */, true /* force_slow_dispatch */);
        tt_metal::detail::LaunchProgram(
            device_receiver, program_receiver, false /* wait_until_cores_done */, true /* force_slow_dispatch */);

        tt_metal::detail::WaitProgramDone(device_sender, program_sender, false);
        tt_metal::detail::WaitProgramDone(device_receiver, program_receiver, false);

        CoreCoord sender_core = {eth_sender_core.x, eth_sender_core.y};
        std::vector<CoreCoord> sender_cores = {
            device_sender->virtual_core_from_logical_core(sender_core, CoreType::ETH)};

        CoreCoord receiver_core = {eth_receiver_core.x, eth_receiver_core.y};
        std::vector<CoreCoord> receiver_cores = {
            device_receiver->virtual_core_from_logical_core(receiver_core, CoreType::ETH)};

        peekDeviceData(device_sender, sender_cores);
        peekDeviceData(device_receiver, receiver_cores);

        TT_ASSERT(
            tt_metal_device_profiler_map.at(device_id_sender).device_sync_new_events.size() ==
            tt_metal_device_profiler_map.at(device_id_receiver).device_sync_new_events.size());

        auto event_receiver = tt_metal_device_profiler_map.at(device_id_receiver).device_sync_new_events.begin();

        for (auto event_sender = tt_metal_device_profiler_map.at(device_id_sender).device_sync_new_events.begin();
             event_sender != tt_metal_device_profiler_map.at(device_id_sender).device_sync_new_events.end();
             event_sender++) {
            TT_ASSERT(event_receiver != tt_metal_device_profiler_map.at(device_id_receiver).device_sync_events.end());
            deviceDeviceTimePair.at(device_id_sender)
                .at(device_id_receiver)
                .push_back({event_sender->timestamp, event_receiver->timestamp});
            event_receiver++;
        }
    }
}

void setSyncInfo(
    chip_id_t device_id,
    std::pair<double, int64_t> syncInfo,
    std::tuple<double, double, double>& root_sync_info,
    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::pair<double, int64_t>>>& deviceDeviceSyncInfo,
    const std::string& parentInfo = "") {
    if (sync_set_devices.find(device_id) == sync_set_devices.end()) {
        sync_set_devices.insert(device_id);
        if (deviceDeviceSyncInfo.find(device_id) != deviceDeviceSyncInfo.end()) {
            std::string parentInfoNew =
                parentInfo + fmt::format("->{}: ({},{})", device_id, syncInfo.second, syncInfo.first);
            for (auto child_device : deviceDeviceSyncInfo.at(device_id)) {
                std::pair<double, int64_t> childSyncInfo = child_device.second;
                childSyncInfo.second *= syncInfo.first;
                childSyncInfo.second += syncInfo.second;
                childSyncInfo.first *= syncInfo.first;
                setSyncInfo(child_device.first, childSyncInfo, root_sync_info, deviceDeviceSyncInfo, parentInfo);
            }
        }
        detail::setShift(device_id, syncInfo.second, syncInfo.first, root_sync_info);
    }
}

void syncAllDevices(chip_id_t host_connected_device) {
    // Check if profiler on host connected device is initilized
    if (tt_metal_device_profiler_map.find(host_connected_device) == tt_metal_device_profiler_map.end()) {
        return;
    }

    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }
    // Update deviceDeviceTimePair
    for (const auto& sender : deviceDeviceTimePair) {
        for (const auto& receiver : sender.second) {
            syncDeviceDevice(sender.first, receiver.first);
        }
    }

    // Run linear regression to calculate scale and bias between devices
    // deviceDeviceSyncInfo[dev0][dev1] = {scale, bias} of dev0 over dev1
    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::pair<double, int64_t>>> deviceDeviceSyncInfo;
    for (auto& sender : deviceDeviceTimePair) {
        for (auto& receiver : sender.second) {
            std::vector<std::pair<uint64_t, uint64_t>> timePairs;
            for (int i = 0; i < receiver.second.size(); i += 2) {
                uint64_t senderTime = (receiver.second[i].first + receiver.second[i + 1].first) / 2;
                timePairs.push_back({senderTime, receiver.second[i].second});
            }
            double senderSum = 0;
            double receiverSum = 0;
            double receiverSquareSum = 0;
            double senderReceiverProductSum = 0;

            // Direct computation causes large error because sqaure of clock is very big
            // So apply linear regression on shifted values
            uint64_t senderBase = 0;
            uint64_t receiverBase = 0;

            if (timePairs.size() > 0) {
                senderBase = timePairs[0].first;
                receiverBase = timePairs[0].second;
            }
            for (auto& timePair : timePairs) {
                double senderTime = timePair.first - senderBase;
                double recieverTime = timePair.second - receiverBase;

                receiverSum += recieverTime;
                senderSum += senderTime;
                receiverSquareSum += (recieverTime * recieverTime);
                senderReceiverProductSum += (senderTime * recieverTime);
            }

            uint16_t accumulateSampleCount = timePairs.size();

            double freqScale = (senderReceiverProductSum * accumulateSampleCount - senderSum * receiverSum) /
                               (receiverSquareSum * accumulateSampleCount - receiverSum * receiverSum);

            uint64_t shift = (double)(senderSum - freqScale * (double)receiverSum) / accumulateSampleCount +
                             (senderBase - freqScale * receiverBase);
            deviceDeviceSyncInfo.emplace(sender.first, (std::unordered_map<chip_id_t, std::pair<double, int64_t>>){});
            deviceDeviceSyncInfo.at(sender.first)
                .emplace(receiver.first, (std::pair<double, int64_t>){freqScale, shift});

            deviceDeviceSyncInfo.emplace(receiver.first, (std::unordered_map<chip_id_t, std::pair<double, int64_t>>){});
            deviceDeviceSyncInfo.at(receiver.first)
                .emplace(sender.first, (std::pair<double, int64_t>){1.0 / freqScale, -1 * shift});
        }
    }

    // Find any sync info from root device
    // Currently, sync info only exists for SYNC_CORE
    std::tuple<double, double, double> root_sync_info;
    for (auto& [core, info] : tt_metal_device_profiler_map.at(host_connected_device).device_core_sync_info) {
        root_sync_info = info;
        break;
    }

    // Propagate sync info with DFS through sync tree
    sync_set_devices.clear();
    setSyncInfo(host_connected_device, (std::pair<double, int64_t>){1.0, 0}, root_sync_info, deviceDeviceSyncInfo);
}

void ProfilerSync(ProfilerSyncState state) {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_sync_enabled()) {
        return;
    }
    if (!getDeviceProfilerState()) {
        return;
    }
    static chip_id_t first_connected_device_id = -1;
    if (state == ProfilerSyncState::INIT) {
        do_sync_on_close = true;
        auto ethernet_connections = tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_connections();
        std::set<chip_id_t> visited_devices = {};
        constexpr int TOTAL_DEVICE_COUNT = 36;
        for (int sender_device_id = 0; sender_device_id < TOTAL_DEVICE_COUNT; sender_device_id++) {
            if (tt::DevicePool::instance().is_device_active(sender_device_id)) {
                auto sender_device = tt::DevicePool::instance().get_active_device(sender_device_id);
                const auto& active_eth_cores = sender_device->get_active_ethernet_cores(false);

                chip_id_t receiver_device_id;
                tt_xy_pair receiver_eth_core;
                bool doSync = true;
                for (auto& sender_eth_core : active_eth_cores) {
                    if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                            sender_device_id, sender_eth_core)) {
                        continue;
                    }
                    doSync = false;
                    std::tie(receiver_device_id, receiver_eth_core) =
                        sender_device->get_connected_ethernet_core(sender_eth_core);

                    // std::cout << sender_device_id << ":" << sender_eth_core.x << "," << sender_eth_core.y;
                    // std::cout << "->" << receiver_device_id << ":" << receiver_eth_core.x << ",";
                    // std::cout << receiver_eth_core.y << std::endl;

                    if (visited_devices.find(sender_device_id) == visited_devices.end() or
                        visited_devices.find(receiver_device_id) == visited_devices.end()) {
                        visited_devices.insert(sender_device_id);
                        visited_devices.insert(receiver_device_id);
                        std::pair<chip_id_t, chip_id_t> ping_pair = {sender_device_id, receiver_device_id};

                        deviceDeviceTimePair.emplace(
                            sender_device_id,
                            (std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>>){});
                        deviceDeviceTimePair.at(sender_device_id)
                            .emplace(receiver_device_id, (std::vector<std::pair<uint64_t, uint64_t>>){});
                    }
                }
                if (doSync or first_connected_device_id == -1) {
                    if (first_connected_device_id == -1 and !doSync) {
                        first_connected_device_id = sender_device_id;
                    }
                    syncDeviceHost(sender_device, SYNC_CORE, true);
                }
            }
        }
        // If at least one sender reciever pair has been found
        if (first_connected_device_id != -1) {
            syncAllDevices(first_connected_device_id);
        }
    }

    if (state == ProfilerSyncState::CLOSE_DEVICE and do_sync_on_close) {
        do_sync_on_close = false;
        // If at least one sender reciever pair has been found
        if (first_connected_device_id != -1) {
            syncAllDevices(first_connected_device_id);
        }
    }
#endif
}

void ClearProfilerControlBuffer(IDevice* device) {
#if defined(TRACY_ENABLE)
    std::vector<uint32_t> control_buffer(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    setControlBuffer(device, control_buffer);
#endif
}

void InitDeviceProfiler(IDevice* device) {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    auto device_id = device->id();
    CoreCoord logical_grid_size = device->logical_grid_size();
    TracySetCpuTime(TracyGetCpuTime());

    if (getDeviceProfilerState()) {
        static std::atomic<bool> firstInit = true;

        auto device_id = device->id();

        if (tt_metal_device_profiler_map.find(device_id) == tt_metal_device_profiler_map.end()) {
            if (firstInit.exchange(false)) {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(device, true));
            } else {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(device, false));
            }
        }

        uint32_t dramBankCount =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_num_dram_views();
        uint32_t coreCountPerDram = tt::tt_metal::MetalContext::instance()
                                        .get_cluster()
                                        .get_soc_desc(device_id)
                                        .profiler_ceiled_core_count_perf_dram_bank;

        uint32_t pageSize = PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * PROFILER_RISC_COUNT * coreCountPerDram;

        auto mesh_device = device->get_mesh_device();
        auto& profiler = tt_metal_device_profiler_map.at(device_id);
        if (profiler.output_dram_buffer.get_buffer() == nullptr && mesh_device) {
            // If buffer is not allocated, trying to re-use a buffer already allocated for another device within a
            // single MeshDevice
            for (auto neighbor_device : mesh_device->get_devices()) {
                auto neighbor_profiler_it = tt_metal_device_profiler_map.find(neighbor_device->id());
                if (neighbor_profiler_it != tt_metal_device_profiler_map.end()) {
                    auto& neighbor_profiler = neighbor_profiler_it->second;
                    if (neighbor_profiler.output_dram_buffer.get_buffer() != nullptr) {
                        profiler.output_dram_buffer = neighbor_profiler.output_dram_buffer;
                        break;
                    }
                }
            }
        }
        if (profiler.output_dram_buffer.get_buffer() == nullptr) {
            tt::tt_metal::InterleavedBufferConfig dram_config{
                .device = mesh_device ? mesh_device.get() : device,
                .size = pageSize * dramBankCount,
                .page_size = pageSize,
                .buffer_type = tt::tt_metal::BufferType::DRAM};
            profiler.output_dram_buffer = distributed::AnyBuffer::create(dram_config);
            profiler.profile_buffer.resize(profiler.output_dram_buffer.get_buffer()->size() / sizeof(uint32_t));
        }
        auto output_dram_buffer_ptr = tt_metal_device_profiler_map.at(device_id).output_dram_buffer.get_buffer();

        std::vector<uint32_t> control_buffer(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
        control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS] = output_dram_buffer_ptr->address();
        setControlBuffer(device, control_buffer);

        std::vector<uint32_t> inputs_DRAM(output_dram_buffer_ptr->size() / sizeof(uint32_t), 0);

        if (device->dispatch_firmware_active()) {
            issue_fd_write_to_profiler_buffer(profiler.output_dram_buffer, device, inputs_DRAM);
        } else {
            tt_metal::detail::WriteToBuffer(*(profiler.output_dram_buffer.get_buffer()), inputs_DRAM);
        }
    }
#endif
}

void DumpDeviceProfileResults(
    IDevice* device, ProfilerDumpState state, const std::optional<ProfilerOptionalMetadata>& metadata) {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    std::vector<CoreCoord> workerCores;
    auto device_id = device->id();
    auto device_num_hw_cqs = device->num_hw_cqs();
    const auto& dispatch_core_config = get_dispatch_core_config();
    for (const CoreCoord& core : tt::get_logical_compute_cores(device_id, device_num_hw_cqs, dispatch_core_config)) {
        const CoreCoord curr_core = device->worker_core_from_logical_core(core);
        workerCores.push_back(curr_core);
    }
    for (const CoreCoord& core : device->get_active_ethernet_cores(true)) {
        auto virtualCore = device->virtual_core_from_logical_core(core, CoreType::ETH);
        workerCores.push_back(virtualCore);
    }

    DumpDeviceProfileResults(device, workerCores, state, metadata);
    if (deviceDeviceTimePair.find(device->id()) != deviceDeviceTimePair.end() and
        state == ProfilerDumpState::CLOSE_DEVICE_SYNC) {
        for (auto& connected_device : deviceDeviceTimePair.at(device->id())) {
            chip_id_t sender_id = device->id();
            chip_id_t receiver_id = connected_device.first;
        }
    }

#endif
}

void DumpDeviceProfileResults(
    IDevice* device,
    std::vector<CoreCoord>& worker_cores,
    ProfilerDumpState state,
    const std::optional<ProfilerOptionalMetadata>& metadata) {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    std::string name = fmt::format("Device Dump {}", device->id());
    ZoneName(name.c_str(), name.size());
    std::scoped_lock<std::mutex> lock(device_mutex);
    const auto& dispatch_core_config = get_dispatch_core_config();
    auto dispatch_core_type = dispatch_core_config.get_core_type();
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_do_dispatch_cores()) {
        auto device_id = device->id();
        auto device_num_hw_cqs = device->num_hw_cqs();
        for (const CoreCoord& core :
             tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_config)) {
            const auto curr_core = device->virtual_core_from_logical_core(core, dispatch_core_type);
            worker_cores.push_back(curr_core);
        }
    }
    if (getDeviceProfilerState()) {
        if (state != ProfilerDumpState::LAST_CLOSE_DEVICE) {
            const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;
            if (USE_FAST_DISPATCH) {
                if (auto mesh_device = device->get_mesh_device()) {
                    mesh_device->mesh_command_queue().finish();
                } else {
                    Finish(device->command_queue());
                }
            }
        } else {
            if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_do_dispatch_cores()) {
                auto device_id = device->id();
                constexpr uint8_t maxLoopCount = 10;
                constexpr uint32_t loopDuration_us = 10000;
                auto device_num_hw_cqs = device->num_hw_cqs();
                std::vector<CoreCoord> dispatchCores =
                    tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_config);
                const auto& hal = MetalContext::instance().hal();
                while (dispatchCores.size() > 0) {
                    bool coreDone = false;

                    auto curr_core = device->virtual_core_from_logical_core(dispatchCores[0], dispatch_core_type);

                    HalProgrammableCoreType CoreType;
                    if (tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(curr_core, device_id)) {
                        CoreType = HalProgrammableCoreType::TENSIX;
                    } else {
                        auto active_eth_cores =
                            tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_id);
                        bool is_active_eth_core =
                            active_eth_cores.find(tt::tt_metal::MetalContext::instance()
                                                      .get_cluster()
                                                      .get_logical_ethernet_core_from_virtual(device_id, curr_core)) !=
                            active_eth_cores.end();

                        CoreType = is_active_eth_core ? tt_metal::HalProgrammableCoreType::ACTIVE_ETH
                                                      : tt_metal::HalProgrammableCoreType::IDLE_ETH;
                    }
                    profiler_msg_t* profiler_msg =
                        hal.get_dev_addr<profiler_msg_t*>(CoreType, HalL1MemAddrType::PROFILER);
                    for (int i = 0; i < maxLoopCount; i++) {
                        std::vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
                            device_id,
                            curr_core,
                            reinterpret_cast<uint64_t>(profiler_msg->control_vector),
                            kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
                        if (control_buffer[kernel_profiler::PROFILER_DONE] == 1) {
                            coreDone = true;
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::microseconds(loopDuration_us));
                    }
                    if (!coreDone) {
                        std::string msg = fmt::format(
                            "Device profiling never finished on device {}, worker core {}, {}",
                            device_id,
                            curr_core.x,
                            curr_core.y);
                        TracyMessageC(msg.c_str(), msg.size(), tracy::Color::Tomato3);
                        log_warning(msg.c_str());
                    }
                    dispatchCores.erase(dispatchCores.begin());
                }
            }
        }
        TT_FATAL(DprintServerIsRunning() == false, "Debug print server is running, cannot dump device profiler data");
        auto device_id = device->id();

        if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end()) {
            if (state != ProfilerDumpState::LAST_CLOSE_DEVICE) {
                if (deviceHostTimePair.find(device_id) != deviceHostTimePair.end()) {
                    syncDeviceHost(device, SYNC_CORE, false);
                }
            }
            tt_metal_device_profiler_map.at(device_id).setDeviceArchitecture(device->arch());
            tt_metal_device_profiler_map.at(device_id).dumpResults(device, worker_cores, state, metadata);
            if (state == ProfilerDumpState::LAST_CLOSE_DEVICE) {
                // Process is ending, no more device dumps are coming, reset your ref on the buffer so deallocate is the
                // last owner. Sync program also contains a buffer so it is safter to release it here
                tt_metal_device_profiler_map.at(device_id).output_dram_buffer = {};
                tt_metal_device_profiler_map.at(device_id).sync_program.reset();
            } else {
                InitDeviceProfiler(device);
            }
            if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_tracy_mid_run_push()) {
                tt_metal_device_profiler_map.at(device_id).pushTracyDeviceResults();
            }
        }
    }
#endif
}

void SetDeviceProfilerDir(const std::string& output_dir) {
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map) {
        tt_metal_device_profiler_map.at(device_id.first).setOutputDir(output_dir);
    }
#endif
}

void FreshProfilerDeviceLog() {
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map) {
        tt_metal_device_profiler_map.at(device_id.first).freshDeviceLog();
    }
#endif
}

uint32_t EncodePerDeviceProgramID(uint32_t base_program_id, uint32_t device_id, bool is_host_fallback_op) {
    // Given the base (host assigned id) for a program running on multiple devices, generate a unique per-device
    // id by coalescing the physical_device id with the program id.
    // For ops running on device, the MSB is 0. For host-fallback ops, the MSB is 1. This avoids aliasing.
    constexpr uint32_t DEVICE_ID_NUM_BITS = 10;
    constexpr uint32_t DEVICE_OP_ID_NUM_BITS = 31;
    return (is_host_fallback_op << DEVICE_OP_ID_NUM_BITS) | (base_program_id << DEVICE_ID_NUM_BITS) | device_id;
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
