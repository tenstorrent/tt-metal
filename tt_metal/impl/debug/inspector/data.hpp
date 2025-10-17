// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/debug/inspector/logger.hpp"
#include "impl/debug/inspector/rpc_server_controller.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal::inspector {

class Data {
public:
    ~Data();

private:
    Data(); // NOLINT - False alarm, tt::tt_metal::Inspector is calling this constructor.

    void serialize_rpc();
    RpcServer& get_rpc_server();
    void rpc_get_programs(rpc::Inspector::GetProgramsResults::Builder& results);
    void rpc_get_mesh_devices(rpc::Inspector::GetMeshDevicesResults::Builder& results);
    void rpc_get_mesh_workloads(rpc::Inspector::GetMeshWorkloadsResults::Builder& results);
    void rpc_get_devices_in_use(rpc::Inspector::GetDevicesInUseResults::Builder& results);
    void rpc_get_kernel(
        rpc::Inspector::GetKernelParams::Reader params, rpc::Inspector::GetKernelResults::Builder results);
    void rpc_get_all_build_envs(rpc::Inspector::GetAllBuildEnvsResults::Builder results);
    void rpc_get_dispatch_core_info(
        rpc::Inspector::GetDispatchCoreInfoParams::Reader params,
        rpc::Inspector::GetDispatchCoreInfoResults::Builder results);
    void rpc_get_dispatch_s_core_info(
        rpc::Inspector::GetDispatchSCoreInfoParams::Reader params,
        rpc::Inspector::GetDispatchSCoreInfoResults::Builder results);
    void rpc_get_prefetch_core_info(
        rpc::Inspector::GetPrefetchCoreInfoParams::Reader params,
        rpc::Inspector::GetPrefetchCoreInfoResults::Builder results);
    void rpc_get_all_dispatch_core_infos(rpc::Inspector::GetAllDispatchCoreInfosResults::Builder results);
    void rpc_get_all_prefetch_core_infos(rpc::Inspector::GetAllPrefetchCoreInfosResults::Builder results);
    void rpc_get_all_dispatch_s_core_infos(rpc::Inspector::GetAllDispatchSCoreInfosResults::Builder results);
    void rpc_all_command_queue_event_infos();

    static rpc::BinaryStatus convert_binary_status(ProgramBinaryStatus status);

    inspector::Logger logger;
    RpcServerController rpc_server_controller;
    std::mutex programs_mutex;
    std::mutex mesh_devices_mutex;
    std::mutex mesh_workloads_mutex;
    // mutex to protect dispatch core info
    std::mutex dispatch_core_info_mutex;
    // mutex to protect dispatch_s core info
    std::mutex dispatch_s_core_info_mutex;
    // mutex to protect prefetcher core info
    std::mutex prefetcher_core_info_mutex;
    // mutex to protect command queue event info
    std::mutex cq_to_event_by_device_mutex;
    std::unordered_map<uint64_t, inspector::ProgramData> programs_data;
    std::unordered_map<int, uint64_t> kernel_id_to_program_id;
    std::unordered_map<int, inspector::MeshDeviceData> mesh_devices_data;
    std::unordered_map<uint64_t, inspector::MeshWorkloadData> mesh_workloads_data;
    // store dispatch core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> dispatch_core_info;
    // store dispatch_s core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> dispatch_s_core_info;
    // store prefetcher core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> prefetcher_core_info;
    // store command queue event info by device and cq id
    std::unordered_map<ChipId, std::vector<uint32_t>> cq_to_event_by_device;

    // fw_compile_hash needs to be atomic because it is set in MetalContext::initialize()
    std::atomic<uint64_t> fw_compile_hash;
    friend class tt::tt_metal::Inspector;
};

}  // namespace tt::tt_metal::inspector
