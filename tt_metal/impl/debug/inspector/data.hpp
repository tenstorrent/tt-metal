// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/debug/inspector/logger.hpp"
#include "impl/debug/inspector/rpc_server_controller.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <deque>

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
    void rpc_get_mesh_workloads_runtime_ids(rpc::Inspector::GetMeshWorkloadsRuntimeIdsResults::Builder& results);
    void rpc_get_devices_in_use(rpc::Inspector::GetDevicesInUseResults::Builder& results);
    void rpc_get_kernel(
        rpc::Inspector::GetKernelParams::Reader params, rpc::Inspector::GetKernelResults::Builder results);
    void rpc_get_all_build_envs(rpc::Inspector::GetAllBuildEnvsResults::Builder results);
    void rpc_get_all_dispatch_core_infos(rpc::Inspector::GetAllDispatchCoreInfosResults::Builder results);
    void rpc_get_metal_device_id_mappings(rpc::Inspector::GetMetalDeviceIdMappingsResults::Builder results);

    static rpc::BinaryStatus convert_binary_status(ProgramBinaryStatus status);
    static void populate_core_info(rpc::CoreInfo::Builder& out, const CoreInfo& info, uint32_t event_id);
    static void populate_core_entry(
        rpc::CoreEntry::Builder& entry, const tt_cxy_pair& k, const CoreInfo& info, uint32_t event_id);
    static uint32_t get_event_id_for_core(
        const CoreInfo& info, const std::unordered_map<ChipId, std::vector<uint32_t>>& cq_to_event_by_device);
    static void populate_core_entries_by_category(
        rpc::CoreEntriesByCategory::Builder& category_builder,
        rpc::CoreCategory category_type,
        const std::unordered_map<tt_cxy_pair, CoreInfo>& core_info,
        const std::unordered_map<ChipId, std::vector<uint32_t>>& cq_to_event_by_device);

    inspector::Logger logger;
    RpcServerController rpc_server_controller;
    std::mutex programs_mutex;
    std::mutex mesh_devices_mutex;
    std::mutex mesh_workloads_mutex;
    std::mutex runtime_ids_mutex;
    // mutex to protect dispatch core info
    std::mutex dispatch_core_info_mutex;
    // mutex to protect dispatch_s core info
    std::mutex dispatch_s_core_info_mutex;
    // mutex to protect prefetcher core info
    std::mutex prefetcher_core_info_mutex;
    std::unordered_map<uint64_t, inspector::ProgramData> programs_data;
    std::unordered_map<int, uint64_t> kernel_id_to_program_id;
    std::unordered_map<int, inspector::MeshDeviceData> mesh_devices_data;
    std::unordered_map<uint64_t, inspector::MeshWorkloadData> mesh_workloads_data;
    std::deque<inspector::MeshWorkloadRuntimeIdEntry> runtime_ids;
    static constexpr size_t MAX_RUNTIME_ID_ENTRIES = 10000;
    // store dispatch core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> dispatch_core_info;
    // store dispatch_s core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> dispatch_s_core_info;
    // store prefetcher core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> prefetcher_core_info;

    // fw_compile_hash needs to be atomic because it is set in MetalContext::initialize()
    std::atomic<uint64_t> fw_compile_hash;
    friend class tt::tt_metal::Inspector;
};

}  // namespace tt::tt_metal::inspector
