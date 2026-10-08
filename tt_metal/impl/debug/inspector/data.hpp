// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/debug/inspector/logger.hpp"
#include "impl/debug/inspector/rpc_server_controller.hpp"
#include <tt-metalium/mesh_trace_id.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <atomic>
#include <cstddef>
#include <optional>
#include <unordered_set>
#include <vector>

namespace tt::tt_metal {
class MetalContext;
class MetalEnvImpl;
}  // namespace tt::tt_metal

namespace tt::tt_metal::inspector {

// An inspector session: the inspector state of one MetalContext (and the MetalEnv it uses).
// It is owned by that MetalContext and registered with the Inspector for as long as it lives.
class Data {
public:
    ~Data();

private:
    // `context` must outlive this object.
    Data(MetalContext& context, std::optional<int> rank, uint64_t fw_hash);

    // Whether tensor specs should be captured on op dispatch.
    bool capture_tensor_specs() const;

    void serialize_rpc();
    RpcServer& get_rpc_server();
    void rpc_get_programs(rpc::Inspector::GetProgramsResults::Builder& results);
    void rpc_get_mesh_devices(rpc::Inspector::GetMeshDevicesResults::Builder& results);
    void rpc_get_sockets(rpc::Inspector::GetSocketsResults::Builder& results);
    void rpc_get_global_semaphores(rpc::Inspector::GetGlobalSemaphoresResults::Builder& results);
    void rpc_get_mesh_workloads(rpc::Inspector::GetMeshWorkloadsResults::Builder& results);
    void rpc_get_mesh_workload_runtime_entries(rpc::Inspector::GetMeshWorkloadRuntimeEntriesResults::Builder& results);
    void rpc_get_devices_in_use(rpc::Inspector::GetDevicesInUseResults::Builder& results);
    void rpc_get_kernel(
        rpc::Inspector::GetKernelParams::Reader params, rpc::Inspector::GetKernelResults::Builder results);
    void rpc_get_all_build_envs(rpc::Inspector::GetAllBuildEnvsResults::Builder results);
    void rpc_get_all_dispatch_core_infos(rpc::Inspector::GetAllDispatchCoreInfosResults::Builder results);
    void rpc_get_blocks_by_type(rpc::Inspector::GetBlocksByTypeResults::Builder results);
    void rpc_get_metal_device_id_mappings(rpc::Inspector::GetMetalDeviceIdMappingsResults::Builder results);
    void rpc_get_configuration(rpc::Inspector::GetConfigurationResults::Builder& results);
    void rpc_get_system_mesh(rpc::Inspector::GetSystemMeshResults::Builder& results);

    static rpc::BinaryStatus convert_binary_status(ProgramBinaryStatus status);
    static void populate_core_ranges(
        ::capnp::List<rpc::LogicalCoreRange>::Builder list, const CoreRangeSet& core_range_set);
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

    // The owning MetalContext, for runtime state (device manager, build envs).
    MetalContext& context_;
    // The MetalEnv used by the owning context, for low level queries (HAL, cluster, rtoptions, control plane,
    // system mesh). Do not cache the control plane or system mesh: the env rebuilds them when fabric is
    // reconfigured. Inspector settings are read through the env's rtoptions on demand (except the error reporting
    // policy used by the TT_INSPECTOR_* macros, see logger.hpp).
    MetalEnvImpl& env_;

    inspector::Logger logger;
    RpcServerController rpc_server_controller;
    std::mutex programs_mutex;
    std::mutex mesh_buffers_mutex;
    std::mutex mesh_devices_mutex;
    std::mutex mesh_workloads_mutex;
    std::mutex runtime_entries_mutex;
    // mutex to protect dispatch core info
    std::mutex dispatch_core_info_mutex;
    // mutex to protect dispatch_s core info
    std::mutex dispatch_s_core_info_mutex;
    // mutex to protect prefetcher core info
    std::mutex prefetcher_core_info_mutex;
    std::unordered_map<uint64_t, inspector::ProgramData> programs_data;
    std::unordered_map<int, uint64_t> kernel_id_to_program_id;
    std::unordered_set<const distributed::MeshBuffer*> mesh_buffers_data;
    bool mesh_buffer_logging_enabled{false};
    bool mesh_socket_logging_enabled{false};
    std::unordered_map<int, inspector::MeshDeviceData> mesh_devices_data;
    std::unordered_map<const distributed::MeshBuffer*, inspector::MeshSocketData> mesh_sockets_data;
    std::unordered_map<const distributed::MeshBuffer*, inspector::GlobalSemaphoreData> global_semaphores_data;
    std::unordered_map<uint64_t, inspector::MeshWorkloadData> mesh_workloads_data;
    static constexpr size_t kRuntimeEntriesCapacity = 8192;
    std::array<inspector::MeshWorkloadRuntimeEntry, kRuntimeEntriesCapacity> runtime_entries{};
    size_t runtime_entries_write_pos{0};
    bool runtime_entries_logging_enabled{false};
    std::mutex trace_runtime_entries_mutex;
    std::unordered_map<tt::tt_metal::distributed::MeshTraceId, std::vector<inspector::MeshWorkloadRuntimeEntry>>
        trace_runtime_entries;
    // store dispatch core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> dispatch_core_info;
    // store dispatch_s core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> dispatch_s_core_info;
    // store prefetcher core info by virtual core
    std::unordered_map<tt_cxy_pair, inspector::CoreInfo> prefetcher_core_info;

    std::atomic<bool> kernel_path_collection_enabled{false};
    std::mutex kernel_path_mutex;
    std::unordered_map<int, std::vector<std::string>> kernel_id_to_processor_elf_paths;

    // Hash of the compile settings that the firmware was built with, reported over RPC.
    // Fixed at construction, i.e. before the RPC server starts.
    const uint64_t fw_compile_hash;
    friend class tt::tt_metal::Inspector;
};

}  // namespace tt::tt_metal::inspector
