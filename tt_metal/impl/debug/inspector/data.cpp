// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data.hpp"
#include <stdexcept>
#include "impl/debug/inspector/rpc_server_controller.hpp"
#include "impl/debug/inspector/logger.hpp"
#include "impl/dispatch/system_memory_manager.hpp"
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include <tt-metalium/device_pool.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal::inspector {


Data::Data()
    : logger(MetalContext::instance().rtoptions().get_inspector_log_path()) {

    // Initialize RPC server if enabled
    const auto& rtoptions = MetalContext::instance().rtoptions();
    if (rtoptions.get_inspector_rpc_server_enabled()) {
        try {
            auto address = rtoptions.get_inspector_rpc_server_address();
            rpc_server_controller.start(address);

            // Connect callbacks that we want to respond to
            get_rpc_server().setGetProgramsCallback([this](auto result) { this->rpc_get_programs(result); });
            get_rpc_server().setGetMeshDevicesCallback([this](auto result) { this->rpc_get_mesh_devices(result); });
            get_rpc_server().setGetMeshWorkloadsCallback([this](auto result) { this->rpc_get_mesh_workloads(result); });
            get_rpc_server().setGetDevicesInUseCallback([this](auto result) { this->rpc_get_devices_in_use(result); });
            get_rpc_server().setGetKernelCallback(
                [this](auto params, auto result) { this->rpc_get_kernel(params, result); });
            get_rpc_server().setGetAllBuildEnvsCallback([this](auto result) { this->rpc_get_all_build_envs(result); });
            get_rpc_server().setGetDispatchCoreInfoCallback(
                [this](auto params, auto result) { this->rpc_get_dispatch_core_info(params, result); });
            get_rpc_server().setGetPrefetchCoreInfoCallback(
                [this](auto params, auto result) { this->rpc_get_prefetch_core_info(params, result); });
            get_rpc_server().setGetAllDispatchCoreInfosCallback(
                [this](auto result) { this->rpc_get_all_dispatch_core_infos(result); });
            get_rpc_server().setGetDispatchSCoreInfoCallback(
                [this](auto params, auto result) { this->rpc_get_dispatch_s_core_info(params, result); });
            get_rpc_server().setGetAllDispatchSCoreInfosCallback(
                [this](auto result) { this->rpc_get_all_dispatch_s_core_infos(result); });
            get_rpc_server().setGetAllPrefetchCoreInfosCallback(
                [this](auto result) { this->rpc_get_all_prefetch_core_infos(result); });
        } catch (const std::exception& e) {
            TT_INSPECTOR_THROW("Failed to start Inspector RPC server: {}", e.what());
        }
    }
}

Data::~Data() {
    rpc_server_controller.stop();
}

RpcServer& Data::get_rpc_server() {
    return rpc_server_controller.get_rpc_server();
}

void Data::serialize_rpc() {
    rpc_server_controller.get_rpc_server().serialize(logger.get_logging_path());
}

void Data::rpc_get_programs(rpc::Inspector::GetProgramsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto programs = results.initPrograms(programs_data.size());
    uint32_t i = 0;

    for (const auto& [program_id, program_data] : programs_data) {
        auto program = programs[i++];

        // Set basic program info
        program.setProgramId(program_id);

        // Check if program is compiled (has finished compilation)
        bool compiled = program_data.compile_finished_timestamp != inspector::time_point{};
        program.setCompiled(compiled);

        // Set binary status per device
        auto binary_status_list = program.initBinaryStatusPerDevice(program_data.binary_status_per_device.size());
        uint32_t j = 0;
        for (const auto& [device_id, status] : program_data.binary_status_per_device) {
            auto device_status = binary_status_list[j++];
            device_status.setDeviceId(static_cast<uint64_t>(device_id));
            device_status.setStatus(convert_binary_status(status));
        }

        // Set kernels
        auto kernels_list = program.initKernels(program_data.kernels.size());
        j = 0;
        for (const auto& [kernel_id, kernel_data] : program_data.kernels) {
            auto kernel = kernels_list[j++];
            kernel.setWatcherKernelId(kernel_data.watcher_kernel_id);
            kernel.setName(kernel_data.name);
            kernel.setPath(kernel_data.path);
            kernel.setSource(kernel_data.source);
            kernel.setProgramId(program_id);
        }
    }
}

void Data::rpc_get_mesh_devices(rpc::Inspector::GetMeshDevicesResults::Builder& results) {
    std::lock_guard<std::mutex> lock(mesh_devices_mutex);
    auto mesh_devices = results.initMeshDevices(mesh_devices_data.size());
    uint32_t i = 0;
    for (const auto& [mesh_id, mesh_device_data] : mesh_devices_data) {
        auto mesh_device = mesh_devices[i++];
        mesh_device.setMeshId(mesh_id);

        uint32_t j = 0;
        auto devices_view = mesh_device_data.mesh_device->get_devices();
        auto devices = mesh_device.initDevices(devices_view.size());
        for (const auto& device : devices_view) {
            devices.set(j++, device->id());
        }

        auto& shape_view = mesh_device_data.mesh_device->get_view().shape();
        auto shape = mesh_device.initShape(shape_view.dims());
        for (size_t k = 0; k < shape_view.dims(); ++k) {
            shape.set(k, shape_view.get_stride(k));
        }

        mesh_device.setParentMeshId(mesh_device_data.parent_mesh_id.value_or(-1));
        mesh_device.setInitialized(mesh_device_data.initialized);
    }
}

void Data::rpc_get_mesh_workloads(rpc::Inspector::GetMeshWorkloadsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(mesh_workloads_mutex);
    auto mesh_workloads = results.initMeshWorkloads(mesh_workloads_data.size());
    uint32_t i = 0;
    for (const auto& [mesh_workload_id, mesh_workload_data] : mesh_workloads_data) {
        auto mesh_workload = mesh_workloads[i++];
        mesh_workload.setMeshWorkloadId(mesh_workload_id);

        auto& programs = mesh_workload_data.mesh_workload->get_programs();
        auto programs_data = mesh_workload.initPrograms(programs.size());
        uint32_t j = 0;
        for (const auto& [device_range, program] : programs) {
            auto program_data = programs_data[j++];
            program_data.setProgramId(program.impl().get_id());
            auto coordinates_list = program_data.initCoordinates(device_range.shape().mesh_size());
            uint32_t k = 0;
            for (auto& device_coordinate : device_range) {
                auto mesh_coordinate = coordinates_list[k++];
                auto coords = device_coordinate.coords();
                auto coordinates = mesh_coordinate.initCoordinates(coords.size());
                for (size_t l = 0; l < coords.size(); ++l) {
                    coordinates.set(l, coords[l]);
                }
            }
        }

        auto binary_status_list = mesh_workload.initBinaryStatusPerMeshDevice(mesh_workload_data.binary_status_per_device.size());
        j = 0;
        for (const auto& [mesh_id, status] : mesh_workload_data.binary_status_per_device) {
            auto binary_status = binary_status_list[j++];
            binary_status.setMeshId(mesh_id);
            binary_status.setStatus(convert_binary_status(status));
        }
    }
}

void Data::rpc_get_devices_in_use(rpc::Inspector::GetDevicesInUseResults::Builder& results) {
    // Get all active device ids
    auto device_ids = DevicePool::instance().get_all_active_device_ids();

    // Write result
    auto result_device_ids = results.initDeviceIds(device_ids.size());
    size_t i = 0;
    for (const auto& device_id : device_ids) {
        result_device_ids.set(i++, device_id);
    }
}

void Data::rpc_get_kernel(rpc::Inspector::GetKernelParams::Reader params, rpc::Inspector::GetKernelResults::Builder results) {
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto kernel_id = params.getWatcherKernelId();
    auto program_id_it = kernel_id_to_program_id.find(kernel_id);
    if (program_id_it == kernel_id_to_program_id.end()) {
        throw std::runtime_error("Kernel not found");
    }
    auto program_id = program_id_it->second;
    auto program_data = programs_data.find(program_id);
    if (program_data == programs_data.end()) {
        throw std::runtime_error("Program not found");
    }
    auto kernel_data_it = program_data->second.kernels.find(kernel_id);
    if (kernel_data_it == program_data->second.kernels.end()) {
        throw std::runtime_error("Kernel not found inside the program");
    }
    auto& kernel_data = kernel_data_it->second;
    auto kernel = results.initKernel();
    kernel.setWatcherKernelId(kernel_data.watcher_kernel_id);
    kernel.setName(kernel_data.name);
    kernel.setPath(kernel_data.path);
    kernel.setSource(kernel_data.source);
    kernel.setProgramId(program_id);
}

// Get build environment information for all devices
// This allows Inspector clients (e.g. tt-triage) to get the correct firmware path
// for each device and build config, enabling correct firmware path resolution
// without relying on relative paths
// Declared here in Data to centralize Inspector RPC callback registration and
// tie it to Inspector Data's lifetime
void Data::rpc_get_all_build_envs(rpc::Inspector::GetAllBuildEnvsResults::Builder results) {
    // Get build environment info for all devices
    // Calls to BuildEnvManager::get_all_build_envs_info are thread-safe as it's protected by an internal mutex
    const auto& build_envs_info = BuildEnvManager::get_instance().get_all_build_envs_info();
    // Populate RPC response with build environment info for all devices
    auto result_build_envs = results.initBuildEnvs(build_envs_info.size());
    const auto fw_compile_hash = this->fw_compile_hash.load(std::memory_order_acquire);
    size_t i = 0;
    for (const auto& build_env : build_envs_info) {
        auto item = result_build_envs[i++];
        item.setDeviceId(build_env.device_id);
        // Populate RPC response with build environment info
        auto build_info = item.initBuildInfo();
        build_info.setBuildKey(build_env.build_key);
        build_info.setFirmwarePath(build_env.firmware_root_path);
        build_info.setFwCompileHash(fw_compile_hash);
    }
}

// Get dispatch core info by virtual core
void Data::rpc_get_dispatch_core_info(
    rpc::Inspector::GetDispatchCoreInfoParams::Reader params,
    rpc::Inspector::GetDispatchCoreInfoResults::Builder results) {
    // This populates the cq_to_event_by_device map with an on-demand snapshot
    // of the command queue event info
    this->rpc_all_command_queue_event_infos();
    std::scoped_lock locks(dispatch_core_info_mutex, cq_to_event_by_device_mutex);
    // Get the key and find the core info
    const auto key = params.getKey();
    const tt_cxy_pair key_cxy{key.getChip(), key.getX(), key.getY()};
    const auto it = dispatch_core_info.find(key_cxy);
    if (it == dispatch_core_info.end()) {
        throw std::runtime_error("Dispatch core info not found");
    }
    const auto& info = it->second;
    // Get the event id for the core's command queue
    uint32_t event_id = this->get_event_id_for_core(info);
    // Populate the results
    auto out = results.initInfo();
    this->populate_core_info(out, info, event_id);
}

// Get dispatch_s core info by virtual core
void Data::rpc_get_dispatch_s_core_info(
    rpc::Inspector::GetDispatchSCoreInfoParams::Reader params,
    rpc::Inspector::GetDispatchSCoreInfoResults::Builder results) {
    // This populates the cq_to_event_by_device map with an on-demand snapshot
    // of the command queue event info
    this->rpc_all_command_queue_event_infos();
    std::scoped_lock locks(dispatch_s_core_info_mutex, cq_to_event_by_device_mutex);
    const auto key = params.getKey();
    const tt_cxy_pair key_cxy{key.getChip(), key.getX(), key.getY()};
    const auto it = dispatch_s_core_info.find(key_cxy);
    if (it == dispatch_s_core_info.end()) {
        throw std::runtime_error("Dispatch_s core info not found");
    }
    const auto& info = it->second;
    // Get the event id for the core's command queue
    uint32_t event_id = this->get_event_id_for_core(info);
    // Populate the results
    auto out = results.initInfo();
    this->populate_core_info(out, info, event_id);
}

// Get prefetch core info by virtual core
void Data::rpc_get_prefetch_core_info(
    rpc::Inspector::GetPrefetchCoreInfoParams::Reader params,
    rpc::Inspector::GetPrefetchCoreInfoResults::Builder results) {
    // This populates the cq_to_event_by_device map with an on-demand snapshot
    // of the command queue event info
    this->rpc_all_command_queue_event_infos();
    std::scoped_lock locks(prefetcher_core_info_mutex, cq_to_event_by_device_mutex);
    const auto key = params.getKey();
    const tt_cxy_pair key_cxy{key.getChip(), key.getX(), key.getY()};
    const auto it = prefetcher_core_info.find(key_cxy);
    if (it == prefetcher_core_info.end()) {
        throw std::runtime_error("Prefetcher core info not found");
    }
    const auto& info = it->second;
    // Get the event id for the core's command queue
    uint32_t event_id = this->get_event_id_for_core(info);
    // Populate the results
    auto out = results.initInfo();
    this->populate_core_info(out, info, event_id);
}

// Get all dispatch core info
void Data::rpc_get_all_dispatch_core_infos(rpc::Inspector::GetAllDispatchCoreInfosResults::Builder results) {
    // This populates the cq_to_event_by_device map with an on-demand snapshot
    // of the command queue event info
    this->rpc_all_command_queue_event_infos();
    // Lock to protect cq_to_event_by_device and dispatch_core_info
    std::scoped_lock locks(dispatch_core_info_mutex, cq_to_event_by_device_mutex);
    // Populate the results with the dispatch core info  and corresponding cq_id event info
    auto list = results.initEntries(dispatch_core_info.size());
    size_t i = 0;
    for (const auto& kv : dispatch_core_info) {
        // Get key, value from dispatch_core_info
        const tt_cxy_pair& k = kv.first;
        const auto& info = kv.second;
        // Get the event id for the core's command queue
        uint32_t event_id = this->get_event_id_for_core(info);
        // Populate the core entry with the key, info, and event id
        auto entry = list[i++];
        this->populate_core_entry(entry, k, info, event_id);
    }
}

// Get all dispatch_s core info
void Data::rpc_get_all_dispatch_s_core_infos(rpc::Inspector::GetAllDispatchSCoreInfosResults::Builder results) {
    // This populates the cq_to_event_by_device map with an on-demand snapshot
    // of the command queue event info
    this->rpc_all_command_queue_event_infos();
    // Lock to protect cq_to_event_by_device and dispatch_core_info
    std::scoped_lock locks(dispatch_s_core_info_mutex, cq_to_event_by_device_mutex);
    // Populate the results with the dispatch core info  and corresponding cq_id event info
    auto list = results.initEntries(dispatch_s_core_info.size());
    size_t i = 0;
    for (const auto& kv : dispatch_s_core_info) {
        // Get key, value from dispatch_s_core_info
        const tt_cxy_pair& k = kv.first;
        const auto& info = kv.second;
        // Get the event id for the core's command queue
        uint32_t event_id = this->get_event_id_for_core(info);
        // Populate the core entry with the key, info, and event id
        auto entry = list[i++];
        this->populate_core_entry(entry, k, info, event_id);
    }
}

// Get all prefetch core info
void Data::rpc_get_all_prefetch_core_infos(rpc::Inspector::GetAllPrefetchCoreInfosResults::Builder results) {
    // This populates the cq_to_event_by_device map with an on-demand snapshot
    // of the command queue event info
    this->rpc_all_command_queue_event_infos();
    // Lock to protect cq_to_event_by_device and prefetcher_core_info
    std::scoped_lock locks(prefetcher_core_info_mutex, cq_to_event_by_device_mutex);
    // Populate the results with the dispatch core info  and corresponding cq_id event info
    auto list = results.initEntries(prefetcher_core_info.size());
    size_t i = 0;
    for (const auto& kv : prefetcher_core_info) {
        // Get key, value from prefetcher_core_info
        const tt_cxy_pair& k = kv.first;
        const auto& info = kv.second;
        // Get the event id for the core's command queue
        uint32_t event_id = this->get_event_id_for_core(info);
        // Populate the core entry with the key, info, and event id
        auto entry = list[i++];
        this->populate_core_entry(entry, k, info, event_id);
    }
}

// Get all devices from mesh on-demand (snapshot)
// For each device, get system manager queue
// Get last issued event_id for each cq
// Append to results
void Data::rpc_all_command_queue_event_infos() {
    // Get all active devices
    auto map = DevicePool::instance().get_all_command_queue_event_infos();
    std::lock_guard<std::mutex> lock(cq_to_event_by_device_mutex);
    cq_to_event_by_device = std::move(map);
}

// Helper function to convert internal enum to Cap'n Proto enum
rpc::BinaryStatus Data::convert_binary_status(ProgramBinaryStatus status) {
    switch (status) {
        case ProgramBinaryStatus::NotSent:
            return rpc::BinaryStatus::NOT_SENT;
        case ProgramBinaryStatus::InFlight:
            return rpc::BinaryStatus::IN_FLIGHT;
        case ProgramBinaryStatus::Committed:
            return rpc::BinaryStatus::COMMITTED;
        default:
            return rpc::BinaryStatus::NOT_SENT;
    }
}

// Helper function to populate the core info
void Data::populate_core_info(rpc::CoreInfo::Builder& out, const CoreInfo& info, uint32_t event_id) {
    out.setDeviceId(info.device_id);
    out.setServicingDeviceId(info.servicing_device_id);
    // Convert enum to string
    std::string worker_type_str(enchantum::to_string(info.worker_type));
    out.setWorkType(worker_type_str);
    out.setEventID(event_id);
    out.setCqId(info.cq_id);
}

// Helper function to get the event id for a core
// If not found, return std::numeric_limits<uint32_t>::max()
uint32_t Data::get_event_id_for_core(const CoreInfo& info) const {
    auto device_it = cq_to_event_by_device.find(info.device_id);
    if (device_it != cq_to_event_by_device.end() && info.cq_id < device_it->second.size()) {
        return device_it->second[info.cq_id];
    }
    return std::numeric_limits<uint32_t>::max();
}

// Helper function to populate the core entry
void Data::populate_core_entry(
    rpc::CoreEntry::Builder& entry, const tt_cxy_pair& k, const CoreInfo& info, uint32_t event_id) {
    // Populate the key
    auto key = entry.initKey();
    key.setChip(k.chip);
    key.setX(k.x);
    key.setY(k.y);
    // Populate the info
    auto out = entry.initInfo();
    this->populate_core_info(out, info, event_id);
}

}  // namespace tt::tt_metal::inspector
