// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data.hpp"
#include <stdexcept>
#include "rpc_server_controller.hpp"
#include "logger.hpp"
#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "device/device_manager.hpp"
#include <tt_stl/reflection.hpp>
#include <llrt/tt_cluster.hpp>

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
            get_rpc_server().setGetMeshWorkloadsRuntimeIdsCallback(
                [this](auto result) { this->rpc_get_mesh_workloads_runtime_ids(result); });
            get_rpc_server().setGetDevicesInUseCallback([this](auto result) { this->rpc_get_devices_in_use(result); });
            get_rpc_server().setGetKernelCallback(
                [this](auto params, auto result) { this->rpc_get_kernel(params, result); });
            get_rpc_server().setGetAllBuildEnvsCallback([this](auto result) { this->rpc_get_all_build_envs(result); });
            get_rpc_server().setGetAllDispatchCoreInfosCallback(
                [this](auto result) { this->rpc_get_all_dispatch_core_infos(result); });
            get_rpc_server().setGetMetalDeviceIdMappingsCallback(
                [this](auto result) { this->rpc_get_metal_device_id_mappings(result); });
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
            device_status.setMetalDeviceId(static_cast<uint64_t>(device_id));
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

        const auto& shape_view = mesh_device_data.mesh_device->get_view().shape();
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
        mesh_workload.setName(mesh_workload_data.name);
        mesh_workload.setParameters(mesh_workload_data.parameters);

        const auto& programs = mesh_workload_data.mesh_workload->get_programs();
        auto programs_data = mesh_workload.initPrograms(programs.size());
        uint32_t j = 0;
        for (const auto& [device_range, program] : programs) {
            auto program_data = programs_data[j++];
            program_data.setProgramId(program.impl().get_id());
            auto coordinates_list = program_data.initCoordinates(device_range.shape().mesh_size());
            uint32_t k = 0;
            for (const auto& device_coordinate : device_range) {
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

void Data::rpc_get_mesh_workloads_runtime_ids(rpc::Inspector::GetMeshWorkloadsRuntimeIdsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(runtime_ids_mutex);
    auto all_runtime_ids = results.initRuntimeIds(runtime_ids.size());
    for (size_t i = 0; i < runtime_ids.size(); ++i) {
        auto entry = all_runtime_ids[i];
        entry.setWorkloadId(runtime_ids[i].workload_id);
        entry.setRuntimeId(runtime_ids[i].runtime_id);
    }
}

void Data::rpc_get_devices_in_use(rpc::Inspector::GetDevicesInUseResults::Builder& results) {
    // Get all active device ids
    auto device_ids = tt_metal::MetalContext::instance().device_manager()->get_all_active_device_ids();

    // Write result
    auto result_device_ids = results.initMetalDeviceIds(device_ids.size());
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
        item.setMetalDeviceId(build_env.device_id);
        // Populate RPC response with build environment info
        auto build_info = item.initBuildInfo();
        build_info.setBuildKey(build_env.build_key);
        build_info.setFirmwarePath(build_env.firmware_root_path);
        build_info.setFwCompileHash(fw_compile_hash);
    }
}

// Get all dispatch core infos for all active devices
// Do an on-demand snapshot of the command queue event info
// Populate the results with the dispatch core info and corresponding cq_id event info
void Data::rpc_get_all_dispatch_core_infos(rpc::Inspector::GetAllDispatchCoreInfosResults::Builder results) {
    if (!tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        // Fast dispatch is not enabled, no dispatch core info to return
        results.initCoresByCategory(0);
        return;
    }
    // This returns a map of command queue id to event id for all active devices
    auto cq_to_event_by_device =
        tt_metal::MetalContext::instance().device_manager()->get_all_command_queue_event_infos();
    // In a single lock, get the number of non-empty categories and initialize the results
    std::scoped_lock locks(dispatch_core_info_mutex, dispatch_s_core_info_mutex, prefetcher_core_info_mutex);

    // Get the number of non-empty categories
    size_t non_empty_categories = 0;
    if (!dispatch_core_info.empty()) {
        non_empty_categories++;
    }
    if (!dispatch_s_core_info.empty()) {
        non_empty_categories++;
    }
    if (!prefetcher_core_info.empty()) {
        non_empty_categories++;
    }

    // Initialize the results with the number of non-empty categories
    auto list = results.initCoresByCategory(non_empty_categories);

    size_t category_index = 0;
    // Populate the dispatch core info
    if (!dispatch_core_info.empty()) {
        auto category = list[category_index++];
        Data::populate_core_entries_by_category(
            category, rpc::CoreCategory::DISPATCH, dispatch_core_info, cq_to_event_by_device);
    }
    // Populate the dispatch_s core info
    if (!dispatch_s_core_info.empty()) {
        auto category = list[category_index++];
        Data::populate_core_entries_by_category(
            category, rpc::CoreCategory::DISPATCH_S, dispatch_s_core_info, cq_to_event_by_device);
    }
    // Populate the prefetcher core info
    if (!prefetcher_core_info.empty()) {
        auto category = list[category_index++];
        Data::populate_core_entries_by_category(
            category, rpc::CoreCategory::PREFETCH, prefetcher_core_info, cq_to_event_by_device);
    }
}

void Data::rpc_get_metal_device_id_mappings(rpc::Inspector::GetMetalDeviceIdMappingsResults::Builder results) {
    // Get cluster descriptor from MetalContext
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& chip_id_to_unique_id = cluster.get_cluster_desc()->get_chip_unique_ids();

    // Populate RPC response
    auto result_mappings = results.initMappings(chip_id_to_unique_id.size());
    size_t i = 0;
    for (const auto& [chip_id, unique_id] : chip_id_to_unique_id) {
        auto entry = result_mappings[i++];
        entry.setMetalDeviceId(static_cast<uint64_t>(chip_id));
        entry.setUniqueId(unique_id);
    }
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
    out.setMetalDeviceId(info.device_id);
    out.setServicingMetalDeviceId(info.servicing_device_id);
    // Convert enum to string
    std::string worker_type_str(enchantum::to_string(info.worker_type));
    out.setWorkType(worker_type_str);
    out.setEventID(event_id);
    out.setCqId(info.cq_id);
}

// Helper function to get the event id for a core
// If not found, return std::numeric_limits<uint32_t>::max()
uint32_t Data::get_event_id_for_core(
    const CoreInfo& info, const std::unordered_map<ChipId, std::vector<uint32_t>>& cq_to_event_by_device) {
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
    Data::populate_core_info(out, info, event_id);
}

// Helper function to populate the core entries by category
void Data::populate_core_entries_by_category(
    rpc::CoreEntriesByCategory::Builder& category_builder,
    rpc::CoreCategory category_type,
    const std::unordered_map<tt_cxy_pair, CoreInfo>& core_info,
    const std::unordered_map<ChipId, std::vector<uint32_t>>& cq_to_event_by_device) {
    // Set the category type
    category_builder.setCategory(category_type);
    // Initialize the entries
    auto entries = category_builder.initEntries(core_info.size());
    size_t i = 0;
    for (const auto& kv : core_info) {
        // Get key, value from core_info
        const tt_cxy_pair& k = kv.first;
        const auto& info = kv.second;
        // Get the event id for the core's command queue
        uint32_t event_id = Data::get_event_id_for_core(info, cq_to_event_by_device);
        // Populate the core entry with the key, info, and event id
        auto entry = entries[i++];
        Data::populate_core_entry(entry, k, info, event_id);
    }
}

}  // namespace tt::tt_metal::inspector
