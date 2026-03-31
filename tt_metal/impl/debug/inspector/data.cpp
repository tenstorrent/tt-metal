// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "data.hpp"
#include <stdexcept>
#include "rpc_server_controller.hpp"
#include "logger.hpp"
#include <tt-metalium/experimental/inspector_config.hpp>
#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "device/device_manager.hpp"
#include <llrt/tt_cluster.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <fmt/format.h>

namespace tt::tt_metal::inspector {

std::string stringify_tensor_specs(const std::vector<TensorSpec>& tensor_specs) {
    if (tensor_specs.empty()) {
        return "Not captured";
    }

    constexpr size_t TENSOR_ARGS_BUFFER_SIZE = 4096;
    fmt::memory_buffer buf;
    buf.reserve(TENSOR_ARGS_BUFFER_SIZE);
    for (size_t i = 0; i < tensor_specs.size(); ++i) {
        if (i > 0) {
            fmt::format_to(std::back_inserter(buf), ", ");
        }
        fmt::format_to(std::back_inserter(buf), "[{}]: {}", i, tensor_specs[i]);
    }
    return std::string(buf.data(), buf.size());
}

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
            get_rpc_server().setGetMeshWorkloadRuntimeEntriesCallback(
                [this](auto result) { this->rpc_get_mesh_workload_runtime_entries(result); });
            get_rpc_server().setGetDevicesInUseCallback([this](auto result) { this->rpc_get_devices_in_use(result); });
            get_rpc_server().setGetKernelCallback(
                [this](auto params, auto result) { this->rpc_get_kernel(params, result); });
            get_rpc_server().setGetAllBuildEnvsCallback([this](auto result) { this->rpc_get_all_build_envs(result); });
            get_rpc_server().setGetAllDispatchCoreInfosCallback(
                [this](auto result) { this->rpc_get_all_dispatch_core_infos(result); });
            get_rpc_server().setGetBlocksByTypeCallback([this](auto result) { this->rpc_get_blocks_by_type(result); });
            get_rpc_server().setGetMetalDeviceIdMappingsCallback(
                [this](auto result) { this->rpc_get_metal_device_id_mappings(result); });
            get_rpc_server().setGetConfigurationCallback([this](auto result) { this->rpc_get_configuration(result); });
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

        auto binary_status_list =
            mesh_workload.initBinaryStatusPerMeshDevice(mesh_workload_data.binary_status_per_device.size());
        j = 0;
        for (const auto& [mesh_id, status] : mesh_workload_data.binary_status_per_device) {
            auto binary_status = binary_status_list[j++];
            binary_status.setMeshId(mesh_id);
            binary_status.setStatus(convert_binary_status(status));
        }
    }
}

void Data::rpc_get_mesh_workload_runtime_entries(
    rpc::Inspector::GetMeshWorkloadRuntimeEntriesResults::Builder& results) {
    std::lock_guard<std::mutex> lock(runtime_entries_mutex);
    auto write_pos = runtime_entries_write_pos;
    size_t count = std::min(write_pos, kRuntimeEntriesCapacity);
    size_t start = write_pos - count;

    auto all_runtime_entries = results.initRuntimeEntries(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& re = runtime_entries[(start + i) % kRuntimeEntriesCapacity];
        auto entry = all_runtime_entries[i];
        entry.setWorkloadId(re.workload_id);
        entry.setRuntimeId(re.runtime_id);
        entry.setOperationName(std::string(re.operation_name));
        entry.setOperationParameters(stringify_tensor_specs(re.tensor_specs));
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

void Data::rpc_get_blocks_by_type(rpc::Inspector::GetBlocksByTypeResults::Builder results) {
    auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    auto device_ids = tt_metal::MetalContext::instance().device_manager()->get_all_active_device_ids();

    auto chips_builder = results.initChips(device_ids.size());
    size_t chip_idx = 0;

    for (ChipId device_id : device_ids) {
        auto chip_entry = chips_builder[chip_idx++];
        chip_entry.setChipId(static_cast<uint64_t>(device_id));

        std::vector<std::pair<uint32_t, uint32_t>> active_eth_xy;
        std::vector<std::pair<uint32_t, uint32_t>> idle_eth_xy;

        for (const CoreCoord& logical_core : control_plane.get_active_ethernet_cores(device_id)) {
            active_eth_xy.emplace_back(logical_core.x, logical_core.y);
        }

        for (const CoreCoord& logical_core : control_plane.get_inactive_ethernet_cores(device_id)) {
            idle_eth_xy.emplace_back(logical_core.x, logical_core.y);
        }

        auto blocks = chip_entry.initBlocks();
        auto set_coords = [](auto list_builder, const std::vector<std::pair<uint32_t, uint32_t>>& xy) {
            auto list = list_builder(xy.size());
            for (size_t i = 0; i < xy.size(); ++i) {
                list[i].setX(xy[i].first);
                list[i].setY(xy[i].second);
            }
        };
        set_coords([&blocks](size_t n) { return blocks.initActiveEth(n); }, active_eth_xy);
        set_coords([&blocks](size_t n) { return blocks.initIdleEth(n); }, idle_eth_xy);
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

// Helper: add an rtoptions entry, catching any exceptions from getters that throw when unset
#define RT_ENTRY(name_str, expr)                                             \
    try {                                                                    \
        entries.push_back({name_str, fmt::format("{}", expr), "RtOptions"}); \
    } catch (...) {                                                          \
        entries.push_back({name_str, "(unset)", "RtOptions"});               \
    }
#define RT_OPT_ENTRY(name_str, expr)                                                                          \
    try {                                                                                                     \
        auto v = expr;                                                                                        \
        entries.push_back({name_str, v.has_value() ? fmt::format("{}", v.value()) : "(unset)", "RtOptions"}); \
    } catch (...) {                                                                                           \
        entries.push_back({name_str, "(unset)", "RtOptions"});                                                \
    }

void collect_environment_entries(std::vector<ConfigurationEntry>& entries) {
    for (auto name : tt::llrt::get_env_var_names()) {
        std::string name_str(name);
        const char* val = std::getenv(name_str.c_str());
        if (val) {
            entries.push_back({std::move(name_str), std::string(val), "Environment"});
        }
    }
}

void collect_rtoptions_entries(std::vector<ConfigurationEntry>& entries, const tt::llrt::RunTimeOptions& rt) {
    // clang-format off
    // Path configuration
    RT_ENTRY("root_dir", rt.get_root_dir());
    // These getters TT_THROW (which logs a critical message) when unset, so guard them
    if (rt.is_cache_dir_specified()) { RT_ENTRY("cache_dir", rt.get_cache_dir()); }
    else { entries.push_back({"cache_dir", "(unset)", "RtOptions"}); }
    RT_ENTRY("logs_dir", rt.get_logs_dir());
    if (rt.is_kernel_dir_specified()) { RT_ENTRY("kernel_dir", rt.get_kernel_dir()); }
    else { entries.push_back({"kernel_dir", "(unset)", "RtOptions"}); }
    RT_ENTRY("system_kernel_dir", rt.get_system_kernel_dir());
    if (rt.is_core_grid_override_todeprecate()) { RT_ENTRY("core_grid_override_todeprecate", rt.get_core_grid_override_todeprecate()); }
    else { entries.push_back({"core_grid_override_todeprecate", "(unset)", "RtOptions"}); }

    // General
    RT_ENTRY("build_map_enabled", rt.get_build_map_enabled());
    RT_ENTRY("fast_dispatch", rt.get_fast_dispatch());
    RT_ENTRY("num_hw_cqs", rt.get_num_hw_cqs());
    RT_ENTRY("dram_backed_cq", rt.get_dram_backed_cq());
    RT_ENTRY("numa_based_affinity", rt.get_numa_based_affinity());
    RT_ENTRY("target_device", static_cast<int>(rt.get_target_device()));
    RT_ENTRY("simulator_enabled", rt.get_simulator_enabled());
    RT_ENTRY("simulator_path", rt.get_simulator_path().string());
    RT_ENTRY("mock_enabled", rt.get_mock_enabled());
    RT_ENTRY("mock_cluster_desc_path", rt.get_mock_cluster_desc_path());
    RT_ENTRY("visible_devices", rt.get_visible_devices());
    RT_ENTRY("arch_name", rt.get_arch_name());

    // Kernel execution
    RT_ENTRY("kernels_nullified", rt.get_kernels_nullified());
    RT_ENTRY("kernels_early_return", rt.get_kernels_early_return());
    RT_ENTRY("skip_loading_fw", rt.get_skip_loading_fw());
    RT_ENTRY("disable_precompiled_fw", rt.get_disable_precompiled_fw());
    RT_ENTRY("force_jit_compile", rt.get_force_jit_compile());
    RT_ENTRY("force_context_reinit", rt.get_force_context_reinit());
    RT_ENTRY("log_kernels_compilation_commands", rt.get_log_kernels_compilation_commands());
    RT_ENTRY("dump_build_commands", rt.get_dump_build_commands());
    RT_ENTRY("compile_hash_string", rt.get_compile_hash_string());
    RT_ENTRY("erisc_iram_enabled", rt.get_erisc_iram_enabled());

    // Memory
    RT_ENTRY("clear_l1", rt.get_clear_l1());
    RT_ENTRY("clear_dram", rt.get_clear_dram());

    // Hardware
    RT_ENTRY("hw_cache_invalidation_enabled", rt.get_hw_cache_invalidation_enabled());
    RT_ENTRY("relaxed_memory_ordering_disabled", rt.get_relaxed_memory_ordering_disabled());
    RT_ENTRY("gathering_enabled", rt.get_gathering_enabled());
    RT_ENTRY("enable_2_erisc_mode", rt.get_enable_2_erisc_mode());
    RT_ENTRY("disable_fabric_2_erisc_mode", rt.get_disable_fabric_2_erisc_mode());
    RT_ENTRY("disable_dma_ops", rt.get_disable_dma_ops());
    RT_ENTRY("disable_sfploadmacro", rt.get_disable_sfploadmacro());
    RT_ENTRY("disable_xip_dump", rt.get_disable_xip_dump());
    RT_ENTRY("skip_eth_cores_with_retrain", rt.get_skip_eth_cores_with_retrain());
    RT_ENTRY("use_mesh_graph_descriptor_2_0", rt.get_use_mesh_graph_descriptor_2_0());
    RT_ENTRY("custom_fabric_mesh_graph_desc_path", rt.get_custom_fabric_mesh_graph_desc_path());
    RT_ENTRY("arc_debug_buffer_size", rt.get_arc_debug_buffer_size());
    RT_ENTRY("validate_kernel_binaries", rt.get_validate_kernel_binaries());
    RT_ENTRY("record_noc_transfers", rt.get_record_noc_transfers());
    RT_ENTRY("use_device_print", rt.get_use_device_print());

    // Timeouts
    RT_ENTRY("timeout_duration_for_operations", fmt::format("{}s", rt.get_timeout_duration_for_operations().count()));
    RT_ENTRY("dispatch_timeout_command_to_execute", rt.get_dispatch_timeout_command_to_execute());
    RT_ENTRY("dispatch_progress_update_ms", rt.get_dispatch_progress_update_ms());

    // Fabric
    RT_ENTRY("enable_fabric_telemetry", rt.get_enable_fabric_telemetry());
    RT_ENTRY("enable_fabric_bw_telemetry", rt.get_enable_fabric_bw_telemetry());
    RT_ENTRY("enable_fabric_code_profiling_rx_ch_fwd", rt.get_enable_fabric_code_profiling_rx_ch_fwd());
    RT_ENTRY("enable_channel_trimming_capture", rt.get_enable_channel_trimming_capture());
    RT_ENTRY("fabric_trimming_profile_path", rt.get_fabric_trimming_profile_path());
    RT_ENTRY("fabric_trimming_override_path", rt.get_fabric_trimming_override_path());
    RT_ENTRY("enable_fabric_vc2", rt.get_enable_fabric_vc2());
    RT_OPT_ENTRY("fabric_router_sync_timeout_ms", rt.get_fabric_router_sync_timeout_ms());
    RT_OPT_ENTRY("fabric_kernel_opt_level", [&]() -> std::optional<int> {
        auto v = rt.get_fabric_kernel_opt_level();
        return v.has_value() ? std::optional<int>(static_cast<int>(v.value())) : std::nullopt;
    }());
    RT_OPT_ENTRY("reliability_mode", [&]() -> std::optional<int> {
        auto v = rt.get_reliability_mode();
        return v.has_value() ? std::optional<int>(static_cast<int>(v.value())) : std::nullopt;
    }());

    // Profiler
    RT_ENTRY("profiler_enabled", rt.get_profiler_enabled());
    RT_ENTRY("profiler_do_dispatch_cores", rt.get_profiler_do_dispatch_cores());
    RT_ENTRY("profiler_sync_enabled", rt.get_profiler_sync_enabled());
    RT_ENTRY("profiler_trace_only", rt.get_profiler_trace_only());
    RT_ENTRY("profiler_trace_tracking", rt.get_profiler_trace_tracking());
    RT_ENTRY("profiler_mid_run_dump", rt.get_profiler_mid_run_dump());
    RT_ENTRY("profiler_cpp_post_process", rt.get_profiler_cpp_post_process());
    RT_ENTRY("profiler_sum", rt.get_profiler_sum());
    RT_OPT_ENTRY("profiler_program_support_count", rt.get_profiler_program_support_count());
    RT_ENTRY("profiler_buffer_usage_enabled", rt.get_profiler_buffer_usage_enabled());
    RT_ENTRY("profiler_noc_events_enabled", rt.get_profiler_noc_events_enabled());
    RT_ENTRY("profiler_perf_counter_mode", rt.get_profiler_perf_counter_mode());
    RT_ENTRY("profiler_noc_events_report_path", rt.get_profiler_noc_events_report_path());
    RT_ENTRY("profiler_disable_dump_to_files", rt.get_profiler_disable_dump_to_files());
    RT_ENTRY("profiler_disable_push_to_tracy", rt.get_profiler_disable_push_to_tracy());
    RT_ENTRY("experimental_noc_debug_dump_enabled", rt.get_experimental_noc_debug_dump_enabled());
    RT_ENTRY("tracy_mid_run_push", rt.get_tracy_mid_run_push());

    // Watcher
    RT_ENTRY("watcher_enabled", rt.get_watcher_enabled());
    RT_ENTRY("watcher_hash", rt.get_watcher_hash());
    RT_ENTRY("watcher_interval", rt.get_watcher_interval());
    RT_ENTRY("watcher_dump_all", rt.get_watcher_dump_all());
    RT_ENTRY("watcher_append", rt.get_watcher_append());
    RT_ENTRY("watcher_auto_unpause", rt.get_watcher_auto_unpause());
    RT_ENTRY("watcher_noinline", rt.get_watcher_noinline());
    RT_ENTRY("watcher_phys_coords", rt.get_watcher_phys_coords());
    RT_ENTRY("watcher_text_start", rt.get_watcher_text_start());
    RT_ENTRY("watcher_skip_logging", rt.get_watcher_skip_logging());
    RT_ENTRY("watcher_noc_sanitize_linked_transaction", rt.get_watcher_noc_sanitize_linked_transaction());
    RT_ENTRY("watcher_debug_delay", rt.get_watcher_debug_delay());
    {
        const auto& disabled = rt.get_watcher_disabled_features();
        std::string joined;
        for (const auto& s : disabled) {
            if (!joined.empty()) joined += ", ";
            joined += s;
        }
        entries.push_back({"watcher_disabled_features", joined.empty() ? "(empty)" : joined, "RtOptions"});
    }

    // Inspector
    RT_ENTRY("inspector_enabled", rt.get_inspector_enabled());
    RT_ENTRY("inspector_initialization_is_important", rt.get_inspector_initialization_is_important());
    RT_ENTRY("inspector_warn_on_write_exceptions", rt.get_inspector_warn_on_write_exceptions());
    RT_ENTRY("inspector_rpc_server_enabled", rt.get_inspector_rpc_server_enabled());
    RT_ENTRY("inspector_rpc_server_host", rt.get_inspector_rpc_server_host());
    RT_ENTRY("inspector_rpc_server_port", rt.get_inspector_rpc_server_port());
    RT_ENTRY("inspector_rpc_server_address", rt.get_inspector_rpc_server_address());
    RT_ENTRY("inspector_capture_tensor_specs", rt.get_inspector_capture_tensor_specs());
    RT_ENTRY("inspector_log_runtime_entries", rt.get_inspector_log_runtime_entries());
    RT_ENTRY("inspector_log_path", rt.get_inspector_log_path().string());
    RT_ENTRY("serialize_inspector_on_dispatch_timeout", rt.get_serialize_inspector_on_dispatch_timeout());
    RT_ENTRY("riscv_debug_info_enabled", rt.get_riscv_debug_info_enabled());
    RT_ENTRY("jit_analytics_enabled", rt.get_jit_analytics_enabled());
    RT_ENTRY("lightweight_kernel_asserts", rt.get_lightweight_kernel_asserts());
    RT_ENTRY("llk_asserts", rt.get_llk_asserts());

    // Dispatch data / testing
    RT_ENTRY("dispatch_data_collection_enabled", rt.get_dispatch_data_collection_enabled());
    RT_ENTRY("test_mode_enabled", rt.get_test_mode_enabled());

    // DispatchCoreConfig
    {
        static const char* dispatch_core_types[] = {"WORKER", "ETH"};
        static const char* dispatch_core_axes[] = {"ROW", "COL"};
        try {
            auto config = rt.get_dispatch_core_config();
            auto type_idx = static_cast<int>(config.get_dispatch_core_type());
            auto axis_idx = static_cast<int>(config.get_dispatch_core_axis());
            entries.push_back({"dispatch_core_config_type", (type_idx < 2) ? dispatch_core_types[type_idx] : fmt::format("{}", type_idx), "RtOptions"});
            entries.push_back({"dispatch_core_config_axis", (axis_idx < 2) ? dispatch_core_axes[axis_idx] : fmt::format("{}", axis_idx), "RtOptions"});
        } catch (...) {
            entries.push_back({"dispatch_core_config", "(unset)", "RtOptions"});
        }
    }

    // FabricTelemetrySettings
    {
        try {
            const auto& fts = rt.get_fabric_telemetry_settings();
            entries.push_back({"fabric_telemetry_enabled", fmt::format("{}", fts.enabled), "RtOptions"});
            entries.push_back({"fabric_telemetry_chips_monitor_all", fmt::format("{}", fts.chips.monitor_all), "RtOptions"});
            entries.push_back({"fabric_telemetry_channels_monitor_all", fmt::format("{}", fts.channels.monitor_all), "RtOptions"});
            entries.push_back({"fabric_telemetry_eriscs_monitor_all", fmt::format("{}", fts.eriscs.monitor_all), "RtOptions"});
            entries.push_back({"fabric_telemetry_stats_mask", fmt::format("{}", fts.stats_mask), "RtOptions"});
        } catch (...) {
            entries.push_back({"fabric_telemetry_settings", "(unset)", "RtOptions"});
        }
    }

    // Per-feature debug settings
    {
        static const char* feature_names[] = {"dprint", "read_debug_delay", "write_debug_delay", "atomic_debug_delay", "enable_l1_data_cache"};
        for (int i = 0; i < tt::llrt::RunTimeDebugFeatureCount; ++i) {
            auto feature = static_cast<tt::llrt::RunTimeDebugFeatures>(i);
            const char* fname = feature_names[i];
            try { entries.push_back({fmt::format("feature_{}_enabled", fname), fmt::format("{}", rt.get_feature_enabled(feature)), "RtOptions"}); } catch (...) {}
            try { entries.push_back({fmt::format("feature_{}_file_name", fname), rt.get_feature_file_name(feature), "RtOptions"}); } catch (...) {}
            try { entries.push_back({fmt::format("feature_{}_one_file_per_risc", fname), fmt::format("{}", rt.get_feature_one_file_per_risc(feature)), "RtOptions"}); } catch (...) {}
            try { entries.push_back({fmt::format("feature_{}_prepend_device_core_risc", fname), fmt::format("{}", rt.get_feature_prepend_device_core_risc(feature)), "RtOptions"}); } catch (...) {}
            try { entries.push_back({fmt::format("feature_{}_all_chips", fname), fmt::format("{}", rt.get_feature_all_chips(feature)), "RtOptions"}); } catch (...) {}
        }
    }
    // clang-format on
}

#undef RT_ENTRY
#undef RT_OPT_ENTRY

void Data::rpc_get_configuration(rpc::Inspector::GetConfigurationResults::Builder& results) {
    std::vector<ConfigurationEntry> all_entries;

    // 1. Environment variables
    collect_environment_entries(all_entries);

    // 2. RtOptions
    const auto& rt = MetalContext::instance().rtoptions();
    collect_rtoptions_entries(all_entries, rt);

    // 3. TTNN config (registered via static callback at library load time)
    auto& ttnn_cb = ttnn_config_callback();
    if (ttnn_cb) {
        auto ttnn_entries = ttnn_cb();
        all_entries.insert(
            all_entries.end(),
            std::make_move_iterator(ttnn_entries.begin()),
            std::make_move_iterator(ttnn_entries.end()));
    }

    // Serialize into Cap'n Proto
    auto entries = results.initEntries(all_entries.size());
    for (size_t i = 0; i < all_entries.size(); ++i) {
        auto entry = entries[i];
        entry.setName(all_entries[i].name);
        entry.setValue(all_entries[i].value);

        if (all_entries[i].scope == "Environment") {
            entry.setScope(rpc::ConfigurationScope::ENVIRONMENT);
        } else if (all_entries[i].scope == "RtOptions") {
            entry.setScope(rpc::ConfigurationScope::RT_OPTIONS);
        } else if (all_entries[i].scope == "TtnnConfig") {
            entry.setScope(rpc::ConfigurationScope::TTNN_CONFIG);
        } else {
            entry.setScope(rpc::ConfigurationScope::UNKNOWN);
        }
    }
}

}  // namespace tt::tt_metal::inspector
