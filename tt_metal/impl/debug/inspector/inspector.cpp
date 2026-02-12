// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "inspector.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/debug/inspector/data.hpp"
#include "impl/debug/inspector/rpc_server_generated.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/jit_build_options.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "program.hpp"
#include <memory>
#include <tt-metalium/experimental/inspector.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

namespace {
inspector::Data* get_inspector_data() { return tt::tt_metal::MetalContext::instance().get_inspector_data(); }
}  // namespace

bool Inspector::is_enabled() { return tt::tt_metal::MetalContext::instance().rtoptions().get_inspector_enabled(); }

std::unique_ptr<inspector::Data> Inspector::initialize() {
    if (!is_enabled()) {
        // Inspector is not enabled, skipping initialization.
        return nullptr;
    }
    try {
        auto* data = new inspector::Data();

        return std::unique_ptr<inspector::Data>(data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to initialize Inspector: {}", e.what());
        throw;
    }
}

void Inspector::serialize_rpc() {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        data->serialize_rpc();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to serialize RPC: {}", e.what());
    }
}

void Inspector::program_created(const detail::ProgramImpl* program) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.program = program->weak_from_this();
        program_data.program_id = program->get_id();
        data->logger.log_program_created(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program created: {}", e.what());
    }
}

void Inspector::program_destroyed(const detail::ProgramImpl* program) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        data->logger.log_program_destroyed(program_data);
        for (const auto& [kernel_id, _] : program_data.kernels) {
            data->kernel_id_to_program_id.erase(kernel_id);
        }
        data->programs_data.erase(program->get_id());
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void Inspector::program_compile_started(
    const detail::ProgramImpl* program, const IDevice* /*device*/, uint64_t /*build_key*/) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.compile_started_timestamp = std::chrono::high_resolution_clock::now();
        data->logger.log_program_compile_started(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void Inspector::program_compile_already_exists(
    const detail::ProgramImpl* program, const IDevice* /*device*/, uint64_t /*build_key*/) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        data->logger.log_program_compile_already_exists(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile already exists: {}", e.what());
    }
}

void Inspector::program_kernel_compile_finished(
    const detail::ProgramImpl* program,
    const IDevice* /*device*/,
    const std::shared_ptr<Kernel>& kernel,
    const tt::tt_metal::JitBuildOptions& build_options) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        auto& kernel_data = program_data.kernels[kernel->get_watcher_kernel_id()];
        kernel_data.kernel = kernel;
        kernel_data.watcher_kernel_id = kernel->get_watcher_kernel_id();
        kernel_data.name = kernel->name();
        kernel_data.path = build_options.path;
        kernel_data.source = kernel->kernel_source().source_;
        data->kernel_id_to_program_id[kernel->get_watcher_kernel_id()] = program->get_id();
        data->logger.log_program_kernel_compile_finished(program_data, kernel_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program kernel compile finished: {}", e.what());
    }
}

void Inspector::program_compile_finished(
    const detail::ProgramImpl* program, const IDevice* /*device*/, uint64_t /*build_key*/) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.compile_finished_timestamp = std::chrono::high_resolution_clock::now();
        data->logger.log_program_compile_finished(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile finished: {}", e.what());
    }
}

void Inspector::program_set_binary_status(
    const detail::ProgramImpl* program, std::size_t device_id, ProgramBinaryStatus status) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.binary_status_per_device[device_id] = status;
        data->logger.log_program_binary_status_change(program_data, device_id, status);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program binary status change: {}", e.what());
    }
}

void Inspector::mesh_device_created(
    const distributed::MeshDeviceImpl* mesh_device, std::optional<int> parent_mesh_id) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        mesh_device_data.mesh_device = mesh_device;
        mesh_device_data.mesh_id = mesh_device->id();
        mesh_device_data.parent_mesh_id = parent_mesh_id;
        data->logger.log_mesh_device_created(mesh_device_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device created: {}", e.what());
    }
}

void Inspector::mesh_device_destroyed(const distributed::MeshDeviceImpl* mesh_device) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        data->logger.log_mesh_device_destroyed(mesh_device_data);
        data->mesh_devices_data.erase(mesh_device->id());
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device destroyed: {}", e.what());
    }
}

void Inspector::mesh_device_initialized(const distributed::MeshDeviceImpl* mesh_device) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        mesh_device_data.initialized = true;
        data->logger.log_mesh_device_initialized(mesh_device_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device initialized: {}", e.what());
    }
}

void Inspector::mesh_workload_created(const distributed::MeshWorkloadImpl* mesh_workload) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        mesh_workload_data.mesh_workload = mesh_workload;
        mesh_workload_data.mesh_workload_id = mesh_workload->get_id();
        data->logger.log_mesh_workload_created(mesh_workload_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload created: {}", e.what());
    }
}

void Inspector::mesh_workload_destroyed(const distributed::MeshWorkloadImpl* mesh_workload) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        data->logger.log_mesh_workload_destroyed(mesh_workload_data);
        data->mesh_workloads_data.erase(mesh_workload->get_id());
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload destroyed: {}", e.what());
    }
}

void Inspector::mesh_workload_add_program(
    const distributed::MeshWorkloadImpl* mesh_workload,
    const distributed::MeshCoordinateRange& device_range,
    std::size_t program_id) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        data->logger.log_mesh_workload_add_program(mesh_workload_data, device_range, program_id);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload add program: {}", e.what());
    }
}

void Inspector::mesh_workload_set_program_binary_status(
    const distributed::MeshWorkloadImpl* mesh_workload, std::size_t mesh_id, ProgramBinaryStatus status) noexcept {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        mesh_workload_data.binary_status_per_device[mesh_id] = status;
        data->logger.log_mesh_workload_set_program_binary_status(mesh_workload_data, mesh_id, status);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload set program binary status: {}", e.what());
    }
}

void Inspector::mesh_workload_set_operation_name_and_parameters(
    const distributed::MeshWorkloadImpl* mesh_workload,
    std::string_view operation_name,
    std::string_view operation_parameters) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        mesh_workload_data.name = std::string(operation_name);
        mesh_workload_data.parameters = std::string(operation_parameters);
        // Keep log/event name stable for tooling compatibility.
        data->logger.log_mesh_workload_operation_name_and_parameters(
            mesh_workload_data, operation_name, operation_parameters);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload set metadata: {}", e.what());
    }
}

void Inspector::mesh_workload_set_runtime_id(
    const distributed::MeshWorkloadImpl* mesh_workload, uint64_t runtime_id) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();

        std::lock_guard<std::mutex> lock(data->runtime_ids_mutex);
        data->runtime_ids.push_back({mesh_workload->get_id(), runtime_id});

        // Keep only the last MAX_RUNTIME_ID_ENTRIES
        if (data->runtime_ids.size() > inspector::Data::MAX_RUNTIME_ID_ENTRIES) {
            data->runtime_ids.pop_front();
        }
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log workload runtime ID: {}", e.what());
    }
}

// Set dispatch core info
void Inspector::set_dispatch_core_info(
    const tt_cxy_pair& virtual_core,
    const tt::tt_metal::DispatchWorkerType& type,
    const uint8_t cq_id,
    const ChipId device_id,
    const ChipId servicing_device_id) {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->dispatch_core_info_mutex);
        data->dispatch_core_info[virtual_core] = {type, device_id, servicing_device_id, cq_id};
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log dispatch core info: {}", e.what());
    }
}

// Set dispatch_s core info
void Inspector::set_dispatch_s_core_info(
    const tt_cxy_pair& virtual_core,
    const tt::tt_metal::DispatchWorkerType& type,
    const uint8_t cq_id,
    const ChipId device_id,
    const ChipId servicing_device_id) {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->dispatch_s_core_info_mutex);
        data->dispatch_s_core_info[virtual_core] = {type, device_id, servicing_device_id, cq_id};
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log dispatch_s core info: {}", e.what());
    }
}

// Set prefetcher core info
void Inspector::set_prefetcher_core_info(
    const tt_cxy_pair& virtual_core,
    const tt::tt_metal::DispatchWorkerType& type,
    const uint8_t cq_id,
    const ChipId device_id,
    const ChipId servicing_device_id) {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->prefetcher_core_info_mutex);
        data->prefetcher_core_info[virtual_core] = {type, device_id, servicing_device_id, cq_id};
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log prefetcher core info: {}", e.what());
    }
}

// Clear dispatch core info to clear stale entries
// Used in MetalContext::teardown() to clear stale entries
void Inspector::clear_all_core_info() {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        std::scoped_lock locks(
            data->dispatch_core_info_mutex, data->dispatch_s_core_info_mutex, data->prefetcher_core_info_mutex);
        data->dispatch_core_info.clear();
        data->dispatch_s_core_info.clear();
        data->prefetcher_core_info.clear();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to clear all core infos: {}", e.what());
    }
}

inspector::RpcServer& Inspector::get_rpc_server() {
    if (is_enabled()) {
        try {
            auto* data = get_inspector_data();
            if (data) {
                return data->get_rpc_server();
            }
        } catch (const std::exception& e) {
            TT_INSPECTOR_LOG("Failed to get RPC server: {}", e.what());
        }
    }
    static inspector::RpcServer empty_rpc_server;
    return empty_rpc_server;
}

void Inspector::set_build_env_fw_compile_hash(const uint64_t fw_compile_hash) {
    if (!is_enabled()) {
        return;
    }
    auto* data = get_inspector_data();
    if (!data) {
        // Inspector failed to initialize, no need to print failure message again.
        return;
    }
    try {
        data->fw_compile_hash.store(fw_compile_hash, std::memory_order_release);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to set FW compile hash: {}", e.what());
    }
}

namespace experimental::inspector {

bool IsEnabled() {
    return Inspector::is_enabled();
}

void EmitMeshWorkloadAnnotation(
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation_name,
    std::string_view operation_parameters) {
    tt::tt_metal::Inspector::mesh_workload_set_operation_name_and_parameters(
        &workload.impl(), operation_name, operation_parameters);
}

void EmitMeshWorkloadRuntimeId(tt::tt_metal::distributed::MeshWorkload& workload, uint64_t runtime_id) {
    tt::tt_metal::Inspector::mesh_workload_set_runtime_id(&workload.impl(), runtime_id);
}

}  // namespace experimental::inspector

}  // namespace tt::tt_metal
