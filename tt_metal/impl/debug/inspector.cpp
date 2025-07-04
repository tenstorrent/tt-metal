// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inspector.hpp"
#include "inspector_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/jit_build_options.hpp"
#include "mesh_device.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "program.hpp"
#include <memory>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

static inspector::Data* get_inspector_data() {
    auto* data = tt::tt_metal::MetalContext::instance().get_inspector_data();
    if (!data) {
        throw std::runtime_error("Inspector data is not initialized.");
    }
    return data;
}

bool Inspector::is_enabled() {
    return tt::tt_metal::MetalContext::instance().rtoptions().get_inspector_enabled();
}

std::unique_ptr<inspector::Data> Inspector::initialize() {
    if (!is_enabled()) {
        // Inspector is not enabled, skipping initialization.
        return nullptr;
    }
    try {
        auto* data = new inspector::Data();

        return std::unique_ptr<inspector::Data>(data);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to initialize Inspector: {}", e.what());
        throw;
    }
}

void Inspector::program_created(
    const detail::ProgramImpl* program) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.program=program->weak_from_this();
        program_data.program_id=program->get_id();
        data->logger.log_program_created(program_data);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program created: {}", e.what());
    }
}

void Inspector::program_destroyed(
    const detail::ProgramImpl* program) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        data->logger.log_program_destroyed(program_data);
        data->programs_data.erase(program->get_id());
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void Inspector::program_compile_started(
    const detail::ProgramImpl* program,
    const IDevice* device,
    uint32_t build_key) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.compile_started_timestamp = std::chrono::high_resolution_clock::now();
        data->logger.log_program_compile_started(program_data);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void Inspector::program_compile_already_exists(
    const detail::ProgramImpl* program,
    const IDevice* device,
    uint32_t build_key) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        data->logger.log_program_compile_already_exists(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile already exists: {}", e.what());
    }
}

void Inspector::program_kernel_compile_finished(
    const detail::ProgramImpl* program,
    const IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const tt::tt_metal::JitBuildOptions& build_options) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        auto& kernel_data = program_data.kernels[kernel->get_watcher_kernel_id()];
        kernel_data.kernel = kernel;
        kernel_data.watcher_kernel_id = kernel->get_watcher_kernel_id();
        kernel_data.name = kernel->name();
        kernel_data.path = build_options.path;
        kernel_data.source = kernel->kernel_source().source_;
        data->logger.log_program_kernel_compile_finished(program_data, kernel_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program kernel compile finished: {}", e.what());
    }
}

void Inspector::program_compile_finished(
    const detail::ProgramImpl* program,
    const IDevice* device,
    uint32_t build_key) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.compile_finished_timestamp = std::chrono::high_resolution_clock::now();
        data->logger.log_program_compile_finished(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile finished: {}", e.what());
    }
}

void Inspector::program_set_binary_status(
    const detail::ProgramImpl* program,
    std::size_t device_id,
    ProgramBinaryStatus status) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.binary_status_per_device[device_id] = status;
        data->logger.log_program_binary_status_change(program_data, device_id, status);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program binary status change: {}", e.what());
    }
}

void Inspector::mesh_device_created(
    const distributed::MeshDevice* mesh_device,
    std::optional<int> parent_mesh_id) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        mesh_device_data.mesh_device = mesh_device->weak_from_this();
        mesh_device_data.mesh_id = mesh_device->id();
        mesh_device_data.parent_mesh_id = parent_mesh_id;
        data->logger.log_mesh_device_created(mesh_device_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device created: {}", e.what());
    }
}

void Inspector::mesh_device_destroyed(
    const distributed::MeshDevice* mesh_device) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        data->logger.log_mesh_device_destroyed(mesh_device_data);
        data->mesh_devices_data.erase(mesh_device->id());
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device destroyed: {}", e.what());
    }
}

void Inspector::mesh_device_initialized(
    const distributed::MeshDevice* mesh_device) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        mesh_device_data.initialized = true;
        data->logger.log_mesh_device_initialized(mesh_device_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device initialized: {}", e.what());
    }
}

void Inspector::mesh_workload_created(
    const distributed::MeshWorkloadImpl* mesh_workload) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        mesh_workload_data.mesh_workload = mesh_workload;
        mesh_workload_data.mesh_workload_id = mesh_workload->get_id();
        data->logger.log_mesh_workload_created(mesh_workload_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload created: {}", e.what());
    }
}

void Inspector::mesh_workload_destroyed(
    const distributed::MeshWorkloadImpl* mesh_workload) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
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
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        data->logger.log_mesh_workload_add_program(mesh_workload_data, device_range, program_id);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload add program: {}", e.what());
    }
}

void Inspector::mesh_workload_set_program_binary_status(
    const distributed::MeshWorkloadImpl* mesh_workload,
    std::size_t mesh_id,
    ProgramBinaryStatus status) noexcept {
    if (!is_enabled()) {
        return;
    }
    try {
        auto* data = get_inspector_data();
        std::lock_guard<std::mutex> lock(data->mesh_workloads_mutex);
        auto& mesh_workload_data = data->mesh_workloads_data[mesh_workload->get_id()];
        mesh_workload_data.binary_status_per_device[mesh_id] = status;
        data->logger.log_mesh_workload_set_program_binary_status(mesh_workload_data, mesh_id, status);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload set program binary status: {}", e.what());
    }
}

}  // namespace tt::tt_metal
