// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inspector.hpp"
#include "inspector_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/jit_build_options.hpp"
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

}  // namespace tt::tt_metal
