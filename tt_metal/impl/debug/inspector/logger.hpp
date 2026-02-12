// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>

#include "impl/debug/inspector/types.hpp"
#include "mesh_coord.hpp"

#define TT_INSPECTOR_THROW(...) \
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_inspector_initialization_is_important()) { \
        TT_THROW(__VA_ARGS__); \
    } else { \
        log_warning(tt::LogInspector, __VA_ARGS__); \
        return; \
    }

#define TT_INSPECTOR_LOG(...) \
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_inspector_warn_on_write_exceptions()) { \
        log_warning(tt::LogInspector, __VA_ARGS__); \
    }

namespace tt::tt_metal::inspector {

class Logger {
private:
    time_point start_time;
    std::ofstream programs_ostream;
    std::ofstream kernels_ostream;
    std::ofstream mesh_devices_ostream;
    std::ofstream mesh_workloads_ostream;
    bool initialized{false};
    std::filesystem::path logging_path;

    int64_t convert_timestamp(const time_point& tp) const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    }

public:
    Logger(const std::filesystem::path& logging_path);

    std::filesystem::path get_logging_path() const noexcept {
        return logging_path;
    }

    void log_program_created(const ProgramData& program_data) noexcept;
    void log_program_destroyed(const ProgramData& program_data) noexcept;
    void log_program_compile_started(const ProgramData& program_data) noexcept;
    void log_program_compile_already_exists(const ProgramData& program_data) noexcept;
    void log_program_kernel_compile_finished(const ProgramData& program_data, const inspector::KernelData& kernel_data) noexcept;
    void log_program_compile_finished(const ProgramData& program_data) noexcept;
    void log_program_binary_status_change(const ProgramData& program_data, std::size_t device_id, ProgramBinaryStatus status) noexcept;

    void log_mesh_device_created(const MeshDeviceData& mesh_device_data) noexcept;
    void log_mesh_device_destroyed(const MeshDeviceData& mesh_device_data) noexcept;
    void log_mesh_device_initialized(const MeshDeviceData& mesh_device_data) noexcept;
    void log_mesh_workload_created(const MeshWorkloadData& mesh_workload_data) noexcept;
    void log_mesh_workload_destroyed(const MeshWorkloadData& mesh_workload_data) noexcept;
    void log_mesh_workload_add_program(const MeshWorkloadData& mesh_workload_data, const distributed::MeshCoordinateRange& device_range, std::size_t program_id) noexcept;
    void log_mesh_workload_set_program_binary_status(
        const MeshWorkloadData& mesh_workload_data, std::size_t mesh_id, ProgramBinaryStatus status) noexcept;
    void log_mesh_workload_operation_name_and_parameters(
        const MeshWorkloadData& mesh_workload_data, std::string_view name, std::string_view parameters) noexcept;
};

}  // namespace tt::tt_metal::inspector
