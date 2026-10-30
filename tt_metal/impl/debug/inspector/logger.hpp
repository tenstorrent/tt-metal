// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <optional>

#include "impl/debug/inspector/types.hpp"
#include "mesh_coord.hpp"

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

// Error reporting of the inspector. `rtoptions` are those of the session's MetalEnv, which select the policy.
// TT_INSPECTOR_THROW returns from the enclosing function when the failure is only warned about.
#define TT_INSPECTOR_THROW(rtoptions, ...)                             \
    do {                                                               \
        if ((rtoptions).get_inspector_initialization_is_important()) { \
            TT_THROW(__VA_ARGS__);                                     \
        }                                                              \
        log_warning(tt::LogInspector, __VA_ARGS__);                    \
        return;                                                        \
    } while (0)

#define TT_INSPECTOR_LOG(rtoptions, ...)                            \
    do {                                                            \
        if ((rtoptions).get_inspector_warn_on_write_exceptions()) { \
            log_warning(tt::LogInspector, __VA_ARGS__);             \
        }                                                           \
    } while (0)

namespace tt::tt_metal::inspector {

class Logger {
private:
    const llrt::RunTimeOptions& rtoptions;
    time_point start_time;
    std::ofstream programs_ostream;
    std::ofstream kernels_ostream;
    std::ofstream mesh_devices_ostream;
    std::ofstream mesh_workloads_ostream;
    std::ofstream mesh_buffers_ostream;
    std::ofstream mesh_sockets_ostream;
    bool initialized{false};
    std::filesystem::path logging_path;

    int64_t convert_timestamp(const time_point& tp) const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    }

public:
    // `rtoptions` selects which optional log files are written and how failures are reported; it must outlive this
    // object.
    Logger(const std::filesystem::path& logging_path, std::optional<int> rank, const llrt::RunTimeOptions& rtoptions);

    std::filesystem::path get_logging_path() const noexcept {
        return logging_path;
    }

    void log_program_created(const ProgramData& program_data) noexcept;
    void log_program_destroyed(const ProgramData& program_data) noexcept;
    void log_program_compile_started(const ProgramData& program_data) noexcept;
    void log_program_compile_already_exists(const ProgramData& program_data) noexcept;
    void log_program_kernel_compile_finished(const ProgramData& program_data, const inspector::KernelData& kernel_data) noexcept;
    void log_program_compile_finished(const ProgramData& program_data) noexcept;

    void log_mesh_buffer_allocated(const distributed::MeshBuffer* mesh_buffer) noexcept;
    void log_mesh_buffer_deallocated(const distributed::MeshBuffer* mesh_buffer) noexcept;

    void log_mesh_socket_created(
        const distributed::MeshBuffer* config_buffer, const MeshSocketData& socket_data) noexcept;
    void log_mesh_socket_destroyed(
        const distributed::MeshBuffer* config_buffer, const MeshSocketData& socket_data) noexcept;

    void log_mesh_device_created(const MeshDeviceData& mesh_device_data) noexcept;
    void log_mesh_device_destroyed(const MeshDeviceData& mesh_device_data) noexcept;
    void log_mesh_device_initialized(const MeshDeviceData& mesh_device_data) noexcept;
    void log_runtime_entry(const MeshWorkloadRuntimeEntry& entry) noexcept;
};

}  // namespace tt::tt_metal::inspector
