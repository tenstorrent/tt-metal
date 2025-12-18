// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/debug/inspector/logger.hpp"
#include "impl/debug/inspector/types.hpp"
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <iomanip>
#include <chrono>

namespace tt::tt_metal::inspector {

Logger::Logger(const std::filesystem::path& logging_path) : logging_path(logging_path) {
    constexpr std::string_view additional_text =
        "\nYou can disable exception by setting TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 in your environment "
        "variables. Note that this will not throw an exception, but will log a warning instead. Running without "
        "Inspector logger will impact tt-triage functionality.";

    try {
        // Recreate the logging directory if it doesn't exist or clear it if it does.
        std::filesystem::remove_all(logging_path);
        std::filesystem::create_directories(logging_path);
    }
    catch (const std::exception& e) {
        TT_INSPECTOR_THROW(
            "Failed to create logging directory: {}. Error: {}\n{}", logging_path.string(), e.what(), additional_text);
    }

    // Write startup information to the inspector files.
    {
        try {
            std::ofstream inspector_startup_ostream(logging_path / "startup.yaml", std::ios::trunc);

            if (!inspector_startup_ostream.is_open()) {
                TT_INSPECTOR_THROW(
                    "Failed to create inspector file: {}\n{}",
                    (logging_path / "startup.yaml").string(),
                    additional_text);
            } else {
                // Log current system time and high_resolution_clock time_point
                auto now_system = std::chrono::system_clock::now();
                auto now_highres = std::chrono::high_resolution_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now_system);
                auto now_system_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now_system.time_since_epoch()).count();
                auto now_highres_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now_highres.time_since_epoch()).count();

                start_time = now_highres;
                inspector_startup_ostream << "startup_time:\n";
                inspector_startup_ostream << "  system_clock_iso: '"
                                          << std::put_time(std::gmtime(&now_c), "%Y-%m-%dT%H:%M:%SZ") << "'\n";
                inspector_startup_ostream << "  system_clock_ns: " << now_system_ns << "\n";
                inspector_startup_ostream << "  high_resolution_clock_ns: " << now_highres_ns << "\n";
            }
        } catch (const std::exception& e) {
            TT_INSPECTOR_THROW("Failed to create inspector startup log. Error: {}\n{}", e.what(), additional_text);
        }
    }

    programs_ostream.open(logging_path / "programs_log.yaml", std::ios::trunc);
    if (!programs_ostream.is_open()) {
        TT_INSPECTOR_THROW(
            "Failed to create inspector file: {}\n{}", (logging_path / "programs_log.yaml").string(), additional_text);
    }
    kernels_ostream.open(logging_path / "kernels.yaml", std::ios::trunc);
    if (!kernels_ostream.is_open()) {
        TT_INSPECTOR_THROW(
            "Failed to create inspector file: {}\n{}", (logging_path / "kernels.yaml").string(), additional_text);
    }
    mesh_devices_ostream.open(logging_path / "mesh_devices_log.yaml", std::ios::trunc);
    if (!mesh_devices_ostream.is_open()) {
        TT_INSPECTOR_THROW(
            "Failed to create inspector file: {}\n{}", (logging_path / "mesh_devices_log.yaml").string(), additional_text);
    }
    mesh_workloads_ostream.open(logging_path / "mesh_workloads_log.yaml", std::ios::trunc);
    if (!mesh_workloads_ostream.is_open()) {
        TT_INSPECTOR_THROW(
            "Failed to create inspector file: {}\n{}", (logging_path / "mesh_workloads_log.yaml").string(), additional_text);
    }

    initialized = true;
}

void Logger::log_program_created(const ProgramData& program_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        programs_ostream << "- program_created:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program created: {}", e.what());
    }
}

void Logger::log_program_destroyed(const ProgramData& program_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        programs_ostream << "- program_destroyed:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program destroyed: {}", e.what());
    }
}

void Logger::log_program_compile_started(const ProgramData& program_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        programs_ostream << "- program_compile_started:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(program_data.compile_started_timestamp) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile started: {}", e.what());
    }
}

void Logger::log_program_compile_already_exists(const ProgramData& /*program_data*/) noexcept {
    // Long running programs call this log entry too many times, so we don't want to log it.
}

void Logger::log_program_kernel_compile_finished(const ProgramData& program_data, const KernelData& kernel_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        auto timestamp = std::chrono::high_resolution_clock::now();
        programs_ostream << "- program_kernel_compile_finished:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(timestamp) << "\n";
        programs_ostream << "    duration_ns: " << std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp - program_data.compile_started_timestamp).count() << "\n";
        programs_ostream << "    watcher_kernel_id: " << kernel_data.watcher_kernel_id << "\n";
        programs_ostream << "    name: " << kernel_data.name << "\n";
        programs_ostream << "    path: " << kernel_data.path << "\n";
        programs_ostream << "    source: " << kernel_data.source << "\n";
        programs_ostream.flush();
        kernels_ostream << "- kernel:\n";
        kernels_ostream << "    watcher_kernel_id: " << kernel_data.watcher_kernel_id << "\n";
        kernels_ostream << "    name: " << kernel_data.name << "\n";
        kernels_ostream << "    path: " << kernel_data.path << "\n";
        kernels_ostream << "    source: " << kernel_data.source << "\n";
        kernels_ostream << "    program_id: " << program_data.program_id << "\n";
        kernels_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program kernel compile finished: {}", e.what());
    }
}

void Logger::log_program_compile_finished(const ProgramData& program_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        programs_ostream << "- program_compile_finished:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(program_data.compile_finished_timestamp) << "\n";
        programs_ostream << "    duration_ns: " << std::chrono::duration_cast<std::chrono::nanoseconds>(program_data.compile_finished_timestamp - program_data.compile_started_timestamp).count() << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile finished: {}", e.what());
    }
}

void Logger::log_program_binary_status_change(const ProgramData& program_data, std::size_t device_id, ProgramBinaryStatus status) noexcept {
    if (!initialized) {
        return;
    }
    try {
        programs_ostream << "- program_binary_status_change:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        programs_ostream << "    device_id: " << device_id << "\n";
        programs_ostream << "    status: ";
        switch (status) {
            case ProgramBinaryStatus::NotSent:
                programs_ostream << "NotSent" << "\n";
                break;
            case ProgramBinaryStatus::InFlight:
                programs_ostream << "InFlight" << "\n";
                break;
            case ProgramBinaryStatus::Committed:
                programs_ostream << "Committed" << "\n";
                break;
        }
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program enqueued: {}", e.what());
    }
}

void Logger::log_mesh_device_created(const MeshDeviceData& mesh_device_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        mesh_devices_ostream << "- mesh_device_created:\n";
        mesh_devices_ostream << "    mesh_id: " << mesh_device_data.mesh_id << "\n";
        mesh_devices_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        if (mesh_device_data.parent_mesh_id.has_value()) {
            mesh_devices_ostream << "    parent_mesh_id: " << *mesh_device_data.parent_mesh_id << "\n";
        }

        const auto* mesh_device = mesh_device_data.mesh_device;
        if (mesh_device) {
            mesh_devices_ostream << "    devices: [";
            bool first = true;
            for (const auto& device : mesh_device->get_view().get_devices()) {
                if (!first) {
                    mesh_devices_ostream << ", ";
                }
                first = false;
                mesh_devices_ostream << device->id();
            }
            mesh_devices_ostream << "]\n";

            const auto& shape = mesh_device->get_view().shape();
            mesh_devices_ostream << "    shape: [";
            for (size_t i = 0; i < shape.dims(); ++i) {
                if (i > 0) {
                    mesh_devices_ostream << ", ";
                }
                mesh_devices_ostream << shape.get_stride(i);
            }
            mesh_devices_ostream << "]\n";
        }
        mesh_devices_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device created: {}", e.what());
    }
}

void Logger::log_mesh_device_destroyed(const MeshDeviceData& mesh_device_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        mesh_devices_ostream << "- mesh_device_destroyed:\n";
        mesh_devices_ostream << "    mesh_id: " << mesh_device_data.mesh_id << "\n";
        mesh_devices_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        mesh_devices_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device destroyed: {}", e.what());
    }
}

void Logger::log_mesh_device_initialized(const MeshDeviceData& mesh_device_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        mesh_devices_ostream << "- mesh_device_initialized:\n";
        mesh_devices_ostream << "    mesh_id: " << mesh_device_data.mesh_id << "\n";
        mesh_devices_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        mesh_devices_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh device initialized: {}", e.what());
    }
}

void Logger::log_mesh_workload_created(const MeshWorkloadData& mesh_workload_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        mesh_workloads_ostream << "- mesh_workload_created:\n";
        mesh_workloads_ostream << "    mesh_workload_id: " << mesh_workload_data.mesh_workload_id << "\n";
        mesh_workloads_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        mesh_workloads_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload created: {}", e.what());
    }
}

void Logger::log_mesh_workload_destroyed(const MeshWorkloadData& mesh_workload_data) noexcept {
    if (!initialized) {
        return;
    }
    try {
        mesh_workloads_ostream << "- mesh_workload_destroyed:\n";
        mesh_workloads_ostream << "    mesh_workload_id: " << mesh_workload_data.mesh_workload_id << "\n";
        mesh_workloads_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        mesh_workloads_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload destroyed: {}", e.what());
    }
}

void Logger::log_mesh_workload_add_program(const MeshWorkloadData& mesh_workload_data, const distributed::MeshCoordinateRange& device_range, std::size_t program_id) noexcept {
    if (!initialized) {
        return;
    }
    try {
        mesh_workloads_ostream << "- mesh_workload_add_program:\n";
        mesh_workloads_ostream << "    mesh_workload_id: " << mesh_workload_data.mesh_workload_id << "\n";
        mesh_workloads_ostream << "    program_id: " << program_id << "\n";
        mesh_workloads_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        mesh_workloads_ostream << "    coordinates:\n";
        for (const auto& coordinate : device_range) {
            auto vector = coordinate.coords();
            mesh_workloads_ostream << "      - [";
            for (size_t i = 0; i < vector.size(); ++i) {
                if (i > 0) {
                    mesh_workloads_ostream << ", ";
                }
                mesh_workloads_ostream << vector[i];
            }
            mesh_workloads_ostream << "]\n";
        }
        mesh_workloads_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload add program: {}", e.what());
    }
}

void Logger::log_mesh_workload_set_program_binary_status(const MeshWorkloadData& mesh_workload_data, std::size_t mesh_id, ProgramBinaryStatus status) noexcept {
    if (!initialized) {
        return;
    }
    try {
        mesh_workloads_ostream << "- mesh_workload_set_program_binary_status:\n";
        mesh_workloads_ostream << "    mesh_workload_id: " << mesh_workload_data.mesh_workload_id << "\n";
        mesh_workloads_ostream << "    mesh_id: " << mesh_id << "\n";
        mesh_workloads_ostream << "    status: ";
        switch (status) {
            case ProgramBinaryStatus::NotSent:
                mesh_workloads_ostream << "NotSent" << "\n";
                break;
            case ProgramBinaryStatus::InFlight:
                mesh_workloads_ostream << "InFlight" << "\n";
                break;
            case ProgramBinaryStatus::Committed:
                mesh_workloads_ostream << "Committed" << "\n";
                break;
        }
        mesh_workloads_ostream << "    timestamp_ns: " << convert_timestamp(std::chrono::high_resolution_clock::now()) << "\n";
        mesh_workloads_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log mesh workload set program binary status: {}", e.what());
    }
}

}  // namespace tt::tt_metal::inspector
