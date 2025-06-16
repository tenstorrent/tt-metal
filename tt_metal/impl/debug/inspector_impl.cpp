// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inspector_impl.hpp"
#include <iomanip>
#include <chrono>
#include <ctime>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::inspector {

Logger::Logger(const std::filesystem::path& logging_path)
    : programs_ostream(logging_path / "programs_log.yaml", std::ios::trunc)
    , kernels_ostream(logging_path / "kernels.yaml", std::ios::trunc) {
    {
        std::ofstream inspector_startup_ostream(logging_path / "startup.yaml", std::ios::trunc);

        if (!inspector_startup_ostream.is_open()) {
            TT_INSPECTOR_THROW("Failed to create inspector file: {}", (logging_path / "startup.yaml").string());
        }
        else {
            // Log current system time and high_resolution_clock time_point
            auto now_system = std::chrono::system_clock::now();
            auto now_highres = std::chrono::high_resolution_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now_system);
            auto now_system_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now_system.time_since_epoch()).count();
            auto now_highres_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now_highres.time_since_epoch()).count();

            start_time = now_highres;
            inspector_startup_ostream << "startup_time:\n";
            inspector_startup_ostream << "  system_clock_iso: '" << std::put_time(std::gmtime(&now_c), "%Y-%m-%dT%H:%M:%SZ") << "'\n";
            inspector_startup_ostream << "  system_clock_ns: " << now_system_ns << "\n";
            inspector_startup_ostream << "  high_resolution_clock_ns: " << now_highres_ns << "\n";
        }
    }
    if (!programs_ostream.is_open()) {
        TT_INSPECTOR_THROW("Failed to create inspector file: {}", (logging_path / "programs_log.yaml").string());
    }
    if (!kernels_ostream.is_open()) {
        TT_INSPECTOR_THROW("Failed to create inspector file: {}", (logging_path / "kernels.yaml").string());
    }
}

void Logger::log_program_created(const ProgramData& program_data) noexcept {
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
    try {
        programs_ostream << "- program_compile_started:\n";
        programs_ostream << "    id: " << program_data.program_id << "\n";
        programs_ostream << "    timestamp_ns: " << convert_timestamp(program_data.compile_started_timestamp) << "\n";
        programs_ostream.flush();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG("Failed to log program compile started: {}", e.what());
    }
}

void Logger::log_program_compile_already_exists(const ProgramData& program_data) noexcept {
    // Long running programs call this log entry too many times, so we don't want to log it.
}

void Logger::log_program_kernel_compile_finished(const ProgramData& program_data, const KernelData& kernel_data) noexcept {
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

Data::Data()
    : logger(MetalContext::instance().rtoptions().get_inspector_log_path()) {
}

}  // namespace tt::tt_metal::inspector
