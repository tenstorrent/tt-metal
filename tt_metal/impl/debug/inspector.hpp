// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <filesystem>
#include <unordered_map>

#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

namespace inspector {
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

    struct KernelData {
        std::weak_ptr<Kernel> kernel;
        std::string name;
        std::string path;
        std::string source;
        int watcher_kernel_id;
    };

    struct ProgramData {
        std::weak_ptr<const detail::ProgramImpl> program;
        uint64_t program_id;
        time_point compile_started_timestamp;
        time_point compile_finished_timestamp;
        std::unordered_map<int, KernelData> kernels;
        std::unordered_map<std::size_t, ProgramBinaryStatus> binary_status_per_device;
    };

    class Logger {
    private:
        time_point start_time;
        std::ofstream programs_ostream;
        std::ofstream kernels_ostream;

        int64_t convert_timestamp(const time_point& tp) const {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
        }

    public:
        Logger(const std::filesystem::path& logging_path);

        void log_program_created(const ProgramData& program_data) noexcept;
        void log_program_destroyed(const ProgramData& program_data) noexcept;
        void log_program_compile_started(const ProgramData& program_data) noexcept;
        void log_program_compile_already_exists(const ProgramData& program_data) noexcept;
        void log_program_kernel_compile_finished(const ProgramData& program_data, const inspector::KernelData& kernel_data) noexcept;
        void log_program_compile_finished(const ProgramData& program_data) noexcept;
        void log_program_binary_status_change(const ProgramData& program_data, std::size_t device_id, ProgramBinaryStatus status) noexcept;
    };
}

class Inspector {
private:
    inspector::Logger logger;

    Inspector();
    Inspector(const Inspector&) = delete;
    Inspector& operator=(const Inspector&) = delete;

    // This is a singleton class
    static Inspector& instance() {
        static Inspector instance;
        return instance;
    }

    std::mutex programs_mutex;
    std::unordered_map<uint64_t, inspector::ProgramData> programs_data;

public:
    static bool is_enabled();

    static void program_created(
        const detail::ProgramImpl* program) noexcept;
    static void program_destroyed(
        const detail::ProgramImpl* program) noexcept;
    static void program_set_binary_status(
        const detail::ProgramImpl* program,
        std::size_t device_id,
        ProgramBinaryStatus status) noexcept;
    static void program_compile_started(
        const detail::ProgramImpl* program,
        const IDevice* device,
        uint32_t build_key) noexcept;
    static void program_compile_already_exists(
        const detail::ProgramImpl* program,
        const IDevice* device,
        uint32_t build_key) noexcept;
    static void program_kernel_compile_finished(
        const detail::ProgramImpl* program,
        const IDevice* device,
        const std::shared_ptr<Kernel>& kernel,
        const tt::tt_metal::JitBuildOptions& build_options) noexcept;
    static void program_compile_finished(
        const detail::ProgramImpl* program,
        const IDevice* device,
        uint32_t build_key) noexcept;
};

}  // namespace tt::tt_metal
