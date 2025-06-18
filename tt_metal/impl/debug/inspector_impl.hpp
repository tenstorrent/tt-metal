// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <fstream>
#include <memory>
#include <filesystem>
#include <unordered_map>

#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {
    class Inspector;
    class MetalContext;
}

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

class Data {
private:
    Data(); // NOLINT - False alarm, tt::tt_metal::Inspector is calling this constructor.

    inspector::Logger logger;
    std::mutex programs_mutex;
    std::unordered_map<uint64_t, inspector::ProgramData> programs_data;

    friend class tt::tt_metal::Inspector;
};

}  // namespace tt::tt_metal::inspector
