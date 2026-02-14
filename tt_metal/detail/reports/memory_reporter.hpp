// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <fstream>
#include <string>

#include <tt-metalium/memory_reporter.hpp>

namespace tt::tt_metal {
class IDevice;
enum class BufferType;
}  // namespace tt::tt_metal

namespace tt::tt_metal::detail {

class MemoryReporter {
public:
    MemoryReporter& operator=(const MemoryReporter&) = delete;
    MemoryReporter& operator=(MemoryReporter&& other) noexcept = delete;
    MemoryReporter(const MemoryReporter&) = delete;
    MemoryReporter(MemoryReporter&& other) noexcept = delete;

    void flush_program_memory_usage(uint64_t program_id, const IDevice* device);

    void dump_memory_usage_state(const IDevice* device, const std::string& prefix = "") const;

    MemoryView get_memory_view(const IDevice* device, const BufferType& buffer_type) const;

    static void toggle(bool state);
    static MemoryReporter& inst();
    static bool enabled();

private:
    MemoryReporter() = default;
    ~MemoryReporter();
    void init_reports();
    static std::atomic<bool> is_enabled_;
    std::ofstream program_l1_usage_summary_report_;
    std::ofstream program_memory_usage_summary_report_;
    std::ofstream program_detailed_memory_usage_report_;
};

}  // namespace tt::tt_metal::detail
