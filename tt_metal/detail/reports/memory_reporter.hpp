#pragma once

#include <filesystem>
#include <mutex>

#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program.hpp"
#include "tt_metal/impl/device/device.hpp"

using std::unique_lock;
using std::mutex;

namespace tt::tt_metal {

namespace detail {

class MemoryReporter {
   public:
    MemoryReporter();

    ~MemoryReporter();

    void flush_program_memory_usage(const Program &program, const Device *device);

    void dump_memory_usage_state(const Device *device) const;

   private:
    void init_reports();

    std::mutex mutex_;
    std::ofstream program_l1_usage_summary_report_;
    std::ofstream program_memory_usage_summary_report_;
    std::ofstream program_detailed_memory_usage_report_;
};

}   // namespace detail

}   // namespace tt::tt_metal
