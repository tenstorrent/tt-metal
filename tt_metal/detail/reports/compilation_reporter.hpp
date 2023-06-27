#pragma once

#include <filesystem>
#include <mutex>

#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program.hpp"

using std::unique_lock;
using std::mutex;

namespace tt::tt_metal {

namespace detail {

class CompilationReporter {
   public:
    CompilationReporter();

    ~CompilationReporter();

    void add_kernel_compile_stats(const Program &program, Kernel *kernel, bool cache_hit, size_t kernel_hash);

    void flush_program_entry(const Program &program, bool persistent_compilation_cache_enabled);

   private:
    void init_reports();

    struct cache_counters {int misses = 0; int hits = 0; };
    std::mutex mutex_;
    size_t total_num_compile_programs_ = 0;
    std::unordered_map<u64, cache_counters> program_id_to_cache_hit_counter_;
    std::unordered_map<u64, std::vector<string>> program_id_to_kernel_stats_;
    std::ofstream detailed_report_;
    std::ofstream summary_report_;
};

}   // namespace detail

}   // namespace tt::tt_metal
