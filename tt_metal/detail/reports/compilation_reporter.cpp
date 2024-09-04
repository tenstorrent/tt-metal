// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <filesystem>
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/report_utils.hpp"

namespace fs = std::filesystem;

namespace tt::tt_metal {

namespace detail {

std::atomic<bool> CompilationReporter::enable_compile_reports_ = false;

void EnableCompilationReports() { CompilationReporter::toggle(true); }
void DisableCompilationReports() { CompilationReporter::toggle(false); }

void CompilationReporter::toggle(bool state)
{
    enable_compile_reports_ = state;
}

bool CompilationReporter::enabled()
{
    return enable_compile_reports_;
}

CompilationReporter& CompilationReporter::inst()
{
    static CompilationReporter inst;
    return inst;
}

CompilationReporter::CompilationReporter() {}

CompilationReporter::~CompilationReporter() {
    if ((this->detailed_report_.is_open() and this->summary_report_.is_open())) {
        string footer = "Number of CompileProgram API calls: " + std::to_string(this->total_num_compile_programs_) + "\n";
        this->detailed_report_ << footer;
        this->summary_report_ << footer;
        this->detailed_report_.close();
        this->summary_report_.close();
    }
}

std::string kernel_attributes_str(std::shared_ptr<Kernel> kernel) {
    std::string attr_str = "{";
    if (not kernel->compile_time_args().empty()) {
        attr_str += "Compile args: [";
        for (const auto compile_arg : kernel->compile_time_args()) {
            attr_str += std::to_string(compile_arg) + " ";
        }
        attr_str += "] ";
    }
    if (not kernel->defines().empty()) {
        attr_str += "Defines: {";
        for (const auto &[k, v] : kernel->defines()) {
            attr_str += "{ " + k + " - " + v + " } ";
        }
        attr_str += "} ";
    }
    auto config = kernel->config();
    if (std::holds_alternative<DataMovementConfig>(config)) {
        attr_str += "NOC: " + std::to_string(std::get<DataMovementConfig>(config).noc) + " ";
    } else {
        TT_ASSERT(std::holds_alternative<ComputeConfig>(config), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(config),__FILE__, __LINE__));
        auto compute_config = std::get<ComputeConfig>(config);
        std::stringstream math_fidel_str;
        math_fidel_str << compute_config.math_fidelity;
        attr_str += "Math fidelity: " + math_fidel_str.str() + " ";
        attr_str += "FP32 dest accumulate enabled: " + std::string(compute_config.fp32_dest_acc_en ? "Y" : "N") + " ";
        attr_str += "Math approx mode enabled: " + std::string(compute_config.math_approx_mode ? "Y" : "N") + " ";
    }

    attr_str += "}";
    return attr_str;
}

void CompilationReporter::add_kernel_compile_stats(const Program &program, std::shared_ptr<Kernel> kernel, bool cache_hit, size_t kernel_hash) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (cache_hit) {
        this->program_id_to_cache_hit_counter_[program.get_id()].hits++;
    } else {
        this->program_id_to_cache_hit_counter_[program.get_id()].misses++;
    }
    std::string kernel_stats = "," + kernel->name() + ",";
    std::string cache_status = cache_hit ? "cache hit" : "cache miss";

    int index = 0;
    for (const auto & core_range : kernel->core_range_set().ranges()) {
        if (index == 0) {
            kernel_stats += "\"" + core_range.str() + "\", " + cache_status + ", " + kernel_attributes_str(kernel) + ", " + std::to_string(kernel_hash) + "\n";
        } else {
            kernel_stats += ",,\"" + core_range.str() + "\", , ,\n";
        }
        index++;
    }
    this->program_id_to_kernel_stats_[program.get_id()].push_back(kernel_stats);
}

void CompilationReporter::flush_program_entry(const Program &program, bool persistent_compilation_cache_enabled) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto num_cache_misses = this->program_id_to_cache_hit_counter_.at(program.get_id()).misses;
    auto num_cache_hits = this->program_id_to_cache_hit_counter_.at(program.get_id()).hits;
    if (this->total_num_compile_programs_ == 0) {
        this->init_reports();
    }

    auto get_num_compute_and_data_movement_kernels = [&]() {
        uint32_t num_compute = 0;
        uint32_t num_data_movement = 0;
        for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
            const auto kernel = detail::GetKernel(program, kernel_id);
            if (kernel->processor() == tt::RISCV::BRISC or kernel->processor() == tt::RISCV::NCRISC) {
                num_data_movement++;
            } else {
                num_compute++;
            }
        }
        return std::make_pair(num_compute, num_data_movement);
    };

    auto [num_compute_kernels, num_data_movement_kernels] = get_num_compute_and_data_movement_kernels();

    this->summary_report_ << program.get_id() << ", "
                            << num_compute_kernels << ", "
                            << num_data_movement_kernels << ", "
                            << (persistent_compilation_cache_enabled ? "Y" : "N") << ", "
                            << num_cache_misses << ", "
                            << num_cache_hits << "\n";

    this->detailed_report_ << "Compiling Program: " << program.get_id() << "\n";
    this->detailed_report_ << "\n,Kernel Creation Report:\n";
    this->detailed_report_ << ",,Number of Compute CreateKernel API calls: " << num_compute_kernels << "\n";
    this->detailed_report_ << ",,Number of Datamovement CreateKernel API calls: " << num_data_movement_kernels << "\n";

    this->detailed_report_ << "\n,Kernel Compilation Report:\n";
    this->detailed_report_ << ",,Persistent kernel compile cache enabled: " << (persistent_compilation_cache_enabled ? "Y\n" : "N\n");
    this->detailed_report_ << ",,Total number of kernel compile cache misses: " << num_cache_misses << "\n";
    this->detailed_report_ << ",,Total number of kernel compile cache hits: " << num_cache_hits << "\n";

    this->detailed_report_ << "\n,Kernel File Name, Core Range, Cache Hit, Kernel Attributes, Hash\n";
    auto kernel_stats_vec = this->program_id_to_kernel_stats_.at(program.get_id());
    for (const auto &kernel_stats : kernel_stats_vec) {
        this->detailed_report_ << kernel_stats;
    }
    this->detailed_report_ << "\n";

    this->summary_report_.flush();
    this->detailed_report_.flush();
    this->total_num_compile_programs_++;
}

void CompilationReporter::init_reports() {
    static const std::string compile_report_path = metal_reports_dir() + "compile_program.csv";
    static const std::string summary_report_path = metal_reports_dir() + "compile_program_summary.csv";
    fs::create_directories(metal_reports_dir());
    this->detailed_report_.open(compile_report_path);
    this->summary_report_.open(summary_report_path);
    this->summary_report_ << "Program, Number of CreateKernel API calls, Number of CreateKernel API calls, Persistent Kernel Compile Cache Enabled, Total Number of Kernel Cache Misses, Total Number of Kernel Cache Hits\n";
}

}   // namespace detail

}   // namespace tt::tt_metal
