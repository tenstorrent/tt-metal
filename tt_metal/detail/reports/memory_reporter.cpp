// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/reports/report_utils.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/device/device_impl.hpp"
#include "tt_metal/impl/program/program.hpp"

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace tt::tt_metal {

namespace detail {

using bank_to_statistics = std::map<uint32_t, allocator::Statistics>;

std::atomic<bool> MemoryReporter::is_enabled_ = false;

MemoryReporter::~MemoryReporter() {
    if (this->program_l1_usage_summary_report_.is_open()) {
        this->program_l1_usage_summary_report_.close();
    }
    if (this->program_memory_usage_summary_report_.is_open()) {
        this->program_memory_usage_summary_report_.close();
    }
    if (this->program_detailed_memory_usage_report_.is_open()) {
        this->program_detailed_memory_usage_report_.close();
    }
}

void write_headers(std::ofstream &memory_usage_summary_report, std::ofstream &l1_usage_summary_report, bool add_program_id) {
    l1_usage_summary_report << "This report indicates available space for interleaving L1 buffers\n";
    if (add_program_id) {
        l1_usage_summary_report << "Program ID";
        memory_usage_summary_report << "Program ID";
    }
    l1_usage_summary_report << ", Largest Contiguous Free Block (B), Total Free L1 Space (B)\n";
    memory_usage_summary_report << ", Total Allocatable Size (B), Total Allocated (B), Total Free (KB), Largest Free Block (KB)\n";
}

void write_detailed_report_info(
    const Device *device,
    const BufferType &buffer_type,
    std::ofstream &detailed_memory_usage_report,
    size_t total_allocatable, size_t total_allocated, size_t total_free,
    const bank_to_statistics &bank_stats
    ) {
    detailed_memory_usage_report << ",Total allocatable (B):," << total_allocatable << "\n"
                                    << ",Total allocated (B):," << total_allocated << "\n"
                                    << ",Total free (B):," << total_free << "\n";

    for (const auto &[bank_id, stats] : bank_stats) {
        detailed_memory_usage_report << "Bank ID:," << bank_id << "\n"
                                        << ",,Total allocatable (B): " << stats.total_allocatable_size_bytes << "\n"
                                        << ",,Total free (B): " << stats.total_free_bytes << "\n"
                                        << ",,Total allocated (B): " << stats.total_allocated_bytes << "\n"
                                        << ",,Largest free block (B): " << stats.largest_free_block_bytes << "\n";
        device->dump_memory_blocks(buffer_type, detailed_memory_usage_report);
    }
}

void write_memory_usage(
    const Device *device,
    const BufferType &buffer_type,
    std::ofstream &memory_usage_summary_report,
    std::ofstream &detailed_memory_usage_report,
    std::ofstream &l1_usage_summary_report
) {
    auto num_banks = device->num_banks(buffer_type);
    auto stats = device->get_memory_allocation_statistics(buffer_type);
    memory_usage_summary_report << ","
                                << stats.total_allocatable_size_bytes << ","
                                << stats.total_allocated_bytes << ","
                                << stats.total_free_bytes << ","
                                << stats.largest_free_block_bytes << "\n";


    detailed_memory_usage_report << "," << (buffer_type == BufferType::DRAM ? "DRAM\n" : "L1\n");
    detailed_memory_usage_report << ",Total allocatable (B):," << (stats.total_allocatable_size_bytes * num_banks) << "\n"
                                 << ",Total allocated (B):," << (stats.total_allocated_bytes * num_banks) << "\n"
                                 << ",Total free (B):," << (stats.total_free_bytes * num_banks) << "\n";
    device->dump_memory_blocks(buffer_type, detailed_memory_usage_report);

    if (buffer_type == BufferType::L1) {
        l1_usage_summary_report << ","
                        << stats.largest_free_block_bytes << ","
                        << (stats.largest_free_block_bytes * num_banks) << "\n";
    }
}

void populate_reports(const Device *device, std::ofstream &memory_usage_summary_report, std::ofstream &detailed_memory_usage_report, std::ofstream &l1_usage_summary_report) {

    write_memory_usage(device, BufferType::DRAM, memory_usage_summary_report, detailed_memory_usage_report, l1_usage_summary_report);

    write_memory_usage(device, BufferType::L1, memory_usage_summary_report, detailed_memory_usage_report, l1_usage_summary_report);
}

void MemoryReporter::flush_program_memory_usage(const Program &program, const Device *device) {
    if (not this->program_memory_usage_summary_report_.is_open()) {
        this->init_reports();
    }

    this->program_memory_usage_summary_report_ << program.get_id();
    this->program_l1_usage_summary_report_ << program.get_id();
    this->program_detailed_memory_usage_report_ << program.get_id();

    populate_reports(device, this->program_memory_usage_summary_report_, this->program_detailed_memory_usage_report_, this->program_l1_usage_summary_report_);
}

void MemoryReporter::dump_memory_usage_state(const Device *device, std::string prefix) const {
    std::ofstream memory_usage_summary_report, l1_usage_summary_report, detailed_memory_usage_report;

    fs::create_directories(metal_reports_dir());
    memory_usage_summary_report.open(metal_reports_dir() + prefix + "memory_usage_summary.csv");
    l1_usage_summary_report.open(metal_reports_dir() + prefix + "l1_usage_summary.csv");
    detailed_memory_usage_report.open(metal_reports_dir() + prefix + "detailed_memory_usage.csv");

    write_headers(memory_usage_summary_report, l1_usage_summary_report, /*add_program_id=*/false);
    populate_reports(device, memory_usage_summary_report, detailed_memory_usage_report, l1_usage_summary_report);

    memory_usage_summary_report.close();
    l1_usage_summary_report.close();
    detailed_memory_usage_report.close();
}

void MemoryReporter::init_reports() {
    fs::create_directories(metal_reports_dir());
    this->program_memory_usage_summary_report_.open(metal_reports_dir() + "program_memory_usage_summary.csv");
    this->program_l1_usage_summary_report_.open(metal_reports_dir() + "program_l1_usage_summary.csv");
    this->program_detailed_memory_usage_report_.open(metal_reports_dir() + "program_detailed_memory_usage.csv");
    write_headers(this->program_memory_usage_summary_report_, this->program_l1_usage_summary_report_, /*add_program_id=*/true);
}
void DumpDeviceMemoryState(const Device *device, std::string prefix) {
    MemoryReporter::inst().dump_memory_usage_state(device, prefix);
}

bool MemoryReporter::enabled() {
    return is_enabled_;
}

void MemoryReporter::toggle(bool state)
{
    is_enabled_ = state;
}

MemoryReporter& MemoryReporter::inst()
{
    static MemoryReporter inst;
    return inst;
}

void EnableMemoryReports() { MemoryReporter::toggle(true); }
void DisableMemoryReports() { MemoryReporter::toggle(false); }

}   // namespace detail

}   // namespace tt::tt_metal
