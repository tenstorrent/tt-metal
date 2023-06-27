#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/reports/report_utils.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

#include <algorithm>

namespace tt::tt_metal {

namespace detail {

using bank_to_statistics = std::map<u32, allocator::Statistics>;

MemoryReporter::MemoryReporter() {}

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
    l1_usage_summary_report << ", Minimum Largest Contiguous Free Block (KB), Total Free L1 Space (KB)\n";
    memory_usage_summary_report << ", DRAM channel/Logical core, Bank ID, Total Allocatable Size (KB), Total Allocated (KB), Total Free (KB), Largest Free Block (KB)\n";
}

void write_detailed_report_info(
    const Device *device,
    const BufferType &buffer_type,
    std::ofstream &detailed_memory_usage_report,
    size_t total_allocatable, size_t total_allocated, size_t total_free,
    const bank_to_statistics &bank_stats
    ) {
    detailed_memory_usage_report << ",Total allocatable (KB):," << total_allocatable << "\n"
                                    << ",Total allocated (KB):," << total_allocated << "\n"
                                    << ",Total free (KB):," << total_free << "\n";

    for (const auto &[bank_id, stats] : bank_stats) {
        detailed_memory_usage_report << "Bank ID:," << bank_id << "\n"
                                        << ",,Total allocatable (KB): " << stats.total_allocatable_size_bytes / 1024 << "\n"
                                        << ",,Total free (KB): " << stats.total_free_bytes / 1024 << "\n"
                                        << ",,Total allocated (KB): " << stats.total_allocated_bytes / 1024 << "\n"
                                        << ",,Largest free block (KB): " << stats.largest_free_block_bytes / 1024 << "\n";
        device->dump_memory_blocks(buffer_type, bank_id, detailed_memory_usage_report);
    }
}

// Writes DRAM memory usage summary and detailed reports
void write_dram_memory_usage(
    const Device *device,
    std::ofstream &memory_usage_summary_report,
    std::ofstream &detailed_memory_usage_report
) {
    std::map<u32, bank_to_statistics> dram_channel_to_bank_stats;
    std::map<u32, allocator::Statistics> dram_channel_totals;
    for (auto dram_bank_id = 0; dram_bank_id < device->num_banks(BufferType::DRAM); dram_bank_id++) {
        auto dram_channel = device->dram_channel_from_bank_id(dram_bank_id);
        auto stats = device->get_memory_allocation_statistics(BufferType::DRAM, dram_bank_id);
        memory_usage_summary_report << ","
                                    << dram_channel << ","
                                    << dram_bank_id << ","
                                    << stats.total_allocatable_size_bytes / 1024 << ","
                                    << stats.total_allocated_bytes / 1024 << ","
                                    << stats.total_free_bytes / 1024 << ","
                                    << stats.largest_free_block_bytes / 1024 << "\n";
        dram_channel_to_bank_stats[dram_channel][dram_bank_id] = stats;
        auto &total_stats = dram_channel_totals[dram_channel];
        total_stats.total_allocatable_size_bytes += stats.total_allocatable_size_bytes;
        total_stats.total_allocated_bytes += stats.total_allocated_bytes;
        total_stats.total_free_bytes += stats.total_free_bytes;
    }

    // Populate detailed memory usage summary
    for (const auto &[dram_channel, dram_bank_stats] : dram_channel_to_bank_stats) {
        size_t total_allocatable = dram_channel_totals.at(dram_channel).total_allocatable_size_bytes / 1024;
        size_t total_allocated = dram_channel_totals.at(dram_channel).total_allocated_bytes / 1024;
        size_t total_free = dram_channel_totals.at(dram_channel).total_free_bytes / 1024;
        detailed_memory_usage_report << ",DRAM channel ID," << dram_channel << "\n";
        write_detailed_report_info(
            device,
            BufferType::DRAM,
            detailed_memory_usage_report,
            total_allocatable, total_allocated, total_free,
            dram_bank_stats
        );
    }
}

// Writes L1 memory usage summary and populates helper maps to write detailed memory usage report
size_t write_l1_memory_usage_summary(
    const Device *device,
    std::ofstream &memory_usage_summary_report,
    std::map<CoreCoord, bank_to_statistics> &logical_core_to_bank_stats,
    std::map<CoreCoord, allocator::Statistics> &l1_core_totals,
    std::map<CoreCoord, size_t> &logical_core_to_max_bank_offset) {
    size_t min_largest_free_block = (size_t)-1;
    for (auto l1_bank_id = 0; l1_bank_id < device->num_banks(BufferType::L1); l1_bank_id++) {
        auto logical_core = device->logical_core_from_bank_id(l1_bank_id);
        auto stats = device->get_memory_allocation_statistics(BufferType::L1, l1_bank_id);
        memory_usage_summary_report << ","
                                    << "\"" + logical_core.str() + "\", "
                                    << l1_bank_id << ","
                                    << stats.total_allocatable_size_bytes / 1024 << ","
                                    << stats.total_allocated_bytes / 1024 << ","
                                    << stats.total_free_bytes / 1024 << ","
                                    << stats.largest_free_block_bytes / 1024 << "\n";
        logical_core_to_bank_stats[logical_core][l1_bank_id] = stats;
        auto &total_stats = l1_core_totals[logical_core];
        total_stats.total_allocatable_size_bytes += stats.total_allocatable_size_bytes;
        total_stats.total_allocated_bytes += stats.total_allocated_bytes;
        total_stats.total_free_bytes += stats.total_free_bytes;

        auto bank_offset = device->l1_bank_offset_from_bank_id(l1_bank_id);
        logical_core_to_max_bank_offset[logical_core] = std::max(logical_core_to_max_bank_offset[logical_core], (size_t)std::abs(bank_offset));
        min_largest_free_block = std::min(min_largest_free_block, stats.largest_free_block_bytes);
    }
    return min_largest_free_block;
}

void write_l1_memory_usage(const Device *device, std::ofstream &memory_usage_summary_report, std::ofstream &detailed_memory_usage_report, std::ofstream &l1_usage_summary_report) {
    std::map<CoreCoord, bank_to_statistics> logical_core_to_bank_stats;
    std::map<CoreCoord, allocator::Statistics> l1_core_totals;
    std::map<CoreCoord, size_t> logical_core_to_max_bank_offset;
    // Populates helper maps for writing detailed l1 memory usage and l1 usage summary report
    auto min_largest_free_block = write_l1_memory_usage_summary(
        device, memory_usage_summary_report, logical_core_to_bank_stats, l1_core_totals, logical_core_to_max_bank_offset);

    std::set<u32> candidate_addrs;
    for (const auto &[logical_core, l1_bank_stats] : logical_core_to_bank_stats) {
        size_t total_allocatable = l1_core_totals.at(logical_core).total_allocatable_size_bytes / 1024;
        size_t total_allocated = l1_core_totals.at(logical_core).total_allocated_bytes / 1024;
        size_t total_free = l1_core_totals.at(logical_core).total_free_bytes / 1024;

        detailed_memory_usage_report << ",\"Logical core " + logical_core.str() + "\"\n";
        write_detailed_report_info(
            device,
            BufferType::L1,
            detailed_memory_usage_report,
            total_allocatable, total_allocated, total_free,
            l1_bank_stats
        );

        // COMMENT
        if (min_largest_free_block == 0) {
            continue;
        }

        auto max_bank_offset = logical_core_to_max_bank_offset.at(logical_core);
        // std::cout << "For core: " << logical_core.str() << " max bank offset: " << max_bank_offset << std::endl;
        std::set<u32> addrs;
        for (const auto &[bank_id, stats] : l1_bank_stats) {
            auto bank_offset = std::abs(device->l1_bank_offset_from_bank_id(bank_id));
            auto offset = max_bank_offset - bank_offset;
            // std::cout << "\tbank id " << bank_id << " offset: " << bank_offset << " offset to add: " << offset << std::endl;
            for (const auto relative_addr : stats.largest_free_block_addrs) {
                auto snapped_up_addr = relative_addr + offset;
                // std::cout << "\t\tadding snapped up addr: " << snapped_up_addr << std::endl;
                addrs.insert(snapped_up_addr);
            }
        }
        if (candidate_addrs.empty()) {
            candidate_addrs = addrs;
        } else {
            std::set<u32> intersect;
            std::set_intersection(
                addrs.begin(), addrs.end(), candidate_addrs.begin(), candidate_addrs.end(), std::inserter(intersect, intersect.begin()));
            candidate_addrs = std::move(intersect);
        }
    }

    min_largest_free_block = candidate_addrs.empty() ? 0 : min_largest_free_block;
    // Populate l1 usage summary
    l1_usage_summary_report << ","
                            << min_largest_free_block / 1024 << ","
                            << (min_largest_free_block/ 1024) * device->num_banks(BufferType::L1) << "\n";
}

void populate_reports(const Device *device, std::ofstream &memory_usage_summary_report, std::ofstream &l1_usage_summary_report, std::ofstream &detailed_memory_usage_report) {

    write_dram_memory_usage(device, memory_usage_summary_report, detailed_memory_usage_report);

    write_l1_memory_usage(device, memory_usage_summary_report, detailed_memory_usage_report, l1_usage_summary_report);
}

void MemoryReporter::flush_program_memory_usage(const Program &program, const Device *device) {
    if (not this->program_memory_usage_summary_report_.is_open()) {
        this->init_reports();
    }

    this->program_memory_usage_summary_report_ << program.get_id();
    this->program_l1_usage_summary_report_ << program.get_id();
    this->program_detailed_memory_usage_report_ << program.get_id();

    populate_reports(device, this->program_memory_usage_summary_report_, this->program_l1_usage_summary_report_, this->program_detailed_memory_usage_report_);
}

void MemoryReporter::dump_memory_usage_state(const Device *device) const {
    std::ofstream memory_usage_summary_report, l1_usage_summary_report, detailed_memory_usage_report;

    fs::create_directories(metal_reports_dir());
    memory_usage_summary_report.open(metal_reports_dir() + "memory_usage_summary.csv");
    l1_usage_summary_report.open(metal_reports_dir() + "l1_usage_summary.csv");
    detailed_memory_usage_report.open(metal_reports_dir() + "detailed_memory_usage.csv");

    write_headers(memory_usage_summary_report, l1_usage_summary_report, /*add_program_id=*/false);
    populate_reports(device, memory_usage_summary_report, l1_usage_summary_report, detailed_memory_usage_report);

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

}   // namespace detail

}   // namespace tt::tt_metal
