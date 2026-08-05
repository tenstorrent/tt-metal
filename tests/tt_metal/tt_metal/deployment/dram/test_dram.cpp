// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "dram_base.hpp"
#include "kernels/common_dram.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include <atomic>
#include <csignal>
#include <cstdlib>

#include <thread>
#include <chrono>
#include <unistd.h>
#include <mutex>
#include <condition_variable>

#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <dirent.h>

#include <iomanip>
#include <array>

namespace tt::tt_metal {

using namespace std;
using namespace tt;

std::atomic<bool> g_stop_requested{false};
std::atomic<bool> g_watchdog_requested{false};
static std::atomic<bool> g_stop_message_printed{false};

static std::string test_dram_trim_copy(std::string s) {
    while (!s.empty() && std::isspace(s.front())) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(s.back())) {
        s.pop_back();
    }
    return s;
}

static std::string test_dram_read_text_file_trimmed(const std::string& path) {
    std::ifstream file(path);
    std::string value;
    std::getline(file, value);
    return test_dram_trim_copy(value);
}

static std::vector<std::string> get_tenstorrent_pci_bdf_lines() {
    std::vector<std::string> lines;

    DIR* dir = opendir("/sys/bus/pci/devices");
    if (!dir) {
        return lines;
    }

    while (auto* entry = readdir(dir)) {
        const std::string bdf = entry->d_name;
        if (bdf.empty() || bdf[0] == '.') {
            continue;
        }

        const std::string base = "/sys/bus/pci/devices/" + bdf;
        std::string vendor = test_dram_read_text_file_trimmed(base + "/vendor");
        std::transform(vendor.begin(), vendor.end(), vendor.begin(), [](char c) { return std::tolower(c); });

        // Tenstorrent PCI vendor id. This catches Blackhole/Galaxy boards exposed on PCIe.
        if (vendor != "0x1e52") {
            continue;
        }

        const std::string pci_device = test_dram_read_text_file_trimmed(base + "/device");
        const std::string bus = bdf.size() >= 7 ? bdf.substr(bdf.size() - 7, 2) : "??";
        const std::string device = bdf.size() >= 4 ? bdf.substr(bdf.size() - 4, 2) : "??";
        const std::string function = !bdf.empty() ? bdf.substr(bdf.size() - 1, 1) : "?";

        lines.push_back(
            fmt::format("bdf={} bus={} device={} function={} pci_device={}", bdf, bus, device, function, pci_device));
    }

    closedir(dir);
    std::sort(lines.begin(), lines.end());
    return lines;
}

static void log_blackhole_galaxy_pci_bdfs_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        const auto lines = get_tenstorrent_pci_bdf_lines();

        if (lines.empty()) {
            log_info(
                tt::LogTest, "Blackhole Galaxy PCI BDF scan: no Tenstorrent PCI devices found in /sys/bus/pci/devices");
            return;
        }

        log_info(tt::LogTest, "Blackhole Galaxy PCI BDF scan: found {} Tenstorrent PCI device(s)", lines.size());
        for (const auto& line : lines) {
            log_info(tt::LogTest, "Blackhole Galaxy PCI: {}", line);
        }
    });
}

struct DramBankSummary {
    bool pass = true;
    uint64_t checked_bytes = 0;
    uint64_t suspected_write_error_bytes = 0;
    uint64_t suspected_read_error_bytes = 0;
};

struct DramChipSummary {
    uint32_t device_id = 0;
    bool pass = true;
    std::vector<DramBankSummary> banks;
};

struct DramGalaxySummary {
    uint32_t chips_tested = 0;
    uint64_t jobs = 0;
    uint64_t checked_bytes = 0;
    uint64_t suspected_write_error_bytes = 0;
    uint64_t suspected_read_error_bytes = 0;
    bool pass = true;

    std::vector<DramChipSummary> chips;
};

struct DramPatternTimingSummary {
    static constexpr uint32_t kMaxPatternId = 32u;
    uint64_t ticks[kMaxPatternId] = {};
    uint64_t jobs[kMaxPatternId] = {};
    uint64_t bytes[kMaxPatternId] = {};
};

static void accumulate_pattern_timing_summary(DramPatternTimingSummary& dst, const DramMultiInstanceSummary& run) {
    for (const auto& entry : run.per_core_results) {
        const DramBaseResult& r = entry.result;

        if (r.pattern_id >= DramPatternTimingSummary::kMaxPatternId) {
            continue;
        }

        dst.ticks[r.pattern_id] += r.job_total_ticks;
        dst.jobs[r.pattern_id] += 1u;
        dst.bytes[r.pattern_id] += (uint64_t)r.words_checked * sizeof(uint32_t);
    }
}

static void merge_pattern_timing_summary(DramPatternTimingSummary& dst, const DramPatternTimingSummary& src) {
    for (uint32_t pattern_id = 0; pattern_id < DramPatternTimingSummary::kMaxPatternId; ++pattern_id) {
        dst.ticks[pattern_id] += src.ticks[pattern_id];
        dst.jobs[pattern_id] += src.jobs[pattern_id];
        dst.bytes[pattern_id] += src.bytes[pattern_id];
    }
}

static uint32_t get_dram_test_loops_from_env(uint32_t default_loops = 1u) {
    const char* env = std::getenv("DRAM_TEST_LOOPS");
    if (!env || env[0] == '\0') {
        return default_loops;
    }

    char* end = nullptr;
    const unsigned long value = std::strtoul(env, &end, 0);
    if (end == env || value == 0ul) {
        return default_loops;
    }

    return static_cast<uint32_t>(value);
}

static bool pattern_timing_enabled() {
    const char* env = std::getenv("DRAM_TEST_PATTERN_TIMING");
    return env && std::atoi(env);
}

static void log_pattern_timing_summary(
    const char* test_name, const DramPatternTimingSummary& summary, double test_wall_ms) {
    if (!pattern_timing_enabled()) {
        return;
    }

    bool any = false;

    for (auto job : summary.jobs) {
        if (job) {
            any = true;
            break;
        }
    }

    if (!any) {
        return;
    }

    // timestamp() ticks are treated as 1 GHz device ticks here: 1 tick = 1 ns.
    // Raw sums are worker-time, so they can exceed wall-clock time when workers run in parallel.
    // Scale all pattern times so the sum of displayed pattern ms matches the real host wall-clock test duration.
    constexpr double kBlackholeClockHz = 1000000000.0;

    double raw_total_ms = 0.0;

    for (auto tick : summary.ticks) {
        raw_total_ms += tick * 1000.0 / kBlackholeClockHz;
    }

    const double scale = raw_total_ms > 0.0 && test_wall_ms > 0.0 ? test_wall_ms / raw_total_ms : 1.0;

    log_info(tt::LogTest, "=== {} BRISC job time by DRAM pattern ===", test_name);
    log_info(
        tt::LogTest,
        "Pattern timing is scaled from BRISC worker-time to host wall-clock: raw_sum_ms={:.3f} wall_ms={:.3f} "
        "scale={:.9f}",
        raw_total_ms,
        test_wall_ms,
        scale);
    log_info(
        tt::LogTest, "| {:>2} | {:<24} | {:>8} | {:>14} | {:>14} |", "ID", "Pattern", "Jobs", "Wall ms", "Checked MB");

    for (uint32_t pattern_id = 0; pattern_id < DramPatternTimingSummary::kMaxPatternId; ++pattern_id) {
        if (summary.jobs[pattern_id] == 0u) {
            continue;
        }

        const double raw_pattern_ms = summary.ticks[pattern_id] * 1000.0 / kBlackholeClockHz;
        const double scaled_pattern_ms = raw_pattern_ms * scale;

        log_info(
            tt::LogTest,
            "| {:>2} | {:<24} | {:>8} | {:>14.3f} | {:>14.2f} |",
            pattern_id,
            pattern_name(pattern_id),
            summary.jobs[pattern_id],
            scaled_pattern_ms,
            summary.bytes[pattern_id] / (1024.0 * 1024.0));
    }
}

static void handle_sigint(int) {
    g_stop_requested.store(true);

    if (!g_stop_message_printed.exchange(true)) {
        const char msg[] = "\nSIGINT received, requesting graceful stop...\n";
        [[maybe_unused]] ssize_t written = write(STDERR_FILENO, msg, sizeof msg - 1); /* NOLINT */
    }
}

[[maybe_unused]]
static uint64_t bytes_to_mb_floor(uint64_t bytes) {
    return bytes / (1024ull * 1024ull);
}

static std::string format_bytes(uint64_t bytes) {
    std::ostringstream oss;

    const double KB = 1024.0;
    const double MB = 1024.0 * 1024.0;

    if (bytes < 1024ull) {
        oss << bytes << "B";
    } else if (bytes < 1024ull * 1024ull) {
        oss << std::fixed << std::setprecision(2) << (bytes / KB) << "KB";
    } else {
        oss << std::fixed << std::setprecision(2) << (bytes / MB) << "MB";
    }

    return oss.str();
}

static std::string format_error_pct(double pct) {
    if (pct == 0.0) {
        return "0%";
    }

    if (pct < 0.001) {
        return "<0.001%";
    }

    std::string out = fmt::format("{:.3f}", pct);

    while (!out.empty() && out.back() == '0') {
        out.pop_back();
    }

    if (!out.empty() && out.back() == '.') {
        out.pop_back();
    }

    return out + "%";
}

static void accumulate_bank_summary(DramBankSummary& dst, const DramBaseResult& result) {
    dst.pass &= (result.failures == 0u);

    dst.checked_bytes += (uint64_t)result.words_checked * sizeof(uint32_t);

    dst.suspected_write_error_bytes += (uint64_t)result.suspected_write_failures * sizeof(uint32_t);

    dst.suspected_read_error_bytes += (uint64_t)result.suspected_read_failures * sizeof(uint32_t);
}

static DramChipSummary make_chip_bank_summary(
    IDevice* device, uint32_t num_dram_channels, const DramMultiInstanceSummary& run) {
    DramChipSummary chip{};

    chip.device_id = device->id();
    chip.pass = run.summary.pass;
    chip.banks.resize(num_dram_channels);

    for (const auto& per_core : run.per_core_results) {
        const uint32_t bank_id = per_core.result.bank_id;

        if (bank_id < chip.banks.size()) {
            accumulate_bank_summary(chip.banks[bank_id], per_core.result);
        }
    }

    return chip;
}

static std::pair<std::string, std::string> test_dram_ubb_tray_and_location_from_bdf(const std::string& bdf) {
    const auto first_colon_pos = bdf.find(':');
    const auto second_colon_pos =
        first_colon_pos == std::string::npos ? std::string::npos : bdf.find(':', first_colon_pos + 1);

    if (first_colon_pos == std::string::npos || second_colon_pos == std::string::npos ||
        second_colon_pos <= first_colon_pos + 1) {
        return {"unknown", "unknown"};
    }

    const std::string d_text = bdf.substr(first_colon_pos + 1, second_colon_pos - first_colon_pos - 1);

    try {
        const uint32_t d = std::stoul(d_text, nullptr, 16);
        const uint32_t upper_nibble = (d >> 4) & 0xF;
        const uint32_t lower_nibble = d & 0xF;

        std::string ubb_tray = "unknown";
        switch (upper_nibble) {
            case 0x0: ubb_tray = "1"; break;
            case 0x4: ubb_tray = "2"; break;
            case 0xC: ubb_tray = "3"; break;
            case 0x8: ubb_tray = "4"; break;
            default: break;
        }

        return {ubb_tray, fmt::format("{}", lower_nibble)};
    } catch (...) {
        return {"unknown", "unknown"};
    }
}

static void log_dram_bank_result_table(const DramGalaxySummary& s) {
    if (s.chips.empty()) {
        return;
    }

    size_t num_dram_channels = 0;
    for (const auto& chip : s.chips) {
        num_dram_channels = std::max(num_dram_channels, chip.banks.size());
    }

    if (num_dram_channels == 0) {
        return;
    }

    const size_t bdf_col_width = std::max(std::string("BDF").size(), std::string("0000:00:00.0").size());
    const size_t ubb_col_width = std::string("UBB tray").size();
    const size_t location_col_width = std::string("Location").size();
    const size_t bank_col_width = 2;

    auto make_separator = [&]() {
        std::string sep = "+";
        sep += std::string(bdf_col_width + 2, '-') + "+";
        sep += std::string(ubb_col_width + 2, '-') + "+";
        sep += std::string(location_col_width + 2, '-') + "+";
        for (size_t bank = 0; bank < num_dram_channels; ++bank) {
            sep += std::string(bank_col_width + 2, '-') + "+";
        }
        return sep;
    };

    auto make_left_cell = [](const std::string& value, size_t width) {
        return " " + value + std::string(width > value.size() ? width - value.size() : 0, ' ') + " ";
    };

    auto make_center_cell = [](const std::string& value, size_t width) {
        const size_t pad = width > value.size() ? width - value.size() : 0;
        const size_t left = pad / 2;
        const size_t right = pad - left;
        return " " + std::string(left, ' ') + value + std::string(right, ' ') + " ";
    };

    log_info(tt::LogTest, "=== DRAM Bank Result Table ===");
    log_info(tt::LogTest, "{}", make_separator());

    std::string header = "|";
    header += make_left_cell("BDF", bdf_col_width) + "|";
    header += make_left_cell("UBB tray", ubb_col_width) + "|";
    header += make_left_cell("Location", location_col_width) + "|";
    for (size_t bank = 0; bank < num_dram_channels; ++bank) {
        header += make_center_cell(fmt::format("D{}", bank), bank_col_width) + "|";
    }
    log_info(tt::LogTest, "{}", header);
    log_info(tt::LogTest, "{}", make_separator());

    for (const auto& chip : s.chips) {
        std::string row = "|";
        const auto bdf = pci_bdf_for_device_id(chip.device_id);
        const auto [ubb_tray, location] = test_dram_ubb_tray_and_location_from_bdf(bdf);
        row += make_left_cell(bdf, bdf_col_width) + "|";
        row += make_left_cell(ubb_tray, ubb_col_width) + "|";
        row += make_left_cell(location, location_col_width) + "|";

        for (size_t bank = 0; bank < num_dram_channels; ++bank) {
            if (bank < chip.banks.size()) {
                row += make_center_cell(chip.banks[bank].pass ? "OK" : "NO", bank_col_width) + "|";
            } else {
                row += make_center_cell("--", bank_col_width) + "|";
            }
        }

        log_info(tt::LogTest, "{}", row);
    }

    log_info(tt::LogTest, "{}", make_separator());
}

static void print_subtest_status(
    uint32_t test_index,
    uint32_t total_tests,
    uint64_t subtest_index,
    uint64_t total_subtests,
    uint32_t mesh_x,
    uint32_t mesh_y,
    uint32_t bank_id,
    uint32_t pattern_id,
    double elapsed_ms,
    const DramRunSummary* summary = nullptr) {
    std::string out = fmt::format(
        "test {}/{} subtest {}/{} mesh({},{}) bank:{} pattern:{} time:{:.2f}ms",
        test_index,
        total_tests,
        subtest_index,
        total_subtests,
        mesh_x,
        mesh_y,
        bank_id,
        pattern_name(pattern_id),
        elapsed_ms);

    if (summary != nullptr && !summary->pass) {
        out = fmt::format(
            "{} {}/{} suspected write errors {}/{} suspected read errors",
            out,
            format_bytes(summary->suspected_write_error_bytes),
            format_bytes(summary->checked_bytes),
            format_bytes(summary->suspected_read_error_bytes),
            format_bytes(summary->checked_bytes));
    }

    log_info(tt::LogTest, "{}", out);
}

static bool get_dram_test_fast_from_env() {
    const char* value = std::getenv("DRAM_TEST_FAST");
    return value != nullptr && std::string(value) == "1";
}

static const std::vector<uint32_t>& get_deployment_patterns_from_env() {
    static const std::vector<uint32_t> all_patterns = {
        DRAM_PATTERN_COUNTER,
        DRAM_PATTERN_CHECKERBOARD,
        DRAM_PATTERN_ADDRESS,
        DRAM_PATTERN_MARCHING_ONES,
        DRAM_PATTERN_MARCHING_ZEROES,
        DRAM_PATTERN_MARCHING_ONE_BITS,
        DRAM_PATTERN_MARCHING_ZERO_BITS,
        DRAM_PATTERN_TOGGLE_BITS,
        DRAM_PATTERN_SATURATION,
        DRAM_PATTERN_REVERSIBLE_RANDOM,
        DRAM_PATTERN_RANDOM,
        DRAM_PATTERN_RANDOM_XOSHIRO128PP,
        DRAM_PATTERN_BYTEWISE_SSN,
    };

    // Fast mode runs a small representative subset for quick deployment checks.
    static const std::vector<uint32_t> fast_patterns = {
        DRAM_PATTERN_CHECKERBOARD,
        DRAM_PATTERN_RANDOM,
        DRAM_PATTERN_COUNTER,
        DRAM_PATTERN_ADDRESS,
    };

    return get_dram_test_fast_from_env() ? fast_patterns : all_patterns;
}

static void log_dram_pattern_mode_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        const bool fast_mode = get_dram_test_fast_from_env();
        const auto& patterns = get_deployment_patterns_from_env();

        std::vector<std::string> names;
        names.reserve(patterns.size());
        for (uint32_t pattern_id : patterns) {
            names.push_back(pattern_name(pattern_id));
        }

        std::string pattern_list;
        for (size_t i = 0; i < names.size(); i++) {
            if (i) {
                pattern_list += ",";
            }
            pattern_list += names[i];
        }

        log_info(
            tt::LogTest,
            "DRAM pattern mode: {} patterns={} env DRAM_TEST_FAST={}",
            fast_mode ? "fast" : "all",
            pattern_list,
            fast_mode ? "1" : "0");
    });
}

class [[maybe_unused]] Watchdog {
public:
    explicit Watchdog(std::chrono::seconds timeout) :
        test_finished(false), thread_([this, timeout]() {
            std::unique_lock<std::mutex> lock(mutex_);
            const bool finished_in_time = cv_.wait_for(lock, timeout, [this]() { return test_finished.load(); });

            if (!finished_in_time) {
                const char msg[] = "\nWatchdog timeout!\n";
                ssize_t rc = ::write(2, msg, sizeof(msg) - 1);
                (void)rc;
                std::raise(SIGINT);
            }
        }) {}

    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            test_finished = true;
        }
        cv_.notify_one();

        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::atomic<bool> test_finished;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
};

static std::vector<CoreCoord> get_worker_cores_for_deployment(IDevice* device) {
    std::vector<CoreCoord> cores;

    const auto grid = device->compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            cores.emplace_back(x, y);
        }
    }

    return cores;
}

[[maybe_unused]]
static uint32_t get_dram_test_bytes_from_env(uint32_t default_bytes) {
    const char* env = std::getenv("DRAM_TEST_MBYTES");

    uint64_t bytes = default_bytes;

    if (env != nullptr) {
        uint64_t value_mb = std::strtoull(env, nullptr, 0);

        TT_FATAL(value_mb > 0, "DRAM_TEST_BYTES must be > 0");

        bytes = value_mb * 1024ull * 1024ull;

        log_info(tt::LogTest, "Using DRAM_TEST_BYTES={} MB ({} bytes)", value_mb, bytes);
    }

    if (bytes > DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES) {
        log_info(
            tt::LogTest,
            "Requested DRAM_TEST_BYTES={} bytes exceeds effective test limit {} bytes; clamping to exclude top {} "
            "bytes",
            bytes,
            DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES,
            DRAM_TEST_RESERVED_TOP_BYTES);
        bytes = DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES;
    }

    return static_cast<uint32_t>(bytes);
}

enum class [[maybe_unused]] DramNocMode { FIXED_0, FIXED_1, ALTERNATE };

[[maybe_unused]]
static DramNocMode get_dram_noc_mode_from_env() {
    const char* env = std::getenv("DRAM_TEST_NOC_MODE");

    if (!env) {
        return DramNocMode::FIXED_0;  // default
    }

    std::string val(env);

    if (val == "0") {
        return DramNocMode::FIXED_0;
    }
    if (val == "1") {
        return DramNocMode::FIXED_1;
    }
    if (val == "alternate") {
        return DramNocMode::ALTERNATE;
    }

    TT_FATAL(false, "Invalid DRAM_TEST_NOC_MODE: {} (expected 0, 1, alternate)", val);
}

[[maybe_unused]]
static uint32_t resolve_noc(DramNocMode mode, uint64_t pattern_index) {
    switch (mode) {
        case DramNocMode::FIXED_0: return 0;

        case DramNocMode::FIXED_1: return 1;

        case DramNocMode::ALTERNATE: return (pattern_index % 2 == 0) ? 0 : 1;
    }

    return 0;
}

[[maybe_unused]]
static bool get_env_flag(const char* name, bool default_val = false) {
    const char* val = std::getenv(name);
    if (!val) {
        return default_val;
    }

    if (std::string(val) == "1" || std::string(val) == "true") {
        return true;
    }
    if (std::string(val) == "0" || std::string(val) == "false") {
        return false;
    }

    TT_FATAL(false, "Invalid value for {} (use 0/1)", name);
    return default_val;
}

[[maybe_unused]]
static std::vector<CoreCoord> get_first_n_worker_cores(IDevice* device, size_t n) {
    const auto all_cores = get_worker_cores_for_deployment(device);
    TT_FATAL(all_cores.size() >= n, "Need at least {} worker cores, found {}", n, all_cores.size());

    return std::vector<CoreCoord>(all_cores.begin(), all_cores.begin() + n);
}

[[maybe_unused]]
static uint32_t get_bank_id_for_core_in_all_controllers_test(size_t core_index, size_t total_cores) {
    constexpr size_t num_controllers = 8u;

    const size_t base_cores_per_controller = total_cores / num_controllers;
    const size_t remainder_cores = total_cores % num_controllers;

    size_t core_begin = 0;

    for (uint32_t bank_id = 0; bank_id < num_controllers; bank_id++) {
        const size_t cores_in_this_controller = base_cores_per_controller + (bank_id < remainder_cores ? 1u : 0u);

        const size_t core_end = core_begin + cores_in_this_controller;

        if (core_index >= core_begin && core_index < core_end) {
            return bank_id;
        }

        core_begin = core_end;
    }

    TT_FATAL(false, "Invalid core_index={} for total_cores={}", core_index, total_cores);
}

[[maybe_unused]]
const bool verbose = get_env_flag("DRAM_TEST_VERBOSE", false);

[[maybe_unused]]
static void print_subtest_status_per_instance(
    uint32_t test_index,
    uint32_t total_tests,
    uint64_t subtest_index,
    uint64_t total_subtests,
    const DramPerCoreResult& per_core,
    double elapsed_ms) {
    /* ======================== */
    DramRunSummary tmp{};
    tmp.pass = (per_core.result.failures == 0u);
    tmp.bank_id = per_core.result.bank_id;
    tmp.checked_bytes = per_core.result.words_checked * sizeof(uint32_t);
    tmp.suspected_write_error_bytes = per_core.result.suspected_write_failures * sizeof(uint32_t);
    tmp.suspected_read_error_bytes = per_core.result.suspected_read_failures * sizeof(uint32_t);

    print_subtest_status(
        test_index,
        total_tests,
        subtest_index,
        total_subtests,
        per_core.core.x,
        per_core.core.y,
        per_core.result.bank_id,
        per_core.result.pattern_id,
        elapsed_ms,
        tmp.pass ? nullptr : &tmp);
}

[[maybe_unused]]
static uint32_t get_dram_chunk_bytes_from_env(uint32_t default_bytes) {
    const char* env = std::getenv("DRAM_TEST_CHUNK_BYTES");

    uint64_t bytes = default_bytes;

    if (env != nullptr) {
        bytes = std::strtoull(env, nullptr, 0);

        TT_FATAL(bytes > 0, "DRAM_TEST_CHUNK_BYTES must be > 0");
        TT_FATAL(bytes % sizeof(uint32_t) == 0, "DRAM_TEST_CHUNK_BYTES must be word aligned");

        log_info(tt::LogTest, "Using DRAM_TEST_CHUNK_BYTES={} bytes", bytes);
    }

    // opciono: enforce neki max ako želiš
    TT_FATAL(bytes <= (1u << 20), "Chunk too large (>1MB)");  // možeš promeniti limit

    return bytes;
}

[[maybe_unused]]
static uint32_t get_dram_max_chips_from_env(uint32_t available_chips) {
    const char* env = std::getenv("DRAM_TEST_MAX_CHIPS");

    if (env == nullptr) {
        return available_chips;
    }

    const uint64_t value = std::strtoull(env, nullptr, 0);

    TT_FATAL(value > 0, "DRAM_TEST_MAX_CHIPS must be > 0");
    TT_FATAL(value <= available_chips, "DRAM_TEST_MAX_CHIPS={} exceeds available chips={}", value, available_chips);

    return static_cast<uint32_t>(value);
}

static void accumulate_galaxy_summary(DramGalaxySummary& dst, const DramMultiInstanceSummary& run, uint64_t jobs) {
    dst.jobs += jobs;
    dst.checked_bytes += run.summary.checked_bytes;
    dst.suspected_write_error_bytes += run.summary.suspected_write_error_bytes;
    dst.suspected_read_error_bytes += run.summary.suspected_read_error_bytes;
    dst.pass &= run.summary.pass;
}

static void log_galaxy_summary(const char* name, const DramGalaxySummary& s, std::chrono::seconds duration) {
    double write_pct = 0.0;
    double read_pct = 0.0;

    if (s.checked_bytes > 0) {
        write_pct = 100.0 * s.suspected_write_error_bytes / s.checked_bytes;
        read_pct = 100.0 * s.suspected_read_error_bytes / s.checked_bytes;
    }

    log_info(tt::LogTest, "=== {} Galaxy/SOC Summary ===", name);

    std::string message = fmt::format(
        "chips={} jobs={} duration={} checked_bytes={} ({:.2f} MB) pass={}",
        s.chips_tested,
        s.jobs,
        format_duration_seconds(duration.count()),
        s.checked_bytes,
        s.checked_bytes / (1024.0 * 1024.0),
        s.pass ? "YES" : "NO");

    if (s.pass) {
        log_info(tt::LogTest, "{}", message);
    } else {
        log_critical(tt::LogTest, "{}", message);
    }

    if (!s.pass) {
        log_info(tt::LogTest, "write_err={} read_err={}", format_error_pct(write_pct), format_error_pct(read_pct));
    }

    log_dram_bank_result_table(s);
}

static void merge_galaxy_summary(DramGalaxySummary& dst, const DramGalaxySummary& src) {
    dst.chips_tested += src.chips_tested;
    dst.jobs += src.jobs;
    dst.checked_bytes += src.checked_bytes;
    dst.suspected_write_error_bytes += src.suspected_write_error_bytes;
    dst.suspected_read_error_bytes += src.suspected_read_error_bytes;
    dst.pass &= src.pass;
    dst.chips.insert(dst.chips.end(), src.chips.begin(), src.chips.end());
}

template <typename RunOneChipFn>
static void run_chips_in_parallel(uint32_t chips_to_test, RunOneChipFn run_one_chip) {
    std::vector<std::thread> threads;
    threads.reserve(chips_to_test);

    for (uint32_t chip_index = 0; chip_index < chips_to_test; chip_index++) {
        threads.emplace_back([&, chip_index]() {
            if (g_stop_requested.load()) {
                return;
            }

            run_one_chip(chip_index);
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

TEST_F(MeshDispatchFixture, DramDeployment_PersistentOptimalWorkersAllDramBanks) {
    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user.";
    }
    g_stop_message_printed.store(false);
    g_watchdog_requested.store(false);

    bool all_pass = true;
    DramGalaxySummary galaxy{};
    const auto galaxy_start = std::chrono::steady_clock::now();

    const uint32_t repeats = get_dram_test_loops_from_env(1u);
    constexpr uint32_t initial_seed = 0x12345678u;
    constexpr uint32_t advance_seed = 1u;

    const auto& kDeploymentPatterns = get_deployment_patterns_from_env();

    const uint32_t chunk_bytes = get_dram_chunk_bytes_from_env(4096u);
    const uint32_t total_bytes_per_controller = get_dram_test_bytes_from_env(DRAM_TEST_BYTES);

    auto noc_mode = get_dram_noc_mode_from_env();

    std::signal(SIGINT, handle_sigint);
    log_blackhole_galaxy_pci_bdfs_once();
    log_dram_pattern_mode_once();
    log_info(tt::LogTest, "DRAM_TEST_LOOPS={} (whole workload repeat count)", repeats);

    log_info(tt::LogTest, "Persistent optimal-worker DRAM deployment running on {} chip(s)", devices_.size());

    const uint32_t chips_to_test = get_dram_max_chips_from_env(devices_.size());
    const bool parallel_chips = true;

    log_info(
        tt::LogTest,
        "Testing {} of {} available chip(s), parallel execution={}",
        chips_to_test,
        devices_.size(),
        parallel_chips ? "YES" : "NO");

    std::mutex summary_mutex;
    DramPatternTimingSummary pattern_timing;

    auto run_one_chip = [&](size_t chip_index) {
        if (g_stop_requested.load()) {
            return;
        }

        DramGalaxySummary chip_summary{};
        DramPatternTimingSummary chip_pattern_timing{};
        bool chip_pass = true;

        const auto& mesh_device = devices_[chip_index];
        auto* const device = mesh_device->get_devices()[0];

        chip_summary.chips_tested = 1;

        log_info(
            tt::LogTest,
            "Starting chip {}/{} bdf={} device_id={}",
            chip_index + 1,
            chips_to_test,
            pci_bdf_for_device_id(device->id()),
            device->id());

        const auto assignments = get_optimal_dram_bank_worker_assignments(mesh_device, tt_metal::NOC::NOC_0);

        log_info(
            tt::LogTest,
            "bdf={} device_id={} persistent optimal-worker test uses {} DRAM channels and {} worker cores",
            pci_bdf_for_device_id(device->id()),
            device->id(),
            device->num_dram_channels(),
            assignments.size());

        if (verbose) {
            for (const auto& a : assignments) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} DRAM bank {} assigned to logical worker core ({}, {})",
                    pci_bdf_for_device_id(device->id()),
                    device->id(),
                    a.bank_id,
                    a.worker_core.x,
                    a.worker_core.y);
            }
        }

        std::vector<DramWorkItem> jobs;
        uint32_t job_id = 1u;
        uint32_t seed = initial_seed;
        uint64_t pattern_toggle_index = 0;

        for (uint32_t repeat_index = 0; repeat_index < repeats; repeat_index++) {
            for (uint32_t pattern_id : kDeploymentPatterns) {
                if (g_stop_requested.load()) {
                    break;
                }

                const uint32_t write_noc = resolve_noc(noc_mode, pattern_toggle_index);

                const uint32_t read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

                const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                for (uint32_t pass_index = 0; pass_index < num_passes; pass_index++) {
                    for (const auto& assignment : assignments) {
                        DramWorkItem job{};

                        job.job_id = job_id++;
                        job.bank_id = assignment.bank_id;
                        job.bank_offset_lo = 0u;
                        job.bank_offset_hi = 0u;
                        job.total_bytes = total_bytes_per_controller;
                        job.chunk_bytes = chunk_bytes;
                        job.pattern_id = pattern_id;
                        job.seed = seed;
                        job.pass_index = pass_index;
                        job.repeat_index = repeat_index;
                        job.write_noc = write_noc;
                        job.read_noc = read_noc;
                        job.max_burst_len = chunk_bytes;
                        job.transfer_len_mode = 0u;
                        job.skip_writes = 0u;
                        job.skip_reads = 0u;

                        jobs.push_back(job);
                    }
                }

                pattern_toggle_index++;
            }

            seed += advance_seed;
        }

        std::vector<CoreCoord> worker_cores;
        worker_cores.reserve(assignments.size());

        for (const auto& assignment : assignments) {
            worker_cores.push_back(assignment.worker_core);
        }

        std::vector<std::vector<DramWorkItem>> jobs_per_core(assignments.size());

        for (const auto& job : jobs) {
            TT_FATAL(
                job.bank_id < assignments.size(),
                "Invalid DRAM bank id {} for assignment size {}",
                job.bank_id,
                assignments.size());

            jobs_per_core[job.bank_id].push_back(job);
        }

        const auto subtest_start = std::chrono::steady_clock::now();

        DramMultiInstanceSummary run = run_dram_persistent_jobs_test_verbose(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device,
            worker_cores,
            jobs_per_core,
            chunk_bytes,
            DataMovementProcessor::RISCV_0);

        const auto subtest_end = std::chrono::steady_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(subtest_end - subtest_start).count();

        accumulate_galaxy_summary(chip_summary, run, jobs.size());
        accumulate_pattern_timing_summary(chip_pattern_timing, run);

        chip_summary.chips.push_back(make_chip_bank_summary(device, assignments.size(), run));

        const auto& s = run.summary;

        if (!s.pass) {
            chip_pass = false;
        }

        {
            std::lock_guard<std::mutex> lock(summary_mutex);

            log_info(
                tt::LogTest,
                "=== Persistent Optimal DRAM Deployment Chip Summary bdf={} device_id={} ===",
                pci_bdf_for_device_id(device->id()),
                device->id());

            if (g_watchdog_requested.load()) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} status=ABORTED reason=stall_watchdog",
                    pci_bdf_for_device_id(device->id()),
                    device->id());
            }

            string message = fmt::format(
                "bdf={} device_id={} dram_channels={} workers={} jobs={} time={:.2f} ms checked_bytes={} pass={}",
                pci_bdf_for_device_id(device->id()),
                device->id(),
                assignments.size(),
                worker_cores.size(),
                jobs.size(),
                elapsed_ms,
                s.checked_bytes,
                s.pass ? "YES" : "NO");

            if (s.pass) {
                log_info(tt::LogTest, "{}", message);
            } else {
                log_critical(tt::LogTest, "{}", message);
            }

            if (s.pass) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} all jobs passed with no errors",
                    pci_bdf_for_device_id(device->id()),
                    device->id());
            }

            merge_galaxy_summary(galaxy, chip_summary);
            merge_pattern_timing_summary(pattern_timing, chip_pattern_timing);
            all_pass &= chip_pass;
        }

        if (g_watchdog_requested.load()) {
            chip_pass = false;
            g_stop_requested.store(true);
        }
    };

    if (parallel_chips) {
        run_chips_in_parallel(chips_to_test, run_one_chip);
    } else {
        for (size_t chip_index = 0; chip_index < chips_to_test; chip_index++) {
            if (g_stop_requested.load()) {
                break;
            }
            run_one_chip(chip_index);
        }
    }

    const auto galaxy_end = std::chrono::steady_clock::now();
    log_pattern_timing_summary(
        "Persistent Optimal DRAM Deployment",
        pattern_timing,
        std::chrono::duration<double, std::milli>(galaxy_end - galaxy_start).count());

    log_info(tt::LogTest, "=== Persistent Optimal DRAM Deployment Loop Summary ===");
    log_info(
        tt::LogTest,
        "loops={} total_jobs={} checked_bytes={} pass={}",
        repeats,
        galaxy.jobs,
        galaxy.checked_bytes,
        galaxy.pass ? "YES" : "NO");

    log_galaxy_summary(
        "Persistent Optimal DRAM Deployment",
        galaxy,
        std::chrono::duration_cast<std::chrono::seconds>(galaxy_end - galaxy_start));

    all_pass &= galaxy.pass;

    if (g_watchdog_requested.load()) {
        FAIL() << "Test aborted by stall watchdog.";
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user.";
    }

    ASSERT_TRUE(all_pass);
}

TEST_F(MeshDispatchFixture, DramDeployment_PersistentAllWorkersSingleDramSequentialSweep) {
    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user.";
    }

    g_stop_message_printed.store(false);
    g_watchdog_requested.store(false);

    bool all_pass = true;
    DramGalaxySummary galaxy{};
    const auto galaxy_start = std::chrono::steady_clock::now();

    constexpr uint64_t controller_bank_offset = 0u;
    const uint32_t repeats = get_dram_test_loops_from_env(1u);
    constexpr uint32_t initial_seed = 0x12345678u;
    constexpr uint32_t advance_seed = 1u;

    const auto& kDeploymentPatterns = get_deployment_patterns_from_env();

    const uint32_t chunk_bytes = get_dram_chunk_bytes_from_env(4096u);
    const uint32_t total_bytes_per_controller = get_dram_test_bytes_from_env(DRAM_TEST_BYTES);

    auto noc_mode = get_dram_noc_mode_from_env();

    std::signal(SIGINT, handle_sigint);
    log_blackhole_galaxy_pci_bdfs_once();
    log_dram_pattern_mode_once();
    log_info(tt::LogTest, "DRAM_TEST_LOOPS={} (whole workload repeat count)", repeats);

    log_info(tt::LogTest, "Persistent all-workers single-DRAM sequential sweep running on {} chip(s)", devices_.size());

    const uint32_t chips_to_test = get_dram_max_chips_from_env(devices_.size());

    const bool parallel_chips = true;

    log_info(
        tt::LogTest,
        "Testing {} of {} available chip(s), parallel execution={}",
        chips_to_test,
        devices_.size(),
        parallel_chips ? "YES" : "NO");

    std::mutex summary_mutex;
    DramPatternTimingSummary pattern_timing;

    auto run_one_chip = [&](size_t chip_index) {
        if (g_stop_requested.load()) {
            return;
        }

        DramGalaxySummary chip_summary{};
        DramPatternTimingSummary chip_pattern_timing{};
        bool chip_pass = true;

        const auto& mesh_device = devices_[chip_index];
        auto* const device = mesh_device->get_devices()[0];

        chip_summary.chips_tested = 1;

        log_info(
            tt::LogTest,
            "Starting chip {}/{} bdf={} device_id={}",
            chip_index + 1,
            chips_to_test,
            pci_bdf_for_device_id(device->id()),
            device->id());

        const uint32_t num_dram_channels = device->num_dram_channels();
        const auto worker_cores = get_worker_cores_for_deployment(device);
        DramChipSummary chip_bank_summary{};
        chip_bank_summary.device_id = device->id();
        chip_bank_summary.pass = true;
        chip_bank_summary.banks.resize(num_dram_channels);

        TT_FATAL(!worker_cores.empty(), "No worker cores found");
        TT_FATAL(num_dram_channels > 0, "No DRAM channels found");

        log_info(
            tt::LogTest,
            "bdf={} device_id={} persistent all-workers single-DRAM sequential sweep: workers={} dram_channels={} "
            "bytes_per_dram={} chunk_bytes={}",
            pci_bdf_for_device_id(device->id()),
            device->id(),
            worker_cores.size(),
            num_dram_channels,
            total_bytes_per_controller,
            chunk_bytes);

        const uint64_t bytes_per_core_base = (total_bytes_per_controller / worker_cores.size()) & ~0xFFFULL;

        TT_FATAL(
            bytes_per_core_base >= chunk_bytes,
            "bytes_per_core_base too small: {} < chunk_bytes {}",
            bytes_per_core_base,
            chunk_bytes);

        TT_FATAL(bytes_per_core_base <= std::numeric_limits<uint32_t>::max(), "bytes_per_core_base must fit uint32_t");

        const uint64_t covered_bytes = bytes_per_core_base * worker_cores.size();
        const uint64_t remainder_bytes = total_bytes_per_controller - covered_bytes;

        const auto full_start = std::chrono::steady_clock::now();

        for (uint32_t bank_id = 0; bank_id < num_dram_channels; bank_id++) {
            if (g_stop_requested.load()) {
                break;
            }

            std::vector<std::vector<DramWorkItem>> jobs_per_core(worker_cores.size());

            uint32_t job_id = 1u;
            uint32_t seed = initial_seed;
            uint64_t pattern_toggle_index = 0;

            for (uint32_t repeat_index = 0; repeat_index < repeats; repeat_index++) {
                for (uint32_t pattern_id : kDeploymentPatterns) {
                    if (g_stop_requested.load()) {
                        break;
                    }

                    const uint32_t write_noc = resolve_noc(noc_mode, pattern_toggle_index);
                    const uint32_t read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

                    const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                    for (uint32_t pass_index = 0; pass_index < num_passes; pass_index++) {
                        for (size_t worker_idx = 0; worker_idx < worker_cores.size(); worker_idx++) {
                            const uint64_t bank_offset = controller_bank_offset + worker_idx * bytes_per_core_base;

                            uint64_t bytes_this_core = bytes_per_core_base;

                            if (worker_idx == worker_cores.size() - 1) {
                                bytes_this_core += remainder_bytes;
                            }

                            TT_FATAL(
                                bytes_this_core >= chunk_bytes,
                                "bytes_this_core too small: {} < chunk_bytes {}",
                                bytes_this_core,
                                chunk_bytes);

                            TT_FATAL(
                                bytes_this_core <= std::numeric_limits<uint32_t>::max(),
                                "bytes_this_core must fit uint32_t");

                            DramWorkItem job{};

                            job.job_id = job_id++;
                            job.bank_id = bank_id;
                            job.bank_offset_lo = bank_offset & 0xFFFFFFFFull;
                            job.bank_offset_hi = (bank_offset >> 32) & 0xFFFFFFFFull;
                            job.total_bytes = bytes_this_core;
                            job.chunk_bytes = chunk_bytes;
                            job.pattern_id = pattern_id;
                            job.seed = seed;
                            job.pass_index = pass_index;
                            job.repeat_index = repeat_index;
                            job.write_noc = write_noc;
                            job.read_noc = read_noc;
                            job.max_burst_len = chunk_bytes;
                            job.transfer_len_mode = 0u;
                            job.skip_writes = 0u;
                            job.skip_reads = 0u;

                            jobs_per_core[worker_idx].push_back(job);
                        }
                    }

                    pattern_toggle_index++;
                }

                seed += advance_seed;
            }

            const uint64_t bank_jobs = job_id - 1u;

            const auto bank_start = std::chrono::steady_clock::now();

            DramMultiInstanceSummary run = run_dram_persistent_jobs_test_verbose(
                static_cast<MeshDispatchFixture*>(this),
                mesh_device,
                worker_cores,
                jobs_per_core,
                chunk_bytes,
                DataMovementProcessor::RISCV_0);

            const auto bank_end = std::chrono::steady_clock::now();

            const auto bank_duration_sec =
                std::chrono::duration_cast<std::chrono::seconds>(bank_end - bank_start).count();

            accumulate_galaxy_summary(chip_summary, run, bank_jobs);
            accumulate_pattern_timing_summary(chip_pattern_timing, run);

            chip_bank_summary.pass &= run.summary.pass;

            for (const auto& per_core : run.per_core_results) {
                accumulate_bank_summary(chip_bank_summary.banks[bank_id], per_core.result);
            }

            const auto& s = run.summary;

            {
                std::lock_guard<std::mutex> lock(summary_mutex);

                std::string message = fmt::format(
                    "bdf={} device_id={} completed DRAM bank {}/{} workers={} jobs={} duration={} checked_bytes={} "
                    "pass={}",
                    pci_bdf_for_device_id(device->id()),
                    device->id(),
                    bank_id + 1,
                    num_dram_channels,
                    worker_cores.size(),
                    bank_jobs,
                    format_duration_seconds(bank_duration_sec),
                    s.checked_bytes,
                    s.pass ? "YES" : "NO");

                if (s.pass) {
                    log_info(tt::LogTest, "{}", message);
                } else {
                    log_critical(tt::LogTest, "{}", message);
                }
            }

            if (!s.pass) {
                chip_pass = false;
            }

            if (g_watchdog_requested.load()) {
                chip_pass = false;
                g_stop_requested.store(true);
                break;
            }
        }

        const auto full_end = std::chrono::steady_clock::now();

        const auto full_duration_sec = std::chrono::duration_cast<std::chrono::seconds>(full_end - full_start).count();

        if (g_watchdog_requested.load()) {
            chip_pass = false;
            g_stop_requested.store(true);
        }

        {
            std::lock_guard<std::mutex> lock(summary_mutex);

            log_info(
                tt::LogTest,
                "=== Persistent All-Workers Single-DRAM Sequential Sweep Chip Summary bdf={} device_id={} ===",
                pci_bdf_for_device_id(device->id()),
                device->id());

            if (g_watchdog_requested.load()) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} status=ABORTED reason=stall_watchdog",
                    pci_bdf_for_device_id(device->id()),
                    device->id());
            }

            std::string message = fmt::format(
                "bdf={} device_id={} workers={} dram_channels={} duration={} pass={}",
                pci_bdf_for_device_id(device->id()),
                device->id(),
                worker_cores.size(),
                num_dram_channels,
                format_duration_seconds(full_duration_sec),
                chip_pass ? "YES" : "NO");

            if (chip_pass) {
                log_info(tt::LogTest, "{}", message);
            } else {
                log_critical(tt::LogTest, "{}", message);
            }

            if (chip_pass) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} all banks passed with no errors",
                    pci_bdf_for_device_id(device->id()),
                    device->id());
            }

            chip_summary.chips.push_back(chip_bank_summary);
            merge_galaxy_summary(galaxy, chip_summary);
            merge_pattern_timing_summary(pattern_timing, chip_pattern_timing);
            all_pass &= chip_pass;
        }
    };

    if (parallel_chips) {
        run_chips_in_parallel(chips_to_test, run_one_chip);
    } else {
        for (size_t chip_index = 0; chip_index < chips_to_test; chip_index++) {
            if (g_stop_requested.load()) {
                break;
            }

            run_one_chip(chip_index);
        }
    }

    const auto galaxy_end = std::chrono::steady_clock::now();

    log_pattern_timing_summary(
        "Persistent All-Workers Single-DRAM Sequential Sweep",
        pattern_timing,
        std::chrono::duration<double, std::milli>(galaxy_end - galaxy_start).count());

    log_info(tt::LogTest, "=== Persistent All-Workers Single-DRAM Sequential Sweep Loop Summary ===");
    log_info(
        tt::LogTest,
        "loops={} total_jobs={} checked_bytes={} pass={}",
        repeats,
        galaxy.jobs,
        galaxy.checked_bytes,
        galaxy.pass ? "YES" : "NO");

    log_galaxy_summary(
        "Persistent All-Workers Single-DRAM Sequential Sweep",
        galaxy,
        std::chrono::duration_cast<std::chrono::seconds>(galaxy_end - galaxy_start));

    all_pass &= galaxy.pass;

    if (g_watchdog_requested.load()) {
        FAIL() << "Test aborted by stall watchdog.";
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user.";
    }

    ASSERT_TRUE(all_pass);
}

TEST_F(MeshDispatchFixture, DramDeployment_PersistentPartitionedWorkersAllDramBanks) {
    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user.";
    }

    g_stop_message_printed.store(false);
    g_watchdog_requested.store(false);

    bool all_pass = true;
    DramGalaxySummary galaxy{};
    const auto galaxy_start = std::chrono::steady_clock::now();

    constexpr uint64_t controller_bank_offset = 0u;
    const uint32_t repeats = get_dram_test_loops_from_env(1u);
    constexpr uint32_t initial_seed = 0x12345678u;
    constexpr uint32_t advance_seed = 1u;

    const auto& kDeploymentPatterns = get_deployment_patterns_from_env();

    const uint32_t chunk_bytes = get_dram_chunk_bytes_from_env(4096u);
    const uint32_t total_bytes_per_controller = get_dram_test_bytes_from_env(DRAM_TEST_BYTES);

    auto noc_mode = get_dram_noc_mode_from_env();

    std::signal(SIGINT, handle_sigint);
    log_blackhole_galaxy_pci_bdfs_once();
    log_dram_pattern_mode_once();
    log_info(tt::LogTest, "DRAM_TEST_LOOPS={} (whole workload repeat count)", repeats);

    log_info(tt::LogTest, "Persistent partitioned-workers all-DRAM test running on {} chip(s)", devices_.size());

    const uint32_t chips_to_test = get_dram_max_chips_from_env(devices_.size());

    const bool parallel_chips = true;

    log_info(
        tt::LogTest,
        "Testing {} of {} available chip(s), parallel execution={}",
        chips_to_test,
        devices_.size(),
        parallel_chips ? "YES" : "NO");

    std::mutex summary_mutex;
    DramPatternTimingSummary pattern_timing;

    auto run_one_chip = [&](size_t chip_index) {
        if (g_stop_requested.load()) {
            return;
        }

        DramGalaxySummary chip_summary{};
        DramPatternTimingSummary chip_pattern_timing{};
        bool chip_pass = true;

        const auto& mesh_device = devices_[chip_index];
        auto* const device = mesh_device->get_devices()[0];

        chip_summary.chips_tested = 1;

        log_info(
            tt::LogTest,
            "Starting chip {}/{} bdf={} device_id={}",
            chip_index + 1,
            chips_to_test,
            pci_bdf_for_device_id(device->id()),
            device->id());

        const uint32_t num_dram_channels = device->num_dram_channels();
        const auto worker_cores = get_worker_cores_for_deployment(device);

        TT_FATAL(!worker_cores.empty(), "No worker cores found");
        TT_FATAL(num_dram_channels > 0, "No DRAM channels found");
        TT_FATAL(
            worker_cores.size() >= num_dram_channels,
            "Need at least one worker per DRAM channel: workers={} dram_channels={}",
            worker_cores.size(),
            num_dram_channels);

        log_info(
            tt::LogTest,
            "bdf={} device_id={} persistent partitioned-workers all-DRAM test: workers={} dram_channels={} "
            "bytes_per_dram={} chunk_bytes={}",
            pci_bdf_for_device_id(device->id()),
            device->id(),
            worker_cores.size(),
            num_dram_channels,
            total_bytes_per_controller,
            chunk_bytes);

        std::vector<std::vector<size_t>> workers_for_bank(num_dram_channels);

        for (size_t worker_idx = 0; worker_idx < worker_cores.size(); worker_idx++) {
            const uint32_t bank_id = worker_idx % num_dram_channels;
            workers_for_bank[bank_id].push_back(worker_idx);
        }

        if (verbose) {
            for (uint32_t bank_id = 0; bank_id < num_dram_channels; bank_id++) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} DRAM bank {} assigned {} worker cores",
                    pci_bdf_for_device_id(device->id()),
                    device->id(),
                    bank_id,
                    workers_for_bank[bank_id].size());
            }
        }

        std::vector<std::vector<DramWorkItem>> jobs_per_core(worker_cores.size());

        uint32_t job_id = 1u;
        uint32_t seed = initial_seed;
        uint64_t pattern_toggle_index = 0;

        for (uint32_t repeat_index = 0; repeat_index < repeats; repeat_index++) {
            for (uint32_t pattern_id : kDeploymentPatterns) {
                if (g_stop_requested.load()) {
                    break;
                }

                const uint32_t write_noc = resolve_noc(noc_mode, pattern_toggle_index);
                const uint32_t read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

                const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                for (uint32_t pass_index = 0; pass_index < num_passes; pass_index++) {
                    for (uint32_t bank_id = 0; bank_id < num_dram_channels; bank_id++) {
                        const auto& bank_workers = workers_for_bank[bank_id];

                        TT_FATAL(!bank_workers.empty(), "No workers assigned to DRAM bank {}", bank_id);

                        const uint64_t bytes_per_core_base =
                            (total_bytes_per_controller / bank_workers.size()) & ~0xFFFULL;

                        TT_FATAL(
                            bytes_per_core_base >= chunk_bytes,
                            "bytes_per_core_base too small: {} < chunk_bytes {}",
                            bytes_per_core_base,
                            chunk_bytes);

                        TT_FATAL(
                            bytes_per_core_base <= std::numeric_limits<uint32_t>::max(),
                            "bytes_per_core_base must fit uint32_t");

                        const uint64_t covered_bytes = bytes_per_core_base * bank_workers.size();
                        const uint64_t remainder_bytes = total_bytes_per_controller - covered_bytes;

                        for (size_t local_idx = 0; local_idx < bank_workers.size(); local_idx++) {
                            const size_t worker_idx = bank_workers[local_idx];

                            const uint64_t bank_offset = controller_bank_offset + local_idx * bytes_per_core_base;

                            uint64_t bytes_this_core = bytes_per_core_base;

                            if (local_idx == bank_workers.size() - 1) {
                                bytes_this_core += remainder_bytes;
                            }

                            TT_FATAL(
                                bytes_this_core >= chunk_bytes,
                                "bytes_this_core too small: {} < chunk_bytes {}",
                                bytes_this_core,
                                chunk_bytes);

                            TT_FATAL(
                                bytes_this_core <= std::numeric_limits<uint32_t>::max(),
                                "bytes_this_core must fit uint32_t");

                            DramWorkItem job{};

                            job.job_id = job_id++;
                            job.bank_id = bank_id;
                            job.bank_offset_lo = bank_offset & 0xFFFFFFFFull;
                            job.bank_offset_hi = (bank_offset >> 32) & 0xFFFFFFFFull;
                            job.total_bytes = bytes_this_core;
                            job.chunk_bytes = chunk_bytes;
                            job.pattern_id = pattern_id;
                            job.seed = seed;
                            job.pass_index = pass_index;
                            job.repeat_index = repeat_index;
                            job.write_noc = write_noc;
                            job.read_noc = read_noc;
                            job.max_burst_len = chunk_bytes;
                            job.transfer_len_mode = 0u;
                            job.skip_writes = 0u;
                            job.skip_reads = 0u;

                            jobs_per_core[worker_idx].push_back(job);
                        }
                    }
                }

                pattern_toggle_index++;
            }

            seed += advance_seed;
        }

        const uint64_t total_jobs_for_chip = job_id - 1u;

        const auto start = std::chrono::steady_clock::now();

        DramMultiInstanceSummary run = run_dram_persistent_jobs_test_verbose(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device,
            worker_cores,
            jobs_per_core,
            chunk_bytes,
            DataMovementProcessor::RISCV_0);

        const auto end = std::chrono::steady_clock::now();

        const auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        accumulate_galaxy_summary(chip_summary, run, total_jobs_for_chip);
        accumulate_pattern_timing_summary(chip_pattern_timing, run);

        chip_summary.chips.push_back(make_chip_bank_summary(device, num_dram_channels, run));

        const auto& s = run.summary;

        if (!s.pass) {
            chip_pass = false;
        }

        if (g_watchdog_requested.load()) {
            chip_pass = false;
            g_stop_requested.store(true);
        }

        {
            std::lock_guard<std::mutex> lock(summary_mutex);

            log_info(
                tt::LogTest,
                "=== Persistent Partitioned Workers All-DRAM Chip Summary bdf={} device_id={} ===",
                pci_bdf_for_device_id(device->id()),
                device->id());

            if (g_watchdog_requested.load()) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} status=ABORTED reason=stall_watchdog",
                    pci_bdf_for_device_id(device->id()),
                    device->id());
            }

            std::string message = fmt::format(
                "bdf={} device_id={} workers={} dram_channels={} jobs={} duration={} checked_bytes={} pass={}",
                pci_bdf_for_device_id(device->id()),
                device->id(),
                worker_cores.size(),
                num_dram_channels,
                total_jobs_for_chip,
                format_duration_seconds(duration_sec),
                s.checked_bytes,
                s.pass ? "YES" : "NO");

            if (s.pass) {
                log_info(tt::LogTest, "{}", message);
            } else {
                log_critical(tt::LogTest, "{}", message);
            }

            if (s.pass) {
                log_info(
                    tt::LogTest,
                    "bdf={} device_id={} all jobs passed with no errors",
                    pci_bdf_for_device_id(device->id()),
                    device->id());
            }

            merge_galaxy_summary(galaxy, chip_summary);
            merge_pattern_timing_summary(pattern_timing, chip_pattern_timing);
            all_pass &= chip_pass;
        }
    };

    if (parallel_chips) {
        run_chips_in_parallel(chips_to_test, run_one_chip);
    } else {
        for (size_t chip_index = 0; chip_index < chips_to_test; chip_index++) {
            if (g_stop_requested.load()) {
                break;
            }

            run_one_chip(chip_index);
        }
    }

    const auto galaxy_end = std::chrono::steady_clock::now();

    log_pattern_timing_summary(
        "Persistent Partitioned Workers All-DRAM",
        pattern_timing,
        std::chrono::duration<double, std::milli>(galaxy_end - galaxy_start).count());

    log_info(tt::LogTest, "=== Persistent Partitioned Workers All-DRAM Loop Summary ===");
    log_info(
        tt::LogTest,
        "loops={} total_jobs={} checked_bytes={} pass={}",
        repeats,
        galaxy.jobs,
        galaxy.checked_bytes,
        galaxy.pass ? "YES" : "NO");

    log_galaxy_summary(
        "Persistent Partitioned Workers All-DRAM",
        galaxy,
        std::chrono::duration_cast<std::chrono::seconds>(galaxy_end - galaxy_start));

    all_pass &= galaxy.pass;

    if (g_watchdog_requested.load()) {
        FAIL() << "Test aborted by stall watchdog.";
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user.";
    }

    ASSERT_TRUE(all_pass);
}

}  // namespace tt::tt_metal
