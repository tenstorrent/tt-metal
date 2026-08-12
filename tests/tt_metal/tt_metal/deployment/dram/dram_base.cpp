// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include "dram_base.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <limits>

#include "command_queue_fixture.hpp"
#include "kernels/common_dram.hpp"
#include "kernels/patterns/sync_mailbox.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <fstream>
#include <cctype>
#include <dirent.h>
#include <cstdlib>
#include <string>
#include <umd/device/types/telemetry.hpp>

static bool get_dram_inject_tensix_heartbeat_stall_from_env_once() {
    const char* env = std::getenv("DRAM_TEST_INJECT_TENSIX_HEARTBEAT_STALL");
    return (env != nullptr) && (std::string(env) == "1");
}

static uint32_t get_dram_insert_errors_pattern_id_from_env_once() {
    const char* env = std::getenv("DRAM_INSERT_ERRORS_PATTERN_ID");
    if (env == nullptr || env[0] == '\0') {
        return 0u;
    }

    char* end = nullptr;
    const unsigned long value = std::strtoul(env, &end, 0);
    if (end == env || value == 0ul) {
        return 0u;
    }

    return value;
}

namespace tt::tt_metal {

extern std::atomic<bool> g_stop_requested;
extern std::atomic<bool> g_watchdog_requested;

using namespace std;
using namespace tt;

static std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(s.front())) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(s.back())) {
        s.pop_back();
    }
    return s;
}

static std::string read_text_file_trimmed(const std::string& path) {
    std::ifstream file(path);
    std::string value;
    std::getline(file, value);
    return trim_copy(value);
}

static const std::vector<std::string>& get_tenstorrent_pci_bdfs_cached() {
    static const std::vector<std::string> bdfs = []() {
        std::vector<std::string> out;

        DIR* dir = opendir("/sys/bus/pci/devices");
        if (dir == nullptr) {
            return out;
        }

        while (auto* entry = readdir(dir)) {
            const std::string bdf = entry->d_name;
            if (bdf.empty() || bdf[0] == '.') {
                continue;
            }

            const std::string base = "/sys/bus/pci/devices/" + bdf;
            std::string vendor = read_text_file_trimmed(base + "/vendor");
            std::transform(vendor.begin(), vendor.end(), vendor.begin(), [](char c) { return std::tolower(c); });

            // Tenstorrent PCI vendor id. Sorted order gives a stable best-effort map:
            // runtime bdf={} device_id=0 -> first Tenstorrent BDF, bdf={} device_id=1 -> second, etc.
            if (vendor == "0x1e52") {
                out.push_back(bdf);
            }
        }

        closedir(dir);
        std::sort(out.begin(), out.end());
        return out;
    }();

    return bdfs;
}

static std::string pci_bdf_for_device_id(uint32_t device_id) {
    const auto& bdfs = get_tenstorrent_pci_bdfs_cached();
    if (device_id < bdfs.size()) {
        return bdfs[device_id];
    }
    return "unknown";
}

struct DramPhysicalLocation {
    std::string bdf;
    std::string ubb_tray;
    std::string location;
};

static std::pair<std::string, std::string> dram_ubb_tray_and_location_from_bdf(const std::string& bdf) {
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

static DramPhysicalLocation dram_physical_location_for_device_id(uint32_t device_id) {
    DramPhysicalLocation loc;
    loc.bdf = pci_bdf_for_device_id(device_id);
    auto decoded = dram_ubb_tray_and_location_from_bdf(loc.bdf);
    loc.ubb_tray = decoded.first;
    loc.location = decoded.second;
    return loc;
}

static std::string device_log_prefix(IDevice* device) {
    const uint32_t device_id = device->id();
    return fmt::format("[bdf={}][device_id={}]", pci_bdf_for_device_id(device_id), device_id);
}

static void ArmWatchdogHardExit(const std::string& reason) {
    std::thread([reason]() {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        log_error(
            tt::LogTest,
            "\n"
            "================ DRAM WATCHDOG HARD EXIT ================\n"
            "Reason: {}\n"
            "Metal/device cleanup did not complete.\n"
            "\n"
            "If the next run cannot acquire the device or topology mapping fails, execute:\n"
            "    source ~/.tenstorrent-venv/bin/activate\n"
            "    tt-smi -r\n"
            "=========================================================",
            reason);
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(2);
    }).detach();
}

static bool dram_test_verbose_enabled() {
    const char* env = std::getenv("DRAM_TEST_VERBOSE");
    return env != nullptr && std::string(env) == "1";
}

static void log_verbose_dram_work_item(
    IDevice* device, const CoreCoord& core, const DramWorkItem& job, uint32_t core_job_index, uint32_t core_job_count) {
    if (!dram_test_verbose_enabled()) {
        return;
    }

    log_info(
        tt::LogTest,
        "{} subtest={} core_job={}/{} core=({}, {}) bank={} pattern={} pass={} repeat={}",
        device_log_prefix(device),
        job.job_id,
        core_job_index + 1u,
        core_job_count,
        core.x,
        core.y,
        job.bank_id,
        pattern_name(job.pattern_id),
        job.pass_index,
        job.repeat_index);
}

std::vector<DramBankWorkerAssignment> get_optimal_dram_bank_worker_assignments(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, tt_metal::NOC noc) {
    auto* const device = mesh_device->get_devices()[0];

    const uint32_t num_dram_channels = device->num_dram_channels();

    std::vector<CoreCoord> optimal_workers = mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(noc);

    TT_FATAL(
        optimal_workers.size() == num_dram_channels,
        "Optimal DRAM worker assignment size mismatch: got {}, expected {}",
        optimal_workers.size(),
        num_dram_channels);

    std::vector<DramBankWorkerAssignment> assignments;
    assignments.reserve(num_dram_channels);

    for (uint32_t bank_id = 0; bank_id < num_dram_channels; bank_id++) {
        assignments.push_back(DramBankWorkerAssignment{
            .bank_id = bank_id,
            .worker_core = optimal_workers[bank_id],
        });
    }

    return assignments;
}

static inline const char* dram_failure_kind_name(uint32_t failure_kind) {
    switch (failure_kind) {
        case DRAM_FAILURE_WRITE: return "write";
        case DRAM_FAILURE_READ: return "read";
        case DRAM_FAILURE_NONE: return "none";
        default: return "unknown";
    }
}

static void log_dram_failure(IDevice* device, const CoreCoord& core, const DramBaseResult* result) {
    log_info(
        tt::LogTest,
        "{} Mismatch on dram_controller={} core {} pattern={} repeat={} pass={}: failures={}, "
        "first_fail_classified_as={}, write_failures={}, read_failures={}",
        device_log_prefix(device),
        result->bank_id,
        core,
        pattern_name(result->pattern_id),
        result->repeat_index,
        result->pass_index,
        result->failures,
        dram_failure_kind_name(result->failure_kind),
        result->suspected_write_failures,
        result->suspected_read_failures);
}

static inline void accumulate_result_into_summary(DramRunSummary& summary, const DramBaseResult* result) {
    summary.pass &= (result->failures == 0u);
    summary.bank_id = result->bank_id;
    summary.checked_bytes += result->words_checked * sizeof(uint32_t);

    summary.suspected_write_error_bytes += result->suspected_write_failures * sizeof(uint32_t);

    summary.suspected_read_error_bytes += result->suspected_read_failures * sizeof(uint32_t);

    summary.prepare_ticks += result->prepare_ticks;
    summary.write_ticks += result->write_ticks;
    summary.read_ticks += result->read_ticks;
    summary.generate_ticks += result->generate_ticks;
    summary.ncrisc_blocked_wait_ticks += result->ncrisc_blocked_wait_ticks;
    summary.compare_brisc_ticks += result->compare_brisc_ticks;
    summary.compare_wait_ticks += result->compare_wait_ticks;
    summary.compare_total_ticks += result->compare_total_ticks;
    summary.ncrisc_idle_ticks += result->ncrisc_idle_ticks;
    summary.ncrisc_write_active_ticks += result->ncrisc_write_active_ticks;
    summary.ncrisc_read_active_ticks += result->ncrisc_read_active_ticks;
    summary.ncrisc_diag_active_ticks += result->ncrisc_diag_active_ticks;
    summary.math_generate_active_ticks += result->math_generate_active_ticks;
    summary.pack_generate_active_ticks += result->pack_generate_active_ticks;
    summary.math_compare_active_ticks += result->math_compare_active_ticks;
    summary.pack_compare_active_ticks += result->pack_compare_active_ticks;
    summary.unpack_compare_active_ticks += result->unpack_compare_active_ticks;
    summary.job_total_ticks += result->job_total_ticks;
}

static inline double dram_result_write_error_pct(const DramBaseResult* result) {
    if (result->words_checked == 0u) {
        return 0.0;
    }
    return 100.0 * result->suspected_write_failures / result->words_checked;
}

static inline double dram_result_read_error_pct(const DramBaseResult* result) {
    if (result->words_checked == 0u) {
        return 0.0;
    }
    return 100.0 * result->suspected_read_failures / result->words_checked;
}

static inline uint64_t read_arc_global_tick(tt::tt_metal::IDevice* device) {
    return MetalContext::instance().get_cluster().get_arc_timer_heartbeat(device->id());
}

static inline const char* dram_watchdog_reason_name(uint32_t reason) {
    switch (reason) {
        case 1: return "kernel_stall";
        case 2: return "system_stall";
        default: return "none";
    }
}

DramRunSummary run_dram_base_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ======================== */
    auto* const device = mesh_device->get_devices()[0];

    TT_FATAL(cfg.bank_id < 8, "bank_id must not exceed the total number of controllers");
    TT_FATAL(cfg.total_bytes <= DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES, "total_bytes must be under (4GB-16MB-2KB)");
    TT_FATAL(cfg.chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(cfg.total_bytes % sizeof(uint32_t) == 0, "total_bytes must be word aligned");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    MetalContext::instance().get_cluster().write_core(
        device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    auto kernel = tt_metal::CreateKernel(
        program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

    DramTestParameters params{
        .bank_id = cfg.bank_id,
        .bank_offset_lo = cfg.bank_offset,
        .bank_offset_hi = (cfg.bank_offset >> 32),
        .total_bytes = cfg.total_bytes,
        .chunk_bytes = cfg.chunk_bytes,
        .pattern_id = cfg.pattern_id,
        .seed = seed,
        .pass_index = pass_index,
        .repeat_index = repeat_index,
        .result_l1_addr = result_l1_address,
        .expect_l1_addr = expect_l1_address,
        .observe_l1_addr = observe_l1_address,
        .write_noc = cfg.write_noc,
        .read_noc = cfg.read_noc,
        .max_burst_len = cfg.max_burst_len,
        .transfer_len_mode = cfg.transfer_len_mode,
        .skip_writes = cfg.skip_writes,
        .skip_reads = cfg.skip_reads,
    };

    //    uint32_t insert_write_errors = get_env_flag("DRAM_TEST_INSERT_WRITE_ERRORS") ? 1u : 0u;
    //    uint32_t insert_read_errors  = get_env_flag("DRAM_TEST_INSERT_READ_ERRORS")  ? 1u : 0u;

    tt_metal::SetRuntimeArgs(
        program,
        kernel,
        core,
        {
            params.bank_id,
            params.bank_offset_lo,
            params.bank_offset_hi,
            params.total_bytes,
            params.chunk_bytes,
            params.pattern_id,
            params.seed,
            params.pass_index,
            params.repeat_index,
            params.result_l1_addr,
            params.expect_l1_addr,
            params.observe_l1_addr,
            params.write_noc,
            params.read_noc,
            params.max_burst_len,
            params.transfer_len_mode,
            params.skip_writes,
            params.skip_reads,
            //            params.insert_write_errors,
            //            params.insert_read_errors,
        });

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    auto raw_result = MetalContext::instance().get_cluster().read_core(
        device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

    const DramBaseResult* result = (const DramBaseResult*)raw_result.data();

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = result->bank_id;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    accumulate_result_into_summary(summary, result);

    if (result->failures > 0u) {
        log_dram_failure(device, core, result);
    }

    return summary;
}

DramRunSummary run_dram_multi_core_single_controller_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ======================== */
    auto* const device = mesh_device->get_devices()[0];

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(cfg.bank_id < 8, "bank_id must not exceed the total number of controllers");
    TT_FATAL(cfg.total_bytes <= DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES, "total_bytes must be under (4GB-16MB-2KB)");
    TT_FATAL(cfg.chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(
        cfg.total_bytes % DRAM_TEST_NOC_WORD_BYTES == 0,
        "total_bytes must be NOC word aligned for multi-core controller mode");

    const uint64_t total_bytes = cfg.total_bytes;
    const uint64_t bytes_per_core_base = (total_bytes / cores.size()) & ~0xFFFULL;

    TT_FATAL(bytes_per_core_base >= cfg.chunk_bytes, "bytes_per_core_base too small");
    TT_FATAL(bytes_per_core_base <= std::numeric_limits<uint32_t>::max(), "bytes_per_core_base must fit into uint32_t");

    const uint64_t covered_bytes = bytes_per_core_base * cores.size();
    const uint64_t remainder_bytes = total_bytes - covered_bytes;

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof zero_result[0], 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    for (size_t i = 0; i < cores.size(); i++) {
        const CoreCoord core = cores[i];
        const uint64_t bank_offset = cfg.bank_offset + i * bytes_per_core_base;

        uint64_t bytes_this_core = bytes_per_core_base;
        if (i == (cores.size() - 1)) {
            bytes_this_core += remainder_bytes;
        }

        TT_FATAL(bytes_this_core >= cfg.chunk_bytes, "bytes_this_core too small");
        TT_FATAL(bytes_this_core <= std::numeric_limits<uint32_t>::max(), "bytes_this_core must fit into uint32_t");

        auto kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                cfg.bank_id,
                bank_offset & 0xFFFFFFFFull,
                (bank_offset >> 32) & 0xFFFFFFFFull,
                bytes_this_core,
                cfg.chunk_bytes,
                cfg.pattern_id,
                seed,
                pass_index,
                repeat_index,
                result_l1_address,
                expect_l1_address,
                observe_l1_address,
                cfg.write_noc,
                cfg.read_noc,
                cfg.max_burst_len,
                cfg.transfer_len_mode,
                cfg.skip_writes,
                cfg.skip_reads,
                //                params.insert_write_errors,
                //                params.insert_read_errors,
            });
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = cfg.bank_id;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = (const DramBaseResult*)raw_result.data();

        accumulate_result_into_summary(summary, result);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return summary;
}

DramRunSummary run_dram_multi_core_all_controllers_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ======================== */
    auto* const device = mesh_device->get_devices()[0];

    constexpr uint32_t num_controllers = 8u;

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(
        total_bytes_per_controller <= DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES,
        "total_bytes_per_controller must be under (4GB-16MB-2KB)");
    TT_FATAL(chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(
        total_bytes_per_controller % DRAM_TEST_NOC_WORD_BYTES == 0,
        "total_bytes_per_controller must be NOC word aligned");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof zero_result[0], 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    const size_t total_cores = cores.size();
    const size_t base_cores_per_controller = total_cores / num_controllers;
    const size_t remainder_cores = total_cores % num_controllers;

    size_t core_begin = 0;

    for (uint32_t bank_id = 0; bank_id < num_controllers; bank_id++) {
        const size_t cores_in_this_controller = base_cores_per_controller + (bank_id < remainder_cores ? 1 : 0);

        if (cores_in_this_controller == 0) {
            continue;
        }

        const uint64_t bytes_per_core_base = (total_bytes_per_controller / cores_in_this_controller) & ~0xFFFULL;

        TT_FATAL(bytes_per_core_base >= chunk_bytes, "bytes_per_core_base too small");
        TT_FATAL(
            bytes_per_core_base <= std::numeric_limits<uint32_t>::max(), "bytes_per_core_base must fit into uint32_t");

        const uint64_t covered_bytes = bytes_per_core_base * cores_in_this_controller;
        const uint64_t remainder_bytes = total_bytes_per_controller - covered_bytes;

        for (size_t local_idx = 0; local_idx < cores_in_this_controller; local_idx++) {
            const size_t global_idx = core_begin + local_idx;
            const CoreCoord core = cores[global_idx];
            const uint64_t bank_offset = local_idx * bytes_per_core_base;

            uint64_t bytes_this_core = bytes_per_core_base;
            if (local_idx == (cores_in_this_controller - 1)) {
                bytes_this_core += remainder_bytes;
            }

            TT_FATAL(bytes_this_core >= chunk_bytes, "bytes_this_core too small");
            TT_FATAL(bytes_this_core <= std::numeric_limits<uint32_t>::max(), "bytes_this_core must fit into uint32_t");

            auto kernel = tt_metal::CreateKernel(
                program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

            tt_metal::SetRuntimeArgs(
                program,
                kernel,
                core,
                {
                    bank_id,
                    bank_offset & 0xFFFFFFFFull,
                    (bank_offset >> 32) & 0xFFFFFFFFull,
                    bytes_this_core,
                    chunk_bytes,
                    pattern_id,
                    seed,
                    pass_index,
                    repeat_index,
                    result_l1_address,
                    expect_l1_address,
                    observe_l1_address,
                    write_noc,
                    read_noc,
                    max_burst_len,
                    transfer_len_mode,
                    skip_writes,
                    skip_reads,
                    //                    params.insert_write_errors,
                    //                    params.insert_read_errors,
                });
        }

        core_begin += cores_in_this_controller;
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = 0;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = (const DramBaseResult*)raw_result.data();

        accumulate_result_into_summary(summary, result);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return summary;
}

DramRunSummary run_dram_eight_single_core_single_controller_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint64_t bank_offset,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ======================== */
    auto* const device = mesh_device->get_devices()[0];

    constexpr uint32_t num_controllers = 8u;

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(
        cores.size() <= num_controllers,
        "This helper supports at most {} cores, got {}",
        num_controllers,
        cores.size());
    TT_FATAL(
        total_bytes_per_controller <= DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES,
        "total_bytes_per_controller must be under (4GB-16MB-2KB)");
    TT_FATAL(chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(
        total_bytes_per_controller % DRAM_TEST_NOC_WORD_BYTES == 0,
        "total_bytes_per_controller must be NOC word aligned");
    TT_FATAL((bank_offset & 0xFFFULL) == 0ULL, "bank_offset must be 4KB aligned");
    TT_FATAL(
        bank_offset + total_bytes_per_controller <= DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES,
        "bank_offset + total_bytes_per_controller exceeds DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES");
    TT_FATAL(bank_offset <= std::numeric_limits<uint64_t>::max(), "bank_offset out of range");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    for (size_t inst_idx = 0; inst_idx < cores.size(); inst_idx++) {
        const CoreCoord core = cores[inst_idx];
        const uint32_t bank_id = inst_idx;

        auto kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                bank_id,
                bank_offset & 0xFFFFFFFFull,
                (bank_offset >> 32) & 0xFFFFFFFFull,
                total_bytes_per_controller,
                chunk_bytes,
                pattern_id,
                seed,
                pass_index,
                repeat_index,
                result_l1_address,
                expect_l1_address,
                observe_l1_address,
                write_noc,
                read_noc,
                max_burst_len,
                transfer_len_mode,
                skip_writes,
                skip_reads,
            });
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = 0;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = (const DramBaseResult*)raw_result.data();

        accumulate_result_into_summary(summary, result);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return summary;
}

[[maybe_unused]]
DramMultiInstanceSummary run_dram_eight_single_core_single_controller_test_verbose(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint64_t bank_offset,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ======================== */
    auto* const device = mesh_device->get_devices()[0];

    constexpr uint32_t num_controllers = 8u;

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(
        cores.size() <= num_controllers,
        "This helper supports at most {} cores, got {}",
        num_controllers,
        cores.size());
    TT_FATAL(
        total_bytes_per_controller <= DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES,
        "total_bytes_per_controller must be under (4GB-16MB-2KB)");
    TT_FATAL(chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(
        total_bytes_per_controller % DRAM_TEST_NOC_WORD_BYTES == 0,
        "total_bytes_per_controller must be NOC word aligned");
    TT_FATAL((bank_offset & 0xFFFULL) == 0ULL, "bank_offset must be 4KB aligned");
    TT_FATAL(
        bank_offset + total_bytes_per_controller <= DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES,
        "bank_offset + total_bytes_per_controller exceeds DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    for (size_t inst_idx = 0; inst_idx < cores.size(); inst_idx++) {
        const CoreCoord core = cores[inst_idx];
        const uint32_t bank_id = inst_idx;

        auto kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                bank_id,
                bank_offset & 0xFFFFFFFFull,
                (bank_offset >> 32) & 0xFFFFFFFFull,
                total_bytes_per_controller,
                chunk_bytes,
                pattern_id,
                seed,
                pass_index,
                repeat_index,
                result_l1_address,
                expect_l1_address,
                observe_l1_address,
                write_noc,
                read_noc,
                max_burst_len,
                transfer_len_mode,
                skip_writes,
                skip_reads,
            });
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramMultiInstanceSummary out{};
    out.summary.pass = true;
    out.summary.bank_id = 0;
    out.summary.checked_bytes = 0;
    out.summary.suspected_write_error_bytes = 0;
    out.summary.suspected_read_error_bytes = 0;

    out.per_core_results.reserve(cores.size());

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = (const DramBaseResult*)raw_result.data();

        accumulate_result_into_summary(out.summary, result);

        DramPerCoreResult per_core{};
        per_core.core = core;
        per_core.result = *result;
        out.per_core_results.push_back(per_core);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return out;
}

static inline void write_core_u32(IDevice* device, const CoreCoord& core, uint32_t l1_addr, uint32_t value) {
    MetalContext::instance().get_cluster().write_core(
        device->id(), device->worker_core_from_logical_core(core), std::vector<uint32_t>{value}, l1_addr);
}

[[maybe_unused]]
DramMultiInstanceSummary run_dram_persistent_jobs_test_verbose(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& worker_cores,
    const std::vector<std::vector<DramWorkItem>>& jobs_per_core,
    uint32_t chunk_bytes,
    DataMovementProcessor processor) {
    /* ======================== */
    auto* const device = mesh_device->get_devices()[0];

    TT_FATAL(!worker_cores.empty(), "No worker cores provided");
    TT_FATAL(!jobs_per_core.empty(), "No per-core jobs provided");
    TT_FATAL(
        worker_cores.size() == jobs_per_core.size(),
        "worker_cores/jobs_per_core size mismatch: workers={} job_lists={}",
        worker_cores.size(),
        jobs_per_core.size());

    uint64_t total_jobs = 0;
    for (const auto& core_jobs : jobs_per_core) {
        total_jobs += core_jobs.size();
    }

    TT_FATAL(total_jobs > 0, "No jobs provided");

    uint32_t queue_capacity = 0;
    for (const auto& core_jobs : jobs_per_core) {
        queue_capacity = std::max<uint32_t>(queue_capacity, core_jobs.size());
    }

    TT_FATAL(queue_capacity > 0, "queue_capacity must be non-zero");

    const uint32_t max_in_flight_jobs_per_core = queue_capacity;
    log_info(
        tt::LogTest,
        "{} persistent queue_capacity={} workers={} total_jobs={}",
        device_log_prefix(device),
        queue_capacity,
        worker_cores.size(),
        total_jobs);

    struct PerCorePersistentResources {
        CoreCoord core{0, 0};

        uint32_t queue_ctrl_l1_addr = 0;
        uint32_t queue_jobs_l1_addr = 0;
        uint32_t status_l1_addr = 0;
        uint32_t result_ring_l1_addr = 0;
        uint32_t expect_l1_addr = 0;
        uint32_t gen_ping_l1_addr = 0;
        uint32_t gen_pong_l1_addr = 0;
        uint32_t observe_l1_addr = 0;
        uint32_t observe_ping_l1_addr = 0;
        uint32_t observe_pong_l1_addr = 0;
        uint32_t wake_flag_l1_addr = 0;
        uint32_t sync_mailbox_l1_addr = 0;
        uint32_t jobs_enqueued = 0;
        uint32_t jobs_observed_done = 0;
        uint32_t host_tail_shadow = 0;
        bool stop_sent = false;

        std::chrono::steady_clock::time_point last_monitor_print_time;
        uint32_t prev_heartbeat_tick = 0;
        uint32_t prev_jobs_completed = 0;
        uint64_t prev_arc_tick = 0;

        bool monitor_initialized = false;
        bool stall_watchdog_armed = false;
        uint32_t stall_watchdog_reason = 0;
        std::chrono::steady_clock::time_point stall_watchdog_start_time;
    };

    std::vector<PerCorePersistentResources> per_core;
    per_core.reserve(worker_cores.size());

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto brisc_kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    auto ncrisc_kernel_config = tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = tt_metal::NOC::NOC_1,
    };

    for (const auto& core : worker_cores) {
        struct l1_allocator alloc = new_tensix_allocator();

        PerCorePersistentResources r{};
        r.core = core;

        r.queue_ctrl_l1_addr = l1_alloc(&alloc, sizeof(DramJobQueueCtrl), 32);
        r.queue_jobs_l1_addr = l1_alloc(&alloc, sizeof(DramWorkItem) * queue_capacity, 32);
        r.status_l1_addr = l1_alloc(&alloc, sizeof(CoreProgressStatus), 32);
        r.result_ring_l1_addr = l1_alloc(&alloc, sizeof(DramBaseResult) * queue_capacity, 32);

        r.gen_ping_l1_addr = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
        r.gen_pong_l1_addr = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
        r.expect_l1_addr = r.gen_ping_l1_addr;

        r.observe_ping_l1_addr = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
        r.observe_pong_l1_addr = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
        r.observe_l1_addr = r.observe_ping_l1_addr;

        r.wake_flag_l1_addr = l1_alloc(&alloc, sizeof(uint32_t), 4);
        r.sync_mailbox_l1_addr = l1_alloc(&alloc, kDramSyncMailboxWords * sizeof(uint32_t), 32);
        DramJobQueueCtrl ctrl{};
        dram_job_queue_ctrl_init(ctrl, queue_capacity);

        MetalContext::instance().get_cluster().write_core(
            device->id(),
            device->worker_core_from_logical_core(core),
            std::vector<uint32_t>((uint32_t*)&ctrl, (uint32_t*)&ctrl + sizeof(DramJobQueueCtrl) / sizeof(uint32_t)),
            r.queue_ctrl_l1_addr);

        CoreProgressStatus status{};
        dram_progress_status_init(status);

        MetalContext::instance().get_cluster().write_core(
            device->id(),
            device->worker_core_from_logical_core(core),
            std::vector<uint32_t>(
                (uint32_t*)&status, (uint32_t*)&status + sizeof(CoreProgressStatus) / sizeof(uint32_t)),
            r.status_l1_addr);

        std::vector<uint32_t> zero_mailbox(kDramSyncMailboxWords, 0u);
        zero_mailbox[MB_MAGIC] = DRAM_SYNC_MAILBOX_MAGIC;
        zero_mailbox[MB_VERSION] = 1u;
        zero_mailbox[MB_INSERT_ERRORS_PATTERN_ID] = get_dram_insert_errors_pattern_id_from_env_once();
        zero_mailbox[MB_STOP] = 0u;
        zero_mailbox[MB_NCRISC_START] = 0u;
        zero_mailbox[MB_NCRISC_DONE] = 0u;
        zero_mailbox[MB_NCRISC_ERROR] = MB_ERROR_NONE;
        zero_mailbox[MB_NCRISC_ACTIVE_OFFSET_BYTES] = 0u;
        zero_mailbox[MB_NCRISC_ACTIVE_TRANSFER_BYTES] = 0u;
        zero_mailbox[MB_OBS_PING_L1_ADDR] = r.observe_ping_l1_addr;
        zero_mailbox[MB_OBS_PONG_L1_ADDR] = r.observe_pong_l1_addr;
        zero_mailbox[MB_OBS_ACTIVE_SLOT] = DRAM_OBS_SLOT_PING;
        zero_mailbox[MB_OBS_ACTIVE_L1_ADDR] = r.observe_ping_l1_addr;

        zero_mailbox[MB_GEN_PING_L1_ADDR] = r.gen_ping_l1_addr;
        zero_mailbox[MB_GEN_PONG_L1_ADDR] = r.gen_pong_l1_addr;
        zero_mailbox[MB_GEN_ACTIVE_SLOT] = DRAM_GEN_SLOT_PING;
        zero_mailbox[MB_GEN_ACTIVE_L1_ADDR] = r.gen_ping_l1_addr;

        zero_mailbox[MB_CURRENT_STAGE] = MB_STAGE_IDLE;

        zero_mailbox[MB_GENERATE_START] = 0u;
        zero_mailbox[MB_GENERATE_DONE] = 0u;

        zero_mailbox[MB_GENERATE_L1_ADDR] = 0u;
        zero_mailbox[MB_GENERATE_WORD_COUNT] = 0u;
        zero_mailbox[MB_GENERATE_BASE_WORD_INDEX] = 0u;

        zero_mailbox[MB_GENERATE_MATH_DONE] = 0u;
        zero_mailbox[MB_GENERATE_PACK_DONE] = 0u;

        zero_mailbox[MB_COMPARE_START] = 0u;
        zero_mailbox[MB_COMPARE_DONE] = 0u;

        zero_mailbox[MB_COMPARE_SOURCE_L1_ADDR] = 0u;
        zero_mailbox[MB_COMPARE_OBSERVED_L1_ADDR] = 0u;
        zero_mailbox[MB_COMPARE_WORD_COUNT] = 0u;
        zero_mailbox[MB_COMPARE_BASE_BYTE_OFFSET] = 0u;

        zero_mailbox[MB_COMPARE_MATH_DONE] = 0u;
        zero_mailbox[MB_COMPARE_MATH_RESULT] = 0u;
        zero_mailbox[MB_COMPARE_MATH_FIRST_ADDR] = 0xFFFFFFFFu;
        zero_mailbox[MB_COMPARE_MATH_FIRST_EXPECTED] = 0u;
        zero_mailbox[MB_COMPARE_MATH_FIRST_OBSERVED] = 0u;

        zero_mailbox[MB_COMPARE_PACK_DONE] = 0u;
        zero_mailbox[MB_COMPARE_PACK_RESULT] = 0u;
        zero_mailbox[MB_COMPARE_PACK_FIRST_ADDR] = 0xFFFFFFFFu;
        zero_mailbox[MB_COMPARE_PACK_FIRST_EXPECTED] = 0u;
        zero_mailbox[MB_COMPARE_PACK_FIRST_OBSERVED] = 0u;

        zero_mailbox[MB_COMPARE_UNPACK_DONE] = 0u;
        zero_mailbox[MB_COMPARE_UNPACK_RESULT] = 0u;
        zero_mailbox[MB_COMPARE_UNPACK_FIRST_ADDR] = 0xFFFFFFFFu;
        zero_mailbox[MB_COMPARE_UNPACK_FIRST_EXPECTED] = 0u;
        zero_mailbox[MB_COMPARE_UNPACK_FIRST_OBSERVED] = 0u;

        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_mailbox, r.sync_mailbox_l1_addr);

        std::vector<uint32_t> zero_results(sizeof(DramBaseResult) * queue_capacity / sizeof zero_results[0], 0u);

        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_results, r.result_ring_l1_addr);

        for (uint32_t i = 0; i < queue_capacity; i++) {
            uint32_t offset = i * sizeof(DramBaseResult) + offsetof(DramBaseResult, job_id);
            write_core_u32(device, core, r.result_ring_l1_addr + offset, 0xFFFFFFFFu);
        }

        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), std::vector<uint32_t>{0}, r.wake_flag_l1_addr);

        const auto now = std::chrono::steady_clock::now();
        r.last_monitor_print_time = now;

        per_core.push_back(r);
    }

    for (const auto& r : per_core) {
        auto brisc_kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", r.core, brisc_kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            brisc_kernel,
            r.core,
            {
                r.queue_ctrl_l1_addr,
                r.queue_jobs_l1_addr,
                r.status_l1_addr,
                r.result_ring_l1_addr,
                r.expect_l1_addr,
                r.observe_l1_addr,
                queue_capacity,
                r.wake_flag_l1_addr,
                r.sync_mailbox_l1_addr,
            });

        auto ncrisc_kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_second_kernel.cpp", r.core, ncrisc_kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            ncrisc_kernel,
            r.core,
            {
                r.sync_mailbox_l1_addr,
            });

        [[maybe_unused]]
        auto compute_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/deployment/kernels/dram_compare_compute_kernel.cpp",
            r.core,
            ComputeConfig{
                .compile_args =
                    {
                        r.sync_mailbox_l1_addr,
                    },
            });
    }

    workload.add_program(device_range, std::move(program));

    for (size_t core_idx = 0; core_idx < per_core.size(); core_idx++) {
        auto& r = per_core[core_idx];
        const auto& core_jobs = jobs_per_core[core_idx];

        const uint32_t preload = std::min<uint32_t>(max_in_flight_jobs_per_core, core_jobs.size());

        if (!preload) {
            continue;
        }

        std::vector<uint32_t> job_words;
        job_words.reserve(sizeof(DramWorkItem) / sizeof job_words[0] * preload);

        for (uint32_t j = 0; j < preload; j++) {
            const DramWorkItem& job = core_jobs[j];
            log_verbose_dram_work_item(device, r.core, job, j, core_jobs.size());

            const uint32_t* p = (const uint32_t*)&job;

            job_words.insert(job_words.end(), p, p + sizeof(DramWorkItem) / sizeof job_words[0]);
        }

        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(r.core), job_words, r.queue_jobs_l1_addr);

        r.jobs_enqueued = preload;
        r.host_tail_shadow = preload;

        DramJobQueueCtrl ctrl{};
        dram_job_queue_ctrl_init(ctrl, queue_capacity);
        ctrl.tail = r.host_tail_shadow;
        ctrl.reserved0 = get_dram_inject_tensix_heartbeat_stall_from_env_once() ? 1u : 0u;

        MetalContext::instance().get_cluster().write_core(
            device->id(),
            device->worker_core_from_logical_core(r.core),
            std::vector<uint32_t>((uint32_t*)&ctrl, (uint32_t*)&ctrl + sizeof(DramJobQueueCtrl) / sizeof(uint32_t)),
            r.queue_ctrl_l1_addr);

        MetalContext::instance().get_cluster().write_core(
            device->id(),
            device->worker_core_from_logical_core(r.core),
            std::vector<uint32_t>{1u},
            r.wake_flag_l1_addr);
    }

    DramMultiInstanceSummary out{};
    out.summary.pass = true;
    out.summary.bank_id = 0;
    out.summary.checked_bytes = 0;
    out.summary.suspected_write_error_bytes = 0;
    out.summary.suspected_read_error_bytes = 0;
    out.per_core_results.reserve(total_jobs);

    constexpr auto kMonitorPrintInterval = std::chrono::seconds(2);
    constexpr auto kStallWatchdogTimeout = std::chrono::seconds(10);

    constexpr bool print_per_core_monitor = false;
    constexpr bool print_persistent_run_start_finish = true;

    uint64_t global_prev_arc_tick = 0;
    bool global_monitor_initialized = false;
    auto global_last_monitor_print_time = std::chrono::steady_clock::now();

    auto broadcast_stop_to_all_cores = [&]() {
        for (auto& r : per_core) {
            if (!r.stop_sent) {
                write_core_u32(device, r.core, r.queue_ctrl_l1_addr + offsetof(DramJobQueueCtrl, stop_requested), 1u);

                write_core_u32(device, r.core, r.wake_flag_l1_addr, 1u);

                r.stop_sent = true;
            }
        }
    };

    auto get_completed_jobs_total = [&]() -> uint64_t {
        uint64_t completed = 0;

        for (size_t i = 0; i < per_core.size(); i++) {
            auto raw_status = MetalContext::instance().get_cluster().read_core(
                device->id(),
                device->worker_core_from_logical_core(per_core[i].core),
                per_core[i].status_l1_addr,
                sizeof(CoreProgressStatus));

            const CoreProgressStatus* status = (const CoreProgressStatus*)raw_status.data();

            completed += std::min<uint64_t>(status->jobs_completed, jobs_per_core[i].size());
        }

        return completed;
    };

    std::thread refill_thread([&]() {
        bool done = false;

        while (!done) {
            done = true;

            if (g_stop_requested.load()) {
                broadcast_stop_to_all_cores();
                done = true;
                break;
            }

            const auto monitor_now = std::chrono::steady_clock::now();

            if ((monitor_now - global_last_monitor_print_time) >= kMonitorPrintInterval) {
                const uint64_t arc_tick = read_arc_global_tick(device);

                bool all_cores_progressing = true;
                bool all_cores_done = true;
                uint64_t completed_jobs_total = 0;

                for (auto& rr : per_core) {
                    const auto& rr_jobs = jobs_per_core[&rr - per_core.data()];

                    auto raw_status = MetalContext::instance().get_cluster().read_core(
                        device->id(),
                        device->worker_core_from_logical_core(rr.core),
                        rr.status_l1_addr,
                        sizeof(CoreProgressStatus));

                    const CoreProgressStatus* status = (const CoreProgressStatus*)raw_status.data();

                    completed_jobs_total += std::min<uint64_t>(status->jobs_completed, rr_jobs.size());

                    const bool core_done = status->jobs_completed >= rr_jobs.size();

                    const bool core_progressing = (status->heartbeat_tick != rr.prev_heartbeat_tick) ||
                                                  (status->jobs_completed != rr.prev_jobs_completed);

                    all_cores_done &= core_done;

                    if (!core_done && !core_progressing) {
                        all_cores_progressing = false;
                    }
                }

                if (dram_test_verbose_enabled()) {
                    if (global_monitor_initialized) {
                        const uint64_t arc_delta = arc_tick - global_prev_arc_tick;
                        log_info(
                            tt::LogTest,
                            "{} monitor: arc={} delta={} {} jobs={}/{}",
                            device_log_prefix(device),
                            arc_tick,
                            arc_delta,
                            (all_cores_progressing || all_cores_done) ? "all cores progressing"
                                                                      : "some cores not progressing",
                            completed_jobs_total,
                            total_jobs);
                    } else {
                        log_info(
                            tt::LogTest,
                            "{} monitor: arc={} {} jobs={}/{}",
                            device_log_prefix(device),
                            arc_tick,
                            (all_cores_progressing || all_cores_done) ? "all cores progressing"
                                                                      : "some cores not progressing",
                            completed_jobs_total,
                            total_jobs);
                    }
                }

                if (!global_monitor_initialized) {
                    global_monitor_initialized = true;
                }

                global_prev_arc_tick = arc_tick;
                global_last_monitor_print_time = monitor_now;
            }

            for (size_t core_idx = 0; core_idx < per_core.size(); core_idx++) {
                auto& r = per_core[core_idx];
                const auto& core_jobs = jobs_per_core[core_idx];

                auto raw_status = MetalContext::instance().get_cluster().read_core(
                    device->id(),
                    device->worker_core_from_logical_core(r.core),
                    r.status_l1_addr,
                    sizeof(CoreProgressStatus));

                const CoreProgressStatus* status = (const CoreProgressStatus*)raw_status.data();

                const auto now = std::chrono::steady_clock::now();

                if ((now - r.last_monitor_print_time) >= kMonitorPrintInterval) {
                    const uint64_t arc_tick = read_arc_global_tick(device);

                    const uint32_t hb_delta = status->heartbeat_tick - r.prev_heartbeat_tick;

                    const uint32_t jobs_delta = status->jobs_completed - r.prev_jobs_completed;

                    const uint64_t arc_delta = arc_tick - r.prev_arc_tick;

                    if (print_per_core_monitor) {
                        log_info(
                            tt::LogTest,
                            "{} monitor: core=({}, {}) jobs={}/{} hb={} delta={} arc={} delta={} stage={} job_id={}",
                            device_log_prefix(device),
                            r.core.x,
                            r.core.y,
                            status->jobs_completed,
                            core_jobs.size(),
                            status->heartbeat_tick,
                            hb_delta,
                            arc_tick,
                            arc_delta,
                            status->current_stage,
                            status->current_job_id);
                    }

                    const bool core_done = status->jobs_completed >= core_jobs.size();

                    const bool tensix_progress = hb_delta || jobs_delta;

                    const bool arc_progress = arc_delta;

                    if (r.monitor_initialized && !core_done && !g_stop_requested.load()) {
                        uint32_t stall_reason = 0;

                        if (!tensix_progress && arc_progress) {
                            stall_reason = 1;
                        } else if (!tensix_progress && !arc_progress) {
                            stall_reason = 2;
                        }

                        if (!stall_reason) {
                            if (r.stall_watchdog_armed) {
                                log_info(
                                    tt::LogTest,
                                    "{} watchdog disarmed: core=({}, {}) progress resumed",
                                    device_log_prefix(device),
                                    r.core.x,
                                    r.core.y);
                            }

                            r.stall_watchdog_armed = false;
                            r.stall_watchdog_reason = 0;
                        } else {
                            if (!r.stall_watchdog_armed || r.stall_watchdog_reason != stall_reason) {
                                r.stall_watchdog_armed = true;
                                r.stall_watchdog_reason = stall_reason;
                                r.stall_watchdog_start_time = now;

                                log_info(
                                    tt::LogTest,
                                    "{} watchdog armed: core=({}, {}) reason={} jobs={}/{}",
                                    device_log_prefix(device),
                                    r.core.x,
                                    r.core.y,
                                    dram_watchdog_reason_name(stall_reason),
                                    status->jobs_completed,
                                    core_jobs.size());

                            } else if ((now - r.stall_watchdog_start_time) >= kStallWatchdogTimeout) {
                                const auto stuck_for_s =
                                    std::chrono::duration_cast<std::chrono::seconds>(now - r.stall_watchdog_start_time)
                                        .count();
                                log_critical(
                                    tt::LogTest,
                                    "\n"
                                    "================ DRAM WATCHDOG TIMEOUT ================\n"
                                    "BDF          : {}\n"
                                    "UBB tray     : {}\n"
                                    "Location     : {}\n"
                                    "DRAM Channel : unknown\n"
                                    "Worker Core  : ({}, {})\n"
                                    "Reason       : {}\n"
                                    "Stuck For    : {}s\n"
                                    "Jobs         : {}/{}\n"
                                    "Heartbeat    : {}\n"
                                    "ARC          : {}\n"
                                    "Stage        : {}\n"
                                    "Job ID       : {}\n"
                                    "\n"
                                    "Requesting graceful stop.\n"
                                    "If the next run cannot acquire the device or topology mapping fails, execute:\n"
                                    "    source ~/.tenstorrent-venv/bin/activate\n"
                                    "    tt-smi -r\n"
                                    "======================================================",
                                    dram_physical_location_for_device_id(device->id()).bdf,
                                    dram_physical_location_for_device_id(device->id()).ubb_tray,
                                    dram_physical_location_for_device_id(device->id()).location,
                                    r.core.x,
                                    r.core.y,
                                    dram_watchdog_reason_name(r.stall_watchdog_reason),
                                    stuck_for_s,
                                    status->jobs_completed,
                                    core_jobs.size(),
                                    status->heartbeat_tick,
                                    arc_tick,
                                    status->current_stage,
                                    status->current_job_id);

                                out.summary.pass = false;
                                g_watchdog_requested.store(true);
                                g_stop_requested.store(true);

                                ArmWatchdogHardExit(fmt::format(
                                    "{} watchdog timeout on worker core=({}, {}) reason={}",
                                    device_log_prefix(device),
                                    r.core.x,
                                    r.core.y,
                                    dram_watchdog_reason_name(r.stall_watchdog_reason)));
                            }
                        }
                    }

                    r.monitor_initialized = true;
                    r.prev_heartbeat_tick = status->heartbeat_tick;
                    r.prev_jobs_completed = status->jobs_completed;
                    r.prev_arc_tick = arc_tick;
                    r.last_monitor_print_time = now;
                }

                while (r.jobs_observed_done < r.jobs_enqueued) {
                    const uint32_t done_index = r.jobs_observed_done;
                    const uint32_t done_slot = done_index;
                    const DramWorkItem& expected_job = core_jobs[done_index];

                    DramBaseResult result_copy{};

                    constexpr auto kResultPollDelay = std::chrono::milliseconds(20);
                    constexpr auto kNoHeartbeatTimeout = std::chrono::seconds(10);
                    constexpr auto kResultReadyTimeout = std::chrono::seconds(10);

                    bool result_ready = false;

                    uint32_t last_heartbeat = status->heartbeat_tick;
                    auto last_heartbeat_change_time = std::chrono::steady_clock::now();
                    auto result_wait_start_time = last_heartbeat_change_time;

                    while (!result_ready) {
                        if (g_stop_requested.load()) {
                            broadcast_stop_to_all_cores();
                            break;
                        }

                        const auto monitor_now = std::chrono::steady_clock::now();

                        if ((monitor_now - global_last_monitor_print_time) >= kMonitorPrintInterval) {
                            const uint64_t arc_tick = read_arc_global_tick(device);
                            const uint64_t completed_jobs_total = get_completed_jobs_total();

                            if (dram_test_verbose_enabled()) {
                                if (global_monitor_initialized) {
                                    const uint64_t arc_delta = arc_tick - global_prev_arc_tick;

                                    log_info(
                                        tt::LogTest,
                                        "{} monitor: arc={} delta={} all cores progressing jobs={}/{}",
                                        device_log_prefix(device),
                                        arc_tick,
                                        arc_delta,
                                        completed_jobs_total,
                                        total_jobs);
                                } else {
                                    log_info(
                                        tt::LogTest,
                                        "{} monitor: arc={} all cores progressing jobs={}/{}",
                                        device_log_prefix(device),
                                        arc_tick,
                                        completed_jobs_total,
                                        total_jobs);
                                }
                            }

                            if (!global_monitor_initialized) {
                                global_monitor_initialized = true;
                            }

                            global_prev_arc_tick = arc_tick;
                            global_last_monitor_print_time = monitor_now;
                        }

                        auto raw_result = MetalContext::instance().get_cluster().read_core(
                            device->id(),
                            device->worker_core_from_logical_core(r.core),
                            r.result_ring_l1_addr + done_slot * sizeof(DramBaseResult),
                            sizeof(DramBaseResult));

                        result_copy = *(const DramBaseResult*)raw_result.data();

                        if (result_copy.job_id == expected_job.job_id &&
                            result_copy.pattern_id == expected_job.pattern_id &&
                            result_copy.pass_index == expected_job.pass_index &&
                            result_copy.repeat_index == expected_job.repeat_index &&
                            result_copy.bank_id == expected_job.bank_id && result_copy.transfers > 0u) {
                            result_ready = true;
                            break;
                        }

                        auto raw_status_poll = MetalContext::instance().get_cluster().read_core(
                            device->id(),
                            device->worker_core_from_logical_core(r.core),
                            r.status_l1_addr,
                            sizeof(CoreProgressStatus));

                        const CoreProgressStatus* status_poll = (const CoreProgressStatus*)raw_status_poll.data();

                        if (status_poll->heartbeat_tick != last_heartbeat) {
                            last_heartbeat = status_poll->heartbeat_tick;
                            last_heartbeat_change_time = std::chrono::steady_clock::now();
                        }

                        const auto now_poll = std::chrono::steady_clock::now();

                        const bool result_overdue = status_poll->jobs_completed > done_index &&
                                                    ((now_poll - result_wait_start_time) >= kResultReadyTimeout);

                        const bool heartbeat_stopped = (now_poll - last_heartbeat_change_time) >= kNoHeartbeatTimeout;

                        if (result_overdue || heartbeat_stopped) {
                            const auto waited_s =
                                std::chrono::duration_cast<std::chrono::seconds>(now_poll - result_wait_start_time)
                                    .count();
                            const auto heartbeat_idle_s =
                                std::chrono::duration_cast<std::chrono::seconds>(now_poll - last_heartbeat_change_time)
                                    .count();
                            log_error(
                                tt::LogTest,
                                "\n"
                                "================ DRAM WATCHDOG TIMEOUT ================\n"
                                "BDF          : {}\n"
                                "UBB tray     : {}\n"
                                "Location     : {}\n"
                                "DRAM Channel : {}\n"
                                "Worker Core  : ({}, {})\n"
                                "Core Index   : {}\n"
                                "Queue Slot   : {}\n"
                                "Waited       : {}s\n"
                                "Heartbeat Idle: {}s\n"
                                "Status Jobs  : {}\n"
                                "\n"
                                "Expected Job : job_id={} pattern={} pass={} repeat={} bank={}\n"
                                "Observed     : job_id={} pattern={} pass={} repeat={} bank={} words={} transfers={}\n"
                                "\n"
                                "Requesting graceful stop.\n"
                                "If the next run cannot acquire the device or topology mapping fails, execute:\n"
                                "    source ~/.tenstorrent-venv/bin/activate\n"
                                "    tt-smi -r\n"
                                "======================================================",
                                dram_physical_location_for_device_id(device->id()).bdf,
                                dram_physical_location_for_device_id(device->id()).ubb_tray,
                                dram_physical_location_for_device_id(device->id()).location,
                                expected_job.bank_id,
                                r.core.x,
                                r.core.y,
                                core_idx,
                                done_slot,
                                waited_s,
                                heartbeat_idle_s,
                                status_poll->jobs_completed,
                                expected_job.job_id,
                                expected_job.pattern_id,
                                expected_job.pass_index,
                                expected_job.repeat_index,
                                expected_job.bank_id,
                                result_copy.job_id,
                                result_copy.pattern_id,
                                result_copy.pass_index,
                                result_copy.repeat_index,
                                result_copy.bank_id,
                                result_copy.words_checked,
                                result_copy.transfers);

                            out.summary.pass = false;
                            g_watchdog_requested.store(true);
                            g_stop_requested.store(true);
                            broadcast_stop_to_all_cores();
                            ArmWatchdogHardExit(fmt::format(
                                "{} result wait timeout on bdf={} ubb_tray={} location={} dram_channel={} core=({}, "
                                "{}) expected_job={} transfers={}",
                                device_log_prefix(device),
                                dram_physical_location_for_device_id(device->id()).bdf,
                                dram_physical_location_for_device_id(device->id()).ubb_tray,
                                dram_physical_location_for_device_id(device->id()).location,
                                expected_job.bank_id,
                                r.core.x,
                                r.core.y,
                                expected_job.job_id,
                                result_copy.transfers));
                            break;
                        }

                        std::this_thread::sleep_for(kResultPollDelay);
                    }

                    if (g_stop_requested.load()) {
                        break;
                    }

                    if (!result_ready) {
                        log_info(
                            tt::LogTest,
                            "{} result not ready: core={} job={} slot={} expected(job_id={}, pattern={}, pass={}, "
                            "repeat={}, bank={}) got(job_id={}, pattern={}, pass={}, repeat={}, bank={}, words={}, "
                            "transfers={})",
                            device_log_prefix(device),
                            core_idx,
                            done_index,
                            done_slot,
                            expected_job.job_id,
                            expected_job.pattern_id,
                            expected_job.pass_index,
                            expected_job.repeat_index,
                            expected_job.bank_id,
                            result_copy.job_id,
                            result_copy.pattern_id,
                            result_copy.pass_index,
                            result_copy.repeat_index,
                            result_copy.bank_id,
                            result_copy.words_checked,
                            result_copy.transfers);
                        break;
                    }

                    const DramBaseResult* result = &result_copy;

                    TT_FATAL(
                        done_index < core_jobs.size(),
                        "done_index {} out of range for core {} job list size {}",
                        done_index,
                        core_idx,
                        core_jobs.size());

                    TT_FATAL(
                        result->job_id == expected_job.job_id,
                        "job_id mismatch on core {}: expected {} got {}",
                        core_idx,
                        expected_job.job_id,
                        result->job_id);

                    TT_FATAL(
                        result->pattern_id == expected_job.pattern_id,
                        "pattern_id mismatch on core {}: expected {} got {}",
                        core_idx,
                        expected_job.pattern_id,
                        result->pattern_id);

                    TT_FATAL(
                        result->pass_index == expected_job.pass_index,
                        "pass_index mismatch on core {}: expected {} got {}",
                        core_idx,
                        expected_job.pass_index,
                        result->pass_index);

                    TT_FATAL(
                        result->repeat_index == expected_job.repeat_index,
                        "repeat_index mismatch on core {}: expected {} got {}",
                        core_idx,
                        expected_job.repeat_index,
                        result->repeat_index);

                    TT_FATAL(
                        result->bank_id == expected_job.bank_id,
                        "bank_id mismatch on core {}: expected {} got {}",
                        core_idx,
                        expected_job.bank_id,
                        result->bank_id);

                    if (expected_job.skip_reads == 0u) {
                        TT_FATAL(
                            result->words_checked > 0u,
                            "words_checked is zero for read-enabled job: core {} job {}",
                            core_idx,
                            done_index);
                    } else {
                        TT_FATAL(
                            result->words_checked == 0u,
                            "words_checked should be zero for write-only job: core {} job {}",
                            core_idx,
                            done_index);
                    }

                    TT_FATAL(result->transfers > 0u, "transfers is zero for core {} job {}", core_idx, done_index);

                    write_core_u32(
                        device,
                        r.core,
                        r.result_ring_l1_addr + done_slot * sizeof(DramBaseResult) + offsetof(DramBaseResult, job_id),
                        0xFFFFFFFFu);

                    accumulate_result_into_summary(out.summary, result);

                    if (result->failures > 0u) {
                        log_info(
                            tt::LogTest,
                            "{} job {}/{} failed: core=({}, {}) bank={} pattern={} pass={} repeat={} kind={} "
                            "first_fail_addr=0x{:08x} write_err={:.6f}% read_err={:.6f}%",
                            device_log_prefix(device),
                            done_index + 1u,
                            core_jobs.size(),
                            r.core.x,
                            r.core.y,
                            result->bank_id,
                            pattern_name(result->pattern_id),
                            result->pass_index,
                            result->repeat_index,
                            dram_failure_kind_name(result->failure_kind),
                            result->first_fail_addr,
                            dram_result_write_error_pct(result),
                            dram_result_read_error_pct(result));
                    }

                    DramPerCoreResult per_core_result{};
                    per_core_result.core = r.core;
                    per_core_result.result = *result;
                    out.per_core_results.push_back(per_core_result);

                    if (result->failures > 0u) {
                        log_dram_failure(device, r.core, result);
                    }

                    r.jobs_observed_done++;
                }

                if (!g_stop_requested.load() && r.jobs_enqueued < core_jobs.size()) {
                    uint32_t tail = r.host_tail_shadow;

                    while (r.jobs_enqueued < core_jobs.size()) {
                        const uint32_t outstanding_jobs = r.jobs_enqueued - r.jobs_observed_done;

                        if (outstanding_jobs >= max_in_flight_jobs_per_core) {
                            break;
                        }

                        const uint32_t slot = tail;
                        const uint32_t next_tail = tail + 1u;
                        const uint32_t next_job_index = r.jobs_enqueued;
                        const DramWorkItem& next_job = core_jobs[next_job_index];
                        log_verbose_dram_work_item(device, r.core, next_job, next_job_index, core_jobs.size());

                        MetalContext::instance().get_cluster().write_core(
                            device->id(),
                            device->worker_core_from_logical_core(r.core),
                            std::vector<uint32_t>(
                                (const uint32_t*)&next_job,
                                (const uint32_t*)&next_job + sizeof(DramWorkItem) / sizeof(uint32_t)),
                            r.queue_jobs_l1_addr + slot * sizeof(DramWorkItem));

                        tail = next_tail;
                        r.jobs_enqueued++;
                    }

                    if (tail != r.host_tail_shadow) {
                        write_core_u32(device, r.core, r.queue_ctrl_l1_addr + offsetof(DramJobQueueCtrl, tail), tail);

                        r.host_tail_shadow = tail;

                        write_core_u32(device, r.core, r.wake_flag_l1_addr, 1u);
                    }
                }

                if ((r.jobs_observed_done == core_jobs.size()) && !r.stop_sent) {
                    write_core_u32(
                        device, r.core, r.queue_ctrl_l1_addr + offsetof(DramJobQueueCtrl, stop_requested), 1u);

                    write_core_u32(device, r.core, r.wake_flag_l1_addr, 1u);

                    r.stop_sent = true;
                }

                if (g_stop_requested.load() && !r.stop_sent) {
                    write_core_u32(
                        device, r.core, r.queue_ctrl_l1_addr + offsetof(DramJobQueueCtrl, stop_requested), 1u);

                    write_core_u32(device, r.core, r.wake_flag_l1_addr, 1u);

                    r.stop_sent = true;
                }

                const bool this_core_done =
                    r.stop_sent && ((r.jobs_observed_done == core_jobs.size()) || g_stop_requested.load());

                done &= this_core_done;
            }

            if (!done) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    });

    auto start_time = std::chrono::steady_clock::now();

    if (print_persistent_run_start_finish) {
        log_info(
            tt::LogTest,
            "{} Starting persistent DRAM test: workers={}, total_jobs={}",
            device_log_prefix(device),
            worker_cores.size(),
            total_jobs);
    }

    fixture->RunProgram(mesh_device, workload, false);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    log_info(
        tt::LogTest,
        "{} Persistent DRAM test finished: workers={}, total_jobs={}, duration={}",
        device_log_prefix(device),
        worker_cores.size(),
        total_jobs,
        format_duration_seconds(duration_sec));

    if (refill_thread.joinable()) {
        refill_thread.join();
    }

    fixture->FinishCommands(mesh_device);

    return out;
}

}  // namespace tt::tt_metal
