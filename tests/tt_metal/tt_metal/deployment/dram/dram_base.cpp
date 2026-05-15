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
#include <umd/device/types/telemetry.hpp>

namespace tt::tt_metal {

extern std::atomic<bool> g_stop_requested;
extern std::atomic<bool> g_watchdog_requested;

using namespace std;
using namespace tt;

[[maybe_unused]] static std::vector<DramBankWorkerAssignment> get_optimal_dram_bank_worker_assignments(
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

    for (uint32_t bank_id = 0; bank_id < num_dram_channels; ++bank_id) {
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
        "Mismatch on device={} dram_controller={} core {} pattern={} repeat={} pass={}: failures={}, "
        "first_fail_classified_as={}, write_failures={}, read_failures={}",
        device->id(),
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
    summary.checked_bytes += static_cast<uint64_t>(result->words_checked) * sizeof(uint32_t);

    summary.suspected_write_error_bytes += static_cast<uint64_t>(result->suspected_write_failures) * sizeof(uint32_t);

    summary.suspected_read_error_bytes += static_cast<uint64_t>(result->suspected_read_failures) * sizeof(uint32_t);
}

static inline double dram_result_write_error_pct(const DramBaseResult* result) {
    if (result->words_checked == 0u) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(result->suspected_write_failures) / static_cast<double>(result->words_checked);
}

static inline double dram_result_read_error_pct(const DramBaseResult* result) {
    if (result->words_checked == 0u) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(result->suspected_read_failures) / static_cast<double>(result->words_checked);
}

static inline uint64_t read_arc_global_tick(tt::tt_metal::IDevice* device) {
    return static_cast<uint64_t>(device->get_arc_timer_heartbeat());
}

static inline const char* dram_watchdog_reason_name(uint32_t reason) {
    switch (reason) {
        case 1: return "kernel_stall";
        case 2: return "system_stall";
        default: return "none";
    }
}

static inline void write_core_u32(IDevice* device, const CoreCoord& core, uint32_t l1_addr, uint32_t value) {
    MetalContext::instance().get_cluster().write_core(
        device->id(), device->worker_core_from_logical_core(core), std::vector<uint32_t>{value}, l1_addr);
}

[[maybe_unused]] DramMultiInstanceSummary run_dram_persistent_jobs_test_verbose(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& worker_cores,
    const std::vector<std::vector<DramWorkItem>>& jobs_per_core,
    uint32_t chunk_bytes,
    DataMovementProcessor processor) {
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
        queue_capacity = std::max<uint32_t>(queue_capacity, static_cast<uint32_t>(core_jobs.size()));
    }

    TT_FATAL(queue_capacity > 0, "queue_capacity must be non-zero");

    const uint32_t max_in_flight_jobs_per_core = queue_capacity;

    log_info(
        tt::LogTest,
        "device_id={} persistent queue_capacity={} workers={} total_jobs={}",
        device->id(),
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

        std::chrono::steady_clock::time_point last_monitor_print_time{};
        uint32_t prev_heartbeat_tick = 0;
        uint32_t prev_jobs_completed = 0;
        uint64_t prev_arc_tick = 0;

        bool monitor_initialized = false;
        bool stall_watchdog_armed = false;
        uint32_t stall_watchdog_reason = 0;
        std::chrono::steady_clock::time_point stall_watchdog_start_time{};
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
            std::vector<uint32_t>(
                reinterpret_cast<uint32_t*>(&ctrl),
                reinterpret_cast<uint32_t*>(&ctrl) + (sizeof(DramJobQueueCtrl) / sizeof(uint32_t))),
            r.queue_ctrl_l1_addr);

        CoreProgressStatus status{};
        dram_progress_status_init(status);

        MetalContext::instance().get_cluster().write_core(
            device->id(),
            device->worker_core_from_logical_core(core),
            std::vector<uint32_t>(
                reinterpret_cast<uint32_t*>(&status),
                reinterpret_cast<uint32_t*>(&status) + (sizeof(CoreProgressStatus) / sizeof(uint32_t))),
            r.status_l1_addr);

        std::vector<uint32_t> zero_mailbox(kDramSyncMailboxWords, 0u);
        zero_mailbox[MB_MAGIC] = DRAM_SYNC_MAILBOX_MAGIC;
        zero_mailbox[MB_VERSION] = 1u;
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

        std::vector<uint32_t> zero_results((sizeof(DramBaseResult) * queue_capacity) / sizeof(uint32_t), 0u);

        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_results, r.result_ring_l1_addr);

        for (uint32_t i = 0; i < queue_capacity; ++i) {
            uint32_t offset = i * sizeof(DramBaseResult) + offsetof(DramBaseResult, job_id);
            write_core_u32(device, core, r.result_ring_l1_addr + offset, 0xFFFFFFFFu);
        }

        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), std::vector<uint32_t>{0u}, r.wake_flag_l1_addr);

        const auto now = std::chrono::steady_clock::now();
        r.last_monitor_print_time = now;

        per_core.push_back(std::move(r));
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

        [[maybe_unused]] auto compute_kernel = tt_metal::CreateKernel(
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

    for (size_t core_idx = 0; core_idx < per_core.size(); ++core_idx) {
        auto& r = per_core[core_idx];
        const auto& core_jobs = jobs_per_core[core_idx];

        const uint32_t preload =
            std::min<uint32_t>(max_in_flight_jobs_per_core, static_cast<uint32_t>(core_jobs.size()));

        if (preload == 0) {
            continue;
        }

        std::vector<uint32_t> job_words;
        job_words.reserve((sizeof(DramWorkItem) / sizeof(uint32_t)) * preload);

        for (uint32_t j = 0; j < preload; ++j) {
            const DramWorkItem& job = core_jobs[j];
            const uint32_t* p = reinterpret_cast<const uint32_t*>(&job);

            job_words.insert(job_words.end(), p, p + (sizeof(DramWorkItem) / sizeof(uint32_t)));
        }

        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(r.core), job_words, r.queue_jobs_l1_addr);

        r.jobs_enqueued = preload;
        r.host_tail_shadow = preload;

        DramJobQueueCtrl ctrl{};
        dram_job_queue_ctrl_init(ctrl, queue_capacity);
        ctrl.tail = r.host_tail_shadow;

        MetalContext::instance().get_cluster().write_core(
            device->id(),
            device->worker_core_from_logical_core(r.core),
            std::vector<uint32_t>(
                reinterpret_cast<uint32_t*>(&ctrl),
                reinterpret_cast<uint32_t*>(&ctrl) + (sizeof(DramJobQueueCtrl) / sizeof(uint32_t))),
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
    out.per_core_results.reserve(static_cast<size_t>(total_jobs));

    constexpr auto kMonitorPrintInterval = std::chrono::seconds(2);
    constexpr auto kStallWatchdogTimeout = std::chrono::seconds(10);
    constexpr auto kWatchdogGraceExitDelay = std::chrono::seconds(3);

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

        for (size_t i = 0; i < per_core.size(); ++i) {
            auto& rr = per_core[i];
            const auto& rr_jobs = jobs_per_core[i];

            auto raw_status = MetalContext::instance().get_cluster().read_core(
                device->id(),
                device->worker_core_from_logical_core(rr.core),
                rr.status_l1_addr,
                sizeof(CoreProgressStatus));

            const CoreProgressStatus* status = reinterpret_cast<const CoreProgressStatus*>(raw_status.data());

            completed += std::min<uint64_t>(
                static_cast<uint64_t>(status->jobs_completed), static_cast<uint64_t>(rr_jobs.size()));
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

                    const CoreProgressStatus* status = reinterpret_cast<const CoreProgressStatus*>(raw_status.data());

                    completed_jobs_total += std::min<uint64_t>(
                        static_cast<uint64_t>(status->jobs_completed), static_cast<uint64_t>(rr_jobs.size()));

                    const bool core_done = status->jobs_completed >= rr_jobs.size();

                    const bool core_progressing = (status->heartbeat_tick != rr.prev_heartbeat_tick) ||
                                                  (status->jobs_completed != rr.prev_jobs_completed);

                    all_cores_done &= core_done;

                    if (!core_done && !core_progressing) {
                        all_cores_progressing = false;
                    }
                }

                if (global_monitor_initialized) {
                    const uint64_t arc_delta = arc_tick - global_prev_arc_tick;

                    log_info(
                        tt::LogTest,
                        "monitor: arc={} delta={} {} jobs={}/{}",
                        arc_tick,
                        arc_delta,
                        (all_cores_progressing || all_cores_done) ? "all cores progressing"
                                                                  : "some cores not progressing",
                        completed_jobs_total,
                        total_jobs);
                } else {
                    log_info(
                        tt::LogTest,
                        "monitor: arc={} {} jobs={}/{}",
                        arc_tick,
                        (all_cores_progressing || all_cores_done) ? "all cores progressing"
                                                                  : "some cores not progressing",
                        completed_jobs_total,
                        total_jobs);

                    global_monitor_initialized = true;
                }

                global_prev_arc_tick = arc_tick;
                global_last_monitor_print_time = monitor_now;
            }

            for (size_t core_idx = 0; core_idx < per_core.size(); ++core_idx) {
                auto& r = per_core[core_idx];
                const auto& core_jobs = jobs_per_core[core_idx];

                auto raw_status = MetalContext::instance().get_cluster().read_core(
                    device->id(),
                    device->worker_core_from_logical_core(r.core),
                    r.status_l1_addr,
                    sizeof(CoreProgressStatus));

                const CoreProgressStatus* status = reinterpret_cast<const CoreProgressStatus*>(raw_status.data());

                const auto now = std::chrono::steady_clock::now();

                if ((now - r.last_monitor_print_time) >= kMonitorPrintInterval) {
                    const uint64_t arc_tick = read_arc_global_tick(device);

                    const uint32_t hb_delta = status->heartbeat_tick - r.prev_heartbeat_tick;

                    const uint32_t jobs_delta = status->jobs_completed - r.prev_jobs_completed;

                    const uint64_t arc_delta = arc_tick - r.prev_arc_tick;

                    if (print_per_core_monitor) {
                        log_info(
                            tt::LogTest,
                            "monitor: core=({}, {}) jobs={}/{} hb={} delta={} arc={} delta={} stage={} job_id={}",
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

                    const bool tensix_progress = (hb_delta != 0u) || (jobs_delta != 0u);

                    const bool arc_progress = (arc_delta != 0u);

                    if (r.monitor_initialized && !core_done && !g_stop_requested.load()) {
                        uint32_t stall_reason = 0;

                        if (!tensix_progress && arc_progress) {
                            stall_reason = 1;
                        } else if (!tensix_progress && !arc_progress) {
                            stall_reason = 2;
                        }

                        if (stall_reason == 0) {
                            if (r.stall_watchdog_armed) {
                                log_info(
                                    tt::LogTest,
                                    "watchdog disarmed: core=({}, {}) progress resumed",
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
                                    "watchdog armed: core=({}, {}) reason={} jobs={}/{}",
                                    r.core.x,
                                    r.core.y,
                                    dram_watchdog_reason_name(stall_reason),
                                    status->jobs_completed,
                                    core_jobs.size());

                            } else if ((now - r.stall_watchdog_start_time) >= kStallWatchdogTimeout) {
                                log_info(
                                    tt::LogTest,
                                    "watchdog timeout: core=({}, {}) reason={} stuck_for={}s jobs={}/{} hb={} arc={} "
                                    "stage={} job_id={}; requesting graceful stop. If the test does not exit, "
                                    "terminate the process manually. If the next run cannot acquire the device or "
                                    "topology mapping fails, run: tt-smi -r",
                                    r.core.x,
                                    r.core.y,
                                    dram_watchdog_reason_name(r.stall_watchdog_reason),
                                    std::chrono::duration_cast<std::chrono::seconds>(now - r.stall_watchdog_start_time)
                                        .count(),
                                    status->jobs_completed,
                                    core_jobs.size(),
                                    status->heartbeat_tick,
                                    arc_tick,
                                    status->current_stage,
                                    status->current_job_id);

                                out.summary.pass = false;
                                g_watchdog_requested.store(true);
                                g_stop_requested.store(true);

                                std::thread([core = r.core, delay = kWatchdogGraceExitDelay]() {
                                    std::this_thread::sleep_for(delay);

                                    log_info(
                                        tt::LogTest,
                                        "watchdog hard-exit: core=({}, {}) exiting process after grace period",
                                        core.x,
                                        core.y);

                                    std::_Exit(2);
                                }).detach();
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

                    bool result_ready = false;

                    uint32_t last_heartbeat = status->heartbeat_tick;
                    auto last_heartbeat_change_time = std::chrono::steady_clock::now();

                    while (!result_ready) {
                        if (g_stop_requested.load()) {
                            broadcast_stop_to_all_cores();
                            break;
                        }

                        const auto monitor_now = std::chrono::steady_clock::now();

                        if ((monitor_now - global_last_monitor_print_time) >= kMonitorPrintInterval) {
                            const uint64_t arc_tick = read_arc_global_tick(device);
                            const uint64_t completed_jobs_total = get_completed_jobs_total();

                            if (global_monitor_initialized) {
                                const uint64_t arc_delta = arc_tick - global_prev_arc_tick;

                                log_info(
                                    tt::LogTest,
                                    "monitor: arc={} delta={} all cores progressing jobs={}/{}",
                                    arc_tick,
                                    arc_delta,
                                    completed_jobs_total,
                                    total_jobs);
                            } else {
                                log_info(
                                    tt::LogTest,
                                    "monitor: arc={} all cores progressing jobs={}/{}",
                                    arc_tick,
                                    completed_jobs_total,
                                    total_jobs);

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

                        result_copy = *reinterpret_cast<const DramBaseResult*>(raw_result.data());

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

                        const CoreProgressStatus* status_poll =
                            reinterpret_cast<const CoreProgressStatus*>(raw_status_poll.data());

                        if (status_poll->heartbeat_tick != last_heartbeat) {
                            last_heartbeat = status_poll->heartbeat_tick;
                            last_heartbeat_change_time = std::chrono::steady_clock::now();
                        }

                        const auto now_poll = std::chrono::steady_clock::now();

                        if ((now_poll - last_heartbeat_change_time) >= kNoHeartbeatTimeout) {
                            log_info(
                                tt::LogTest,
                                "result wait timeout: core={} job={} slot={} no heartbeat change for {}s "
                                "expected(job_id={}, pattern={}, pass={}, repeat={}, bank={}) got(job_id={}, "
                                "pattern={}, pass={}, repeat={}, bank={}, words={}, transfers={})",
                                core_idx,
                                done_index,
                                done_slot,
                                std::chrono::duration_cast<std::chrono::seconds>(now_poll - last_heartbeat_change_time)
                                    .count(),
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

                        std::this_thread::sleep_for(kResultPollDelay);
                    }

                    if (g_stop_requested.load()) {
                        break;
                    }

                    if (!result_ready) {
                        log_info(
                            tt::LogTest,
                            "result not ready: core={} job={} slot={} expected(job_id={}, pattern={}, pass={}, "
                            "repeat={}, bank={}) got(job_id={}, pattern={}, pass={}, repeat={}, bank={}, words={}, "
                            "transfers={})",
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
                            "job {}/{} failed: core=({}, {}) bank={} pattern={} pass={} repeat={} kind={} "
                            "first_fail_addr=0x{:08x} write_err={:.6f}% read_err={:.6f}%",
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
                        const DramWorkItem& next_job = core_jobs[r.jobs_enqueued];

                        MetalContext::instance().get_cluster().write_core(
                            device->id(),
                            device->worker_core_from_logical_core(r.core),
                            std::vector<uint32_t>(
                                reinterpret_cast<const uint32_t*>(&next_job),
                                reinterpret_cast<const uint32_t*>(&next_job) +
                                    (sizeof(DramWorkItem) / sizeof(uint32_t))),
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
        log_info(tt::LogTest, "Starting persistent DRAM test: workers={}, total_jobs={}", per_core.size(), total_jobs);
    }

    fixture->RunProgram(mesh_device, workload, false);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    log_info(
        tt::LogTest,
        "Persistent DRAM test finished: workers={}, total_jobs={}, duration={}",
        per_core.size(),
        total_jobs,
        format_duration_seconds(duration_sec));

    if (refill_thread.joinable()) {
        refill_thread.join();
    }

    fixture->FinishCommands(mesh_device);

    return out;
}

}  // namespace tt::tt_metal
