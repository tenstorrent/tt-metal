// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// CB memory tracking: what it reports, and what it costs.
//
// Device::get_total_cb_allocated() used to recompute the device-wide CB footprint on every
// program registration, walking every live program and expanding per core. Programs only
// leave that set in ~ProgramImpl, so a host holding many programs alive -- exactly what an
// op-level program cache does across a parameter sweep -- made each call O(live programs) and
// the whole sweep O(N^2). It also answered the wrong question: a registered program
// occupies no CB space until it is dispatched.
//
// The tests here cover both halves. The correctness ones check the reported figure against
// the CB config table read back from the device's own L1 (and DRAM/L1 buffers against the
// allocator), under fast dispatch, slow dispatch and trace replay. The cost one keeps N
// programs alive and asserts that a tracking query does not get more expensive as N grows;
// it creates no kernels, since kernel compilation is disk-cached and not what is measured.
//
// Run just this file's tests:
//   ./build/test/tt_metal/unit_tests_api --gtest_filter='*CircularBufferTracking*:*BufferTracking*'
// The heavier perf characterisations are DISABLED_ and need --gtest_also_run_disabled_tests.

#include <algorithm>
#include <thread>
#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "circular_buffer_test_utils.hpp"
#include "device_fixture.hpp"
#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"

// Internal APIs under measurement.
#include "impl/device/device_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
#include <tt-metalium/buffer.hpp>

using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

namespace {

constexpr uint32_t kCbsPerProgram = 4;

// Tracking is unusable either because Device built no provider (TT_METAL_SHM_TRACKING_DISABLED)
// or because the provider could not attach. Both must skip: the second is a property of the
// machine's /dev/shm, not of the code under test.
const char* kTrackingUnavailable =
    "SHM tracking is not available on this device: either disabled via "
    "TT_METAL_SHM_TRACKING_DISABLED, or the provider could not attach to the region (a stale "
    "/dev/shm/tt_device_*_memory from an older build can cause this; see the Metal log)";

bool tracking_available(const Device* device) {
    const auto* provider = device->get_shm_stats_provider();
    return provider != nullptr && provider->is_initialized();
}

// Median of a timing sample, to keep the assertion robust against scheduler noise.
double median_ms(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Cost of one tracking query. Each sample times a batch and divides: once the total is
// incrementally maintained a query is an atomic load, and a single-call measurement rounds to
// zero -- making the ratio below a division by zero rather than the ~1x it should report.
constexpr int kQueriesPerSample = 500;

double time_tracking_query_ms(const Device* device, int samples_wanted = 5) {
    std::vector<double> samples;
    samples.reserve(samples_wanted);
    for (int i = 0; i < samples_wanted; i++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int q = 0; q < kQueriesPerSample; q++) {
            volatile uint64_t total = device->get_total_cb_allocated();
            (void)total;
        }
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / kQueriesPerSample);
    }
    return median_ms(std::move(samples));
}

}  // namespace

// Runs by default: the regression guard for the bottleneck, sized for CI at 256 live programs
// -- already far past where recompute-per-call shows up (145x on a p150a, 115x on an n300).
// The heavier characterisations below stay opt-in.
TEST_F(AnyDispatchMeshDeviceSingleCardFixture, TensixCircularBufferTrackingCostScaling) {
    auto mesh_device = devices_.at(0);
    auto* idevice = mesh_device->get_devices().at(0);
    auto* device = dynamic_cast<Device*>(idevice);
    ASSERT_NE(device, nullptr) << "expected a concrete Device to query CB tracking on";

    if (!tracking_available(device)) {
        GTEST_SKIP() << kTrackingUnavailable << ". Tracking must be on to reproduce the bottleneck";
    }

    // CBs over the whole compute grid, which is what a real op does.
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRange full_grid(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
    const CoreRangeSet cr_set({full_grid});
    const CBConfig cb_config;

    log_info(
        tt::LogTest,
        "device {}: compute grid {}x{} ({} cores), {} CBs/program",
        device->id(),
        grid.x,
        grid.y,
        grid.x * grid.y,
        kCbsPerProgram);

    // Programs are held alive for the whole test, mirroring an op-level program cache
    // holding one entry per parameter variant.
    std::vector<Program> cached_programs;
    cached_programs.reserve(256);

    const std::vector<size_t> checkpoints = {1, 16, 64, 256};
    const size_t max_programs = checkpoints.back();

    double baseline_ms = 0.0;
    double final_ms = 0.0;
    size_t next_checkpoint = 0;

    log_info(tt::LogTest, "{:>10} {:>18} {:>14}", "programs", "query (ms, median)", "vs baseline");

    for (size_t n = 1; n <= max_programs; n++) {
        Program program;
        for (uint32_t cb_id = 0; cb_id < kCbsPerProgram; cb_id++) {
            CircularBufferConfig config = CircularBufferConfig(cb_config.page_size, {{cb_id, cb_config.data_format}})
                                              .set_page_size(cb_id, cb_config.page_size);
            CreateCircularBuffer(program, cr_set, config);
        }

        // Lays out the CBs. This is the step that used to register the program with the
        // device, putting it in scope for every later tracking query.
        program.impl().allocate_circular_buffers(idevice);
        cached_programs.push_back(std::move(program));

        if (next_checkpoint < checkpoints.size() && n == checkpoints[next_checkpoint]) {
            const double ms = time_tracking_query_ms(device);
            if (next_checkpoint == 0) {
                baseline_ms = ms;
            }
            final_ms = ms;
            log_info(tt::LogTest, "{:>10} {:>18.3f} {:>13.1f}x", n, ms, baseline_ms > 0 ? ms / baseline_ms : 0.0);
            next_checkpoint++;
        }
    }

    ASSERT_GT(baseline_ms, 0.0) << "baseline query was unmeasurably fast even over " << kQueriesPerSample
                                << " queries; raise kQueriesPerSample";

    // The tracked total is incrementally maintained, so a query is an atomic load and its
    // cost does not depend on how many programs are alive. Recomputing it per call instead
    // makes this ratio track the program count: 145x at 256 live programs, measured on a
    // p150a against the code this replaced (115x on an n300).
    const double growth = final_ms / baseline_ms;
    log_info(tt::LogTest, "tracking query cost grew {:.1f}x between 1 and {} live programs", growth, max_programs);
    EXPECT_LT(growth, 10.0) << "Device::get_total_cb_allocated() cost scales with the number of live "
                               "programs ("
                            << growth << "x from 1 to " << max_programs
                            << " programs). "
                               "CB tracking must be incrementally maintained, not recomputed per call.";
}

// Measures the *user-visible* cost: total wall-clock to register N programs'
// circular buffers, which is what a warmup sweep actually pays.
//
// Distinct from the query-cost test above because it is paid per registration rather than
// per query. Against the code this replaced it also understates the damage: that version
// admitted at most one query per 100 ms per device, so a tight loop like this one diluted the
// cost that a compile-paced warmup pays in full. The paced test below is the honest one.
//
// Run twice to compare:
//   unit_tests_api --gtest_filter='*CircularBufferRegistrationWallClock*' \
//     --gtest_also_run_disabled_tests
//   TT_METAL_SHM_TRACKING_DISABLED=1 unit_tests_api --gtest_filter=... (same)
TEST_F(AnyDispatchMeshDeviceSingleCardFixture, DISABLED_TensixCircularBufferRegistrationWallClock) {
    auto mesh_device = devices_.at(0);
    auto* idevice = mesh_device->get_devices().at(0);
    auto* device = dynamic_cast<Device*>(idevice);
    ASSERT_NE(device, nullptr);

    const bool tracking_on = device->get_shm_stats_provider() != nullptr;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRange full_grid(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
    const CoreRangeSet cr_set({full_grid});
    const CBConfig cb_config;

    constexpr size_t kPrograms = 4096;
    std::vector<Program> cached_programs;
    cached_programs.reserve(kPrograms);

    // Pre-build the CB configs so only registration is timed.
    auto t_start = std::chrono::steady_clock::now();
    std::vector<double> window_ms;
    double total_ms = 0.0;

    log_info(
        tt::LogTest,
        "SHM tracking is {} -- registering {} programs on a {}x{} grid",
        tracking_on ? "ON" : "OFF",
        kPrograms,
        grid.x,
        grid.y);
    log_info(tt::LogTest, "{:>10} {:>20} {:>16}", "programs", "window mean (ms)", "elapsed (s)");

    for (size_t n = 1; n <= kPrograms; n++) {
        Program program;
        for (uint32_t cb_id = 0; cb_id < kCbsPerProgram; cb_id++) {
            CircularBufferConfig config = CircularBufferConfig(cb_config.page_size, {{cb_id, cb_config.data_format}})
                                              .set_page_size(cb_id, cb_config.page_size);
            CreateCircularBuffer(program, cr_set, config);
        }

        auto t0 = std::chrono::steady_clock::now();
        program.impl().allocate_circular_buffers(idevice);
        auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        window_ms.push_back(ms);
        cached_programs.push_back(std::move(program));

        if (n % 512 == 0) {
            const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            double sum = 0.0;
            for (double v : window_ms) {
                sum += v;
            }
            log_info(tt::LogTest, "{:>10} {:>20.4f} {:>16.2f}", n, sum / window_ms.size(), elapsed_s);
            window_ms.clear();
        }
    }

    log_info(
        tt::LogTest,
        "TOTAL registration time for {} programs with tracking {}: {:.1f} ms",
        kPrograms,
        tracking_on ? "ON" : "OFF",
        total_ms);
}

// Cost per registration with 4096 programs already live and registrations arriving at a
// compile-like cadence -- the regime #52010 reports, and the one a tight loop hides. Measured
// on one p150a: 67.9 ms per registration before this rework, 0.007 ms after, which is what
// tracking-disabled also costs.
TEST_F(AnyDispatchMeshDeviceSingleCardFixture, DISABLED_TensixCircularBufferPacedRegistrationCost) {
    auto mesh_device = devices_.at(0);
    auto* idevice = mesh_device->get_devices().at(0);
    auto* device = dynamic_cast<Device*>(idevice);
    ASSERT_NE(device, nullptr);

    const bool tracking_on = device->get_shm_stats_provider() != nullptr;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRange full_grid(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
    const CoreRangeSet cr_set({full_grid});
    const CBConfig cb_config;

    constexpr size_t kWarmup = 4096;                           // live programs already in the cache
    constexpr size_t kPaced = 25;                              // additional registrations, paced
    constexpr auto kCadence = std::chrono::milliseconds(110);  // stands in for kernel compilation

    std::vector<Program> cached_programs;
    cached_programs.reserve(kWarmup + kPaced);

    auto make_program = [&]() {
        Program program;
        for (uint32_t cb_id = 0; cb_id < kCbsPerProgram; cb_id++) {
            CircularBufferConfig config = CircularBufferConfig(cb_config.page_size, {{cb_id, cb_config.data_format}})
                                              .set_page_size(cb_id, cb_config.page_size);
            CreateCircularBuffer(program, cr_set, config);
        }
        return program;
    };

    for (size_t n = 0; n < kWarmup; n++) {
        Program p = make_program();
        p.impl().allocate_circular_buffers(idevice);
        cached_programs.push_back(std::move(p));
    }

    log_info(
        tt::LogTest,
        "SHM tracking {} -- {} live programs, now registering {} more at a {} ms cadence",
        tracking_on ? "ON" : "OFF",
        kWarmup,
        kPaced,
        kCadence.count());

    std::vector<double> paced_ms;
    paced_ms.reserve(kPaced);
    for (size_t n = 0; n < kPaced; n++) {
        Program p = make_program();
        std::this_thread::sleep_for(kCadence);  // stand in for kernel compilation

        auto t0 = std::chrono::steady_clock::now();
        p.impl().allocate_circular_buffers(idevice);
        auto t1 = std::chrono::steady_clock::now();

        paced_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        cached_programs.push_back(std::move(p));
    }

    const double med = median_ms(paced_ms);
    double sum = 0.0;
    for (double v : paced_ms) {
        sum += v;
    }
    log_info(
        tt::LogTest,
        "tracking {}: paced registration cost median {:.4f} ms, mean {:.4f} ms, total {:.1f} ms over {} programs",
        tracking_on ? "ON" : "OFF",
        med,
        sum / paced_ms.size(),
        sum,
        kPaced);
}

// Cost of the tracking hook on the dispatch path, which is where the work moved to.
//
// Re-dispatching the same program hits a fast path -- the footprint is compared by pointer
// and nothing is recomputed -- so a benchmark that loops on one program measures nothing.
// This alternates two programs with different CB footprints, so every dispatch changes the
// resident set and pays the full cost: apply the per-core footprint, then publish to shared
// memory. That is the shape of a real workload, where consecutive dispatches are different
// ops.
//
// Run twice to compare:
//   unit_tests_api --gtest_filter='*DispatchOverhead*' --gtest_also_run_disabled_tests
//   TT_METAL_SHM_TRACKING_DISABLED=1 unit_tests_api --gtest_filter=... (same)
TEST_F(UnitMeshCQSingleCardProgramFixture, DISABLED_TensixCircularBufferTrackingDispatchOverhead) {
    for (auto& mesh_device : this->devices_) {
        auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
        ASSERT_NE(device, nullptr);
        const bool tracking_on = device->get_shm_stats_provider() != nullptr;

        const CoreCoord grid = device->compute_with_storage_grid_size();
        const CoreRangeSet cr_set({CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1))});
        const CBConfig cb_config;
        auto zero = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero, zero);
        auto& cq = mesh_device->mesh_command_queue();

        // Two programs whose CB footprints differ, so the resident set changes every dispatch.
        auto make_workload = [&](uint32_t size_multiple) {
            auto workload = std::make_shared<distributed::MeshWorkload>();
            Program p;
            workload->add_program(device_range, std::move(p));
            auto& prog = workload->get_programs().at(device_range);
            for (uint32_t cb_id = 0; cb_id < 4; cb_id++) {
                CircularBufferConfig config =
                    CircularBufferConfig(cb_config.page_size * size_multiple, {{cb_id, cb_config.data_format}})
                        .set_page_size(cb_id, cb_config.page_size);
                CreateCircularBuffer(prog, cr_set, config);
            }
            initialize_program(prog, cr_set);
            return workload;
        };
        auto workload_a = make_workload(1);
        auto workload_b = make_workload(4);

        // Warm up: compile, populate caches, and let the first dispatches settle.
        for (int i = 0; i < 4; i++) {
            distributed::EnqueueMeshWorkload(cq, *workload_a, false);
            distributed::EnqueueMeshWorkload(cq, *workload_b, false);
        }
        distributed::Finish(cq);

        // Host-side enqueue cost is what the hook adds to; finish once at the end so device
        // execution does not dominate the measurement.
        constexpr int kDispatches = 2000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kDispatches; i++) {
            distributed::EnqueueMeshWorkload(cq, (i % 2) ? *workload_b : *workload_a, false);
        }
        auto t1 = std::chrono::steady_clock::now();
        distributed::Finish(cq);

        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        log_info(
            tt::LogTest,
            "device {}: SHM tracking {} -- {} alternating dispatches, {:.3f} us per enqueue ({:.1f} ms total, "
            "{} cores, 4 CBs/program)",
            device->id(),
            tracking_on ? "ON" : "OFF",
            kDispatches,
            total_ms * 1000.0 / kDispatches,
            total_ms,
            grid.x * grid.y);
    }
}

// ---------------------------------------------------------------------------
// Ground-truth check: does the tracked CB figure match what is actually
// configured on the device?
//
// Everything else in this file compares the tracking code against itself. This
// test compares it against the device: after a workload is dispatched, each
// core's CB config table in L1 holds [address, size, num_pages, page_size] per
// buffer index (see dispatch.cpp:1302). Reading it back gives the CB byte ranges
// the hardware is actually using, independent of any host-side bookkeeping.
//
// Note the allocator cannot serve as ground truth here: locally-allocated CBs are
// program-managed, not allocator-managed, so the allocator does not know about them.
// The L1 config table is the only authoritative source.
// ---------------------------------------------------------------------------
namespace {

// Sum, per core, of the union of [address, address+size) over all configured CBs.
// `configured_indices` must list the buffer indices the program actually created.
// It cannot be recovered from L1: the dispatch command only writes the slots in use
// (dispatch.cpp:1302), so unused slots hold stale data from whatever ran before --
// they are NOT zeroed, and a naive scan of all max_cbs slots reads garbage sizes.
uint64_t read_cb_bytes_from_device_l1(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    detail::ProgramImpl& program,
    const CoreRangeSet& cr_set,
    const std::vector<uint32_t>& configured_indices) {
    auto* idevice = mesh_device->get_devices().at(0);
    const uint32_t max_cbs = MetalContext::instance().hal().get_arch_num_circular_buffers();
    const uint32_t cb_config_bytes = max_cbs * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    uint64_t total = 0;
    for (const CoreRange& core_range : cr_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                const CoreCoord core(x, y);
                const auto base = program.get_cb_base_addr(idevice, core, tt::CoreType::WORKER);

                std::vector<uint32_t> cb_config;
                tt::tt_metal::detail::ReadFromDeviceL1(idevice, core, base, cb_config_bytes, cb_config);

                // Collect [start,end) for the buffer indices this program configured.
                std::vector<std::pair<uint64_t, uint64_t>> regions;
                for (const uint32_t cb_id : configured_indices) {
                    const uint32_t i = UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * cb_id;
                    const uint64_t addr = cb_config.at(i);
                    const uint64_t size = cb_config.at(i + 1);
                    if (size == 0) {
                        continue;
                    }
                    regions.emplace_back(addr, addr + size);
                }
                if (regions.empty()) {
                    continue;
                }
                // Union, matching how get_total_cb_allocated() accounts for address reuse.
                std::sort(regions.begin(), regions.end());
                std::vector<std::pair<uint64_t, uint64_t>> merged{regions.front()};
                for (size_t i = 1; i < regions.size(); i++) {
                    auto& last = merged.back();
                    if (regions[i].first <= last.second) {
                        last.second = std::max(last.second, regions[i].second);
                    } else {
                        merged.push_back(regions[i]);
                    }
                }
                for (const auto& [s, e] : merged) {
                    total += e - s;
                }
            }
        }
    }
    return total;
}

}  // namespace

// The checks below are written as helpers so that each one runs under BOTH dispatch
// modes. That matters because the two hooks that feed this figure live in different
// places: fast dispatch records CB residency from the enqueue loop in
// fd_mesh_command_queue.cpp, slow dispatch from LaunchProgram in tt_metal.cpp. A test
// bound to a single fixture would leave one of the two entirely unverified against the
// device -- and fast dispatch is the default path.
namespace {

void check_cb_tracking_matches_device_l1(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
    ASSERT_NE(device, nullptr);
    if (!tracking_available(device)) {
        GTEST_SKIP() << kTrackingUnavailable << "; nothing to validate against";
    }

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRange cr(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
    const CoreRangeSet cr_set({cr});
    const CBConfig cb_config;

    auto zero = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero, zero);
    distributed::MeshWorkload workload;
    Program program;
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    // A few CBs of differing sizes, so a wrong answer is unlikely to coincide.
    uint64_t expected_bytes_per_core = 0;
    std::vector<uint32_t> configured_indices;
    for (uint32_t cb_id = 0; cb_id < 4; cb_id++) {
        configured_indices.push_back(cb_id);
        const uint32_t total_size = cb_config.page_size * (cb_id + 1);
        CircularBufferConfig config = CircularBufferConfig(total_size, {{cb_id, cb_config.data_format}})
                                          .set_page_size(cb_id, cb_config.page_size);
        CreateCircularBuffer(program_, cr_set, config);
        expected_bytes_per_core += total_size;
    }
    initialize_program(program_, cr_set);

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    const uint64_t from_device = read_cb_bytes_from_device_l1(mesh_device, program_.impl(), cr_set, configured_indices);
    const uint64_t from_tracking = device->get_total_cb_allocated();
    const uint64_t analytic = expected_bytes_per_core * grid.x * grid.y;

    log_info(
        tt::LogTest,
        "device {}: CB bytes -- device L1 readback={}, tracking={}, analytic={} ({} cores x {} B)",
        device->id(),
        from_device,
        from_tracking,
        analytic,
        grid.x * grid.y,
        expected_bytes_per_core);

    // The device is the authority: CBs pack contiguously from the CB base, so the union
    // of what the hardware has configured must equal the sum of CB sizes.
    EXPECT_EQ(from_device, analytic) << "device's own CB config table disagrees with the requested CB sizes";

    EXPECT_EQ(from_tracking, from_device)
        << "tracked CB total (" << from_tracking << ") disagrees with the device's CB config table (" << from_device
        << ") for a single live program";

    // ...and that the value actually reaches shared memory, which is the entire point of
    // the feature: an in-process number nobody can read is not memory tracking. Publishing
    // happens on the dispatching thread as the figure changes, so this should be immediate;
    // the retry loop only keeps the test from being a race if that ever stops being true.
    auto* provider = device->get_shm_stats_provider();
    ASSERT_NE(provider, nullptr);
    uint64_t from_shm = 0;
    for (int attempt = 0; attempt < 50; attempt++) {
        from_shm = provider->get_device_stats().cb_allocated;
        if (from_shm == from_device) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_EQ(from_shm, from_device) << "CB total never reached shared memory (" << from_shm << " vs " << from_device
                                     << "); tt-smi/tt-mgmt would report the wrong value";
}

void check_cb_tracking_is_current_not_peak(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* idevice = mesh_device->get_devices().at(0);
    auto* device = dynamic_cast<Device*>(idevice);
    ASSERT_NE(device, nullptr);
    if (!tracking_available(device)) {
        GTEST_SKIP() << kTrackingUnavailable;
    }

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRange cr(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
    const CoreRangeSet cr_set({cr});
    const CBConfig cb_config;
    const uint32_t num_cores = grid.x * grid.y;

    // Program A: small CBs. This is the one we actually dispatch.
    auto zero = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero, zero);
    distributed::MeshWorkload workload;
    Program program_a;
    workload.add_program(device_range, std::move(program_a));
    auto& a = workload.get_programs().at(device_range);

    std::vector<uint32_t> indices;
    uint64_t a_bytes_per_core = 0;
    for (uint32_t cb_id = 0; cb_id < 2; cb_id++) {
        indices.push_back(cb_id);
        CircularBufferConfig config = CircularBufferConfig(cb_config.page_size, {{cb_id, cb_config.data_format}})
                                          .set_page_size(cb_id, cb_config.page_size);
        CreateCircularBuffer(a, cr_set, config);
        a_bytes_per_core += cb_config.page_size;
    }
    initialize_program(a, cr_set);

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    const uint64_t device_view = read_cb_bytes_from_device_l1(mesh_device, a.impl(), cr_set, indices);
    const uint64_t tracked_one = device->get_total_cb_allocated();
    EXPECT_EQ(device_view, a_bytes_per_core * num_cores);
    EXPECT_EQ(tracked_one, device_view) << "single live program should match the device";

    // Program B: 8x larger CBs, CB layout computed but NEVER dispatched, held alive the
    // way a program cache entry would be.
    Program program_b;
    uint64_t b_bytes_per_core = 0;
    for (uint32_t cb_id = 0; cb_id < 2; cb_id++) {
        const uint32_t total_size = cb_config.page_size * 8;
        CircularBufferConfig config = CircularBufferConfig(total_size, {{cb_id, cb_config.data_format}})
                                          .set_page_size(cb_id, cb_config.page_size);
        CreateCircularBuffer(program_b, cr_set, config);
        b_bytes_per_core += total_size;
    }
    program_b.impl().allocate_circular_buffers(idevice);

    const uint64_t device_view_after = read_cb_bytes_from_device_l1(mesh_device, a.impl(), cr_set, indices);
    const uint64_t tracked_two = device->get_total_cb_allocated();

    log_info(
        tt::LogTest,
        "device {}: device L1 (unchanged, only A dispatched)={}, tracked after caching B={} "
        "(A={}/core, B={}/core, {} cores)",
        device->id(),
        device_view_after,
        tracked_two,
        a_bytes_per_core,
        b_bytes_per_core,
        num_cores);

    EXPECT_EQ(device_view_after, device_view) << "dispatching nothing must not change the device's CB config";

    // Caching B must NOT move the tracked figure: B occupies no CB space until it is
    // dispatched. Before the dispatch-time hook this reported B's (8x larger) footprint,
    // because program registration -- not dispatch -- drove the accounting.
    EXPECT_EQ(tracked_two, device_view_after)
        << "caching an undispatched program changed the reported CB total; the figure is tracking "
           "program registration rather than what is resident on the device";
    EXPECT_GT(b_bytes_per_core, a_bytes_per_core) << "test is only meaningful if B is larger than A";
}

}  // namespace

// --- slow dispatch: exercises the LaunchProgram hook in tt_metal.cpp ---------------
TEST_F(MeshDeviceFixture, TensixCircularBufferTrackingMatchesDeviceL1) {
    for (auto& mesh_device : this->devices_) {
        check_cb_tracking_matches_device_l1(mesh_device);
    }
}

TEST_F(MeshDeviceFixture, TensixCircularBufferTrackingIsCurrentNotPeak) {
    for (auto& mesh_device : this->devices_) {
        check_cb_tracking_is_current_not_peak(mesh_device);
    }
}

// --- fast dispatch: exercises the enqueue hook in fd_mesh_command_queue.cpp --------
// This is the default dispatch path, so leaving it unverified against the device was
// the larger gap of the two.
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixCircularBufferTrackingMatchesDeviceL1FastDispatch) {
    for (auto& mesh_device : this->devices_) {
        check_cb_tracking_matches_device_l1(mesh_device);
    }
}

TEST_F(UnitMeshCQSingleCardProgramFixture, TensixCircularBufferTrackingIsCurrentNotPeakFastDispatch) {
    for (auto& mesh_device : this->devices_) {
        check_cb_tracking_is_current_not_peak(mesh_device);
    }
}

// Trace replay records nothing per-program on the host: enqueue_trace plays back a
// captured command stream. So the CB footprint a trace produces is computed once during
// capture and carried on the MeshTraceDescriptor, then re-applied on every replay.
// Without that, traced execution reports whatever was dispatched before the replay --
// which is the common case in production, where inference runs are trace-based.
TEST_F(UnitMeshCQSingleCardTraceFixture, TensixCircularBufferTrackingAcrossTraceReplay) {
    for (auto& mesh_device : this->devices_) {
        auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
        ASSERT_NE(device, nullptr);
        if (!tracking_available(device)) {
            GTEST_SKIP() << kTrackingUnavailable;
        }

        const CoreCoord grid = device->compute_with_storage_grid_size();
        const CoreRange cr(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
        const CoreRangeSet cr_set({cr});
        const CBConfig cb_config;
        const uint32_t num_cores = grid.x * grid.y;
        auto zero = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero, zero);
        auto& cq = mesh_device->mesh_command_queue();

        auto make_workload = [&](uint32_t mult, std::vector<uint32_t>& indices, uint64_t& bytes_per_core) {
            auto workload = std::make_shared<distributed::MeshWorkload>();
            Program p;
            workload->add_program(device_range, std::move(p));
            auto& prog = workload->get_programs().at(device_range);
            bytes_per_core = 0;
            for (uint32_t cb_id = 0; cb_id < 2; cb_id++) {
                indices.push_back(cb_id);
                const uint32_t total_size = cb_config.page_size * mult;
                CircularBufferConfig config = CircularBufferConfig(total_size, {{cb_id, cb_config.data_format}})
                                                  .set_page_size(cb_id, cb_config.page_size);
                CreateCircularBuffer(prog, cr_set, config);
                bytes_per_core += total_size;
            }
            initialize_program(prog, cr_set);
            return workload;
        };

        std::vector<uint32_t> t_indices;
        uint64_t t_bytes_per_core = 0;
        auto workload_t = make_workload(4, t_indices, t_bytes_per_core);
        std::vector<uint32_t> a_indices;
        uint64_t a_bytes_per_core = 0;
        auto workload_a = make_workload(1, a_indices, a_bytes_per_core);

        // Trace capture cannot load new binaries ("warm up before capturing a trace"), so
        // T must run once normally first.
        distributed::EnqueueMeshWorkload(cq, *workload_t, false);
        distributed::Finish(cq);
        const uint64_t tracked_t = device->get_total_cb_allocated();
        EXPECT_EQ(tracked_t, t_bytes_per_core * num_cores);

        const auto tid = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
        distributed::EnqueueMeshWorkload(cq, *workload_t, false);
        mesh_device->end_mesh_trace(cq.id(), tid);

        // Capture executes nothing, so residency must not move.
        EXPECT_EQ(device->get_total_cb_allocated(), tracked_t)
            << "trace capture changed reported CB usage, but capture dispatches nothing";

        // Displace residency with a smaller program dispatched the normal way.
        distributed::EnqueueMeshWorkload(cq, *workload_a, false);
        distributed::Finish(cq);
        const uint64_t tracked_a = device->get_total_cb_allocated();
        EXPECT_EQ(tracked_a, a_bytes_per_core * num_cores);
        ASSERT_NE(tracked_a, tracked_t) << "A and T must differ for this test to mean anything";

        // Replay: the trace rewrites T's CB config into L1, so residency must return to T.
        mesh_device->replay_mesh_trace(cq.id(), tid, true);
        distributed::Finish(cq);

        const uint64_t tracked_after_replay = device->get_total_cb_allocated();
        const uint64_t device_view_after_replay = read_cb_bytes_from_device_l1(
            mesh_device, workload_t->get_programs().at(device_range).impl(), cr_set, t_indices);

        log_info(
            tt::LogTest,
            "device {}: T={} A={} after replay: tracked={} device L1={} ({} cores)",
            device->id(),
            tracked_t,
            tracked_a,
            tracked_after_replay,
            device_view_after_replay,
            num_cores);

        // Ground truth: tracked must match what the device actually has configured.
        EXPECT_EQ(tracked_after_replay, device_view_after_replay)
            << "after trace replay the tracked CB total disagrees with the device's CB config table";
        EXPECT_EQ(tracked_after_replay, tracked_t)
            << "trace replay did not restore the traced program's CB residency; it is still reporting "
               "the program dispatched before the replay";

        mesh_device->release_mesh_trace(tid);
    }
}

// A replay must report the most CB space any of the trace's programs needs per core, not
// whichever was captured last -- invisible in the test above, whose trace holds one program.
// The device's CB config table is deliberately not the oracle: it holds the last program,
// which is exactly the value this asserts we do not report.
TEST_F(UnitMeshCQSingleCardTraceFixture, TensixCircularBufferTrackingReportsTracePeak) {
    for (auto& mesh_device : this->devices_) {
        auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
        ASSERT_NE(device, nullptr);
        if (!tracking_available(device)) {
            GTEST_SKIP() << kTrackingUnavailable;
        }

        const CoreCoord grid = device->compute_with_storage_grid_size();
        const CoreRange cr(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
        const CoreRangeSet cr_set({cr});
        const CBConfig cb_config;
        const uint32_t num_cores = grid.x * grid.y;
        auto zero = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero, zero);
        auto& cq = mesh_device->mesh_command_queue();

        auto make_workload = [&](uint32_t mult, uint64_t& bytes_per_core) {
            auto workload = std::make_shared<distributed::MeshWorkload>();
            Program p;
            workload->add_program(device_range, std::move(p));
            auto& prog = workload->get_programs().at(device_range);
            bytes_per_core = 0;
            for (uint32_t cb_id = 0; cb_id < 2; cb_id++) {
                const uint32_t total_size = cb_config.page_size * mult;
                CircularBufferConfig config = CircularBufferConfig(total_size, {{cb_id, cb_config.data_format}})
                                                  .set_page_size(cb_id, cb_config.page_size);
                CreateCircularBuffer(prog, cr_set, config);
                bytes_per_core += total_size;
            }
            initialize_program(prog, cr_set);
            return workload;
        };

        // The large program is captured FIRST, so "last captured" and "largest" differ.
        uint64_t big_bytes_per_core = 0;
        uint64_t small_bytes_per_core = 0;
        auto workload_big = make_workload(8, big_bytes_per_core);
        auto workload_small = make_workload(1, small_bytes_per_core);
        ASSERT_GT(big_bytes_per_core, small_bytes_per_core);

        // Trace capture cannot load new binaries, so both must have run once already.
        distributed::EnqueueMeshWorkload(cq, *workload_big, false);
        distributed::EnqueueMeshWorkload(cq, *workload_small, false);
        distributed::Finish(cq);

        const auto tid = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
        distributed::EnqueueMeshWorkload(cq, *workload_big, false);
        distributed::EnqueueMeshWorkload(cq, *workload_small, false);
        mesh_device->end_mesh_trace(cq.id(), tid);

        mesh_device->replay_mesh_trace(cq.id(), tid, true);
        distributed::Finish(cq);

        const uint64_t tracked = device->get_total_cb_allocated();
        const uint64_t expected_peak = big_bytes_per_core * num_cores;
        const uint64_t last_captured = small_bytes_per_core * num_cores;

        log_info(
            tt::LogTest,
            "device {}: after replay tracked={} peak={} last-captured={} ({} cores)",
            device->id(),
            tracked,
            expected_peak,
            last_captured,
            num_cores);

        EXPECT_EQ(tracked, expected_peak)
            << "a trace replay reports " << tracked << " where its largest program needs " << expected_peak
            << "; the per-core peak across the trace is what each core must accommodate";
        EXPECT_NE(tracked, last_captured) << "the reported total is the trace's last captured program ("
                                          << last_captured << "), which says nothing about what the trace needs";

        mesh_device->release_mesh_trace(tid);
    }
}

// The footprint is cached per program and a re-dispatch of the resident one is skipped by
// pointer identity. Two ways that goes stale, neither visible above where each program is
// dispatched once: alternating programs must not report the previously resident one, and
// resizing a CB must drop the cache or the figure is pinned to the old size.
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixCircularBufferTrackingFollowsProgramChanges) {
    for (auto& mesh_device : this->devices_) {
        auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
        ASSERT_NE(device, nullptr);
        if (!tracking_available(device)) {
            GTEST_SKIP() << kTrackingUnavailable;
        }

        const CoreCoord grid = device->compute_with_storage_grid_size();
        const CoreRange cr(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
        const CoreRangeSet cr_set({cr});
        const CBConfig cb_config;
        const uint32_t num_cores = grid.x * grid.y;
        auto zero = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero, zero);
        auto& cq = mesh_device->mesh_command_queue();

        // Two programs of different CB size on the same cores, plus the handle of one of A's
        // CBs so we can resize it later.
        std::vector<CBHandle> a_handles;
        auto make_workload = [&](uint32_t mult, std::vector<CBHandle>* handles, uint64_t& bytes_per_core) {
            auto workload = std::make_shared<distributed::MeshWorkload>();
            Program p;
            workload->add_program(device_range, std::move(p));
            auto& prog = workload->get_programs().at(device_range);
            bytes_per_core = 0;
            for (uint32_t cb_id = 0; cb_id < 2; cb_id++) {
                const uint32_t total_size = cb_config.page_size * mult;
                CircularBufferConfig config = CircularBufferConfig(total_size, {{cb_id, cb_config.data_format}})
                                                  .set_page_size(cb_id, cb_config.page_size);
                const CBHandle handle = CreateCircularBuffer(prog, cr_set, config);
                if (handles != nullptr) {
                    handles->push_back(handle);
                }
                bytes_per_core += total_size;
            }
            initialize_program(prog, cr_set);
            return workload;
        };

        uint64_t a_bytes_per_core = 0;
        uint64_t b_bytes_per_core = 0;
        auto workload_a = make_workload(1, &a_handles, a_bytes_per_core);
        auto workload_b = make_workload(4, nullptr, b_bytes_per_core);
        ASSERT_NE(a_bytes_per_core, b_bytes_per_core);

        auto dispatch = [&](distributed::MeshWorkload& workload) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
            distributed::Finish(cq);
            return device->get_total_cb_allocated();
        };

        // Unlike the tests above, this one dispatches several times, so the device's CB config
        // table cannot be used as the oracle: the helper reads the address reported by
        // ProgramImpl::get_cb_base_addr, which is the static KERNEL_CONFIG base rather than the
        // rotating config slot the dispatcher actually wrote (program_base_addr_on_core takes
        // get_last_slot_addr only on the MeshWorkloadImpl path). After the ring has rotated, that
        // address holds a previous program's config. The sum of the program's own CB sizes is an
        // unambiguous expectation here, and the single-dispatch tests above cover the device.
        //
        // A, then B, then A again. The last one is the interesting one: the footprint A is
        // being asked to record is no longer the one resident, even though A was resident
        // two dispatches ago.
        EXPECT_EQ(dispatch(*workload_a), a_bytes_per_core * num_cores) << "first dispatch of A";
        EXPECT_EQ(dispatch(*workload_b), b_bytes_per_core * num_cores) << "dispatch of B did not displace A";
        const uint64_t back_to_a = dispatch(*workload_a);
        EXPECT_EQ(back_to_a, a_bytes_per_core * num_cores)
            << "re-dispatching A after B still reports B's footprint; the cached per-program "
               "footprint is being treated as resident when it is not";

        auto& program_a = workload_a->get_programs().at(device_range);

        // Now resize one of A's circular buffers and dispatch it again. The cached footprint was
        // computed from the old layout and must have been dropped when the CB allocation was
        // invalidated.
        const uint32_t grown = cb_config.page_size * 8;
        UpdateCircularBufferTotalSize(program_a, a_handles.at(0), grown);
        const uint64_t expected_after_resize = grown + cb_config.page_size;  // resized CB + the other one
        const uint64_t after_resize = dispatch(*workload_a);

        log_info(
            tt::LogTest,
            "device {}: A={} B={} A-again={} A-after-resize={} (expected {}, {} cores)",
            device->id(),
            a_bytes_per_core * num_cores,
            b_bytes_per_core * num_cores,
            back_to_a,
            after_resize,
            expected_after_resize * num_cores,
            num_cores);

        EXPECT_EQ(after_resize, expected_after_resize * num_cores)
            << "resizing a circular buffer did not move the reported total; the footprint cached "
               "from the previous layout was not invalidated";
    }
}

// A program whose circular buffers are ALL globally allocated reports a footprint of zero on
// the cores it covers -- not "no footprint".
//
// Global CB bytes belong in the L1 column: they are backed by a real L1 Buffer and counted
// through the buffer path. What is easy to get wrong is concluding from that that such a
// program is invisible to CB residency. It is not: dispatch builds a CB config payload for
// every range circular_buffers_unique_coreranges() reports -- a globally-allocated CB occupies
// a local CB index like any other -- and writes it over the CB config region. So the previous
// program's locally-allocated bytes stop being resident on those cores, and omitting the entry
// leaves them recorded there for good.
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixGlobalOnlyCircularBufferDispatchClearsLocalResidency) {
    for (auto& mesh_device : this->devices_) {
        auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
        ASSERT_NE(device, nullptr);
        if (!tracking_available(device)) {
            GTEST_SKIP() << kTrackingUnavailable;
        }

        // One core: a globally-allocated CB must fit inside a single bank of its backing
        // buffer, which a replicated single-page L1 buffer gives us without a shard spec.
        const CoreCoord core(0, 0);
        const CoreRangeSet cr_set(CoreRange(core, core));
        const CBConfig cb_config;
        auto zero = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero, zero);
        auto& cq = mesh_device->mesh_command_queue();

        // A: two ordinary, locally-allocated CBs.
        uint64_t local_bytes = 0;
        auto workload_a = std::make_shared<distributed::MeshWorkload>();
        {
            Program p;
            workload_a->add_program(device_range, std::move(p));
            auto& prog = workload_a->get_programs().at(device_range);
            for (uint32_t cb_id = 0; cb_id < 2; cb_id++) {
                CircularBufferConfig config =
                    CircularBufferConfig(cb_config.page_size, {{cb_id, cb_config.data_format}})
                        .set_page_size(cb_id, cb_config.page_size);
                CreateCircularBuffer(prog, cr_set, config);
                local_bytes += cb_config.page_size;
            }
            initialize_program(prog, cr_set);
        }
        ASSERT_GT(local_bytes, 0u);

        // B: one CB on the same core, backed by an L1 buffer, so it is globally allocated and
        // contributes nothing to the CB column.
        constexpr uint32_t kPages = 2;
        const uint32_t backing_bytes = kPages * cb_config.page_size;
        auto backing = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = backing_bytes},
            {.page_size = backing_bytes, .buffer_type = BufferType::L1},
            mesh_device.get());
        auto workload_b = std::make_shared<distributed::MeshWorkload>();
        {
            Program p;
            workload_b->add_program(device_range, std::move(p));
            auto& prog = workload_b->get_programs().at(device_range);
            CircularBufferConfig config = CircularBufferConfig(backing_bytes, {{0, cb_config.data_format}})
                                              .set_page_size(0, cb_config.page_size)
                                              .set_globally_allocated_address(*backing->get_reference_buffer());
            CreateCircularBuffer(prog, cr_set, config);
            initialize_program(prog, cr_set);
        }

        auto dispatch = [&](distributed::MeshWorkload& workload) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
            distributed::Finish(cq);
            return device->get_total_cb_allocated();
        };

        // Assertions are on the CHANGE, not the absolute total. Both programs here cover a
        // single core, and the device-wide figure legitimately includes whatever is still
        // resident on the cores they do not touch -- earlier tests in this binary dispatch over
        // the whole grid, and nothing since has overwritten those cores' CB config. Only core
        // (0, 0) moves, and by exactly A's footprint.
        const uint64_t after_a = dispatch(*workload_a);
        const uint64_t after_b = dispatch(*workload_b);
        log_info(
            tt::LogTest,
            "device {}: after local-CB program={} then global-only program={} (a drop of {} expected)",
            device->id(),
            after_a,
            after_b,
            local_bytes);

        EXPECT_EQ(after_b + local_bytes, after_a)
            << "the reported total went from " << after_a << " to " << after_b << " when it should have dropped by A's "
            << local_bytes
            << " bytes: dispatching a program whose circular buffers are all globally allocated has "
               "overwritten the CB config on the core it covers, so none of A's local CB bytes are "
               "resident there any more";

        // And back: the zero entry must not have poisoned the per-program footprint cache either.
        EXPECT_EQ(dispatch(*workload_a), after_a) << "re-dispatching A after the global-only program";
    }
}

// Ground truth for the NON-CB columns. Everything else in this file validates circular
// buffer accounting; DRAM / L1 / L1_SMALL / TRACE come from a different mechanism (the
// GraphTracker buffer alloc/dealloc path) and had never been checked against anything.
//
// Unlike circular buffers, real buffers ARE allocator-managed, so the allocator's own
// statistics are an independent ground truth: allocate known buffers, then compare what
// SHM reports for this process against allocator.get_statistics(<type>).
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixBufferTrackingMatchesAllocator) {
    for (auto& mesh_device : this->devices_) {
        auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
        ASSERT_NE(device, nullptr);
        auto* provider = device->get_shm_stats_provider();
        if (!tracking_available(device)) {
            GTEST_SKIP() << kTrackingUnavailable;
        }

        const auto pid = getpid();
        auto shm_for_this_process = [&](ShmBufferType /*unused*/) {
            for (const auto& p : provider->get_process_stats()) {
                if (p.pid == pid) {
                    return p;
                }
            }
            return SharedMemoryStatsProvider::ProcessInfo{};
        };

        const auto allocator_dram = device->allocator()->get_statistics(BufferType::DRAM).total_allocated_bytes;
        const auto allocator_l1 = device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
        const auto before = shm_for_this_process(ShmBufferType::DRAM);

        // Allocate distinctive amounts so a coincidental match is unlikely.
        constexpr DeviceAddr kDramBytes = 64 * 1024;
        constexpr DeviceAddr kL1Bytes = 16 * 1024;
        auto dram_buffer = Buffer::create(device, kDramBytes, kDramBytes, BufferType::DRAM);
        auto l1_buffer = Buffer::create(device, kL1Bytes, kL1Bytes, BufferType::L1);

        const auto allocator_dram_after = device->allocator()->get_statistics(BufferType::DRAM).total_allocated_bytes;
        const auto allocator_l1_after = device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
        const auto after = shm_for_this_process(ShmBufferType::DRAM);

        const uint64_t allocator_dram_delta = allocator_dram_after - allocator_dram;
        const uint64_t allocator_l1_delta = allocator_l1_after - allocator_l1;
        const uint64_t shm_dram_delta = after.dram_allocated - before.dram_allocated;
        const uint64_t shm_l1_delta = after.l1_allocated - before.l1_allocated;

        log_info(
            tt::LogTest,
            "device {}: DRAM delta -- allocator={} shm={} | L1 delta -- allocator={} shm={}",
            device->id(),
            allocator_dram_delta,
            shm_dram_delta,
            allocator_l1_delta,
            shm_l1_delta);

        EXPECT_EQ(shm_dram_delta, allocator_dram_delta)
            << "SHM DRAM tracking disagrees with the allocator about what was just allocated";
        EXPECT_EQ(shm_l1_delta, allocator_l1_delta)
            << "SHM L1 tracking disagrees with the allocator about what was just allocated";

        // ...and that deallocation is symmetric, which is where an underflow-clamped
        // counter would drift permanently.
        dram_buffer.reset();
        l1_buffer.reset();
        const auto after_free = shm_for_this_process(ShmBufferType::DRAM);
        EXPECT_EQ(after_free.dram_allocated, before.dram_allocated) << "DRAM tracking did not return to its baseline";
        EXPECT_EQ(after_free.l1_allocated, before.l1_allocated) << "L1 tracking did not return to its baseline";
    }
}

// GraphTracker's capture processor stack is per-thread, so a memory-tracking processor
// registered on the device-init thread never observes buffers allocated on any other
// thread. Single-threaded tests cannot see that: the allocation and the registration
// happen on the same thread. This allocates from a worker thread instead.
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixBufferTrackingSeesOtherThreads) {
    for (auto& mesh_device : this->devices_) {
        auto* device = dynamic_cast<Device*>(mesh_device->get_devices().at(0));
        ASSERT_NE(device, nullptr);
        auto* provider = device->get_shm_stats_provider();
        if (!tracking_available(device)) {
            GTEST_SKIP() << kTrackingUnavailable;
        }

        const auto pid = getpid();
        auto shm_dram = [&]() -> uint64_t {
            for (const auto& p : provider->get_process_stats()) {
                if (p.pid == pid) {
                    return p.dram_allocated;
                }
            }
            return 0;
        };

        const uint64_t before = shm_dram();
        constexpr DeviceAddr kDramBytes = 128 * 1024;
        uint64_t during = 0;

        std::thread worker([&] {
            auto buffer = Buffer::create(device, kDramBytes, kDramBytes, BufferType::DRAM);
            during = shm_dram();
        });
        worker.join();

        const uint64_t after = shm_dram();
        log_info(
            tt::LogTest,
            "device {}: DRAM on worker thread -- before={} during={} after={} (allocated {})",
            device->id(),
            before,
            during,
            after,
            kDramBytes);

        EXPECT_EQ(during - before, static_cast<uint64_t>(kDramBytes))
            << "a buffer allocated on a non-device-init thread was not tracked; the memory-tracking "
               "processor is only visible to the thread that registered it";
        EXPECT_EQ(after, before) << "worker-thread deallocation was not tracked";
    }
}

}  // namespace basic_tests::circular_buffer
