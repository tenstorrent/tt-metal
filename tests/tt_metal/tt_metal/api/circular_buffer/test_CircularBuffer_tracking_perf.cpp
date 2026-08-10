// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// CB memory tracking: what it reports, and what it costs.
//
// Device::get_total_cb_allocated() used to recompute the device-wide CB footprint on every
// program registration, walking every live program and expanding per core. Programs only
// leave that set in ~ProgramImpl, so a host holding many programs alive -- exactly what
// ttnn's program cache does across a parameter sweep -- made each call O(live programs) and
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

// Median of a timing sample, to keep the assertion robust against scheduler noise.
double median_ms(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Cost of one tracking query. This isolates the cost from the 100 ms rate limiter in
// update_allocator_stats.cpp, which bounds how often the query runs but not how expensive
// each run is.
//
// Each sample times a batch of queries and divides, rather than timing a single call: once
// the total is incrementally maintained a query is an atomic load in the tens of
// nanoseconds, and a single-call measurement can round to zero -- which would make the
// ratio below a division by zero rather than the ~1x it should report.
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

// This one runs by default: it is the regression guard for the bottleneck, and a guard
// that never executes guards nothing. It is sized for CI -- 256 live programs, well under
// a second of registration -- which is already far past the point where the old
// recompute-per-call behaviour shows up (115x at 256 programs, measured). The heavier
// characterisations below stay opt-in.
TEST_F(AnyDispatchMeshDeviceSingleCardFixture, TensixCircularBufferTrackingCostScaling) {
    auto mesh_device = devices_.at(0);
    auto* idevice = mesh_device->get_devices().at(0);
    auto* device = dynamic_cast<Device*>(idevice);
    ASSERT_NE(device, nullptr) << "expected a concrete Device to query CB tracking on";

    if (device->get_shm_stats_provider() == nullptr) {
        GTEST_SKIP() << "SHM tracking is disabled (TT_METAL_SHM_TRACKING_DISABLED=1); "
                        "unset it to reproduce the bottleneck";
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

    // Programs are held alive for the whole test, mirroring ttnn's program cache
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
    // makes this ratio track the program count: 115x at 256 live programs, measured on an
    // n300 before the rework.
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
// This is distinct from the query-cost test above, because
// update_allocator_stats.cpp rate-limits the query to one per 100 ms per device.
// The limiter bounds tracking to ~10 x cost(N) per second, so the damage depends
// on the wall-clock duration of the warmup, not on N alone: a fast loop dilutes
// it, while a long compile-bound warmup pays it repeatedly at a large N.
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

// The decisive measurement. The two tests above show that a tight registration
// loop is too fast for the 100 ms rate limiter to admit more than ~one query, so
// the tracking cost hides. A real warmup is compile-bound: programs appear tens
// to hundreds of ms apart, so nearly every registration is *admitted* by the
// limiter and pays the full O(live_programs) recompute.
//
// This test builds up a realistic number of live programs, then registers a few
// more at a compile-like cadence and reports the cost of those registrations --
// which is what the reporter's workload actually experiences per program.
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
    constexpr auto kCadence = std::chrono::milliseconds(110);  // > the 100 ms limiter window

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
    if (device->get_shm_stats_provider() == nullptr) {
        GTEST_SKIP() << "SHM tracking disabled; nothing to validate against";
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
    // the feature: an in-process number nobody can read is not memory tracking. The SHM
    // publish is rate-limited (one per 100 ms per device), so allow it a moment to land.
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
    if (device->get_shm_stats_provider() == nullptr) {
        GTEST_SKIP() << "SHM tracking disabled";
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
        if (device->get_shm_stats_provider() == nullptr) {
            GTEST_SKIP() << "SHM tracking disabled";
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
        if (provider == nullptr) {
            GTEST_SKIP() << "SHM tracking disabled";
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
        if (provider == nullptr) {
            GTEST_SKIP() << "SHM tracking disabled";
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
