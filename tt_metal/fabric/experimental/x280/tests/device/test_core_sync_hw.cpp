// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tensix hardware verification for core_sync_barrier over MetalTransport.

#include "harness/artifacts/artifacts.hpp"
#include "harness/metal_runner.hpp"
#include "harness/regions/core_sync_roster.hpp"
#include "tests/device/suites/core_sync/drisc_worker.hpp"
#include "tests/device/suites/core_sync/erisc_worker.hpp"
#include "tests/device/suites/core_sync/probe_args.hpp"
#include "tests/device/suites/core_sync/tensix_directed_run.hpp"

#include "core_sync_barrier.hpp"
#include "core_sync_layout.h"
#include "core_sync_transport_metal.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hreg = x280::harness::regions;
namespace trun = x280::suites::core_sync::tensix_directed;

TEST(CoreSyncMemoryBindings, CounterOnlyRosterNeedsNoRawStorage) {
    const hreg::CoreSyncMemoryRequirements requirements{
        .director_ready_bytes = 0u,
        .director_counter_required = true,
        .raw_release_count = 0u,
    };

    EXPECT_NO_THROW(hreg::detail::validate_core_sync_memory_bindings(requirements, hreg::CoreSyncMemoryBindings{}));
}

TEST(CoreSyncMemoryBindings, AcceptsExactReadyAndReleaseRegions) {
    const hreg::CoreSyncMemoryRequirements requirements{
        .director_ready_bytes = 2u * CORE_SYNC_SLOT_BYTES,
        .director_counter_required = false,
        .raw_release_count = 1u,
    };
    hreg::CoreSyncMemoryBindings memory;
    memory.director_ready = hreg::Region{0x1000u, 2u * CORE_SYNC_SLOT_BYTES};
    memory.raw_releases.push_back(hreg::Region{0x2000u, CORE_SYNC_SLOT_BYTES});

    EXPECT_NO_THROW(hreg::detail::validate_core_sync_memory_bindings(requirements, memory));
}

TEST(CoreSyncMemoryBindings, RejectsInvalidRegions) {
    struct Mutation {
        const char* name;
        void (*apply)(hreg::CoreSyncMemoryBindings&);
        const char* error_fragment;
    };
    const std::array<Mutation, 7> mutations{{
        {"undersized ready region",
         [](auto& memory) { memory.director_ready.bytes = CORE_SYNC_SLOT_BYTES; },
         "smaller than required size"},
        {"misaligned ready size", [](auto& memory) { memory.director_ready.bytes = 15u; }, "size 15"},
        {"misaligned ready base", [](auto& memory) { memory.director_ready.addr += 1u; }, "base 4097"},
        {"missing raw release", [](auto& memory) { memory.raw_releases.clear(); }, "binding count 0"},
        {"extra raw release",
         [](auto& memory) { memory.raw_releases.push_back(hreg::Region{0x3000u, CORE_SYNC_SLOT_BYTES}); },
         "binding count 2"},
        {"undersized raw release",
         [](auto& memory) { memory.raw_releases[0].bytes = CORE_SYNC_SLOT_BYTES - 1u; },
         "size 7"},
        {"misaligned raw release", [](auto& memory) { memory.raw_releases[0].addr += 1u; }, "base 8193"},
    }};

    const hreg::CoreSyncMemoryRequirements requirements{
        .director_ready_bytes = 2u * CORE_SYNC_SLOT_BYTES,
        .raw_release_count = 1u,
    };
    const hreg::CoreSyncMemoryBindings valid{
        .director_ready = hreg::Region{0x1000u, 2u * CORE_SYNC_SLOT_BYTES},
        .raw_releases = {hreg::Region{0x2000u, CORE_SYNC_SLOT_BYTES}},
    };

    for (const auto& mutation : mutations) {
        SCOPED_TRACE(mutation.name);
        auto memory = valid;
        mutation.apply(memory);
        try {
            hreg::detail::validate_core_sync_memory_bindings(requirements, memory);
            FAIL() << "invalid binding was accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string(error.what()).find(mutation.error_fragment), std::string::npos);
        }
    }
}

// The probe kernel reads each shared value from its named slot, behind the
// layout cookie, and a role's args from CORE_SYNC_PROBE_ARG_ROLE_BASE.
TEST(CoreSyncProbeArgs, CommonArgsFillTheirNamedSlotsBehindTheCookie) {
    const auto args = x280::suites::core_sync::probe_common_args({11u, 22u, 33u, 44u}, {55u, 66u});
    ASSERT_EQ(args.size(), static_cast<size_t>(CORE_SYNC_PROBE_ARG_ROLE_BASE) + 2u);
    EXPECT_EQ(args[CORE_SYNC_PROBE_ARG_ROLE_BASE], 55u);
    EXPECT_EQ(args[CORE_SYNC_PROBE_ARG_ROLE_BASE + 1], 66u);
    EXPECT_EQ(args[CORE_SYNC_PROBE_ARG_LAYOUT_COOKIE], CORE_SYNC_PROBE_LAYOUT_COOKIE);
    EXPECT_EQ(args[CORE_SYNC_PROBE_ARG_RUN_NONCE], 11u);
    EXPECT_EQ(args[CORE_SYNC_PROBE_ARG_DIRECTOR_READY_BASE], 22u);
    EXPECT_EQ(args[CORE_SYNC_PROBE_ARG_RESULT_L1_ADDR], 33u);
    EXPECT_EQ(args[CORE_SYNC_PROBE_ARG_ARRIVAL_SEMAPHORE_ID], 44u);
}

const std::filesystem::path kKernelDir = hart_::kernel_dir_of(__FILE__);
const std::vector<std::filesystem::path> kProbeIncludePaths = {
    kKernelDir,
    hart_::shared_kernel_dir(kKernelDir, "shared"),
    hart_::x280_root_of(kKernelDir) / "include",
};

using CoreSyncHwTest = x280::harness::MetalRunner<int>;

TEST_P(CoreSyncHwTest, TwoCoreTensixBarrier) {
    ASSERT_NE(mesh, nullptr) << "unit mesh not initialized";

    const ttm::CoreCoord director_core{0, 0};
    const ttm::CoreCoord worker_core{0, 1};
    const ttm::CoreRange worker_range(worker_core, worker_core);
    const ttm::CoreRangeSet worker_set(worker_range);

    auto result_buffer = trun::create_zeroed_l1_buffer(mesh.get());
    const uint32_t director_ready_base = 0u;
    const uint32_t result_l1_addr = static_cast<uint32_t>(result_buffer->address());

    ttm::Program program = ttm::CreateProgram();

    const auto roster = hreg::make_core_sync_roster(
        program, *mesh, worker_set, director_core, ttm::NOC::NOC_0, hreg::CoreSyncMemoryBindings{});

    ASSERT_EQ(roster.workers.size(), 1u);

    const uint32_t nonce = trun::mint_run_nonce();

    const auto kpath = kKernelDir / "core_sync_probe.cpp";
    ASSERT_TRUE(std::filesystem::exists(kpath)) << "probe kernel not found at " << kpath;

    trun::add_kernels(
        program,
        kpath,
        kProbeIncludePaths,
        director_core,
        worker_set,
        roster,
        nonce,
        director_ready_base,
        result_l1_addr,
        roster.counter_arrivals);
    trun::execute(mesh.get(), std::move(program));

    const auto director_result = trun::read_result(mesh.get(), director_core, result_l1_addr, nonce);
    const auto worker_result = trun::read_result(mesh.get(), worker_core, result_l1_addr, nonce);

    EXPECT_EQ(director_result.outcome, static_cast<uint32_t>(core_sync::Outcome::Success))
        << "director outcome was " << director_result.outcome;
    EXPECT_EQ(worker_result.outcome, static_cast<uint32_t>(core_sync::Outcome::Success))
        << "worker outcome was " << worker_result.outcome;

    const uint64_t director_ts = director_result.exit_timestamp;
    const uint64_t worker_ts = worker_result.exit_timestamp;

    EXPECT_GT(director_ts, 0ull) << "director arrival timestamp is zero";
    EXPECT_GT(worker_ts, 0ull) << "worker arrival timestamp is zero";

    const uint64_t skew = (director_ts > worker_ts) ? (director_ts - worker_ts) : (worker_ts - director_ts);
    std::cout << "Two-core Tensix barrier success:\n"
              << "  director arrival: " << director_ts << " cycles\n"
              << "  worker arrival:   " << worker_ts << " cycles\n"
              << "  arrival skew:     " << skew << " cycles\n";
}

struct GridRun {
    bool barrier_enabled = true;

    /* How many counter arrivals the director waits for. Zero takes the
     * roster's own count; a larger value names an arrival no worker will make,
     * which is what drives the director onto its abandonment path. */
    uint32_t counter_arrivals = 0;
};

struct GridResult {
    /* One barrier Outcome per worker, in roster order. */
    std::vector<uint32_t> worker_outcomes;
};

GridResult run_grid(
    ttmd::MeshDevice* mesh,
    const ttm::CoreCoord& director_core,
    const ttm::CoreRangeSet& worker_set,
    const GridRun& cfg) {
    auto result_buffer = trun::create_zeroed_l1_buffer(mesh);
    const uint32_t director_ready_base = 0u;
    const uint32_t result_l1_addr = static_cast<uint32_t>(result_buffer->address());

    ttm::Program program = ttm::CreateProgram();
    const auto roster = hreg::make_core_sync_roster(
        program,
        *mesh,
        worker_set,
        director_core,
        ttm::NOC::NOC_0,
        hreg::CoreSyncMemoryBindings{},
        hreg::ReleaseMode::RequireMulticast);

    const uint32_t nonce = trun::mint_run_nonce();
    const auto kpath = kKernelDir / "core_sync_probe.cpp";

    const uint32_t counter_arrivals = cfg.counter_arrivals != 0u ? cfg.counter_arrivals : roster.counter_arrivals;
    trun::add_kernels(
        program,
        kpath,
        kProbeIncludePaths,
        director_core,
        worker_set,
        roster,
        nonce,
        director_ready_base,
        result_l1_addr,
        counter_arrivals,
        cfg.barrier_enabled);
    trun::execute(mesh, std::move(program));

    GridResult out;
    out.worker_outcomes = trun::read_outcomes(mesh, worker_set, result_l1_addr, nonce);
    return out;
}

void expect_outcome(const GridResult& r, core_sync::Outcome expected, const char* what) {
    for (size_t i = 0; i < r.worker_outcomes.size(); ++i) {
        EXPECT_EQ(r.worker_outcomes[i], static_cast<uint32_t>(expected))
            << what << ": worker " << i << " outcome was " << r.worker_outcomes[i];
    }
}

/* A Tensix director told to wait for one more arrival than the grid will make.
 *
 * Its wait is bounded by an iteration cap rather than by host STOP, which the
 * X280 director's is, so this is the other of the two abandonment mechanisms.
 * What it has to show is that the cap expires: a
 * director that spins forever holds the release, and every worker behind it
 * spins too, which wedges the grid the first time a participant dies. */
TEST_P(CoreSyncHwTest, WithheldArrivalExpiresTheDirectorsWait) {
    ASSERT_NE(mesh, nullptr) << "unit mesh not initialized";

    const ttm::CoreCoord director_core{0, 0};
    const ttm::CoreRangeSet worker_set = trun::worker_grid();

    const GridResult abandoned =
        run_grid(mesh.get(), director_core, worker_set, GridRun{/*barrier_enabled=*/true, /*counter_arrivals=*/16u});

    std::cout << "Tensix-director abandonment: 16 arrivals awaited over 15 workers\n";

    /* The director never released, so every worker's own bounded poll expired
     * too. Both halves unwinding is the property; a worker still spinning would
     * not have published an outcome at all. */
    expect_outcome(abandoned, core_sync::Outcome::StopRequested, "withheld participant");
}

/* A Tensix director releasing a Tensix rectangle and one idle ERISC in the
 * same barrier. The implementation lives under suites/core_sync/erisc_worker.hpp,
 * because its skip conditions and its two address spaces are its own. */
TEST_P(CoreSyncHwTest, IdleEriscWorkerJoinsATensixRelease) {
    x280::suites::core_sync::erisc_worker::run_erisc_worker(
        mesh.get(), kKernelDir / "core_sync_probe.cpp", kProbeIncludePaths);
}

TEST_P(CoreSyncHwTest, DriscWorkerJoinsATensixRelease) {
    x280::suites::core_sync::drisc_worker::run_drisc_worker(
        mesh.get(), kKernelDir / "core_sync_probe.cpp", kProbeIncludePaths);
}

INSTANTIATE_TEST_SUITE_P(CoreSyncHardware, CoreSyncHwTest, ::testing::Values(0));

}  // namespace
