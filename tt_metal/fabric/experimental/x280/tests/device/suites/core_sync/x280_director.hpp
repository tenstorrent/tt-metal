// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/core_sync/x280_director.hpp -- an X280 hart directs a core_sync
// barrier over a grid of Tensix workers.
//
// The X280 runs firmware-core-sync-director.bin as director and the Tensix
// grid runs tests/device/kernels/core_sync_probe.cpp as workers, co-scheduled
// by enqueue_tensix -> run_x280 -> finish_tensix ordering.
//
// The default cases require multicast completion, STOP-aware abandonment, and
// mixed counter/word arrivals. run_characterization separately records the
// null, unicast, and multicast runs' raw worker exit timestamps without a
// timing threshold.
//
// Coordinates: the X280's outbound TLB window takes virtual coordinates for
// the Tensix release rectangle and endpoints, a participant's arrival target,
// and the release-probe groups. A Tensix kernel addresses the L2CPU at its
// virtual coordinate. Both come from X280Runtime. The active ERISC case aims
// the window at the Ethernet tile's virtual coordinate.

#ifndef X280_SUITES_CORE_SYNC_X280_DIRECTOR_HPP
#define X280_SUITES_CORE_SYNC_X280_DIRECTOR_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/output/output.hpp"
#include "harness/execution/execution.hpp"
#include "harness/regions/regions.hpp"
#include "harness/regions/core_sync_roster.hpp"
#include "harness/regions/release_selector.hpp"
#include "harness/runner.hpp"
#include "tests/device/suites/core_sync/probe_args.hpp"
#include "tests/device/suites/core_sync/tensix_directed_run.hpp"
#include "tests/device/suites/core_sync/erisc_worker.hpp"
#include "harness/regions/l1_window.hpp"
#include "harness/regions/metal_l1_region.hpp"
#include "harness/regions/tensix_l1.hpp"
#include "harness/regions/release_guard.hpp"
#include "tests/device/suites/core_sync/records.hpp"
#include "harness/metal_context.hpp"

#include "core_sync_barrier.hpp"
#include "core_sync_core_family.h"
#include "core_sync_director_args.h"
#include "core_sync_release_probe_args.h"
#include "core_sync_layout.h"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace x280::suites::core_sync::x280_director {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hreg = x280::harness::regions;
namespace trun = x280::suites::core_sync::tensix_directed;

using Axis = int;

constexpr uint32_t kL1BlockBytes = 64u;
constexpr uint32_t kReleaseOffset = 0u;
constexpr uint32_t kResultOffset = 32u;

/* How the director is asked to release. Null issues no release at all: the
 * X280 is not launched, and the Tensix grid is
 * built with the barrier compiled out, which is the floor the other two are
 * read against. Unicast and Multicast select the release by
 * ReleaseMode::UnicastOnly and ReleaseMode::RequireMulticast. */
enum class Release { Null, Unicast, Multicast };

inline hreg::ReleaseMode release_mode(Release r) {
    return r == Release::Multicast ? hreg::ReleaseMode::RequireMulticast : hreg::ReleaseMode::UnicastOnly;
}

inline hreg::ReleasePlan select_tensix_release(
    const ttmd::MeshDevice& mesh, const std::vector<ttm::CoreCoord>& workers, hreg::ReleaseMode mode) {
    std::vector<hreg::ReleaseParticipant> participants;
    participants.reserve(workers.size());
    for (const auto& w : workers) {
        participants.push_back({tt::CoreType::TENSIX, w});
    }
    return hreg::select_release(mesh, participants, hreg::kX280ReleaseSender, mode, CORE_SYNC_DIRECTOR_MAX_ENDPOINTS);
}

inline const char* release_name(Release r) {
    switch (r) {
        case Release::Null: return "null run, barrier compiled out";
        case Release::Unicast: return "X280 director, unicast release";
        case Release::Multicast: return "X280 director, multicast release";
    }
    return "?";
}

inline const char* state_name(uint32_t s) {
    switch (s) {
        case CORE_SYNC_DIRECTOR_STATE_INIT: return "INIT";
        case CORE_SYNC_DIRECTOR_STATE_DONE: return "DONE";
        case CORE_SYNC_DIRECTOR_STATE_ARG_LAYOUT_MISMATCH: return "ARG_LAYOUT_MISMATCH";
        case CORE_SYNC_DIRECTOR_STATE_SYNC_FAILURE: return "SYNC_FAILURE";
        default: return "?";
    }
}

struct X280Regions {
    hreg::Region args;
    hreg::Region ready;
    hreg::Region result;

    static X280Regions compute(x280::X280Tile& tile, uint32_t worker_count) {
        const auto avail = tile.available_lim();
        const hreg::Region w{avail.begin, avail.size()};
        hreg::Arena arena(w);
        X280Regions R{};
        R.args = arena.allocate(sizeof(core_sync_director_section_t));
        R.ready = arena.allocate(static_cast<uint64_t>(worker_count) * CORE_SYNC_SLOT_BYTES);
        R.result = arena.allocate(sizeof(core_sync_director_result_t));
        return R;
    }
};

struct RunResult {
    std::vector<uint64_t> timestamps;
    uint32_t run_nonce = 0;
    uint32_t grid_num_dests = 0;
    uint32_t endpoint_count = 0;
    uint32_t arrival_count = 0;
    uint32_t director_terminal_state = CORE_SYNC_DIRECTOR_STATE_INIT;
    std::vector<uint32_t> worker_outcomes;
    bool x280_ran = false;
    /* Guard cores whose release word no longer held the sentinel after the run. */
    std::vector<x280::harness::regions::OverwrittenGuard> overwritten_guards;
};

/* Builds the args block for one run from the selector's plan; ready_base and
 * release_addr are the addresses the two sides meet at. */
inline core_sync_director_section_t director_section(
    Release release, uint32_t run_nonce, const X280Regions& XR, uint32_t release_addr, const hreg::ReleasePlan& plan) {
    core_sync_director_section_t s{};
    s.cookie_common = core_sync_director_cookie_for(UBENCH_SECTION_COMMON);
    s.cookie_backend = core_sync_director_cookie_for(UBENCH_SECTION_BACKEND);
    s.common.run_nonce = run_nonce;
    s.common.role = CORE_SYNC_DIRECTOR_ROLE_DIRECTOR;
    s.backend.result_addr = XR.result.addr;
    s.backend.ready_base = static_cast<uint32_t>(XR.ready.addr);

    s.backend.ready_bytes = static_cast<uint32_t>(XR.ready.bytes);
    if (release == Release::Null) {
        return s;
    }

    const uint32_t count = static_cast<uint32_t>(plan.translated.size());
    s.backend.arrival_count = count;
    for (uint32_t i = 0; i < count; ++i) {
        s.backend.arrivals[i].addr = core_sync_slot_addr(static_cast<uint32_t>(XR.ready.addr), i);
        s.backend.arrivals[i].worker_index = i;
    }

    if (plan.group) {
        s.backend.grid = {plan.group->rect(), release_addr, plan.group->num_dests};
    }
    s.backend.endpoint_count = static_cast<uint32_t>(plan.endpoints.size());
    for (uint32_t i = 0; i < plan.endpoints.size(); ++i) {
        s.backend.endpoints[i] = {plan.translated[plan.endpoints[i]].noc(), release_addr, 0u};
    }
    return s;
}

/* Runs one release shape end to end and harvests both sides.
 *
 * abandon withholds the X280's counterparty rather than the other way round:
 * the roster names one more arrival than any worker will write, so the
 * director's STOP-aware wait is the only way out of the barrier. The caller
 * gets the director's terminal state either way.
 *
 * guards are logical Tensix cores outside the roster: each holds
 * x280::harness::regions::kGuardSentinel at the release address through the run, and the ones that
 * lost it are reported in overwritten_guards. */
inline RunResult run_once(
    x280::harness::session::Fixture& f,
    const x280::harness::RunSpec& spec,
    const std::vector<ttm::CoreCoord>& workers,
    const ttm::CoreRangeSet& worker_set,
    Release release,
    bool abandon,
    std::chrono::seconds wait_timeout,
    const std::vector<ttm::CoreCoord>& guards = {}) {
    RunResult out;
    const uint32_t count = static_cast<uint32_t>(workers.size());

    /* Every worker's release word and result start zeroed: the release token
     * is derived from the run nonce, but a result left from an earlier run
     * would otherwise be read as this one's. */
    const auto l1 = hreg::allocate_tile_l1_buffer(f.mesh.get(), worker_set, kL1BlockBytes);
    const uint32_t release_addr = l1.l1_tile_base + kReleaseOffset;
    const uint32_t result_addr = l1.l1_tile_base + kResultOffset;

    x280::harness::regions::write_guards(*f.mesh, guards, release_addr);

    const auto l2cpu_translated = f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::RISCV_0_default);

    const X280Regions XR = X280Regions::compute(*f.tile, count);
    EXPECT_LE(XR.ready.end(), std::numeric_limits<uint32_t>::max())
        << "a ready-table slot address is a 32-bit field in the args block";
    hreg::zero_region(*f.tile, XR.ready);
    hreg::zero_region(*f.tile, XR.result);

    const uint32_t nonce = trun::mint_run_nonce();
    out.run_nonce = nonce;
    const core_sync_director_section_t section =
        release == Release::Null
            ? director_section(release, nonce, XR, release_addr, hreg::ReleasePlan{})
            : director_section(
                  release, nonce, XR, release_addr, select_tensix_release(*f.mesh, workers, release_mode(release)));
    if (release != Release::Null) {
        hreg::write_args(*f.tile, XR.args, section);
    }

    const uint32_t barrier_enabled = (release == Release::Null) ? 0u : 1u;
    const std::vector<std::filesystem::path> include_paths = {
        spec.open.kernel_dir,
        hart_::shared_kernel_dir(spec.open.kernel_dir, "shared"),
        hart_::x280_root_of(spec.open.kernel_dir) / "include",
    };

    ttm::Program program = ttm::CreateProgram();
    /* Metal requires the semaphores the probe's counter path names to exist
     * even where this run never reaches it: get_semaphore resolves an id the
     * program allocated. */
    (void)ttm::CreateSemaphore(program, worker_set, 0);
    (void)ttm::CreateSemaphore(program, worker_set, 0);

    const uint32_t launched_workers = count - (abandon ? 1u : 0u);
    for (uint32_t i = 0; i < launched_workers; ++i) {
        const ttm::CoreRangeSet one(ttm::CoreRange(workers[i], workers[i]));
        const auto kernel = ttm::CreateKernel(
            program,
            (spec.open.kernel_dir / "core_sync_probe.cpp").string(),
            one,
            ttm::DataMovementConfig{
                .processor = ttm::DataMovementProcessor::RISCV_0,
                .noc = ttm::NOC::RISCV_0_default,
                .compile_args = {0u, i, barrier_enabled, 1u, 0u},
                .compiler_include_paths = include_paths,
            });
        ttm::SetRuntimeArgs(
            program,
            kernel,
            one,
            word_worker_args(
                {nonce, static_cast<uint32_t>(XR.ready.addr), result_addr, 0u}, l2cpu_translated, 1u, i, release_addr));
    }

    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_CORE_SYNC_DIRECTOR_FW_IMAGE", "firmware-core-sync-director.bin"),
        .run_value = (static_cast<uint64_t>(0xC5C5u) << 32) | nonce,
        .args_addr = XR.args.addr,
        .args_bytes = XR.args.bytes,
        .cleanup_timeout = std::chrono::seconds{10},
        .wait_timeout = wait_timeout,
    };

    hexec::enqueue_tensix(*f.mesh, workload);
    if (release == Release::Null) {
        hexec::finish_tensix(*f.mesh);
    } else {
        const auto outcome = hexec::run_x280(f.handle(), launch);
        hexec::finish_tensix(*f.mesh);

        out.x280_ran = outcome.enqueued();
        EXPECT_TRUE(outcome.enqueued()) << "enqueue rejected";
        if (!abandon) {
            EXPECT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
        }
        EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);
    }

    if (release != Release::Null) {
        const auto director = hreg::read_struct<core_sync_director_result_t>(*f.tile, XR.result);
        out.director_terminal_state = director.terminal_state;
        out.grid_num_dests = director.released_dests;
        out.endpoint_count = director.released_endpoints;
        out.arrival_count = director.awaited_arrivals;
        EXPECT_EQ(director.cookie, CORE_SYNC_DIRECTOR_RESULT_COOKIE) << "the director never initialized its result";
    }

    std::vector<uint64_t> timestamps;
    timestamps.reserve(launched_workers);
    for (uint32_t i = 0; i < launched_workers; ++i) {
        const auto worker = trun::read_result(f.mesh.get(), workers[i], result_addr, nonce);
        out.worker_outcomes.push_back(worker.outcome);
        timestamps.push_back(worker.exit_timestamp);
    }
    out.timestamps = std::move(timestamps);
    out.overwritten_guards = x280::harness::regions::overwritten_guards(*f.mesh, guards, release_addr);
    return out;
}

inline void report(Release release, const RunResult& r) {
    std::printf("  %s:\n", release_name(release));
    std::printf(
        "    rectangle destinations %u, tier-2 writes %u, arrivals awaited %u\n",
        r.grid_num_dests,
        r.endpoint_count,
        r.arrival_count);
    if (r.x280_ran) {
        std::printf("    director terminal state: %s\n", state_name(r.director_terminal_state));
    } else {
        std::printf("    director: not launched\n");
    }
}

inline void expect_all_workers_succeeded(const RunResult& r, const char* what) {
    for (size_t i = 0; i < r.worker_outcomes.size(); ++i) {
        EXPECT_EQ(r.worker_outcomes[i], static_cast<uint32_t>(::core_sync::Outcome::Success))
            << what << ": worker " << i << " outcome was " << r.worker_outcomes[i];
    }
}

/* The 4x4 block of logical Tensix cores this suite releases. Sixteen is the
 * most the args block carries as unicast endpoints, which is what makes the
 * two release shapes comparable over one roster. */
inline std::vector<ttm::CoreCoord> worker_block(uint32_t x0 = 0) {
    std::vector<ttm::CoreCoord> workers;
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            workers.push_back(ttm::CoreCoord{x0 + x, y});
        }
    }
    return workers;
}

/* Releases the 4x4 block at logical (x0..x0+3, 0..3) by one required
 * multicast and checks every worker arrived and was released exactly once. */
inline void run_multicast_block(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, uint32_t x0) {
    ASSERT_NE(f.mesh, nullptr);
    const auto workers = worker_block(x0);
    const ttm::CoreRangeSet worker_set(ttm::CoreRange(ttm::CoreCoord{x0, 0}, ttm::CoreCoord{x0 + 3, 3}));

    const auto multicast = run_once(
        f,
        spec,
        workers,
        worker_set,
        Release::Multicast,
        /*abandon=*/false,
        std::chrono::seconds{60});

    EXPECT_EQ(multicast.grid_num_dests, workers.size());
    EXPECT_EQ(multicast.endpoint_count, 0u) << "a Tensix released by the rectangle must not also take a write";
    EXPECT_EQ(multicast.arrival_count, workers.size());
    EXPECT_EQ(multicast.director_terminal_state, CORE_SYNC_DIRECTOR_STATE_DONE);
    expect_all_workers_succeeded(multicast, "multicast release");
}

/* Runs the X280-director multicast correctness case. */
inline void run_axis(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, Axis /*axis*/) {
    run_multicast_block(f, spec, 0u);
}

/* Records null, unicast, and multicast worker exit timestamps in raw timer ticks. */
inline void run_characterization(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec) {
    ASSERT_NE(f.mesh, nullptr);
    const auto workers = worker_block();
    const ttm::CoreRangeSet worker_set(ttm::CoreRange(ttm::CoreCoord{0, 0}, ttm::CoreCoord{3, 3}));

    const auto plan = select_tensix_release(*f.mesh, workers, hreg::ReleaseMode::RequireMulticast);

    const auto null_run = run_once(
        f,
        spec,
        workers,
        worker_set,
        Release::Null,
        /*abandon=*/false,
        std::chrono::seconds{60});
    const auto unicast = run_once(
        f,
        spec,
        workers,
        worker_set,
        Release::Unicast,
        /*abandon=*/false,
        std::chrono::seconds{60});
    const auto multicast = run_once(
        f,
        spec,
        workers,
        worker_set,
        Release::Multicast,
        /*abandon=*/false,
        std::chrono::seconds{60});

    std::printf(
        "X280-director release skew, %zu Tensix workers, virtual rectangle "
        "(%u, %u)-(%u, %u):\n",
        workers.size(),
        plan.group->start.x(),
        plan.group->start.y(),
        plan.group->end.x(),
        plan.group->end.y());
    report(Release::Null, null_run);
    report(Release::Unicast, unicast);
    report(Release::Multicast, multicast);

    EXPECT_EQ(unicast.grid_num_dests, 0u) << "a unicast release must build no rectangle";
    EXPECT_EQ(multicast.endpoint_count, 0u) << "a Tensix released by the rectangle must not also take a write";
    EXPECT_FALSE(null_run.x280_ran);
    EXPECT_EQ(unicast.director_terminal_state, CORE_SYNC_DIRECTOR_STATE_DONE);
    EXPECT_EQ(multicast.director_terminal_state, CORE_SYNC_DIRECTOR_STATE_DONE);
    expect_all_workers_succeeded(null_run, "null release");
    expect_all_workers_succeeded(unicast, "unicast release");
    expect_all_workers_succeeded(multicast, "multicast release");
    for (const auto& entry :
         {std::pair{"null", &null_run}, std::pair{"unicast", &unicast}, std::pair{"multicast", &multicast}}) {
        std::vector<ReleasedWorker> released;
        for (size_t i = 0; i < entry.second->timestamps.size(); ++i) {
            released.push_back(
                {.logical_x = static_cast<uint32_t>(workers[i].x),
                 .logical_y = static_cast<uint32_t>(workers[i].y),
                 .raw_timestamp = entry.second->timestamps[i],
                 .outcome = entry.second->worker_outcomes[i]});
        }
        x280::harness::output::write_jsonl(
            spec.suite_label,
            x280::harness::output::record(
                characterization_record(entry.first, entry.second->run_nonce, f.tile_id.chip_id, released)));
    }
}

/* A roster naming one arrival no worker will ever write. The director's wait is
 * STOP-aware rather than bounded, so the host's cleanup publishes STOP for the
 * run generation and x280_hart_wait_word_or_stop returns 0; the barrier reports
 * StopRequested and the image reaches SYNC_FAILURE rather than hanging. */
inline void run_abandonment(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec) {
    ASSERT_NE(f.mesh, nullptr);
    const auto workers = worker_block();
    const ttm::CoreRangeSet worker_set(ttm::CoreRange(ttm::CoreCoord{0, 0}, ttm::CoreCoord{3, 3}));

    const auto abandoned = run_once(
        f,
        spec,
        workers,
        worker_set,
        Release::Multicast,
        /*abandon=*/true,
        std::chrono::seconds{5});

    std::printf(
        "X280-director abandonment: arrivals awaited %u over %zu workers, "
        "director terminal state %s\n",
        abandoned.arrival_count,
        workers.size(),
        state_name(abandoned.director_terminal_state));

    EXPECT_EQ(abandoned.director_terminal_state, CORE_SYNC_DIRECTOR_STATE_SYNC_FAILURE)
        << "the director reached " << state_name(abandoned.director_terminal_state)
        << " with one arrival withheld; a barrier whose abandonment path does not run "
        << "wedges the chip the first time a participant dies";
    for (size_t i = 0; i < abandoned.worker_outcomes.size(); ++i) {
        EXPECT_EQ(abandoned.worker_outcomes[i], static_cast<uint32_t>(::core_sync::Outcome::StopRequested))
            << "worker " << i << " was never released and must report StopRequested, not "
            << abandoned.worker_outcomes[i];
    }
}

/* Both arrival tiers into one Tensix director, in one run.
 *
 * A Tensix worker announces arrival with noc_semaphore_inc onto the director's
 * counter; the X280 cannot issue a NoC atomic -- System Port is uncached with
 * no atomics per PMA (x280/include/noc.h:25) -- so it writes a private ready
 * word instead. wait_for_arrivals waits on the counter and then walks the
 * words, and neither tier has met the other on silicon before.
 *
 * Release is tiered the same way: the Tensix grid takes one broadcast, and the
 * X280, which no broadcast reaches, takes a notify_write into LIM.
 */
inline void run_mixed_tier(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, uint32_t x0 = 0) {
    ASSERT_NE(f.mesh, nullptr);
    namespace hregions = x280::harness::regions;

    const ttm::CoreCoord director_core{x0, 0};
    const ttm::CoreRangeSet worker_set = trun::worker_grid(false, x0);
    const auto tensix_workers = ttm::corerange_to_cores(worker_set);

    ttm::Program program = ttm::CreateProgram();
    hregions::CoreSyncRosterBuilder<> builder(program, *f.mesh, director_core, ttm::NOC::NOC_0);
    builder.add_workers<::core_sync::MetalTransport<ProgrammableCoreType::TENSIX>>(worker_set);

    const X280Regions XR = X280Regions::compute(*f.tile, 1u);
    hreg::zero_region(*f.tile, XR.ready);
    hreg::zero_region(*f.tile, XR.result);
    const uint32_t x280_release_addr = static_cast<uint32_t>(XR.ready.addr);

    const auto l2cpu_translated = f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::RISCV_0_default);

    /* The X280 joins the roster as its own group: one tier-2 ready word in the
     * director's L1, and one endpoint the broadcast does not reach. */
    builder.add_external_worker<::core_sync::X280Transport<>>(l2cpu_translated);

    const auto requirements = builder.memory_requirements();
    ASSERT_EQ(requirements.director_ready_bytes, CORE_SYNC_SLOT_BYTES);
    ASSERT_TRUE(requirements.director_counter_required);
    ASSERT_EQ(requirements.raw_release_count, 1u);

    auto ready_buffer = trun::create_zeroed_l1_buffer(f.mesh.get(), kL1BlockBytes);
    auto result_buffer = trun::create_zeroed_l1_buffer(f.mesh.get(), kL1BlockBytes);
    const uint32_t director_ready_base = static_cast<uint32_t>(ready_buffer->address());
    const uint32_t result_l1_addr = static_cast<uint32_t>(result_buffer->address());

    hregions::CoreSyncMemoryBindings memory;
    memory.director_ready = hreg::Region{director_ready_base, kL1BlockBytes};
    memory.raw_releases.push_back(XR.ready);
    const auto roster = builder.build(memory);
    ASSERT_EQ(roster.counter_arrivals, tensix_workers.size());
    ASSERT_EQ(roster.word_arrivals.size(), 1u)
        << "a roster of Tensix workers arrives entirely on the counter, and the X280 "
           "contributes the only ready word";
    const uint32_t x280_word_index = roster.word_arrivals.front().worker_index;

    const auto director_virtual = x280::X280Runtime::virtual_core_from_logical_core(
        *f.mesh, director_core, tt::CoreType::TENSIX, ttm::NOC::NOC_0);

    const uint32_t nonce = trun::mint_run_nonce();

    core_sync_director_section_t section{};
    section.cookie_common = core_sync_director_cookie_for(UBENCH_SECTION_COMMON);
    section.cookie_backend = core_sync_director_cookie_for(UBENCH_SECTION_BACKEND);
    section.common.run_nonce = nonce;
    section.common.role = CORE_SYNC_DIRECTOR_ROLE_PARTICIPANT;
    section.backend.result_addr = XR.result.addr;
    section.backend.local_release_addr = x280_release_addr;
    section.backend.arrival.target = {director_virtual.noc(), director_ready_base, x280_word_index, 0u};
    section.backend.arrival.ready_bytes = kL1BlockBytes;
    hreg::write_args(*f.tile, XR.args, section);

    const std::vector<std::filesystem::path> include_paths = {
        spec.open.kernel_dir,
        hart_::shared_kernel_dir(spec.open.kernel_dir, "shared"),
        hart_::x280_root_of(spec.open.kernel_dir) / "include",
    };
    const auto kpath = spec.open.kernel_dir / "core_sync_probe.cpp";
    trun::add_kernels(
        program,
        kpath,
        include_paths,
        director_core,
        worker_set,
        roster,
        nonce,
        director_ready_base,
        result_l1_addr,
        roster.counter_arrivals);

    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_CORE_SYNC_DIRECTOR_FW_IMAGE", "firmware-core-sync-director.bin"),
        .run_value = (static_cast<uint64_t>(0xC5C6u) << 32) | nonce,
        .args_addr = XR.args.addr,
        .args_bytes = XR.args.bytes,
        .cleanup_timeout = std::chrono::seconds{10},
        .wait_timeout = std::chrono::seconds{60},
    };

    hexec::enqueue_tensix(*f.mesh, workload);
    const auto outcome = hexec::run_x280(f.handle(), launch);
    hexec::finish_tensix(*f.mesh);

    ASSERT_TRUE(outcome.enqueued()) << "enqueue rejected";
    EXPECT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    const auto x280_result = hreg::read_struct<core_sync_director_result_t>(*f.tile, XR.result);
    std::printf(
        "Mixed-tier arrival into a Tensix director: %zu counter arrivals, "
        "%zu word arrivals; X280 participant terminal state %s\n",
        tensix_workers.size(),
        roster.word_arrivals.size(),
        state_name(x280_result.terminal_state));
    EXPECT_EQ(x280_result.terminal_state, CORE_SYNC_DIRECTOR_STATE_DONE)
        << "the X280's word arrival or its release did not complete";

    const auto director_result = trun::read_result(f.mesh.get(), director_core, result_l1_addr, nonce);
    EXPECT_EQ(director_result.outcome, static_cast<uint32_t>(::core_sync::Outcome::Success))
        << "the Tensix director did not see both tiers arrive; outcome " << director_result.outcome;

    const auto worker_outcomes = trun::read_outcomes(f.mesh.get(), worker_set, result_l1_addr, nonce);
    for (size_t i = 0; i < worker_outcomes.size(); ++i) {
        EXPECT_EQ(worker_outcomes[i], static_cast<uint32_t>(::core_sync::Outcome::Success))
            << "Tensix worker " << tensix_workers[i].str() << " outcome was " << worker_outcomes[i];
    }
}

/* Allocates the probe request and result from one tile available LIM. */
struct ReleaseProbeRegions {
    hreg::Region args;
    hreg::Region result;

    static ReleaseProbeRegions compute(x280::X280Tile& tile) {
        const auto available = tile.available_lim();
        hreg::Arena arena(hreg::Region{available.begin, available.size()});
        return ReleaseProbeRegions{
            .args = arena.allocate(sizeof(core_sync_release_probe_args_t)),
            .result = arena.allocate(sizeof(core_sync_release_probe_result_t)),
        };
    }
};

/* Executes the X280 release transport over one or two ordered singleton groups
 * at logical (x0,0) and (x0+2,0) and verifies every selected Tensix destination
 * received the token. */
inline void run_release_probe(x280::harness::session::Fixture& f, uint32_t group_count, uint32_t x0 = 0) {
    ASSERT_TRUE(group_count == 1u || group_count == 2u)
        << "the hardware probe covers singleton and ordered two-group release";

    const std::vector<ttm::CoreCoord> workers = {{x0, 0u}, {x0 + 2u, 0u}};
    const auto l1 = hreg::allocate_tile_l1_buffer(
        f.mesh.get(),
        ttm::CoreRangeSet(std::vector<ttm::CoreRange>{
            ttm::CoreRange(workers[0], workers[0]), ttm::CoreRange(workers[1], workers[1])}),
        kL1BlockBytes);
    const uint32_t release_addr = l1.l1_tile_base;

    const ReleaseProbeRegions regions = ReleaseProbeRegions::compute(*f.tile);
    ASSERT_LE(regions.result.addr, std::numeric_limits<uint32_t>::max());
    hreg::zero_region(*f.tile, regions.result);

    const uint32_t nonce = trun::mint_run_nonce();
    core_sync_release_probe_args_t args{};
    args.result_addr = static_cast<uint32_t>(regions.result.addr);
    args.release_value = core_sync_release_token(nonce);
    args.group_count = group_count;
    for (uint32_t i = 0u; i < group_count; ++i) {
        const auto virtual_coord = x280::X280Runtime::virtual_core_from_logical_core(
            *f.mesh, workers[i], tt::CoreType::TENSIX, ttm::NOC::NOC_0);
        args.groups[i] = {{virtual_coord.noc(), virtual_coord.noc()}, release_addr, 1u};
    }
    hreg::write_args(*f.tile, regions.args, args);

    hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_CORE_SYNC_RELEASE_PROBE_FW_IMAGE", "firmware-core-sync-release-probe.bin"),
        .run_value = (static_cast<uint64_t>(0xC5C7u) << 32) | nonce,
        .args_addr = regions.args.addr,
        .args_bytes = regions.args.bytes,
        .cleanup_timeout = std::chrono::seconds{10},
        .wait_timeout = std::chrono::seconds{30},
    };
    const auto outcome = hexec::run_x280(f.handle(), launch);
    ASSERT_TRUE(outcome.enqueued()) << "release-probe enqueue rejected";
    EXPECT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    const auto result = hreg::read_struct<core_sync_release_probe_result_t>(*f.tile, regions.result);
    EXPECT_EQ(result.cookie, CORE_SYNC_RELEASE_PROBE_RESULT_COOKIE);
    EXPECT_EQ(result.state, CORE_SYNC_RELEASE_PROBE_DONE);
    EXPECT_EQ(result.group_count, group_count);
    EXPECT_EQ(result.release_value, args.release_value);

    auto& cluster = x280::harness::cluster();
    for (uint32_t i = 0u; i < workers.size(); ++i) {
        uint32_t observed = 0xFFFFFFFFu;
        cluster.read_core(
            &observed,
            sizeof(observed),
            tt_cxy_pair(f.mesh->build_id(), f.mesh->worker_core_from_logical_core(workers[i])),
            release_addr);
        EXPECT_EQ(observed, i < group_count ? args.release_value : 0u)
            << "singleton group " << i << " had the wrong release value";
    }
}

/* Runs the singleton probe, then the ordered two-group probe, on one fixture. */
inline void run_release_groups(x280::harness::session::Fixture& f) {
    ASSERT_NE(f.mesh, nullptr);
    run_release_probe(f, 1u);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    run_release_probe(f, 2u);
}

/* Returns the first logical x whose 4x4 block (x..x+3, 0..3) has consecutive
 * virtual columns and whose columns x and x+2 have virtual coordinates
 * different from NOC0 on the fixture's device.
 * std::nullopt means no such block exists, as on an unharvested part. */
inline std::optional<uint32_t> first_renumbered_block_x(const x280::harness::session::Fixture& f) {
    const auto& soc = x280::harness::cluster().get_soc_desc(f.tile_id.chip_id);
    const auto grid = f.mesh->compute_with_storage_grid_size();
    const auto renumbered = [&](uint32_t x) {
        const auto noc0 = soc.translate_coord_to(
            tt::umd::CoreCoord(x, 0, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL), tt::CoordSystem::NOC0);
        const auto virtual_core = f.mesh->worker_core_from_logical_core(ttm::CoreCoord{x, 0});
        return static_cast<size_t>(noc0.x) != virtual_core.x || static_cast<size_t>(noc0.y) != virtual_core.y;
    };
    for (uint32_t x = 0; x + 3u < grid.x && grid.y >= 4u; ++x) {
        bool consecutive = true;
        for (uint32_t i = 0; i < 3u && consecutive; ++i) {
            consecutive = f.mesh->worker_core_from_logical_core(ttm::CoreCoord{x + i + 1u, 0}).x ==
                          f.mesh->worker_core_from_logical_core(ttm::CoreCoord{x + i, 0}).x + 1u;
        }
        if (consecutive && renumbered(x) && renumbered(x + 2u)) {
            return x;
        }
    }
    return std::nullopt;
}

/* Runs the mixed-tier arrival, the ordered two-group release probe and the
 * block's multicast release on a block whose virtual coordinates differ from
 * NOC0, where a NOC0 destination would name another tile. Skips where the
 * device has no such block. */
inline void run_renumbered_column(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec) {
    ASSERT_NE(f.mesh, nullptr);
    const auto x0 = first_renumbered_block_x(f);
    if (!x0) {
        GTEST_SKIP() << "availability: every Tensix column's virtual coordinate equals NOC0 "
                        "on this device (no harvesting renumbering), so NOC0 and virtual "
                        "destinations are indistinguishable";
    }
    std::printf("renumbered-column block at logical x %u..%u\n", *x0, *x0 + 3u);
    run_mixed_tier(f, spec, *x0);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    run_release_probe(f, 2u, *x0);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    run_multicast_block(f, spec, *x0);
}

/* A 2 x 2 logical block at columns x0, x0 + 1 and rows 0..1, and why the
 * block was chosen. */
struct GuardedSelection {
    const char* label;
    uint32_t x0;

    ttm::CoreRange block() const { return ttm::CoreRange(ttm::CoreCoord{x0, 0}, ttm::CoreCoord{x0 + 1u, 1}); }
};

/* Picks, from the fixture's own mapping, the first column pair of each shape
 * the mapping contract must hold for, in this order: virtually and
 * NOC0-adjacent ("contiguous control"), virtually split across translated
 * columns 8-9, and virtually adjacent with a harvested column between them in
 * NOC0. A shape the part lacks is absent from the result and reported on
 * stdout. Needs an open mesh. */
inline std::vector<GuardedSelection> guarded_selections(const x280::harness::session::Fixture& f) {
    const auto& soc = x280::harness::cluster().get_soc_desc(f.tile_id.chip_id);
    const auto grid = f.mesh->compute_with_storage_grid_size();
    const auto virt_x = [&](uint32_t x) { return f.mesh->worker_core_from_logical_core(ttm::CoreCoord{x, 0}).x; };
    const auto noc0_x = [&](uint32_t x) {
        return static_cast<size_t>(
            soc.translate_coord_to(
                   tt::umd::CoreCoord(x, 0, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL), tt::CoordSystem::NOC0)
                .x);
    };
    std::optional<uint32_t> control, columns_8_9, harvested;
    for (uint32_t x = 0; x + 1u < grid.x && grid.y >= 2u; ++x) {
        const bool virt_adjacent = virt_x(x + 1u) == virt_x(x) + 1u;
        if (!control && virt_adjacent && noc0_x(x + 1u) == noc0_x(x) + 1u) {
            control = x;
        }
        if (!columns_8_9 && virt_x(x) <= 7u && virt_x(x + 1u) >= 10u) {
            columns_8_9 = x;
        }
        if (!harvested && virt_adjacent && noc0_x(x + 1u) > noc0_x(x) + 1u) {
            harvested = x;
        }
    }
    std::vector<GuardedSelection> out;
    const auto take = [&](const std::optional<uint32_t>& x0, const char* label) {
        if (x0) {
            out.push_back({label, *x0});
        } else {
            std::printf("guarded release: no column pair on this part %s\n", label);
        }
    };
    take(control, "contiguous control");
    take(columns_8_9, "spans translated columns 8-9");
    take(harvested, "spans a harvested NOC0 column");
    return out;
}

/* Releases each guarded selection by a required multicast and checks every
 * worker was released, and that no live Tensix core outside the selection
 * took the token: the recipient-level check of the mapping contract. Each
 * selection shape the part lacks is reported, not failed. */
inline void run_guarded_multicast(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec) {
    ASSERT_NE(f.mesh, nullptr);
    for (const auto& sel : guarded_selections(f)) {
        SCOPED_TRACE(sel.label);
        const ttm::CoreRange block = sel.block();
        const auto workers = ttm::corerange_to_cores(ttm::CoreRangeSet(block));
        const auto guards = x280::harness::regions::guards_outside(*f.mesh, ttm::CoreRangeSet(block));
        const auto plan = select_tensix_release(*f.mesh, workers, hreg::ReleaseMode::RequireMulticast);
        const auto r = run_once(
            f,
            spec,
            workers,
            ttm::CoreRangeSet(block),
            Release::Multicast,
            /*abandon=*/false,
            std::chrono::seconds{60},
            guards);
        std::printf(
            "guarded multicast, %s: logical (%u, 0)-(%u, 1), virtual (%u, %u)-(%u, %u), "
            "destinations %u, endpoints %u, director %s, guards %zu overwritten %zu\n",
            sel.label,
            sel.x0,
            sel.x0 + 1u,
            plan.group->start.x(),
            plan.group->start.y(),
            plan.group->end.x(),
            plan.group->end.y(),
            r.grid_num_dests,
            r.endpoint_count,
            state_name(r.director_terminal_state),
            guards.size(),
            r.overwritten_guards.size());
        EXPECT_EQ(r.director_terminal_state, CORE_SYNC_DIRECTOR_STATE_DONE);
        EXPECT_EQ(r.grid_num_dests, workers.size());
        EXPECT_EQ(r.endpoint_count, 0u);
        expect_all_workers_succeeded(r, sel.label);
        for (const auto& g : r.overwritten_guards) {
            ADD_FAILURE() << "logical core " << g.core.str() << " outside the selection took the release";
        }
    }
}

/* What one active-ERISC run produced, per ERISC processor. */
struct ActiveEriscResult {
    uint32_t director_terminal_state = CORE_SYNC_DIRECTOR_STATE_INIT;
    uint32_t erisc_outcome[2] = {trun::kProbeNotWritten, trun::kProbeNotWritten};
    ttm::CoreCoord logical{};
    ttm::CoreCoord translated{};
};

/* The X280 director releases both ERISCs of one active Ethernet core: ERISC0
 * on NoC 0 and ERISC1 on NoC 1, each a word-arrival worker writing into the
 * director's ready table in LIM and polling its own release word in ERISC L1.
 *
 * The director's window reaches the Ethernet tile at its translated
 * coordinate, as p2p_latency's metal_peer_runner does; each ERISC reaches the
 * L2CPU at its translated coordinate, which is valid on both NoCs for an L2CPU
 * (X280Runtime::virtual_core_from_tile refuses it otherwise). `misaim_erisc1` releases a word ERISC1 does not poll,
 * so its poll must expire: the negative control. */
inline ActiveEriscResult run_active_erisc_once(
    x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, bool misaim_erisc1) {
    ActiveEriscResult out;
    ttm::IDevice* device = f.mesh->get_devices()[0];
    auto& cluster = x280::harness::cluster();
    const auto& hal = ttm::MetalContext::instance().hal();

    std::optional<ttm::CoreCoord> chosen;
    for (const auto& core : device->get_active_ethernet_cores(true)) {
        if (cluster.is_ethernet_link_up(device->id(), core)) {
            chosen = core;
            break;
        }
    }
    if (!chosen) {
        ADD_FAILURE() << "chip " << device->id() << " has no active Ethernet core with its link up";
        return out;
    }
    out.logical = *chosen;
    out.translated = device->ethernet_core_from_logical_core(*chosen);

    const hreg::L1Window window =
        hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::ACTIVE_ETH, hreg::kCacheLineBytes);
    const uint32_t base = static_cast<uint32_t>(window.local.base);
    const uint32_t result_addr[2] = {base, base + 64u};
    const uint32_t release_addr[2] = {base + 128u, base + 192u};
    const uint32_t unpolled_addr = base + 256u;
    const tt_cxy_pair eth(f.mesh->build_id(), out.translated);
    const std::vector<uint32_t> zeros(80u, 0u);
    cluster.write_core(zeros.data(), 320u, eth, base);

    const X280Regions XR = X280Regions::compute(*f.tile, 2u);
    hreg::zero_region(*f.tile, XR.ready);
    hreg::zero_region(*f.tile, XR.result);

    const uint32_t nonce = trun::mint_run_nonce();
    core_sync_director_section_t section{};
    section.cookie_common = core_sync_director_cookie_for(UBENCH_SECTION_COMMON);
    section.cookie_backend = core_sync_director_cookie_for(UBENCH_SECTION_BACKEND);
    section.common.run_nonce = nonce;
    section.common.role = CORE_SYNC_DIRECTOR_ROLE_DIRECTOR;
    section.backend.result_addr = XR.result.addr;
    section.backend.ready_base = static_cast<uint32_t>(XR.ready.addr);
    section.backend.ready_bytes = static_cast<uint32_t>(XR.ready.bytes);
    /* Both ERISCs of the tile are participants; an Ethernet tile is always an
     * endpoint, so the selector returns two. */
    const auto plan = hreg::select_release(
        *f.mesh,
        {{tt::CoreType::ETH, *chosen}, {tt::CoreType::ETH, *chosen}},
        hreg::kX280ReleaseSender,
        hreg::ReleaseMode::Auto,
        CORE_SYNC_DIRECTOR_MAX_ENDPOINTS);
    section.backend.arrival_count = 2u;
    section.backend.endpoint_count = static_cast<uint32_t>(plan.endpoints.size());
    for (uint32_t i = 0; i < 2u; ++i) {
        section.backend.arrivals[i].addr = core_sync_slot_addr(static_cast<uint32_t>(XR.ready.addr), i);
        section.backend.arrivals[i].worker_index = i;
    }
    for (uint32_t i = 0; i < plan.endpoints.size(); ++i) {
        section.backend.endpoints[i] = {
            plan.translated[plan.endpoints[i]].noc(), (i == 1u && misaim_erisc1) ? unpolled_addr : release_addr[i], 0u};
    }
    hreg::write_args(*f.tile, XR.args, section);

    const ttm::CoreRangeSet core_set(ttm::CoreRange(*chosen, *chosen));
    const auto entry = spec.open.kernel_dir / "core_sync_probe.cpp";
    ttm::Program program = ttm::CreateProgram();
    for (uint32_t i = 0; i < 2u; ++i) {
        const ttm::NOC noc = i == 0u ? ttm::NOC::NOC_0 : ttm::NOC::NOC_1;
        const auto director = f.runtime->virtual_core_from_tile(f.tile_id, noc);
        const auto kernel = ttm::CreateKernel(
            program,
            entry.string(),
            core_set,
            ttm::EthernetConfig{
                .eth_mode = ttm::Eth::SENDER,
                .noc = noc,
                .processor = i == 0u ? ttm::DataMovementProcessor::RISCV_0 : ttm::DataMovementProcessor::RISCV_1,
                .compile_args = {0u, i, 1u, 1u, CORE_SYNC_CORE_FAMILY_ACTIVE_ETH},
            });
        ttm::SetRuntimeArgs(
            program,
            kernel,
            core_set,
            word_worker_args(
                {nonce, static_cast<uint32_t>(XR.ready.addr), result_addr[i], 0u}, director, 0u, i, release_addr[i]));
    }
    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    const hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_CORE_SYNC_DIRECTOR_FW_IMAGE", "firmware-core-sync-director.bin"),
        .run_value = (static_cast<uint64_t>(0xC5C6u) << 32) | nonce,
        .args_addr = XR.args.addr,
        .args_bytes = XR.args.bytes,
        .cleanup_timeout = std::chrono::seconds{10},
        .wait_timeout = std::chrono::seconds{30},
    };
    hexec::enqueue_tensix(*f.mesh, workload);
    const auto outcome = hexec::run_x280(f.handle(), launch);
    hexec::finish_tensix(*f.mesh);
    EXPECT_TRUE(outcome.enqueued()) << "enqueue rejected";
    EXPECT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    out.director_terminal_state = hreg::read_struct<core_sync_director_result_t>(*f.tile, XR.result).terminal_state;
    for (uint32_t i = 0; i < 2u; ++i) {
        out.erisc_outcome[i] = trun::read_result_at(f.mesh.get(), out.translated, result_addr[i], nonce).outcome;
    }
    return out;
}

/* Active ERISCs as word-arrival workers under an X280 director, with the
 * misaimed release as the negative control that the check can fail. */
inline void run_active_erisc_workers(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec) {
    ASSERT_NE(f.mesh, nullptr);
    const auto& hal = ttm::MetalContext::instance().hal();
    if (hal.get_num_risc_processors(ttm::HalProgrammableCoreType::ACTIVE_ETH) < 2u) {
        GTEST_SKIP() << "ACTIVE_ETH reports fewer than two processors (single-ERISC mode)";
    }
    const uint32_t success = static_cast<uint32_t>(::core_sync::Outcome::Success);

    const auto good = run_active_erisc_once(f, spec, /*misaim_erisc1=*/false);
    std::printf(
        "X280 director, active ERISC at logical (%zu, %zu) translated (%zu, %zu): "
        "director %s, ERISC0 outcome %u, ERISC1 outcome %u\n",
        good.logical.x,
        good.logical.y,
        good.translated.x,
        good.translated.y,
        state_name(good.director_terminal_state),
        good.erisc_outcome[0],
        good.erisc_outcome[1]);
    EXPECT_EQ(good.director_terminal_state, CORE_SYNC_DIRECTOR_STATE_DONE);
    EXPECT_EQ(good.erisc_outcome[0], success) << "ERISC0 (NoC 0)";
    EXPECT_EQ(good.erisc_outcome[1], success) << "ERISC1 (NoC 1)";

    const auto misaimed = run_active_erisc_once(f, spec, /*misaim_erisc1=*/true);
    std::printf(
        "negative control, ERISC1 release misaimed: director %s, ERISC0 outcome %u, "
        "ERISC1 outcome %u\n",
        state_name(misaimed.director_terminal_state),
        misaimed.erisc_outcome[0],
        misaimed.erisc_outcome[1]);
    EXPECT_EQ(misaimed.director_terminal_state, CORE_SYNC_DIRECTOR_STATE_DONE)
        << "both ERISCs still arrive, so the director releases and finishes";
    EXPECT_EQ(misaimed.erisc_outcome[0], success) << "ERISC0 is released as before";
    EXPECT_EQ(misaimed.erisc_outcome[1], static_cast<uint32_t>(::core_sync::Outcome::StopRequested))
        << "ERISC1 polls a word nobody writes, so its bounded poll must expire";
}

}  // namespace x280::suites::core_sync::x280_director

#endif  // X280_SUITES_CORE_SYNC_X280_DIRECTOR_HPP
