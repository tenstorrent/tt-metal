// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/core_sync/erisc_worker.hpp -- a Tensix director releases a grid of
// Tensix workers and one idle ERISC in the same barrier.
//
// The ERISC is the roster's only non-Tensix Metal-launched participant, and
// every step it takes is a different one: its logical coordinate resolves
// through ethernet_core_from_logical_core, its semaphores are allocated against
// CoreType::ETH, its release semaphore resolves in the IDLE_ETH address space,
// and no broadcast reaches it, so it takes a unicast write while the Tensix
// grid takes the rectangle.
//
// The test compares barrier outcomes only. A release resolved in the wrong
// address space leaves the ERISC poll to expire as Outcome::StopRequested.

#ifndef X280_SUITES_CORE_SYNC_ERISC_WORKER_HPP
#define X280_SUITES_CORE_SYNC_ERISC_WORKER_HPP

#include "harness/metal_context.hpp"
#include "harness/regions/core_sync_roster.hpp"
#include "harness/regions/metal_l1_region.hpp"
#include "tests/device/suites/core_sync/probe_args.hpp"
#include "tests/device/suites/core_sync/tensix_directed_run.hpp"

#include "core_sync_barrier.hpp"
#include "core_sync_core_family.h"
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
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>

namespace x280::suites::core_sync::erisc_worker {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hreg = x280::harness::regions;
namespace trun = x280::suites::core_sync::tensix_directed;

/* What one mixed-family run produced. tensix_outcomes follows roster order;
 * erisc_outcome is the single Ethernet participant. */
struct RunResult {
    std::vector<uint32_t> tensix_outcomes;
    uint32_t erisc_outcome = trun::kProbeNotWritten;
    uint32_t director_outcome = trun::kProbeNotWritten;
    uint32_t grid_num_dests = 0;
    uint32_t grid_x0 = 0;
    uint32_t grid_y0 = 0;
    uint32_t grid_x1 = 0;
    uint32_t grid_y1 = 0;
    uint32_t endpoint_count = 0;
    ProgrammableCoreType endpoint_core_type = ProgrammableCoreType::TENSIX;
    uint32_t erisc_noc_x = 0;
    uint32_t erisc_noc_y = 0;
};

/* Runs the barrier once over the Tensix grid plus one idle ERISC, and harvests
 * every participant.
 *
 * erisc_logical is the idle ethernet logical core the caller selected; the
 * roster refuses it if it is not of the ethernet family. */
inline RunResult run_once(
    ttmd::MeshDevice* mesh,
    const ttm::CoreCoord& director_core,
    const ttm::CoreRangeSet& tensix_workers,
    const ttm::CoreCoord& erisc_logical,
    const std::filesystem::path& kernel_path,
    const std::vector<std::filesystem::path>& include_paths) {
    const ttm::CoreRangeSet erisc_set(ttm::CoreRange(erisc_logical, erisc_logical));

    auto result_buffer = trun::create_zeroed_l1_buffer(mesh);

    const uint32_t director_ready_base = 0u;
    const uint32_t result_l1_addr = static_cast<uint32_t>(result_buffer->address());

    /* A MeshBuffer is allocated in the Tensix L1 space, which the ERISC's own
     * L1 does not share, so its result word goes at the base of the unreserved
     * ethernet window instead. */
    const auto& hal = ttm::MetalContext::instance().hal();
    const hreg::L1Window eth_window = hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::IDLE_ETH);
    const uint32_t erisc_result_addr = static_cast<uint32_t>(eth_window.local.base);

    ttm::IDevice* device = mesh->get_devices()[0];
    auto& cluster = x280::harness::cluster();
    const ttm::CoreCoord erisc_virtual = device->ethernet_core_from_logical_core(erisc_logical);
    const std::vector<uint32_t> zeros(CORE_SYNC_PROBE_RESULT_WORDS, 0u);
    cluster.write_core(
        zeros.data(),
        sizeof(uint32_t) * CORE_SYNC_PROBE_RESULT_WORDS,
        tt_cxy_pair(mesh->build_id(), erisc_virtual),
        erisc_result_addr);

    ttm::Program program = ttm::CreateProgram();
    hreg::CoreSyncRosterBuilder<> builder(
        program, *mesh, director_core, ttm::NOC::NOC_0, hreg::ReleaseMode::RequireMulticast);
    builder.add_workers<::core_sync::MetalTransport<ProgrammableCoreType::TENSIX>>(tensix_workers);
    builder.add_workers<::core_sync::MetalTransport<ProgrammableCoreType::IDLE_ETH>>(erisc_set);
    const auto roster = builder.build();

    const uint32_t nonce = trun::mint_run_nonce();
    trun::add_kernels(
        program,
        kernel_path,
        include_paths,
        director_core,
        tensix_workers,
        roster,
        nonce,
        director_ready_base,
        result_l1_addr,
        roster.counter_arrivals);

    const auto erisc_kernel = ttm::CreateKernel(
        program,
        kernel_path.string(),
        erisc_set,
        ttm::EthernetConfig{
            .eth_mode = ttm::Eth::IDLE,
            .noc = ttm::NOC::NOC_0,
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .compile_args = {0u, 0u, 1u, 0u, CORE_SYNC_CORE_FAMILY_IDLE_ETH},
        });
    ttm::SetRuntimeArgs(
        program,
        erisc_kernel,
        erisc_set,
        counter_worker_args({nonce, director_ready_base, erisc_result_addr, roster.arrival_semaphore_id}, roster));

    trun::execute(mesh, std::move(program));

    RunResult out;
    out.grid_num_dests = roster.grid.num_dests;
    out.grid_x0 = roster.grid.rect.start.x;
    out.grid_y0 = roster.grid.rect.start.y;
    out.grid_x1 = roster.grid.rect.end.x;
    out.grid_y1 = roster.grid.rect.end.y;
    out.endpoint_count = static_cast<uint32_t>(roster.worker_endpoints.size());
    if (!roster.worker_endpoints.empty()) {
        out.endpoint_core_type = roster.worker_endpoints.front().core_type;
        out.erisc_noc_x = roster.worker_endpoints.front().noc.x;
        out.erisc_noc_y = roster.worker_endpoints.front().noc.y;
    }

    out.director_outcome = trun::read_result(mesh, director_core, result_l1_addr, nonce).outcome;
    out.tensix_outcomes = trun::read_outcomes(mesh, tensix_workers, result_l1_addr, nonce);
    out.erisc_outcome = trun::read_result_at(mesh, erisc_virtual, erisc_result_addr, nonce).outcome;
    return out;
}

inline void expect_all_succeeded(const RunResult& r, const char* what) {
    EXPECT_EQ(r.director_outcome, static_cast<uint32_t>(::core_sync::Outcome::Success))
        << what << ": the director outcome was " << r.director_outcome
        << "; it waits for one counter arrival per participant, the ERISC's included";
    for (size_t i = 0; i < r.tensix_outcomes.size(); ++i) {
        EXPECT_EQ(r.tensix_outcomes[i], static_cast<uint32_t>(::core_sync::Outcome::Success))
            << what << ": Tensix worker " << i << " outcome was " << r.tensix_outcomes[i];
    }
    EXPECT_EQ(r.erisc_outcome, static_cast<uint32_t>(::core_sync::Outcome::Success))
        << what << ": the idle ERISC outcome was " << r.erisc_outcome
        << "; StopRequested means its poll expired, which is what a release resolved "
           "at the Tensix semaphore base rather than the ethernet one looks like";
}

/* A Tensix director, a Tensix grid released by one broadcast, and one idle
 * ERISC released by the unicast that follows it.
 *
 * Skips, rather than fails, on a part with no idle ethernet core: there is
 * nothing to place the participant on. */
inline void run_erisc_worker(
    ttmd::MeshDevice* mesh,
    const std::filesystem::path& kernel_path,
    const std::vector<std::filesystem::path>& include_paths) {
    ASSERT_NE(mesh, nullptr) << "unit mesh not initialized";

    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::IDLE_ETH)) {
        GTEST_SKIP() << "IDLE_ETH programmable cores not enabled";
    }
    ASSERT_FALSE(mesh->get_devices().empty()) << "unit mesh has no devices";
    ttm::IDevice* device = mesh->get_devices()[0];
    const auto inactive_cores = device->get_inactive_ethernet_cores();
    if (inactive_cores.empty()) {
        GTEST_SKIP() << "no inactive Ethernet core available";
    }
    const ttm::CoreCoord erisc_logical = *inactive_cores.begin();

    /* The same 4x4 block of logical Tensix cores the grid case releases, with
     * the director in one corner, so the rectangle covers the director and
     * grid.num_dests is one short of its tile count. */
    const ttm::CoreCoord director_core{0, 0};
    const ttm::CoreRangeSet tensix_workers = trun::worker_grid();
    const uint32_t tensix_count = static_cast<uint32_t>(ttm::corerange_to_cores(tensix_workers).size());

    const RunResult barrier = run_once(mesh, director_core, tensix_workers, erisc_logical, kernel_path, include_paths);

    std::cout << "Tensix director, " << tensix_count << " Tensix workers and one idle "
              << "ERISC at logical (" << erisc_logical.x << ", " << erisc_logical.y << "), translated ("
              << barrier.erisc_noc_x << ", " << barrier.erisc_noc_y << "):\n"
              << "  rectangle (" << barrier.grid_x0 << ", " << barrier.grid_y0 << ")-(" << barrier.grid_x1 << ", "
              << barrier.grid_y1 << "), destinations " << barrier.grid_num_dests << ", unicast endpoints "
              << barrier.endpoint_count << "\n";

    EXPECT_EQ(barrier.grid_num_dests, tensix_count)
        << "the broadcast rectangle must cover the Tensix workers alone; an ERISC inside "
           "it is both unreachable by the broadcast and far outside the Tensix grid";
    ASSERT_EQ(barrier.endpoint_count, 1u)
        << "the ERISC is the only participant no broadcast reaches, so it is the only "
           "unicast endpoint";
    EXPECT_EQ(static_cast<uint32_t>(barrier.endpoint_core_type), static_cast<uint32_t>(ProgrammableCoreType::IDLE_ETH))
        << "the ERISC's endpoint must carry its own family, which is what resolves its "
           "release semaphore in the ethernet address space";
    EXPECT_TRUE(
        barrier.erisc_noc_x < barrier.grid_x0 || barrier.erisc_noc_x > barrier.grid_x1 ||
        barrier.erisc_noc_y < barrier.grid_y0 || barrier.erisc_noc_y > barrier.grid_y1)
        << "the ERISC at translated (" << barrier.erisc_noc_x << ", " << barrier.erisc_noc_y
        << ") lies inside the broadcast rectangle (" << barrier.grid_x0 << ", " << barrier.grid_y0 << ")-("
        << barrier.grid_x1 << ", " << barrier.grid_y1 << "), so it would be released twice or not at all";

    expect_all_succeeded(barrier, "mixed-family barrier");
}

}  // namespace x280::suites::core_sync::erisc_worker

#endif  // X280_SUITES_CORE_SYNC_ERISC_WORKER_HPP
