// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// A Tensix director releases a Tensix grid and one DRISC worker. Tensix
// workers arrive through the counter and take multicast release; the DRISC
// arrives by word and takes a raw-L1 unicast release.

#ifndef X280_SUITES_CORE_SYNC_DRISC_WORKER_HPP
#define X280_SUITES_CORE_SYNC_DRISC_WORKER_HPP

#include "harness/metal_context.hpp"
#include "harness/regions/core_sync_roster.hpp"
#include "harness/regions/metal_l1_region.hpp"
#include "harness/targets/drisc_endpoint.hpp"
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
#include <distributed/mesh_device_impl.hpp>
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace x280::suites::core_sync::drisc_worker {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hreg = x280::harness::regions;
namespace htar = x280::harness::targets;
namespace trun = x280::suites::core_sync::tensix_directed;

constexpr uint32_t kDriscStorageBytes = 64u;

struct RunResult {
    std::vector<uint32_t> tensix_outcomes;
    uint32_t director_outcome = trun::kProbeNotWritten;
    uint32_t drisc_outcome = trun::kProbeNotWritten;
    uint32_t counter_arrivals = 0u;
    uint32_t word_arrivals = 0u;
    uint32_t grid_num_dests = 0u;
    uint32_t endpoint_count = 0u;
    uint32_t release_word = 0u;
    uint32_t expected_release = 0u;
};

/* Executes one mixed Tensix/DRISC barrier. abandon omits one rostered Tensix
 * kernel, so the director releases nobody and every launched worker expires. */
inline RunResult run_once(
    ttmd::MeshDevice* mesh,
    const ttm::CoreCoord& drisc_logical,
    const std::filesystem::path& kernel_path,
    const std::vector<std::filesystem::path>& include_paths,
    bool abandon) {
    const ttm::CoreCoord director_core{0, 0};
    const ttm::CoreRangeSet roster_tensix = trun::worker_grid();
    const ttm::CoreRangeSet launched_tensix = trun::worker_grid(abandon);
    const auto roster_tensix_cores = ttm::corerange_to_cores(roster_tensix);
    const ttm::CoreRangeSet drisc_set(ttm::CoreRange(drisc_logical, drisc_logical));

    auto ready_buffer = trun::create_zeroed_l1_buffer(mesh);
    auto result_buffer = trun::create_zeroed_l1_buffer(mesh);

    const uint32_t director_ready_base = static_cast<uint32_t>(ready_buffer->address());
    const uint32_t result_l1_addr = static_cast<uint32_t>(result_buffer->address());
    const auto& hal = ttm::MetalContext::instance().hal();
    const hreg::L1Window drisc_window = hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::DRAM);
    auto drisc_storage = mesh->impl().drisc_l1_arena().allocate(kDriscStorageBytes, /*alignment=*/64u);
    if (drisc_storage == nullptr) {
        throw std::runtime_error("core_sync DRISC worker: failed to allocate DRISC L1 storage");
    }
    const uint64_t drisc_host_addr = hreg::host_noc_address(drisc_window, drisc_storage->addr(), drisc_storage->size());

    const uint32_t drisc_release_addr = static_cast<uint32_t>(drisc_storage->addr());
    const uint32_t drisc_result_addr = drisc_release_addr + CORE_SYNC_SLOT_BYTES;

    auto& cluster = x280::harness::cluster();
    const ttm::CoreCoord drisc_virtual = mesh->virtual_core_from_logical_core(drisc_logical, tt::CoreType::DRAM);
    const std::array<uint32_t, kDriscStorageBytes / sizeof(uint32_t)> drisc_zeros{};
    cluster.write_core(
        drisc_zeros.data(), kDriscStorageBytes, tt_cxy_pair(mesh->build_id(), drisc_virtual), drisc_host_addr);

    ttm::Program program = ttm::CreateProgram();
    hreg::CoreSyncRosterBuilder<> builder(
        program, *mesh, director_core, ttm::NOC::NOC_0, hreg::ReleaseMode::RequireMulticast);
    builder.add_workers<::core_sync::MetalTransport<ProgrammableCoreType::TENSIX>>(roster_tensix);
    builder.add_workers<::core_sync::MetalTransport<ProgrammableCoreType::DRAM, ::core_sync::ArrivalKind::Word>>(
        drisc_set);

    const auto requirements = builder.memory_requirements();
    if (requirements.director_ready_bytes != CORE_SYNC_SLOT_BYTES || !requirements.director_counter_required ||
        requirements.raw_release_count != 1u) {
        throw std::runtime_error("core_sync DRISC worker: unexpected storage requirements");
    }

    hreg::CoreSyncMemoryBindings memory;
    memory.director_ready = hreg::Region{director_ready_base, trun::kL1BufferBytes};
    memory.raw_releases.push_back(hreg::Region{drisc_release_addr, CORE_SYNC_SLOT_BYTES});
    const auto roster = builder.build(memory);

    if (roster.counter_arrivals != roster_tensix_cores.size() || roster.word_arrivals.size() != 1u ||
        roster.grid.num_dests != roster_tensix_cores.size() || roster.worker_endpoints.size() != 1u) {
        throw std::runtime_error("core_sync DRISC worker: roster partition is inconsistent");
    }
    EXPECT_EQ(roster.worker_endpoints.front().core_type, ProgrammableCoreType::DRAM);
    EXPECT_FALSE(roster.worker_endpoints.front().is_semaphore);
    EXPECT_EQ(roster.worker_endpoints.front().semaphore_id, drisc_release_addr);

    const uint32_t nonce = trun::mint_run_nonce();
    const uint32_t drisc_word_index = roster.word_arrivals.front().worker_index;

    trun::add_kernels(
        program,
        kernel_path,
        include_paths,
        director_core,
        launched_tensix,
        roster,
        nonce,
        director_ready_base,
        result_l1_addr,
        roster.counter_arrivals);

    const auto drisc_kernel = ttm::CreateKernel(
        program,
        kernel_path.string(),
        drisc_logical,
        ttm::DramConfig{
            .noc = ttm::NOC::NOC_0,
            .compile_args = {0u, drisc_word_index, 1u, 1u, CORE_SYNC_CORE_FAMILY_DRAM},
            .defines = {{"CORE_SYNC_PROBE_DRISC", "1"}},
        });
    ttm::SetRuntimeArgs(
        program,
        drisc_kernel,
        drisc_logical,
        word_worker_args(
            {nonce, director_ready_base, drisc_result_addr, roster.arrival_semaphore_id},
            roster.director_noc,
            0u,
            drisc_word_index,
            drisc_release_addr));

    trun::execute(mesh, std::move(program));

    RunResult result;
    result.counter_arrivals = roster.counter_arrivals;
    result.word_arrivals = static_cast<uint32_t>(roster.word_arrivals.size());
    result.grid_num_dests = roster.grid.num_dests;
    result.endpoint_count = static_cast<uint32_t>(roster.worker_endpoints.size());
    result.expected_release = core_sync_release_token(nonce);

    result.director_outcome = trun::read_result(mesh, director_core, result_l1_addr, nonce).outcome;
    result.tensix_outcomes = trun::read_outcomes(mesh, launched_tensix, result_l1_addr, nonce);
    result.drisc_outcome =
        trun::read_result_at(mesh, drisc_virtual, drisc_host_addr + CORE_SYNC_SLOT_BYTES, nonce).outcome;

    cluster.read_core(
        &result.release_word,
        sizeof(result.release_word),
        tt_cxy_pair(mesh->build_id(), drisc_virtual),
        drisc_host_addr);
    return result;
}

inline void expect_outcome(const RunResult& result, ::core_sync::Outcome expected, const char* what) {
    EXPECT_EQ(result.director_outcome, static_cast<uint32_t>(expected))
        << what << ": director outcome " << result.director_outcome;
    for (size_t i = 0; i < result.tensix_outcomes.size(); ++i) {
        EXPECT_EQ(result.tensix_outcomes[i], static_cast<uint32_t>(expected))
            << what << ": Tensix worker " << i << " outcome " << result.tensix_outcomes[i];
    }
    EXPECT_EQ(result.drisc_outcome, static_cast<uint32_t>(expected))
        << what << ": DRISC outcome " << result.drisc_outcome << ", release word 0x" << std::hex << result.release_word;
}

/* Runs success, abandonment, and recovery on one launchable DRISC endpoint. */
inline void run_drisc_worker(
    ttmd::MeshDevice* mesh,
    const std::filesystem::path& kernel_path,
    const std::vector<std::filesystem::path>& include_paths) {
    ASSERT_NE(mesh, nullptr) << "unit mesh not initialized";
    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::DRAM)) {
        GTEST_SKIP() << "DRAM programmable cores not enabled";
    }

    const auto& soc = x280::harness::cluster().get_soc_desc(mesh->build_id());
    const auto drisc = htar::select_drisc_endpoint(soc);
    if (!drisc.has_value()) {
        GTEST_SKIP() << "every endpoint of DRAM bank 0 is firmware-owned";
    }

    const RunResult success = run_once(mesh, *drisc, kernel_path, include_paths, /*abandon=*/false);
    EXPECT_EQ(success.counter_arrivals, 15u);
    EXPECT_EQ(success.word_arrivals, 1u);
    EXPECT_EQ(success.grid_num_dests, 15u);
    EXPECT_EQ(success.endpoint_count, 1u);
    expect_outcome(success, ::core_sync::Outcome::Success, "mixed DRISC barrier");
    EXPECT_EQ(success.release_word, success.expected_release);

    const RunResult abandoned = run_once(mesh, *drisc, kernel_path, include_paths, /*abandon=*/true);
    expect_outcome(abandoned, ::core_sync::Outcome::StopRequested, "withheld Tensix arrival");
    EXPECT_EQ(abandoned.release_word, 0u);

    const RunResult recovered = run_once(mesh, *drisc, kernel_path, include_paths, /*abandon=*/false);
    expect_outcome(recovered, ::core_sync::Outcome::Success, "post-abandonment recovery");
    EXPECT_EQ(recovered.release_word, recovered.expected_release);
}

}  // namespace x280::suites::core_sync::drisc_worker

#endif  // X280_SUITES_CORE_SYNC_DRISC_WORKER_HPP
