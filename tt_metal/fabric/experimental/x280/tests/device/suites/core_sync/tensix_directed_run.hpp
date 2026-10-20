// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_CORE_SYNC_TENSIX_DIRECTED_RUN_HPP
#define X280_SUITES_CORE_SYNC_TENSIX_DIRECTED_RUN_HPP

#include "harness/metal_context.hpp"
#include "harness/execution/execution.hpp"
#include "harness/regions/core_sync_roster.hpp"
#include "tests/device/suites/core_sync/probe_args.hpp"

#include "core_sync_core_family.h"

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

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace x280::suites::core_sync::tensix_directed {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hreg = x280::harness::regions;

constexpr uint32_t kL1BufferBytes = 1024u;

/* The outcome a harvest reports for a probe record this run never wrote; it
 * is no core_sync::Outcome, so every comparison with one fails. */
constexpr uint32_t kProbeNotWritten = 0xFFFFFFFFu;

struct ProbeResult {
    uint32_t outcome = kProbeNotWritten;
    uint64_t exit_timestamp = 0u;
};

/* Returns the fifteen-worker 4x4 grid around director (x0,0). omit_last removes
 * logical core (x0+3,3) from the launched set without changing roster topology. */
inline ttm::CoreRangeSet worker_grid(bool omit_last = false, uint32_t x0 = 0) {
    std::vector<ttm::CoreRange> ranges = {
        ttm::CoreRange(ttm::CoreCoord{x0 + 1, 0}, ttm::CoreCoord{x0 + 3, 0}),
        ttm::CoreRange(ttm::CoreCoord{x0, 1}, ttm::CoreCoord{x0 + 3, 2}),
    };
    ranges.push_back(
        omit_last ? ttm::CoreRange(ttm::CoreCoord{x0, 3}, ttm::CoreCoord{x0 + 2, 3})
                  : ttm::CoreRange(ttm::CoreCoord{x0, 3}, ttm::CoreCoord{x0 + 3, 3}));
    return ttm::CoreRangeSet(ranges);
}

inline uint32_t mint_run_nonce() {
    static uint32_t seq = 1u;
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    return ((seq++ & 0xFFu) << 24) | (static_cast<uint32_t>(ns) & 0x00FFFFFFu) | 1u;
}

/* Allocates and zeroes one replicated Tensix-L1 buffer. The returned owner
 * must outlive every workload using its address. */
inline std::shared_ptr<ttmd::MeshBuffer> create_zeroed_l1_buffer(
    ttmd::MeshDevice* mesh, uint32_t bytes = kL1BufferBytes) {
    auto buffer = ttmd::MeshBuffer::create(
        ttmd::ReplicatedBufferConfig{.size = bytes},
        ttmd::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = ttm::BufferType::L1, .bottom_up = true},
        mesh);
    if (buffer == nullptr) {
        throw std::runtime_error("core_sync: failed to allocate Tensix L1 buffer");
    }
    const std::vector<uint32_t> zeros(bytes / sizeof(uint32_t), 0u);
    ttmd::EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), buffer, zeros, /*blocking=*/true);
    return buffer;
}

/* Adds the Tensix director and counter-worker kernels for one roster. */
inline void add_kernels(
    ttm::Program& program,
    const std::filesystem::path& kernel_path,
    const std::vector<std::filesystem::path>& include_paths,
    const ttm::CoreCoord& director_core,
    const ttm::CoreRangeSet& worker_set,
    const hreg::CoreSyncRoster& roster,
    uint32_t nonce,
    uint32_t director_ready_base,
    uint32_t result_l1_addr,
    uint32_t counter_arrivals,
    bool barrier_enabled = true) {
    const auto kernel = ttm::CreateKernel(
        program,
        kernel_path.string(),
        director_core,
        ttm::DataMovementConfig{
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .noc = ttm::NOC::RISCV_0_default,
            .compile_args =
                {
                    1u,
                    static_cast<uint32_t>(roster.workers.size()),
                    barrier_enabled ? 1u : 0u,
                    0u,
                    CORE_SYNC_CORE_FAMILY_TENSIX,
                },
            .compiler_include_paths = include_paths,
        });
    const ProbeCommon common{nonce, director_ready_base, result_l1_addr, roster.arrival_semaphore_id};
    ttm::SetRuntimeArgs(program, kernel, director_core, director_args(common, roster, counter_arrivals));

    const auto worker_kernel = ttm::CreateKernel(
        program,
        kernel_path.string(),
        worker_set,
        ttm::DataMovementConfig{
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .noc = ttm::NOC::RISCV_0_default,
            .compile_args =
                {
                    0u,
                    0u,
                    barrier_enabled ? 1u : 0u,
                    0u,
                    CORE_SYNC_CORE_FAMILY_TENSIX,
                },
            .compiler_include_paths = include_paths,
        });
    ttm::SetRuntimeArgs(program, worker_kernel, worker_set, counter_worker_args(common, roster));
}

/* Enqueues program across the mesh and waits for completion. */
inline void execute(ttmd::MeshDevice* mesh, ttm::Program&& program) {
    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(mesh->shape()), std::move(program));
    x280::harness::execution::run_tensix_and_drain(*mesh, workload);
}

/* Reads the probe result at result_addr of the core at virtual_core (any
 * core family). result_addr is the address read_core takes for that core,
 * 64-bit for a DRISC's host NoC address. A record whose nonce word is not
 * run_nonce reads as kProbeNotWritten with a zero exit timestamp. */
inline ProbeResult read_result_at(
    ttmd::MeshDevice* mesh, const ttm::CoreCoord& virtual_core, uint64_t result_addr, uint32_t run_nonce) {
    uint32_t words[CORE_SYNC_PROBE_RESULT_WORDS] = {};
    x280::harness::cluster().read_core(words, sizeof(words), tt_cxy_pair(mesh->build_id(), virtual_core), result_addr);
    if (words[CORE_SYNC_PROBE_RESULT_RUN_NONCE] != run_nonce) {
        return ProbeResult{};
    }
    return ProbeResult{
        .outcome = words[CORE_SYNC_PROBE_RESULT_OUTCOME],
        .exit_timestamp = static_cast<uint64_t>(words[CORE_SYNC_PROBE_RESULT_EXIT_LO]) |
                          (static_cast<uint64_t>(words[CORE_SYNC_PROBE_RESULT_EXIT_HI]) << 32),
    };
}

/* read_result_at for a Tensix logical core. */
inline ProbeResult read_result(
    ttmd::MeshDevice* mesh, const ttm::CoreCoord& logical_core, uint32_t result_l1_addr, uint32_t run_nonce) {
    return read_result_at(mesh, mesh->worker_core_from_logical_core(logical_core), result_l1_addr, run_nonce);
}

/* Reads outcomes for worker_set in CoreRangeSet enumeration order. */
inline std::vector<uint32_t> read_outcomes(
    ttmd::MeshDevice* mesh, const ttm::CoreRangeSet& worker_set, uint32_t result_l1_addr, uint32_t run_nonce) {
    std::vector<uint32_t> outcomes;
    const auto cores = ttm::corerange_to_cores(worker_set);
    outcomes.reserve(cores.size());
    for (const auto& core : cores) {
        outcomes.push_back(read_result(mesh, core, result_l1_addr, run_nonce).outcome);
    }
    return outcomes;
}

}  // namespace x280::suites::core_sync::tensix_directed

#endif  // X280_SUITES_CORE_SYNC_TENSIX_DIRECTED_RUN_HPP
