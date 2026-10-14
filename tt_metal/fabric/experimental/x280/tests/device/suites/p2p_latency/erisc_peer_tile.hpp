// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host descriptor for the idle-ERISC p2p_latency peer member.
#ifndef X280_SUITES_P2P_LATENCY_ERISC_PEER_TILE_HPP
#define X280_SUITES_P2P_LATENCY_ERISC_PEER_TILE_HPP

#include "metal_peer_runner.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"

#include <cstdint>
#include <stdexcept>

namespace x280::suites::p2p_latency::erisc_peer_tile {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hreg = x280::harness::regions;

// ETH_CLOCK_CYCLE_1MS nominal scheduling rate; not a measured frequency.
constexpr uint64_t kEriscNominalClockHz = 1000000000ull;
constexpr uint64_t kWarmupBudgetEriscTicks = 3ull * kEriscNominalClockHz;

using Axis = metal_peer::Axis;

// Runs both protocol phases on an inactive idle ERISC and validates both
// participants. Skips when idle Ethernet kernels or cores are unavailable.
inline void run_axis(
    x280::harness::session::Fixture& fixture, const x280::harness::RunSpec& spec, bool x280_initiates_first) {
    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::IDLE_ETH)) {
        GTEST_SKIP() << "idle Ethernet programmable cores not enabled";
        return;
    }
    ASSERT_FALSE(fixture.mesh->get_devices().empty()) << "unit mesh has no devices";
    const auto* device = fixture.mesh->get_devices()[0];
    const auto inactive_cores = device->get_inactive_ethernet_cores();
    if (inactive_cores.empty()) {
        GTEST_SKIP() << "no inactive Ethernet core available";
        return;
    }
    const ttm::CoreCoord logical_core = *inactive_cores.begin();
    // The X280 window reaches the ERISC on NoC 0.
    const auto virtual_core = x280::X280Runtime::virtual_core_from_logical_core(
        *fixture.mesh, logical_core, tt::CoreType::ETH, ttm::NOC::NOC_0);
    const auto reported_window = hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::IDLE_ETH);
    if (reported_window.local.size == 0u) {
        GTEST_SKIP() << "idle-ERISC unreserved L1 window is empty";
        return;
    }

    hreg::L1Window window{};
    try {
        window = hreg::align_window_start(reported_window, hreg::kCacheLineBytes);
    } catch (const std::invalid_argument& error) {
        FAIL() << "invalid idle-ERISC unreserved L1 window: " << error.what();
        return;
    }

    const metal_peer::Target target{
        .kind = x280::harness::ExecutionTarget::IdleEthernet,
        .logical_core = logical_core,
        .virtual_core = virtual_core,
        .window = window,
        .warmup_budget = kWarmupBudgetEriscTicks,
        .backend_id = P2P_LATENCY_BACKEND_ID_ERISC_ERISC_PEER_TILE,
        .member = "erisc_peer_tile",
        .peer_name = "ERISC",
        .warmup_env = "P2P_LATENCY_ERISC_PEER_TILE_W",
        .samples_env = "P2P_LATENCY_ERISC_PEER_TILE_N",
        .counter = x280::harness::measurement::Counter::EriscWallClock,
    };
    metal_peer::run_axis(
        fixture,
        spec,
        x280_initiates_first,
        target,
        [&](const std::filesystem::path& source,
            const std::vector<uint32_t>& runtime_args,
            const x280::harness::execution::X280Launch& launch) {
            ttm::Program program = ttm::CreateProgram();
            const ttm::KernelHandle kernel = ttm::CreateKernel(
                program,
                source.string(),
                logical_core,
                ttm::EthernetConfig{
                    .eth_mode = ttm::Eth::IDLE,
                    .noc = ttm::NOC::NOC_0,
                    .processor = ttm::DataMovementProcessor::RISCV_0,
                    .defines = {{"P2P_LATENCY_TARGET_IDLE_ERISC", "1"}},
                    .opt_level = ttm::KernelBuildOptLevel::Os,
                });
            ttm::SetRuntimeArgs(program, kernel, logical_core, runtime_args);
            ttmd::MeshWorkload workload;
            workload.add_program(ttmd::MeshCoordinateRange(ttmd::MeshCoordinate(0, 0)), std::move(program));
            x280::harness::execution::enqueue_tensix(*fixture.mesh, workload);
            const auto outcome = x280::harness::execution::run_x280(fixture.handle(), launch);
            x280::harness::execution::finish_tensix(*fixture.mesh);
            return outcome;
        });
}

}  // namespace x280::suites::p2p_latency::erisc_peer_tile

#endif  // X280_SUITES_P2P_LATENCY_ERISC_PEER_TILE_HPP
