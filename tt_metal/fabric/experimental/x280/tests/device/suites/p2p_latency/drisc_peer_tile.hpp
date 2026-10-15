// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host descriptor for the free-DRISC p2p_latency peer member.
#ifndef X280_SUITES_P2P_LATENCY_DRISC_PEER_TILE_HPP
#define X280_SUITES_P2P_LATENCY_DRISC_PEER_TILE_HPP

#include "metal_peer_runner.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"

#include <cstdint>
#include <stdexcept>

namespace x280::suites::p2p_latency::drisc_peer_tile {

namespace ttm = tt::tt_metal;
namespace hreg = x280::harness::regions;

constexpr uint64_t kWarmupBudgetDriscTicks = 3ull * 1000 * 1000 * 1000;

using Axis = metal_peer::Axis;

// Runs both protocol phases on a free non-endpoint DRISC and validates both
// participants. Skips when DRISC kernels or a free bank-0 core are unavailable.
inline void run_axis(
    x280::harness::session::Fixture& fixture, const x280::harness::RunSpec& spec, bool x280_initiates_first) {
    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::DRAM)) {
        GTEST_SKIP() << "DRAM programmable cores require Blackhole firmware 19.12 or newer";
        return;
    }
    ASSERT_FALSE(fixture.mesh->get_devices().empty()) << "unit mesh has no devices";
    auto* device = fixture.mesh->get_devices()[0];
    ttm::CoreCoord logical_core{};
    try {
        logical_core = fixture.mesh->impl().pick_unused_dram_logical_core(device, 0);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "bank 0 has no free non-endpoint DRISC: " << error.what();
        return;
    }
    // The X280 window reaches the DRISC on NoC 0.
    const auto virtual_core = x280::X280Runtime::virtual_core_from_logical_core(
        *fixture.mesh, logical_core, tt::CoreType::DRAM, ttm::NOC::NOC_0);
    const auto reported_window = hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::DRAM);
    if (reported_window.local.size == 0u) {
        GTEST_SKIP() << "DRISC unreserved L1 window is empty";
        return;
    }

    hreg::L1Window window{};
    try {
        const auto& arena = fixture.mesh->impl().drisc_l1_arena();
        window = hreg::align_window_start(
            hreg::subwindow(reported_window, arena.kernel_working_region_base(), arena.kernel_working_region_size()),
            hreg::kCacheLineBytes);
    } catch (const std::invalid_argument& error) {
        FAIL() << "invalid DRISC kernel working region: " << error.what();
        return;
    }

    const metal_peer::Target target{
        .kind = x280::harness::ExecutionTarget::Drisc,
        .logical_core = logical_core,
        .virtual_core = virtual_core,
        .window = window,
        .warmup_budget = kWarmupBudgetDriscTicks,
        .backend_id = P2P_LATENCY_BACKEND_ID_DRISC_DRISC_PEER_TILE,
        .member = "drisc_peer_tile",
        .peer_name = "DRISC",
        .warmup_env = "P2P_LATENCY_DRISC_PEER_TILE_W",
        .samples_env = "P2P_LATENCY_DRISC_PEER_TILE_N",
        .counter = x280::harness::measurement::Counter::RiscDebugWallClock,
    };
    metal_peer::run_axis(
        fixture,
        spec,
        x280_initiates_first,
        target,
        [&](const std::filesystem::path& source,
            const std::vector<uint32_t>& runtime_args,
            p2p_latency_transmit_mode_t tx_mode,
            const x280::harness::execution::X280Launch& launch) {
            ttm::Program program = ttm::CreateProgram();
            const ttm::KernelHandle kernel = ttm::CreateKernel(
                program,
                source.string(),
                logical_core,
                ttm::DramConfig{
                    .noc = ttm::NOC::NOC_0,
                    .defines = {{"P2P_LATENCY_TARGET_DRISC", "1"}, metal_peer::tx_mode_define(tx_mode)},
                    .opt_level = ttm::KernelBuildOptLevel::O2,
                });
            ttm::SetRuntimeArgs(program, kernel, logical_core, runtime_args);
            program.impl().compile(device, /*force_slow_dispatch=*/true);
            ttm::slow_dispatch::WriteRuntimeArgsToDevice(*device, program, /*force_slow_dispatch=*/true);
            ttm::slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/true);
            const auto outcome = x280::harness::execution::run_x280(fixture.handle(), launch);
            try {
                ttm::slow_dispatch::WaitProgramDone(*device, program);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("DRISC wait failed; reset the device before another workload: ") + error.what());
            }
            return outcome;
        });
}

}  // namespace x280::suites::p2p_latency::drisc_peer_tile

#endif  // X280_SUITES_P2P_LATENCY_DRISC_PEER_TILE_HPP
