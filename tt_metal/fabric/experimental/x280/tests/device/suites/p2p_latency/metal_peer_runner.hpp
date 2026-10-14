// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_P2P_LATENCY_METAL_PEER_RUNNER_HPP
#define X280_SUITES_P2P_LATENCY_METAL_PEER_RUNNER_HPP

#include "metal_peer.hpp"

#include "tests/device/suites/p2p_latency/metal_peer_arguments.hpp"
#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/regions/metal_l1_region.hpp"
#include "harness/runner.hpp"
#include "harness/targets/execution_target.hpp"
#include "harness/metal_context.hpp"

#include "metal_peer_args.h"
#include "rt/x280_noc_map.h"
#include "kernels/shared/ubench_units.h"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/host_device_transfer.hpp"
#include "llrt/tt_cluster.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <vector>

namespace x280::suites::p2p_latency::metal_peer {

namespace hart_ = x280::harness::artifacts;
namespace hargs = x280::harness::arguments;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace ttm = tt::tt_metal;

constexpr uint32_t kDefaultWarmup = 8u;
constexpr uint32_t kDefaultSamples = 64u;
constexpr uint32_t kPayloadSize = 4u;
constexpr uint32_t kTlbIndex = X280_TLB_PAIRED_BASE;
constexpr uint64_t kWarmupBudgetX280Cycles = 500u * 1000000ull;

using Axis = bool;

// Describes an already selected core-local peer. Addresses are local L1 byte
// addresses; warmup_budget is in the peer's raw wall-clock ticks.
struct Target {
    x280::harness::ExecutionTarget kind;
    ttm::CoreCoord logical_core;
    x280::TranslatedCoord virtual_core;
    hreg::L1Window window;
    uint64_t warmup_budget;
    uint32_t backend_id;
    const char* member;
    const char* peer_name;
    const char* warmup_env;
    const char* samples_env;
    // The peer's wall counter, as its records name it.
    hmeas::Counter counter;
};

// Produces a non-zero protocol nonce and the correlated X280 lifecycle value.
inline uint64_t mint_run_value(uint32_t& run_nonce) {
    static uint32_t sequence = 0u;
    run_nonce = 0x400u + ++sequence;
    return (static_cast<uint64_t>(0xE11Cu) << 48) | (static_cast<uint64_t>(0xE4u) << 40) | run_nonce;
}

// One paired launch's arguments and results. clock is set only when the launch
// assessed AICLK around the workload.
struct PeerLaunch {
    p2p_latency_x280_noc_section_t x280_args;
    p2p_latency_result_t result_x;
    p2p_latency_result_t result_peer;
    uint64_t expected_aiclk_hz;
    std::optional<hmeas::AiclkRunEvidence> clock;
};

// Launches the X280 and the core-local peer once with `warmup` warm-up and
// `samples` timed exchanges per phase and validates both results. Fatal
// failures leave `out` partial; a peer window too small for the layout throws
// std::invalid_argument from Arena::allocate. `run_concurrently` launches the peer,
// runs the supplied X280 launch while it is active, waits for the peer, and
// returns the X280 outcome.
template <typename RunConcurrently>
inline void launch_peer(
    x280::harness::session::Fixture& fixture,
    const x280::harness::RunSpec& spec,
    p2p_latency_direction_t first_direction,
    const Target& target,
    RunConcurrently& run_concurrently,
    uint32_t warmup,
    uint32_t samples,
    uint32_t run_nonce,
    uint64_t run_value,
    bool assess_clock,
    PeerLaunch& out) {
    ASSERT_LE(static_cast<uint64_t>(run_nonce) + 2ull * (static_cast<uint64_t>(warmup) + samples) - 1ull, UINT32_MAX);

    auto& cluster = x280::harness::cluster();
    ASSERT_GT(target.window.local.size, 0u) << target.peer_name << " L1 window is empty";
    ASSERT_LE(target.window.local.base, std::numeric_limits<uint32_t>::max());
    ASSERT_EQ(target.window.local.base % hreg::kCacheLineBytes, 0u)
        << target.peer_name << " L1 window base is not 64-byte aligned";
    ASSERT_LE(
        target.window.local.size,
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ull - target.window.local.base)
        << target.peer_name << " L1 window exceeds the local u32 address space";

    const MetalRegions peer = MetalRegions::compute(target.window);
    for (const auto& region : {peer.mailbox, peer.staging, peer.result}) {
        ASSERT_LE(region.addr, std::numeric_limits<uint32_t>::max());
        ASSERT_LE(region.bytes, std::numeric_limits<uint32_t>::max());
        ASSERT_LE(region.addr + region.bytes, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ull);
        ASSERT_EQ(region.addr % hreg::kCacheLineBytes, 0u) << target.peer_name << " region base is not 64-byte aligned";
    }
    const X280Regions x280_regions = X280Regions::compute(*fixture.tile);
    ASSERT_TRUE(ubench_noc_congruent(peer.staging.addr, x280_regions.mailbox.addr))
        << target.peer_name << " NoC source 0x" << std::hex << peer.staging.addr << " and X280 mailbox destination 0x"
        << x280_regions.mailbox.addr << std::dec << " are not congruent";
    ASSERT_TRUE(ubench_noc_congruent(x280_regions.mailbox.addr, peer.mailbox.addr))
        << "X280 aperture 0x" << std::hex << x280_regions.mailbox.addr << " and " << target.peer_name << " mailbox 0x"
        << peer.mailbox.addr << std::dec << " addresses are not congruent";

    const auto l2cpu_translated = fixture.runtime->virtual_core_from_tile(fixture.tile_id, ttm::NOC::NOC_0);
    hreg::zero_region(*fixture.tile, x280_regions.mailbox);
    hreg::zero_region(*fixture.tile, x280_regions.result);

    out.x280_args = make_x280_args(
        {
            .self = P2P_LATENCY_PARTICIPANT_A,
            .first_direction = static_cast<uint32_t>(first_direction),
            .payload_size = kPayloadSize,
            .warmup_count = warmup,
            .sample_count = samples,
            .run_nonce = run_nonce,
            .warmup_max_poll_cycles = kWarmupBudgetX280Cycles,
        },
        {
            .own_inbound_addr = x280_regions.mailbox.addr,
            .own_result_addr = x280_regions.result.addr,
            .peer_inbound_addr = peer.mailbox.addr,
            .peer_noc = target.virtual_core.noc(),
            .peer_tlb_index = kTlbIndex,
        });
    hreg::write_args(*fixture.tile, x280_regions.args, out.x280_args);

    const tt_cxy_pair peer_coord(fixture.mesh->build_id(), target.virtual_core.x(), target.virtual_core.y());
    const auto host_addr = [&](const hreg::Region& region) {
        return hreg::host_noc_address(target.window, region.addr, region.bytes);
    };
    for (const auto& region : {peer.mailbox, peer.staging, peer.result}) {
        const std::vector<uint8_t> zeros(region.bytes, 0u);
        cluster.write_core(zeros.data(), static_cast<uint32_t>(zeros.size()), peer_coord, host_addr(region));
        std::vector<uint8_t> readback(region.bytes, 0xFFu);
        cluster.read_core(readback.data(), static_cast<uint32_t>(readback.size()), peer_coord, host_addr(region));
        ASSERT_EQ(readback, zeros) << target.peer_name << " region zeroing did not read back at local address 0x"
                                   << std::hex << region.addr << std::dec;
    }

    const auto peer_args = hargs::pack_metal_peer_runtime_args({
        .layout_cookie = P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE,
        .common =
            {
                .self = P2P_LATENCY_PARTICIPANT_B,
                .first_direction = static_cast<uint32_t>(first_direction),
                .payload_size = kPayloadSize,
                .warmup_count = warmup,
                .sample_count = samples,
                .run_nonce = run_nonce,
                .warmup_max_poll_cycles = target.warmup_budget,
            },
        .own_inbound_addr = static_cast<uint32_t>(peer.mailbox.addr),
        .own_staging_addr = static_cast<uint32_t>(peer.staging.addr),
        .own_result_addr = static_cast<uint32_t>(peer.result.addr),
        .peer_inbound_addr = x280_regions.mailbox.addr,
        .peer_noc = l2cpu_translated,
    });
    const auto source = hart_::shared_kernel_dir(spec.open.kernel_dir, "p2p_latency") / "metal_peer_entry.cpp";
    ASSERT_TRUE(std::filesystem::exists(source)) << target.peer_name << " kernel source not found at " << source;
    const hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_P2P_LATENCY_METAL_PEER_TILE_FW_IMAGE"),
        .run_value = run_value,
        .args_addr = x280_regions.args.addr,
        .args_bytes = x280_regions.args.bytes,
        .cleanup_timeout = std::chrono::seconds{5},
        .wait_timeout = kX280WaitTimeout,
    };
    auto* clock_device = fixture.mesh->get_devices()[0];
    const ttm::CoreCoord peer_core{target.virtual_core.x(), target.virtual_core.y()};
    out.expected_aiclk_hz = hexec::programmed_aiclk_hz(x280::harness::cluster(), *clock_device);
    hmeas::AiclkEndpoint clock_start;
    if (assess_clock) {
        clock_start = hexec::read_aiclk_start(x280::harness::cluster(), *clock_device, peer_core);
        ASSERT_TRUE(clock_start.errors.empty()) << "pre-run AICLK evidence is unavailable";
    }
    const auto outcome = run_concurrently(source, peer_args, launch);
    if (assess_clock) {
        const auto clock_end = hexec::read_aiclk_end(x280::harness::cluster(), *clock_device, peer_core);
        out.clock = hmeas::assess_aiclk_run(out.expected_aiclk_hz, clock_start, clock_end);
    }
    ASSERT_TRUE(outcome.enqueued()) << "X280 enqueue rejected";
    ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    out.result_x = hreg::read_struct<p2p_latency_result_t>(*fixture.tile, x280_regions.result);
    cluster.read_core(&out.result_peer, sizeof(out.result_peer), peer_coord, host_addr(peer.result));

    hexec::expect_window_cleared(*fixture.tile, kTlbIndex);

    EXPECT_EQ(out.result_x.backend_id, P2P_LATENCY_BACKEND_ID_X280_METAL_PEER_TILE);
    EXPECT_EQ(out.result_peer.backend_id, target.backend_id);
    ASSERT_EQ(out.result_x.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << "X280 terminal state: " << state_name(out.result_x.terminal_state);
    ASSERT_EQ(out.result_peer.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << target.peer_name << " terminal state: " << state_name(out.result_peer.terminal_state);
    EXPECT_EQ(out.result_x.path_sentinel, run_nonce);
    EXPECT_EQ(out.result_peer.path_sentinel, run_nonce);

    if (target.kind == x280::harness::ExecutionTarget::Drisc) {
        auto* device = fixture.mesh->get_devices()[0];
        std::vector<uint32_t> dram_write = {0xD4A00001u};
        std::vector<uint32_t> dram_read;
        const uint32_t dram_addr = device->allocator()->get_base_allocator_addr(ttm::HalMemType::DRAM);
        ASSERT_TRUE(ttm::slow_dispatch::WriteToDeviceDRAMChannel(*device, 0, dram_addr, dram_write));
        ASSERT_TRUE(ttm::slow_dispatch::ReadFromDeviceDRAMChannel(*device, 0, dram_addr, sizeof(uint32_t), dram_read));
        ASSERT_EQ(dram_read, dram_write);
    }

    ASSERT_NO_FATAL_FAILURE(validate_phases(out.result_x, P2P_LATENCY_PARTICIPANT_A, first_direction, samples));
    ASSERT_NO_FATAL_FAILURE(validate_phases(out.result_peer, P2P_LATENCY_PARTICIPANT_B, first_direction, samples));
}

// Runs the shared X280/core-local peer protocol for one first-direction axis.
// An unset warm-up env var is calibrated from a default-W pilot; a set one is
// used as given. `run_concurrently` is as for launch_peer().
template <typename RunConcurrently>
inline void run_axis(
    x280::harness::session::Fixture& fixture,
    const x280::harness::RunSpec& spec,
    bool x280_initiates_first,
    const Target& target,
    RunConcurrently&& run_concurrently) {
    ASSERT_TRUE(
        target.kind == x280::harness::ExecutionTarget::Drisc ||
        target.kind == x280::harness::ExecutionTarget::IdleEthernet)
        << "Metal peer target must be DRISC or idle Ethernet";
    const auto first_direction = x280_initiates_first ? P2P_LATENCY_DIRECTION_A_TO_B : P2P_LATENCY_DIRECTION_B_TO_A;
    const auto explicit_warmup = hart_::env_u32(target.warmup_env);
    const uint32_t samples = hart_::env_u32(target.samples_env, kDefaultSamples);
    ASSERT_GT(samples, 0u);
    ASSERT_LE(samples, P2P_LATENCY_MAX_SAMPLES);
    hmeas::WarmupSelection selection{.warmup = explicit_warmup.value_or(kDefaultWarmup)};
    ASSERT_GT(selection.warmup, 0u);

    uint32_t run_nonce = 0u;
    const uint64_t run_value = mint_run_value(run_nonce);
    if (!explicit_warmup) {
        uint32_t pilot_nonce = 0u;
        const uint64_t pilot_value = mint_run_value(pilot_nonce);
        PeerLaunch pilot{};
        ASSERT_NO_FATAL_FAILURE(launch_peer(
            fixture,
            spec,
            first_direction,
            target,
            run_concurrently,
            kDefaultWarmup,
            samples,
            pilot_nonce,
            pilot_value,
            false,
            pilot));
        selection = hmeas::calibrate_warmup(warmup_calibration_request(
            pilot.result_x, pilot.result_peer, pilot.expected_aiclk_hz, kDefaultWarmup, samples, run_nonce));
        ASSERT_FALSE(selection.refusal.has_value())
            << "warm-up calibration refused: " << hmeas::warmup_refusal_name(*selection.refusal);
    }
    const uint32_t warmup = static_cast<uint32_t>(selection.warmup);

    PeerLaunch run{};
    ASSERT_NO_FATAL_FAILURE(launch_peer(
        fixture, spec, first_direction, target, run_concurrently, warmup, samples, run_nonce, run_value, true, run));
    const auto& x280_args = run.x280_args;
    const auto& result_x = run.result_x;
    const auto& result_peer = run.result_peer;
    ASSERT_TRUE(run.clock.has_value()) << "measured launch assessed no AICLK";
    const auto& clock = *run.clock;
    const uint64_t expected_aiclk_hz = run.expected_aiclk_hz;

    std::vector<hout::Field> placement{
        {"chip_id", fixture.tile_id.chip_id},
        {"x280_hw_index", fixture.tile_id.hw_index},
        {"x280_noc_x", fixture.tile_id.noc_x},
        {"x280_noc_y", fixture.tile_id.noc_y},
    };
    if (target.kind == x280::harness::ExecutionTarget::Drisc) {
        placement.insert(
            placement.end(),
            {
                {"drisc_logical_x", target.logical_core.x},
                {"drisc_logical_y", target.logical_core.y},
                {"drisc_virtual_x", target.virtual_core.x()},
                {"drisc_virtual_y", target.virtual_core.y()},
                {"dram_bank", 0u},
                {"drisc_role", "free"},
                {"drisc_noc", 0u},
                {"mode_during_run", "stream"},
                {"mode_restoration_contract", "admission_test"},
            });
    } else {
        placement.insert(
            placement.end(),
            {
                {"erisc_logical_x", target.logical_core.x},
                {"erisc_logical_y", target.logical_core.y},
                {"erisc_virtual_x", target.virtual_core.x()},
                {"erisc_virtual_y", target.virtual_core.y()},
            });
    }
    const auto report = [&](const p2p_latency_result_t& result, const char* participant, uint32_t phase, bool is_x280) {
        std::vector<hout::Field> identity{{"participant", participant}};
        identity.insert(identity.end(), placement.begin(), placement.end());
        hout::write_jsonl(
            spec.suite_label,
            hout::record(phase_record(
                target.member,
                std::move(identity),
                result,
                phase,
                x280_args.common,
                x280_args.backend.peer_tlb_index,
                is_x280 ? std::optional{P2P_LATENCY_X280_NOC_ORDERING} : std::nullopt,
                is_x280 ? std::nullopt : std::optional{MetalClock{target.counter, clock}},
                selection)));
    };

    std::printf(
        "\n[p2p_latency %s smoke] first_direction=%s W=%u N=%u payload=%u "
        "same-chip NoC; %s raw counter ticks expected_aiclk_hz=%llu\n",
        target.member,
        direction_name(first_direction),
        warmup,
        samples,
        kPayloadSize,
        target.peer_name,
        static_cast<unsigned long long>(expected_aiclk_hz));
    report(result_x, "X280", 0u, true);
    report(result_peer, target.peer_name, 0u, false);
    report(result_x, "X280", 1u, true);
    report(result_peer, target.peer_name, 1u, false);

    for (uint32_t p = 0; p < 2; ++p) {
        EXPECT_TRUE(phase_samples_nonzero(result_x.phases[p])) << "X280 phase " << p;
        EXPECT_TRUE(phase_samples_nonzero(result_peer.phases[p])) << target.peer_name << " phase " << p;
    }
    EXPECT_TRUE(clock.errors.empty()) << hout::clock_json(clock);
}

}  // namespace x280::suites::p2p_latency::metal_peer

#endif  // X280_SUITES_P2P_LATENCY_METAL_PEER_RUNNER_HPP
