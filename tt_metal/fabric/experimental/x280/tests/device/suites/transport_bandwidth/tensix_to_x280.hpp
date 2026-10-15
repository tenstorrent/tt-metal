// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/transport_bandwidth/tensix_to_x280.hpp -- Tensix -> X280 member.
//
// One Tensix data-movement core issues noc_async_write() at a fixed X280
// LIM destination; X280 is a passive addressable target (no image
// submitted). The descriptor owns the LIM+L1 layouts, args pack, harvest,
// and records; the host test wires the fixture and sweep.

#ifndef X280_SUITES_TRANSPORT_BW_TENSIX_TO_X280_HPP
#define X280_SUITES_TRANSPORT_BW_TENSIX_TO_X280_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/regions.hpp"
#include "harness/regions/tensix_l1.hpp"
#include "harness/runner.hpp"
#include "harness/metal_context.hpp"
#include "tests/device/suites/transport_bandwidth/records.hpp"

#include "transport_bandwidth_common.h"
#include "tensix_to_x280.h"
#include "rt/x280_noc_map.h"

#include <gtest/gtest.h>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <vector>

namespace x280::suites::transport_bandwidth::tensix_to_x280 {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hreg = x280::harness::regions;

constexpr uint32_t kTlbIndex = X280_TLB_PAIRED_BASE;
constexpr uint32_t kDefaultWarmupCount = 128u;
constexpr uint32_t kDefaultTransferCount = 16384u;

struct SweepPoint {
    uint32_t transfer_size;
    uint32_t boundary;
    const char* label;
};
using Axis = SweepPoint;

// One Metal sender's L1 block. The gate line is used only by the ERISC
// sender's start gate (TB_TX2X_ARG_GATE_ADDR).
struct L1Layout {
    uint32_t block_bytes;
    uint32_t staging_offset;
    uint32_t scratch_offset;
    uint32_t result_offset;
    uint32_t gate_offset;

    static L1Layout compute() {
        L1Layout L{};
        auto ru = [](uint64_t n, uint64_t to) { return static_cast<uint32_t>((n + to - 1u) & ~(to - 1u)); };
        uint32_t cursor = 0;
        L.staging_offset = cursor;
        cursor += ru(TB_TX2X_STAGING_MAX_BYTES, hreg::kCacheLineBytes);
        L.scratch_offset = cursor;
        cursor += static_cast<uint32_t>(hreg::kCacheLineBytes);
        L.result_offset = cursor;
        cursor += ru(sizeof(transport_bw_result_t), hreg::kCacheLineBytes);
        L.gate_offset = cursor;
        cursor += static_cast<uint32_t>(hreg::kCacheLineBytes);
        L.block_bytes = cursor;
        return L;
    }
};

struct LimRegions {
    hreg::Region dst;
    hreg::Region verify;

    static LimRegions compute(x280::X280Tile& tile) {
        const auto avail = tile.available_lim();
        const hreg::Region w{avail.begin, avail.size()};
        hreg::Arena arena(w);
        LimRegions R{};
        R.dst = arena.allocate(TB_TX2X_STAGING_MAX_BYTES);
        R.verify = arena.allocate(hreg::kCacheLineBytes);
        return R;
    }
};

// One Metal sender's runtime arguments (TB_TX2X_ARG_*). L1 addresses are the
// sender's own; gate_addr is 0 for no start gate.
struct SenderArgs {
    uint32_t transfer_size;
    uint32_t warmup_count;
    uint32_t transfer_count;
    uint32_t boundary;
    uint32_t run_nonce;
    uint32_t staging_addr;
    uint32_t scratch_addr;
    uint32_t result_addr;
    uint32_t gate_addr;
    uint64_t peer_dst_addr;
    uint64_t peer_verify_addr;
    uint32_t peer_noc_x;
    uint32_t peer_noc_y;
};

inline std::vector<uint32_t> sender_args(const SenderArgs& a) {
    std::vector<uint32_t> args(TB_TX2X_ARG_COUNT, 0u);
    args[TB_TX2X_ARG_LAYOUT_COOKIE] = TB_TX2X_LAYOUT_COOKIE;
    args[TB_TX2X_ARG_TRANSFER_SIZE] = a.transfer_size;
    args[TB_TX2X_ARG_WARMUP_COUNT] = a.warmup_count;
    args[TB_TX2X_ARG_TRANSFER_COUNT] = a.transfer_count;
    args[TB_TX2X_ARG_BOUNDARY] = a.boundary;
    args[TB_TX2X_ARG_RUN_NONCE] = a.run_nonce;
    args[TB_TX2X_ARG_OWN_STAGING_ADDR] = a.staging_addr;
    args[TB_TX2X_ARG_OWN_SCRATCH_ADDR] = a.scratch_addr;
    args[TB_TX2X_ARG_OWN_RESULT_ADDR] = a.result_addr;
    args[TB_TX2X_ARG_PEER_DST_ADDR_LO] = static_cast<uint32_t>(a.peer_dst_addr & 0xFFFFFFFFu);
    args[TB_TX2X_ARG_PEER_DST_ADDR_HI] = static_cast<uint32_t>(a.peer_dst_addr >> 32);
    args[TB_TX2X_ARG_PEER_VERIFY_ADDR_LO] = static_cast<uint32_t>(a.peer_verify_addr & 0xFFFFFFFFu);
    args[TB_TX2X_ARG_PEER_VERIFY_ADDR_HI] = static_cast<uint32_t>(a.peer_verify_addr >> 32);
    args[TB_TX2X_ARG_PEER_NOC_X] = a.peer_noc_x;
    args[TB_TX2X_ARG_PEER_NOC_Y] = a.peer_noc_y;
    args[TB_TX2X_ARG_GATE_ADDR] = a.gate_addr;
    return args;
}

// Checks one sender's completion record: `backend_id`, DONE, and the echoed
// size, boundary and counts; a fatal failure for an unusable interval.
inline void check_result(
    const transport_bw_result_t& result,
    uint32_t backend_id,
    uint32_t transfer_size,
    uint32_t boundary,
    uint32_t W,
    uint32_t N) {
    EXPECT_EQ(result.backend_id, backend_id);
    EXPECT_EQ(result.boundary, boundary);
    ASSERT_EQ(result.terminal_state, static_cast<uint32_t>(TRANSPORT_BW_STATE_DONE))
        << "terminal state: " << state_name(result.terminal_state) << " (path_sentinel=0x" << std::hex
        << result.path_sentinel << std::dec << ")";

    EXPECT_EQ(result.transfer_size, transfer_size);
    EXPECT_EQ(result.transfer_count, N);
    EXPECT_EQ(result.warmup_completed, W);
    EXPECT_EQ(result.logical_bytes, static_cast<uint64_t>(N) * static_cast<uint64_t>(transfer_size));
    ASSERT_GT(result.raw_batch_delta, 0ull);
    // Checked here, before the pilot's delta reaches warm-up calibration.
    ASSERT_NE(result.raw_batch_delta, UINT64_MAX) << "wall-timer interval refused on device";
    ASSERT_NE(result.secondary_batch_delta, UINT64_MAX) << "cycle interval refused on device";
}

// One launch's result. clock is set only when the launch assessed AICLK
// around the workload.
struct TransportLaunch {
    transport_bw_result_t result{};
    uint64_t expected_aiclk_hz = 0u;
    std::optional<hmeas::AiclkRunEvidence> clock;
};

// Launches the Tensix issuer once with W warm-up and N timed transfers of
// point's size and boundary and checks its completion record. Fatal failures
// leave `out` partial.
inline void launch(
    x280::harness::session::Fixture& f,
    const x280::harness::RunSpec& spec,
    const SweepPoint& point,
    uint32_t W,
    uint32_t N,
    bool assess_clock,
    TransportLaunch& out) {
    const uint32_t transfer_size = point.transfer_size;
    const uint32_t boundary = point.boundary;

    const L1Layout TL = L1Layout::compute();
    const ttm::CoreCoord worker{0, 0};
    auto l1 = x280::harness::regions::allocate_tile_l1_buffer(
        f.mesh.get(), ttm::CoreRangeSet(ttm::CoreRange(worker, worker)), TL.block_bytes);
    const ttm::CoreRangeSet& worker_set = l1.worker_set;
    const uint32_t l1_base = l1.l1_tile_base;

    const auto l2cpu_translated = f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::RISCV_0_default);

    const LimRegions XR = LimRegions::compute(*f.tile);
    hreg::zero_region(*f.tile, XR.dst);
    hreg::zero_region(*f.tile, XR.verify);

    static uint32_t s_run_seq = 0u;
    ++s_run_seq;
    const uint32_t run_nonce = 0x900u + s_run_seq;

    const std::vector<uint32_t> tensix_args = sender_args({
        .transfer_size = transfer_size,
        .warmup_count = W,
        .transfer_count = N,
        .boundary = boundary,
        .run_nonce = run_nonce,
        .staging_addr = l1_base + TL.staging_offset,
        .scratch_addr = l1_base + TL.scratch_offset,
        .result_addr = l1_base + TL.result_offset,
        .gate_addr = 0u,
        .peer_dst_addr = XR.dst.addr,
        .peer_verify_addr = XR.verify.addr,
        .peer_noc_x = l2cpu_translated.x(),
        .peer_noc_y = l2cpu_translated.y(),
    });

    ttm::Program program = ttm::CreateProgram();
    const ttm::KernelHandle kernel = ttm::CreateKernel(
        program,
        (spec.open.kernel_dir / "transport_bandwidth_tensix_to_x280.cpp").string(),
        worker_set,
        ttm::DataMovementConfig{
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .noc = ttm::NOC::RISCV_0_default,
            .compiler_include_paths = {
                hart_::shared_kernel_dir(spec.open.kernel_dir, "transport_bandwidth"),
                hart_::shared_kernel_dir(spec.open.kernel_dir, "transport_bandwidth/backends"),
                hart_::shared_kernel_dir(spec.open.kernel_dir, "shared"),
                spec.open.kernel_dir,
            }});
    ttm::SetRuntimeArgs(program, kernel, worker_set, tensix_args);

    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    auto* device = f.mesh->get_device(0, 0);
    const auto timer_core = device->worker_core_from_logical_core(worker);
    out.expected_aiclk_hz = hexec::programmed_aiclk_hz(x280::harness::cluster(), *device);
    hmeas::AiclkEndpoint clock_start;
    if (assess_clock) {
        clock_start = hexec::read_aiclk_start(x280::harness::cluster(), *device, timer_core);
        ASSERT_TRUE(clock_start.errors.empty()) << "pre-run AICLK evidence is unavailable";
    }
    hexec::run_tensix_and_drain(*f.mesh, workload);
    if (assess_clock) {
        const auto clock_end = hexec::read_aiclk_end(x280::harness::cluster(), *device, timer_core);
        out.clock = hmeas::assess_aiclk_run(out.expected_aiclk_hz, clock_start, clock_end);
    }

    std::vector<uint32_t> l1_readback;
    ttmd::EnqueueReadMeshBuffer(f.mesh->mesh_command_queue(), l1_readback, l1.buffer, /*blocking=*/true);
    ASSERT_EQ(l1_readback.size(), TL.block_bytes / sizeof(uint32_t));
    transport_bw_result_t& result = out.result;
    std::memcpy(&result, l1_readback.data() + TL.result_offset / sizeof(uint32_t), sizeof(result));
    ASSERT_NO_FATAL_FAILURE(
        check_result(result, TRANSPORT_BW_BACKEND_ID_TENSIX_TO_X280, transfer_size, boundary, W, N));
}

inline void run_axis(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, const SweepPoint& point) {
    const uint32_t transfer_size = point.transfer_size;
    const uint32_t boundary = point.boundary;

    // An explicit W is used as given; otherwise a default-W pilot of the same
    // point sizes it.
    const auto explicit_warmup = hart_::env_u32("TRANSPORT_BW_W");
    hmeas::WarmupSelection selection{.warmup = explicit_warmup.value_or(kDefaultWarmupCount)};
    const uint32_t N = hart_::env_u32("TRANSPORT_BW_N", kDefaultTransferCount);
    ASSERT_GT(selection.warmup, 0u);
    ASSERT_GT(N, 0u);

    if (!explicit_warmup) {
        TransportLaunch pilot;
        ASSERT_NO_FATAL_FAILURE(launch(f, spec, point, kDefaultWarmupCount, N, false, pilot));
        selection = hmeas::calibrate_warmup(hmeas::batch_warmup_calibration_request(
            pilot.result.raw_batch_delta, N, pilot.expected_aiclk_hz, kDefaultWarmupCount, N));
        ASSERT_FALSE(selection.refusal.has_value())
            << "warm-up calibration refused: " << hmeas::warmup_refusal_name(*selection.refusal);
    }
    const uint32_t W = static_cast<uint32_t>(selection.warmup);

    TransportLaunch run;
    ASSERT_NO_FATAL_FAILURE(launch(f, spec, point, W, N, true, run));
    const auto& result = run.result;
    ASSERT_TRUE(run.clock.has_value()) << "measured launch assessed no AICLK";
    const auto& clock = *run.clock;

    const auto body = x280::harness::output::record(tensix_to_x280_record(
        result, {.transfer_size = transfer_size, .boundary = boundary, .warmup = W, .samples = N}, selection, clock));
    x280::harness::output::write_jsonl(spec.suite_label, body);
    std::printf("%s\n", body.c_str());
    EXPECT_TRUE(clock.errors.empty()) << x280::harness::output::clock_json(clock);
}

}  // namespace x280::suites::transport_bandwidth::tensix_to_x280

#endif  // X280_SUITES_TRANSPORT_BW_TENSIX_TO_X280_HPP
