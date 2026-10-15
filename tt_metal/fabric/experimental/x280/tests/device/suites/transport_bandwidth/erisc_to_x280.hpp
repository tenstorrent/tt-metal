// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/transport_bandwidth/erisc_to_x280.hpp -- idle-ERISC -> X280 member.
//
// One to eight idle ERISCs each run the Tensix -> X280 member's kernel
// (built with TRANSPORT_BW_SENDER_IDLE_ERISC) and issue noc_async_write() into
// their own range of X280 LIM; X280 is a passive target with no image, and
// nothing replies. A host start gate releases the writers' timed loops
// together after warm-up. The host checks every writer's LIM range after the
// run.

#ifndef X280_SUITES_TRANSPORT_BW_ERISC_TO_X280_HPP
#define X280_SUITES_TRANSPORT_BW_ERISC_TO_X280_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/l1_window.hpp"
#include "harness/regions/metal_l1_region.hpp"
#include "harness/regions/regions.hpp"
#include "harness/runner.hpp"
#include "harness/metal_context.hpp"
#include "kernels/shared/ubench_units.h"
#include "tests/device/suites/packet_injection/regions.hpp"
#include "tests/device/suites/transport_bandwidth/records.hpp"
#include "tests/device/suites/transport_bandwidth/tensix_to_x280.hpp"

#include "transport_bandwidth_common.h"
#include "tensix_to_x280.h"

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

namespace x280::suites::transport_bandwidth::erisc_to_x280 {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hreg = x280::harness::regions;
namespace pi = x280::harness::regions::packet_injection;
using tensix_to_x280::L1Layout;

constexpr uint32_t kDefaultWarmupCount = 128u;
constexpr uint32_t kDefaultTransferCount = 16384u;

// How long the host waits for every writer to reach the start gate, counted
// from the enqueue: below a writer's own wait (kGateBoundTicks, 2^31 ERISC
// wall ticks, about 1.59 s at 1.35 GHz), so the host gives up before any
// writer does.
constexpr auto kGateTimeout = std::chrono::seconds{1};

// INJECTION only: `writers` idle ERISCs, each issuing `transfer_size` bytes.
struct SweepPoint {
    uint32_t transfer_size;
    uint32_t writers;
    const char* label;
};
using Axis = SweepPoint;

// Writer w's run nonce: its LIM range holds TB_TX2X_DST_WORD(nonce, k).
inline uint32_t writer_nonce(uint32_t run_nonce, uint32_t w) { return run_nonce + (w << 20); }

// One launch's results, writer w at index w. clock is set only when the launch
// assessed AICLK on writer 0's wall clock.
struct EriscLaunch {
    std::vector<transport_bw_result_t> results;
    uint64_t expected_aiclk_hz = 0u;
    std::optional<hmeas::AiclkRunEvidence> clock;
    uint64_t gate_release_ns = 0u;  // host time to write every writer's start word
};

// Launches `writers` once with W warm-up and N timed transfers each, and checks
// every completion record and LIM range. Fatal failures leave `out` partial.
inline void launch(
    x280::harness::session::Fixture& f,
    const x280::harness::RunSpec& spec,
    const SweepPoint& point,
    const std::vector<pi::LogicalCoord>& writers,
    uint32_t W,
    uint32_t N,
    bool assess_clock,
    EriscLaunch& out) {
    auto& cluster = x280::harness::cluster();
    const auto& hal = ttm::MetalContext::instance().hal();
    auto* device = f.mesh->get_device(0, 0);
    const uint32_t P = static_cast<uint32_t>(writers.size());
    const uint32_t transfer_size = point.transfer_size;
    const uint32_t boundary = TRANSPORT_BW_BOUNDARY_INJECTION;

    const L1Layout TL = L1Layout::compute();
    const hreg::L1Window window =
        hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::IDLE_ETH, hreg::kCacheLineBytes);
    ASSERT_LE(TL.block_bytes, window.local.size) << "the sender's L1 block does not fit the idle-ERISC window";
    const uint32_t l1_base = static_cast<uint32_t>(window.local.base);

    const auto avail = f.tile->available_lim();
    hreg::Arena arena(hreg::Region{avail.begin, avail.size()});
    const hreg::Region dst = arena.allocate(uint64_t{P} * TB_TX2X_STAGING_MAX_BYTES);
    const hreg::Region verify = arena.allocate(uint64_t{P} * hreg::kCacheLineBytes);
    // Each writer's first transfer_size bytes are what it writes and what is checked.
    for (uint32_t w = 0; w < P; ++w) {
        hreg::zero_region(*f.tile, hreg::Region{dst.addr + uint64_t{w} * TB_TX2X_STAGING_MAX_BYTES, transfer_size});
    }
    hreg::zero_region(*f.tile, verify);
    ASSERT_TRUE(ubench_noc_congruent(l1_base + TL.staging_offset, dst.addr))
        << "staging and LIM destination are not congruent mod " << UBENCH_NOC_UNIT_BYTES;

    static uint32_t s_run_seq = 0u;
    ++s_run_seq;
    const uint32_t run_nonce = 0xE00u + s_run_seq;
    const auto l2cpu_translated = f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::NOC_0);

    std::vector<tt_cxy_pair> cores;
    std::vector<ttm::CoreRange> ranges;
    for (const auto& w : writers) {
        const auto virt = x280::X280Runtime::virtual_core_from_logical_core(
            *f.mesh, ttm::CoreCoord{w.x, w.y}, tt::CoreType::ETH, ttm::NOC::NOC_0);
        cores.emplace_back(device->id(), virt.x(), virt.y());
        ranges.emplace_back(ttm::CoreCoord{w.x, w.y});
        // Zeroed so the gate and result words start clear.
        std::vector<uint8_t> zeros(TL.block_bytes, 0u);
        cluster.write_core(zeros.data(), TL.block_bytes, cores.back(), l1_base);
    }

    ttm::Program program = ttm::CreateProgram();
    const ttm::KernelHandle kernel = ttm::CreateKernel(
        program,
        (spec.open.kernel_dir / "transport_bandwidth_tensix_to_x280.cpp").string(),
        ttm::CoreRangeSet(ranges),
        ttm::EthernetConfig{
            .eth_mode = ttm::Eth::IDLE,
            .noc = ttm::NOC::NOC_0,
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .defines = {{"TRANSPORT_BW_SENDER_IDLE_ERISC", "1"}},
        });
    for (uint32_t w = 0; w < P; ++w) {
        ttm::SetRuntimeArgs(
            program,
            kernel,
            ttm::CoreCoord{writers[w].x, writers[w].y},
            tensix_to_x280::sender_args({
                .transfer_size = transfer_size,
                .warmup_count = W,
                .transfer_count = N,
                .boundary = boundary,
                .run_nonce = writer_nonce(run_nonce, w),
                .staging_addr = l1_base + TL.staging_offset,
                .scratch_addr = l1_base + TL.scratch_offset,
                .result_addr = l1_base + TL.result_offset,
                .gate_addr = l1_base + TL.gate_offset,
                .peer_dst_addr = dst.addr + uint64_t{w} * TB_TX2X_STAGING_MAX_BYTES,
                .peer_verify_addr = verify.addr + uint64_t{w} * hreg::kCacheLineBytes,
                .peer_noc_x = l2cpu_translated.x(),
                .peer_noc_y = l2cpu_translated.y(),
            }));
    }
    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    // The host reads writer 0's wall clock only while no kernel runs: a host
    // read re-latches the high word eth_read_wall_clock() returns.
    const ttm::CoreCoord reference{cores[0].x, cores[0].y};
    out.expected_aiclk_hz = hexec::programmed_aiclk_hz(cluster, *device);
    hmeas::AiclkEndpoint clock_start;
    if (assess_clock) {
        clock_start = hexec::read_aiclk_start(cluster, *device, reference);
        ASSERT_TRUE(clock_start.errors.empty()) << "pre-run AICLK evidence is unavailable";
    }

    hexec::enqueue_tensix(*f.mesh, workload);
    // Start gate: every writer has published its nonce (warmed up), then the
    // host writes each one's start word.
    // A writer that has published stays published, so each pass reads only the rest.
    std::vector<bool> published(P, false);
    uint32_t n_published = 0;
    const auto gate_deadline = std::chrono::steady_clock::now() + kGateTimeout;
    while (n_published < P && std::chrono::steady_clock::now() < gate_deadline) {
        for (uint32_t w = 0; w < P; ++w) {
            if (published[w]) {
                continue;
            }
            uint32_t ready = 0;
            cluster.read_core(&ready, sizeof(ready), cores[w], l1_base + TL.gate_offset);
            if (ready == writer_nonce(run_nonce, w)) {
                published[w] = true;
                ++n_published;
            }
        }
    }
    const bool all_ready = n_published == P;
    if (all_ready) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t w = 0; w < P; ++w) {
            const uint32_t go = writer_nonce(run_nonce, w);
            cluster.write_core(&go, sizeof(go), cores[w], l1_base + TL.gate_offset + sizeof(uint32_t));
        }
        out.gate_release_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    // A writer left at the gate gives up after its own bound and ends
    // RENDEZVOUS_TIMEOUT.
    hexec::finish_tensix(*f.mesh);
    ASSERT_TRUE(all_ready) << "not every writer reached the start gate within " << kGateTimeout.count() << " s";
    if (assess_clock) {
        const auto clock_end = hexec::read_aiclk_end(cluster, *device, reference);
        out.clock = hmeas::assess_aiclk_run(out.expected_aiclk_hz, clock_start, clock_end);
    }

    out.results.assign(P, transport_bw_result_t{});
    for (uint32_t w = 0; w < P; ++w) {
        SCOPED_TRACE("writer " + std::to_string(w));
        cluster.read_core(&out.results[w], sizeof(transport_bw_result_t), cores[w], l1_base + TL.result_offset);
        ASSERT_NO_FATAL_FAILURE(tensix_to_x280::check_result(
            out.results[w], TRANSPORT_BW_BACKEND_ID_ERISC_TO_X280, transfer_size, boundary, W, N));
    }

    for (uint32_t w = 0; w < P; ++w) {
        const auto lim = f.tile->read_available_lim(dst.addr + uint64_t{w} * TB_TX2X_STAGING_MAX_BYTES, transfer_size);
        ASSERT_EQ(lim.size(), transfer_size);
        const uint8_t* range = lim.data();
        uint32_t mismatches = 0;
        uint32_t first = 0;
        uint32_t first_read = 0;
        for (uint32_t k = 0; k < transfer_size / sizeof(uint32_t); ++k) {
            uint32_t word = 0;
            std::memcpy(&word, range + k * sizeof(uint32_t), sizeof(word));
            if (word != TB_TX2X_DST_WORD(writer_nonce(run_nonce, w), k) && mismatches++ == 0) {
                first = k;
                first_read = word;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "writer " << w << "'s LIM range differs from its staging pattern, first at word "
                                  << first << " (read 0x" << std::hex << first_read << ", expected 0x"
                                  << TB_TX2X_DST_WORD(writer_nonce(run_nonce, w), first) << std::dec << ")";
    }
}

inline void run_axis(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, const SweepPoint& point) {
    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::IDLE_ETH)) {
        GTEST_SKIP() << "IDLE_ETH programmable cores are not enabled, so no ERISC writer exists";
    }
    auto* device = f.mesh->get_device(0, 0);
    std::vector<pi::LogicalCoord> inactive;
    for (const auto& c : device->get_inactive_ethernet_cores()) {
        inactive.push_back(pi::LogicalCoord{static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)});
    }
    const pi::WriterSelection selection = pi::select_ethernet_writers(inactive, point.writers);
    if (selection.outcome == pi::WriterSelection::Outcome::Shortfall) {
        GTEST_SKIP() << "writer availability: " << selection.describe();
    }
    ASSERT_TRUE(selection.ok()) << "writer selection rejected: " << selection.describe();

    // An explicit W is used as given; otherwise a default-W pilot of the same
    // point sizes it from writer 0's interval.
    const auto explicit_warmup = hart_::env_u32("TRANSPORT_BW_W");
    hmeas::WarmupSelection warmup{.warmup = explicit_warmup.value_or(kDefaultWarmupCount)};
    const uint32_t N = hart_::env_u32("TRANSPORT_BW_N", kDefaultTransferCount);
    ASSERT_GT(warmup.warmup, 0u);
    ASSERT_GT(N, 0u);
    if (!explicit_warmup) {
        EriscLaunch pilot;
        ASSERT_NO_FATAL_FAILURE(launch(f, spec, point, selection.writers, kDefaultWarmupCount, N, false, pilot));
        warmup = hmeas::calibrate_warmup(hmeas::batch_warmup_calibration_request(
            pilot.results[0].raw_batch_delta, N, pilot.expected_aiclk_hz, kDefaultWarmupCount, N));
        ASSERT_FALSE(warmup.refusal.has_value())
            << "warm-up calibration refused: " << hmeas::warmup_refusal_name(*warmup.refusal);
    }
    const uint32_t W = static_cast<uint32_t>(warmup.warmup);

    EriscLaunch run;
    ASSERT_NO_FATAL_FAILURE(launch(f, spec, point, selection.writers, W, N, true, run));
    ASSERT_TRUE(run.clock.has_value()) << "measured launch assessed no AICLK";
    std::vector<WriterAt> at;
    for (const auto& w : selection.writers) {
        at.push_back({w.x, w.y});
    }
    const auto body = x280::harness::output::record(erisc_to_x280_record(
        run.results,
        at,
        {.transfer_size = point.transfer_size, .boundary = TRANSPORT_BW_BOUNDARY_INJECTION, .warmup = W, .samples = N},
        warmup,
        *run.clock,
        run.gate_release_ns));
    x280::harness::output::write_jsonl(spec.suite_label, body);
    std::printf("%s\n", body.c_str());
    EXPECT_TRUE(run.clock->errors.empty()) << x280::harness::output::clock_json(*run.clock);
}

}  // namespace x280::suites::transport_bandwidth::erisc_to_x280

#endif  // X280_SUITES_TRANSPORT_BW_ERISC_TO_X280_HPP
