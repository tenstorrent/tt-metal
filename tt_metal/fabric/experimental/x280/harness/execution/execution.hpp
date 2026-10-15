// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/execution/execution.hpp -- enqueue-wait-cleanup wrapper for X280,
// and lightweight Metal launch helpers for a Tensix data-movement program.
//
// Cases that submit an X280 image share one active-run pattern:
//   1) build an ActiveRunRequest from an X280 image and an in-LIM args block;
//   2) enqueue it on the resident-idle tile;
//   3) wait for the session to reach ResultReady with a hard timeout;
//   4) cleanup with a timeout;
//   5) surface both the wait outcome and the cleanup outcome to gtest.
//
// The Tensix members additionally run a data-movement kernel workload that
// enqueues non-blocking and drains via Finish() on the mesh command queue.
//
// This header names both flows so the tests share one implementation.
// Failures fall through as gtest ADD_FAILURE / EXPECT_* to keep the
// existing skip/fail semantics.

#ifndef X280_HARNESS_EXECUTION_HPP
#define X280_HARNESS_EXECUTION_HPP

#include "x280/device/x280_runtime.hpp"
#include "harness/targets/x280_tile.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "kernels/shared/tensix_live_timer.h"
#include "internal/ethernet/tt_eth_ss_regs.h"
#include "llrt/tt_cluster.hpp"

#include <umd/device/arc/firmware_telemetry_reader.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/telemetry.hpp>

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <string>
#include <thread>
#include <iostream>

namespace x280::harness::execution {

using namespace std::chrono_literals;
namespace ttmd = tt::tt_metal::distributed;

// Returns steady-clock nanoseconds. Values are process-local and used only for
// elapsed host brackets; they are neither UTC nor stable across process runs.
inline uint64_t host_monotonic_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// Resolves the UMD device underlying an opened Metal device. The returned
// object is owned by Cluster and remains valid only while that device is open.
inline tt::umd::TTDevice& umd_device(const tt::Cluster& cluster, const tt::tt_metal::IDevice& device) {
    auto* chip = cluster.get_driver()->get_chip(device.id());
    if (chip == nullptr || chip->get_tt_device() == nullptr) {
        throw std::runtime_error("opened Metal device has no UMD telemetry device");
    }
    return *chip->get_tt_device();
}

// Reads the firmware-programmed maximum AICLK in Hz. The required CMFW tag has
// no compiled fallback; absence and MHz-to-Hz overflow throw.
inline uint64_t programmed_aiclk_hz(const tt::Cluster& cluster, const tt::tt_metal::IDevice& device) {
    auto* telemetry = umd_device(cluster, device).get_firmware_telemetry_reader();
    if (telemetry == nullptr || !telemetry->is_entry_available(tt::umd::TelemetryTag::AICLK_LIMIT_MAX)) {
        throw std::runtime_error("CMFW AICLK_LIMIT_MAX telemetry is unavailable");
    }
    const uint32_t mhz = telemetry->read_entry(tt::umd::TelemetryTag::AICLK_LIMIT_MAX);
    if (mhz == 0) {
        throw std::runtime_error("CMFW AICLK_LIMIT_MAX is invalid: " + std::to_string(mhz));
    }
    return static_cast<uint64_t>(mhz) * 1000000ull;
}

// The live timer at translated `core`, or nullopt for a core that has none.
inline std::optional<measurement::Counter> clock_reference_at(
    const tt::Cluster& cluster, const tt::tt_metal::IDevice& device, const tt::tt_metal::CoreCoord& core) {
    const auto type = cluster.get_soc_desc(device.id())
                          .get_coord_at(tt_xy_pair(core.x, core.y), tt::CoordSystem::TRANSLATED)
                          .core_type;
    switch (type) {
        case tt::CoreType::TENSIX: return measurement::Counter::TensixLiveTimer;
        case tt::CoreType::ETH:
        case tt::CoreType::ACTIVE_ETH:
        case tt::CoreType::IDLE_ETH: return measurement::Counter::EriscWallClock;
        default: return std::nullopt;
    }
}

// The NoC0 coordinate of the core at translated (Metal virtual) `core` on
// `chip`, from the chip's SoC descriptor. Throws std::runtime_error naming the
// coordinate when the descriptor resolves it to no core.
inline std::array<uint32_t, 2> noc0_at(const tt::Cluster& cluster, int chip, const tt::tt_metal::CoreCoord& core) {
    const auto& soc = cluster.get_soc_desc(chip);
    tt::umd::CoreCoord found;
    try {
        found = soc.get_coord_at(tt_xy_pair(core.x, core.y), tt::CoordSystem::TRANSLATED);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "translated (" + std::to_string(core.x) + "," + std::to_string(core.y) + ") on chip " +
            std::to_string(chip) + " resolves to no core: " + error.what());
    }
    const auto noc0 = soc.translate_coord_to(found, tt::CoordSystem::NOC0);
    return {static_cast<uint32_t>(noc0.x), static_cast<uint32_t>(noc0.y)};
}

// The core at `logical` and translated `virt` on `chip`, issuing on `noc`, as
// records place it: its NoC0 coordinate from noc0_at(), which throws for a
// translated coordinate the descriptor does not resolve.
inline x280::harness::output::CorePlacement placement_at(
    const tt::Cluster& cluster,
    int chip,
    const tt::tt_metal::CoreCoord& logical,
    const tt::tt_metal::CoreCoord& virt,
    uint32_t noc) {
    return {
        .logical = std::array<uint32_t, 2>{static_cast<uint32_t>(logical.x), static_cast<uint32_t>(logical.y)},
        .virt = std::array<uint32_t, 2>{static_cast<uint32_t>(virt.x), static_cast<uint32_t>(virt.y)},
        .noc0 = noc0_at(cluster, chip, virt),
        .noc = noc,
    };
}

// The translated core AICLK evidence for a workload at translated `core`
// samples, and its live timer: `core` itself when the host reads a live timer
// there (a Tensix or Ethernet core), otherwise logical Tensix (0,0). A DRISC's
// wall counter has no host sampler, so its evidence is Tensix-referenced; the
// evidence names the core it sampled.
inline std::pair<tt::tt_metal::CoreCoord, measurement::Counter> aiclk_reference(
    const tt::Cluster& cluster, const tt::tt_metal::IDevice& device, const tt::tt_metal::CoreCoord& core) {
    if (const auto counter = clock_reference_at(cluster, device, core)) {
        return {core, *counter};
    }
    return {device.worker_core_from_logical_core(tt::tt_metal::CoreCoord{0, 0}), measurement::Counter::TensixLiveTimer};
}

// Reads one coherent value of `reference`'s counter at translated `core` with
// an MMIO live-high/low/live-high sequence.
//
// TensixLiveTimer is the Tensix debug timestamper, whose live high word is
// documented (WormholeB0 DebugTimestamper.md). EriscWallClock is the counter
// eth_read_wall_clock() reads; that reads low then the latched _1_AT
// (tt_eth_api.h:40-44), while this reads the live _1. That _1 is live is
// inferred from the timestamper layout and the header's naming: the Ethernet
// ISA docs do not document it. A host low-word read re-latches _1_AT, so
// eth_credit samples only while none of its kernels runs; base firmware's own
// eth_read_wall_clock() (eth_fw_api.h:262) can still race it.
//
// Each attempt is host-bracketed; after eight std::runtime_error is thrown.
inline measurement::HostClockSample read_live_timer_sample(
    const tt::Cluster& cluster,
    const tt::tt_metal::IDevice& device,
    const tt::tt_metal::CoreCoord& core,
    measurement::Counter reference) {
    uint32_t low_addr = 0;
    uint32_t high_addr = 0;
    switch (reference) {
        case measurement::Counter::TensixLiveTimer:
            low_addr = TENSIX_TIMER_LOCAL_LOW_ADDR;
            high_addr = TENSIX_TIMER_LOCAL_HIGH_ADDR;
            break;
        case measurement::Counter::EriscWallClock:
            low_addr = ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0;
            high_addr = ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1;
            break;
        default:
            throw std::logic_error(
                "read_live_timer_sample: no live timer for Counter " +
                std::to_string(static_cast<uint32_t>(reference)));
    }
    const tt_cxy_pair target(device.id(), core);
    for (unsigned attempt = 0; attempt < 8; ++attempt) {
        uint32_t high, low, check;
        const auto before = host_monotonic_ns();
        cluster.read_reg(&high, target, high_addr);
        cluster.read_reg(&low, target, low_addr);
        cluster.read_reg(&check, target, high_addr);
        const auto after = host_monotonic_ns();
        if (high == check) {
            return {before, (uint64_t(high) << 32) | low, after};
        }
    }
    throw std::runtime_error(std::string(measurement::counter_name(reference)) + " H/L/H retry exhausted");
}

// Captures all required CMFW clock fields in one host-time bracket. Missing
// tags and read failures throw; the caller records TelemetryUnavailable.
inline measurement::CmfwClockSample read_cmfw_clock_sample(
    const tt::Cluster& cluster, const tt::tt_metal::IDevice& device) {
    auto* telemetry = umd_device(cluster, device).get_firmware_telemetry_reader();
    constexpr tt::umd::TelemetryTag tags[] = {
        tt::umd::TelemetryTag::AICLK_LIMIT_MAX,
        tt::umd::TelemetryTag::AICLK,
        tt::umd::TelemetryTag::AICLK_ARB_MAX,
        tt::umd::TelemetryTag::THERM_TRIP_COUNT,
        tt::umd::TelemetryTag::UPDATE_TELEM_SPEED,
    };
    if (telemetry == nullptr) {
        throw std::runtime_error("CMFW telemetry reader is unavailable");
    }
    for (auto tag : tags) {
        if (!telemetry->is_entry_available(tag)) {
            throw std::runtime_error("required CMFW clock telemetry tag is unavailable");
        }
    }
    measurement::CmfwClockSample sample;
    sample.host_before_ns = host_monotonic_ns();
    sample.limit_max_mhz = telemetry->read_entry(tt::umd::TelemetryTag::AICLK_LIMIT_MAX);
    sample.current_mhz = telemetry->read_entry(tt::umd::TelemetryTag::AICLK);
    sample.arb_max_raw = telemetry->read_entry(tt::umd::TelemetryTag::AICLK_ARB_MAX);
    sample.thermal_trip_count = telemetry->read_entry(tt::umd::TelemetryTag::THERM_TRIP_COUNT);
    sample.update_period_ms = telemetry->read_entry(tt::umd::TelemetryTag::UPDATE_TELEM_SPEED);
    sample.host_after_ns = host_monotonic_ns();
    return sample;
}

// Captures telemetry followed by S0/S1 immediately before a workload at
// translated `core`, on aiclk_reference()'s core. The call blocks for one CMFW
// update period plus 50 ms and records acquisition failures in the returned
// endpoint instead of manufacturing usable samples.
inline measurement::AiclkEndpoint read_aiclk_start(
    const tt::Cluster& cluster, const tt::tt_metal::IDevice& device, const tt::tt_metal::CoreCoord& core) {
    const auto [sampled, reference] = aiclk_reference(cluster, device, core);
    measurement::AiclkEndpoint endpoint;
    endpoint.reference = reference;
    endpoint.reference_core =
        std::array<uint32_t, 2>{static_cast<uint32_t>(sampled.x), static_cast<uint32_t>(sampled.y)};
    try {
        auto initial = read_cmfw_clock_sample(cluster, device);
        if (initial.update_period_ms == 0 || initial.update_period_ms > 1000) {
            endpoint.errors.push_back(measurement::ClockError::InvalidTelemetryPeriod);
            return endpoint;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(initial.update_period_ms));
        endpoint.telemetry = read_cmfw_clock_sample(cluster, device);
    } catch (const std::exception& error) {
        std::cerr << "start CMFW clock acquisition failed: " << error.what() << '\n';
        endpoint.errors.push_back(measurement::ClockError::TelemetryUnavailable);
        return endpoint;
    }
    try {
        endpoint.first = read_live_timer_sample(cluster, device, sampled, reference);
        std::this_thread::sleep_for(std::chrono::nanoseconds(measurement::kAiclkSampleGapNs));
        endpoint.second = read_live_timer_sample(cluster, device, sampled, reference);
    } catch (const std::exception& error) {
        std::cerr << "start AICLK reference acquisition failed: " << error.what() << '\n';
        endpoint.errors.push_back(measurement::ClockError::ReferenceReadFailed);
    }
    return endpoint;
}

// Captures E0/E1 immediately after a workload at translated `core`, on
// aiclk_reference()'s core, then waits one CMFW update period and records end
// telemetry. It blocks for 50 ms plus that update period
// and returns explicit acquisition errors for postmeasurement serialization.
inline measurement::AiclkEndpoint read_aiclk_end(
    const tt::Cluster& cluster, const tt::tt_metal::IDevice& device, const tt::tt_metal::CoreCoord& core) {
    const auto [sampled, reference] = aiclk_reference(cluster, device, core);
    measurement::AiclkEndpoint endpoint;
    endpoint.reference = reference;
    endpoint.reference_core =
        std::array<uint32_t, 2>{static_cast<uint32_t>(sampled.x), static_cast<uint32_t>(sampled.y)};
    try {
        endpoint.first = read_live_timer_sample(cluster, device, sampled, reference);
        std::this_thread::sleep_for(std::chrono::nanoseconds(measurement::kAiclkSampleGapNs));
        endpoint.second = read_live_timer_sample(cluster, device, sampled, reference);
    } catch (const std::exception& error) {
        std::cerr << "end AICLK reference acquisition failed: " << error.what() << '\n';
        endpoint.errors.push_back(measurement::ClockError::ReferenceReadFailed);
    }
    try {
        auto initial = read_cmfw_clock_sample(cluster, device);
        if (initial.update_period_ms == 0 || initial.update_period_ms > 1000) {
            endpoint.errors.push_back(measurement::ClockError::InvalidTelemetryPeriod);
            return endpoint;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(initial.update_period_ms));
        endpoint.telemetry = read_cmfw_clock_sample(cluster, device);
    } catch (const std::exception& error) {
        std::cerr << "end CMFW clock acquisition failed: " << error.what() << '\n';
        endpoint.errors.push_back(measurement::ClockError::TelemetryUnavailable);
    }
    return endpoint;
}

// A single X280 active-run request built from its logical parts.
struct X280Launch {
    std::string image_path;
    uint64_t run_value = 0;
    uint64_t args_addr = 0;
    uint64_t args_bytes = 0;
    std::chrono::seconds cleanup_timeout = 5s;
    std::chrono::seconds wait_timeout = 30s;
};

// Result of an X280 active-run: the enqueue result (session included) +
// terminal outcomes. On failure the caller can still read the enqueue,
// wait and cleanup fields to describe what happened.
struct X280Outcome {
    x280::EnqueueResult enqueue;
    x280::WaitResult wait;
    x280::CleanupResult cleanup;
    bool enqueued() const { return enqueue.enqueued(); }
    bool wait_ready() const { return wait.outcome == x280::WaitOutcome::ResultReady; }
    bool cleaned_up() const { return cleanup.cleaned_up(); }
    bool ok() const { return enqueued() && wait_ready() && cleaned_up(); }
};

// Enqueue without waiting. On failure, any created session is cleaned up;
// wait and cleanup remain unpopulated.
inline X280Outcome enqueue_x280(const targets::X280TileHandle& handle, const X280Launch& L) {
    X280Outcome out{};
    x280::ActiveRunRequest request;
    request.image_path = L.image_path;
    request.run_value = L.run_value;
    request.cleanup_timeout = L.cleanup_timeout;
    request.args_addr = L.args_addr;
    request.args_bytes = L.args_bytes;

    out.enqueue = handle.runtime->enqueue(*handle.tile, request);
    if (!out.enqueued() && out.enqueue.session) {
        out.cleanup = handle.runtime->cleanup(out.enqueue.session, L.cleanup_timeout);
    }
    return out;
}

// Wait for ResultReady or L.wait_timeout. Requires out.enqueued() and leaves
// the session live for cleanup_x280.
inline void wait_x280(const targets::X280TileHandle& handle, const X280Launch& L, X280Outcome& out) {
    const auto& session = handle.runtime->session(out.enqueue.session);
    out.wait = session.wait_until_result_ready(L.wait_timeout);
}

// Release an enqueued session exactly once, including after a wait timeout.
inline void cleanup_x280(const targets::X280TileHandle& handle, const X280Launch& L, X280Outcome& out) {
    out.cleanup = handle.runtime->cleanup(out.enqueue.session, L.cleanup_timeout);
}

/**
 * Fails the test unless window `index` on `tile` reads back cleared, as every
 * image owes on exit: a window still aimed at an address the allocator hands
 * out again is a standing path from this tile to that address's next owner.
 * `index` must be a slot the run programmed: a slot it never opened keeps what
 * it held, which after an ASIC reset is not zero (observed; no document states it).
 */
inline void expect_window_cleared(const X280Tile& tile, uint32_t index) {
    const auto window = tile.read_noc_window(index);
    EXPECT_TRUE(window_reads_cleared(window))
        << "TLB slot " << index << " is not cleared after the image returned: words " << std::hex
        << window.words.data[0] << " " << window.words.data[1] << " " << window.words.data[2] << " "
        << window.words.data[3] << std::dec << (window.diagnostic.empty() ? "" : "; ") << window.diagnostic;
}

inline X280Outcome run_x280(const targets::X280TileHandle& handle, const X280Launch& L) {
    X280Outcome out = enqueue_x280(handle, L);
    if (!out.enqueued()) {
        return out;
    }
    wait_x280(handle, L, out);
    cleanup_x280(handle, L, out);
    return out;
}

// Enqueue a Tensix MeshWorkload non-blocking, then drain via Finish().
inline void run_tensix_and_drain(ttmd::MeshDevice& mesh, ttmd::MeshWorkload& workload) {
    ttmd::EnqueueMeshWorkload(mesh.mesh_command_queue(), workload, /*blocking=*/false);
    ttmd::Finish(mesh.mesh_command_queue());
}

// Enqueue a Tensix MeshWorkload non-blocking; the caller is responsible for
// draining the queue later (paired-kernel members call Finish() after the
// X280 session completes so the two really overlap).
inline void enqueue_tensix(ttmd::MeshDevice& mesh, ttmd::MeshWorkload& workload) {
    ttmd::EnqueueMeshWorkload(mesh.mesh_command_queue(), workload, /*blocking=*/false);
}

// Drain the mesh command queue.
inline void finish_tensix(ttmd::MeshDevice& mesh) { ttmd::Finish(mesh.mesh_command_queue()); }

}  // namespace x280::harness::execution

#endif  // X280_HARNESS_EXECUTION_HPP
