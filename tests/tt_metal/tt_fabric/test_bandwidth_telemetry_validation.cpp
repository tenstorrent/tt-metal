// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Fabric Bandwidth Telemetry Validation - Clean & Parameterized
 *
 * Validates bandwidth telemetry across multiple transfer sizes.
 * Modular design with helper functions for readability.
 */

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <chrono>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <llrt/hal.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_fabric;
using tt::ChipId;

namespace {

constexpr size_t DEFAULT_NUM_ITERATIONS = 100;
constexpr size_t DEFAULT_TRACE_ITERS = 10;
constexpr double TOLERANCE_PERCENT = 15.0;
constexpr uint64_t BYTES_PER_WORD = 4;

struct ChannelCounters {
    uint64_t tx_words = 0;
    uint64_t tx_cycles = 0;

    ChannelCounters& operator+=(const ChannelCounters& other) {
        tx_words += other.tx_words;
        tx_cycles += other.tx_cycles;
        return *this;
    }

    ChannelCounters operator-(const ChannelCounters& other) const {
        return {tx_words - other.tx_words, tx_cycles - other.tx_cycles};
    }
};

struct ValidationMetrics {
    double telemetry_bandwidth_mbps = 0.0;
    double bench_bandwidth_mbps = 0.0;
    double error_percent = 0.0;
    double word_count_ratio = 0.0;
};

// Fixture
struct FabricBandwidthTelemetryFixture : public MeshDeviceFixtureBase {
    FabricBandwidthTelemetryFixture() :
        MeshDeviceFixtureBase(Config{
            .num_cqs = 1, .trace_region_size = 1u << 20, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}) {}

    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }

    tt::umd::Cluster& get_cluster() {
        auto& metal_ctx = tt::tt_metal::MetalContext::instance();
        return const_cast<tt::umd::Cluster&>(*metal_ctx.get_cluster().get_driver());
    }

    const Hal& get_hal() { return tt::tt_metal::MetalContext::instance().hal(); }
};

// Read TX counters from a channel (only if has valid cycles)
ChannelCounters read_tx_counters(tt::umd::Cluster& cluster, const Hal& hal, ChipId chip, uint8_t channel) {
    ChannelCounters counters;
    try {
        auto snap = tt::tt_fabric::read_fabric_telemetry(cluster, hal, chip, channel);
        if (snap.dynamic_info.has_value()) {
            const auto& di = snap.dynamic_info.value();
            if (di.tx_bandwidth.elapsed_active_cycles > 0) {
                counters.tx_words = di.tx_bandwidth.words_sent;
                counters.tx_cycles = di.tx_bandwidth.elapsed_active_cycles;
            }
        }
    } catch (const std::exception& e) {
        // Ignore read errors for inactive/unconfigured channels
        log_trace(tt::LogTest, "Failed to read telemetry for chip {} channel {}: {}", chip, channel, e.what());
    } catch (...) {
        log_trace(tt::LogTest, "Failed to read telemetry for chip {} channel {}: unknown error", chip, channel);
    }
    return counters;
}

// Sum TX counters across all 16 channels
ChannelCounters sum_all_tx_counters(tt::umd::Cluster& cluster, const Hal& hal, ChipId chip) {
    ChannelCounters total;
    for (uint8_t ch = 0; ch < 16; ++ch) {
        total += read_tx_counters(cluster, hal, chip, ch);
    }
    return total;
}

// Run benchmark and measure telemetry for a given transfer size
ValidationMetrics run_bandwidth_validation(
    FabricBandwidthTelemetryFixture& fixture,
    size_t transfer_size_bytes,
    size_t num_iterations,
    ChipId src_chip,
    ChipId dst_chip) {
    ValidationMetrics metrics;

    // Configure workload
    tt::tt_fabric::bench::PerfParams params;
    params.mesh_id = 0;
    params.src_chip = src_chip;
    params.dst_chip = dst_chip;
    params.tensor_bytes = transfer_size_bytes;
    params.page_size = std::min(static_cast<size_t>(4096), transfer_size_bytes);
    params.sender_core = {0, 0};
    params.receiver_core = {0, 0};
    params.trace_iters = DEFAULT_TRACE_ITERS;
    params.use_dram_dst = false;

    size_t total_payload_bytes = transfer_size_bytes * num_iterations * params.trace_iters;

    log_info(
        tt::LogTest,
        "Config: {} MB × {} iters × {} trace = {:.2f} GB payload",
        transfer_size_bytes / (1024 * 1024),
        num_iterations,
        params.trace_iters,
        total_payload_bytes / 1e9);

    // Warmup to load fabric router kernel
    log_info(tt::LogTest, "Running warmup transfer...");
    tt::tt_fabric::bench::run_unicast_once(&fixture, params);
    log_info(tt::LogTest, "Warmup complete");

    // Baseline measurement
    log_info(tt::LogTest, "Reading baseline counters...");
    ChannelCounters baseline = sum_all_tx_counters(fixture.get_cluster(), fixture.get_hal(), src_chip);

    float aiclk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(src_chip);
    log_info(tt::LogTest, "Using AICLK = {:.1f} MHz", aiclk_mhz);

    // Run measured transfers
    log_info(tt::LogTest, "Running {} measured transfers...", num_iterations);
    std::vector<tt::tt_fabric::bench::PerfPoint> results;
    for (size_t i = 0; i < num_iterations; ++i) {
        results.push_back(tt::tt_fabric::bench::run_unicast_once(&fixture, params));
        if ((i + 1) % 25 == 0) {
            log_info(tt::LogTest, "  Completed {}/{}", i + 1, num_iterations);
        }
    }
    log_info(tt::LogTest, "Transfers complete");

    // Calculate bench bandwidth
    double total_time_sec = 0.0;
    for (const auto& r : results) {
        total_time_sec += r.sec;
    }
    metrics.bench_bandwidth_mbps = (total_payload_bytes / (total_time_sec * params.trace_iters)) / 1e6;

    // Read telemetry after
    ChannelCounters after = sum_all_tx_counters(fixture.get_cluster(), fixture.get_hal(), src_chip);
    ChannelCounters delta = after - baseline;

    // Calculate telemetry bandwidth
    double telemetry_bytes = delta.tx_words * BYTES_PER_WORD;
    double telemetry_time_sec = delta.tx_cycles / (aiclk_mhz * 1e6);
    metrics.telemetry_bandwidth_mbps = (telemetry_bytes / telemetry_time_sec) / 1e6;

    metrics.error_percent = std::abs(metrics.telemetry_bandwidth_mbps - metrics.bench_bandwidth_mbps) /
                            metrics.bench_bandwidth_mbps * 100.0;
    metrics.word_count_ratio = telemetry_bytes / total_payload_bytes;

    log_info(tt::LogTest, "Results:");
    log_info(
        tt::LogTest,
        "  Counted:  {:.2f} GB ({:.1f}% of payload)",
        telemetry_bytes / 1e9,
        metrics.word_count_ratio * 100.0);
    log_info(tt::LogTest, "  Expected: {:.2f} GB payload", total_payload_bytes / 1e9);
    log_info(tt::LogTest, "  Telemetry BW: {:.2f} MB/s", metrics.telemetry_bandwidth_mbps);
    log_info(tt::LogTest, "  Bench BW:     {:.2f} MB/s", metrics.bench_bandwidth_mbps);
    log_info(tt::LogTest, "  Error:        {:.1f}%", metrics.error_percent);

    return metrics;
}

}  // namespace

// Test all sizes with a single fixture instance (avoids reinitialization hangs)
TEST(FabricBandwidthTelemetry, ValidateMultipleSizes) {
    if (!std::getenv("TT_METAL_FABRIC_TELEMETRY") || !std::getenv("TT_METAL_FABRIC_BW_TELEMETRY")) {
        GTEST_SKIP() << "Requires both TT_METAL_FABRIC_TELEMETRY and TT_METAL_FABRIC_BW_TELEMETRY";
    }

    if (tt::tt_metal::GetNumAvailableDevices() < 2) {
        GTEST_SKIP() << "Need at least 2 devices";
    }

    log_info(tt::LogTest, "========================================");
    log_info(tt::LogTest, "Multi-Size Bandwidth Validation");
    log_info(tt::LogTest, "========================================");

    FabricBandwidthTelemetryFixture fixture;
    fixture.setup();

    std::vector<size_t> test_sizes = {
        1 * 1024 * 1024,  // 1 MB - ~25% error
        5 * 1024 * 1024,  // 5 MB - ~9% error
        10 * 1024 * 1024  // 10 MB - ~4% error
    };

    for (size_t transfer_size : test_sizes) {
        log_info(tt::LogTest, "\n--- Testing {} MB transfers ---", transfer_size / (1024 * 1024));

        auto metrics = run_bandwidth_validation(fixture, transfer_size, DEFAULT_NUM_ITERATIONS, 0, 1);

        double tolerance = (transfer_size < 10 * 1024 * 1024) ? 30.0 : TOLERANCE_PERCENT;

        EXPECT_LT(metrics.error_percent, tolerance) << fmt::format(
            "{} MB: Error {:.1f}% exceeds tolerance {:.1f}%",
            transfer_size / (1024 * 1024),
            metrics.error_percent,
            tolerance);
    }

    fixture.teardown();
}
