// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <array>
#include <sstream>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/types/arch.hpp>

#include "context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include "fabric_mux_v2_benchmark_program.hpp"

namespace tt::tt_fabric::bench {
namespace {

std::string benchmark_name_for_case(const MuxV2ThroughputCase& benchmark_case) {
    return "mux_v2/standalone_mux_throughput/" + benchmark_case.name_suffix;
}

std::vector<MuxV2ThroughputCase> get_standalone_mux_v2_throughput_cases() {
    struct ForwarderNocConfig {
        const char* name = "";
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
    };

    constexpr uint8_t kDefaultBufferCount = 8;
    // Downstream drainer depth is fixed (kDefaultDrainerNumBuffers=16): not a client-facing
    // knob, and in production it mirrors fabric-router free slots rather than a mux setting.
    constexpr std::array<ForwarderNocConfig, 2> kForwarderNocSweep = {{
        ForwarderNocConfig{.name = "r0", .noc = tt::tt_metal::NOC::RISCV_0_default},
        ForwarderNocConfig{.name = "r1", .noc = tt::tt_metal::NOC::RISCV_1_default},
    }};
    constexpr std::array<uint8_t, 5> kBufferSweep = {1, 2, 4, 8, 16};
    constexpr std::array<uint32_t, 5> kPayloadSweep = {64, 1024, 2048, 4096, 0};
    constexpr std::array<uint32_t, 6> kSenderSweep = {1, 2, 4, 8, 16, 32};
    constexpr std::array<uint32_t, 2> kHighSenderSweep = {48, 64};

    std::vector<MuxV2ThroughputCase> cases;
    cases.reserve(
        kForwarderNocSweep.size() *
        (kBufferSweep.size() + kPayloadSweep.size() + kSenderSweep.size() + kHighSenderSweep.size()));

    for (const auto& noc_config : kForwarderNocSweep) {
        for (const auto buffer_count : kBufferSweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "buffer_sweep_1s_max_buf" + std::to_string(buffer_count) + "_" + noc_config.name,
                .num_buffers_per_channel = buffer_count,
                .forwarder_noc = noc_config.noc,
            });
        }

        for (const auto payload_bytes : kPayloadSweep) {
            const auto payload_name = payload_bytes == 0 ? std::string("max") : std::to_string(payload_bytes) + "B";
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "payload_sweep_1s_" + payload_name + "_buf8_" + noc_config.name,
                .packet_payload_size_bytes = payload_bytes,
                .num_buffers_per_channel = kDefaultBufferCount,
                .forwarder_noc = noc_config.noc,
            });
        }

        for (const auto sender_count : kSenderSweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "sender_sweep_" + std::to_string(sender_count) + "s_max_buf8_" + noc_config.name,
                .num_senders = sender_count,
                .num_buffers_per_channel = kDefaultBufferCount,
                .forwarder_noc = noc_config.noc,
            });
        }

        // High-sender cases use fewer per-channel buffers to stay within mux-core L1.
        for (const auto sender_count : kHighSenderSweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "high_sender_sweep_" + std::to_string(sender_count) + "s_max_buf4_" + noc_config.name,
                .num_senders = sender_count,
                .num_buffers_per_channel = 4,
                .forwarder_noc = noc_config.noc,
            });
        }
    }

    return cases;
}

void BM_StandaloneMuxV2Throughput(
    benchmark::State& state, FabricMuxV2BenchmarkContext* context, const MuxV2ThroughputCase& benchmark_case) {
    const auto resolved_payload_size_bytes = resolve_packet_payload_size_bytes(benchmark_case);
    const auto num_packets = benchmark_case.num_packets;

    state.counters["senders"] = benchmark::Counter(static_cast<double>(benchmark_case.num_senders));
    state.counters["payload_bytes"] = benchmark::Counter(static_cast<double>(resolved_payload_size_bytes));
    state.counters["num_packets"] = benchmark::Counter(static_cast<double>(num_packets));
    state.counters["buffers_per_channel"] =
        benchmark::Counter(static_cast<double>(benchmark_case.num_buffers_per_channel));
    state.counters["drainer_buffers"] = benchmark::Counter(static_cast<double>(benchmark_case.num_drainer_buffers));
    state.counters["clock_mhz"] = benchmark::Counter(static_cast<double>(get_tt_npu_clock(context->get_device())));

    std::string rejection_reason;
    if (!context->can_support_case(benchmark_case, &rejection_reason)) {
        state.SkipWithError(rejection_reason);
        return;
    }

    const double device_clock_hz = static_cast<double>(get_tt_npu_clock(context->get_device())) * 1.0e6;
    for ([[maybe_unused]] auto _ : state) {
        const auto run_result = run_standalone_mux_v2_benchmark_once(*context, benchmark_case);
        if (!run_result.success) {
            state.SkipWithError(run_result.error_message);
            return;
        }

        state.counters["aggregate_case_bytes"] = benchmark::Counter(static_cast<double>(run_result.aggregate_bytes));
        state.counters["max_sender_cycles"] = benchmark::Counter(static_cast<double>(run_result.max_sender_cycles));
        state.counters["bytes_per_cycle"] = benchmark::Counter(
            static_cast<double>(run_result.aggregate_bytes) / static_cast<double>(run_result.max_sender_cycles));
        state.counters["throughput_bytes_per_s"] =
            benchmark::Counter(static_cast<double>(run_result.aggregate_bytes), benchmark::Counter::kIsRate);

        state.SetIterationTime(static_cast<double>(run_result.max_sender_cycles) / device_clock_hz);
    }
}

}  // namespace

void register_and_run_standalone_mux_v2_throughput_benchmarks(FabricMuxV2BenchmarkContext& context) {
    auto benchmark_cases = get_standalone_mux_v2_throughput_cases();
    std::size_t registered_case_count = 0;
    for (const auto& benchmark_case : benchmark_cases) {
        std::string rejection_reason;
        if (!context.can_support_case(benchmark_case, &rejection_reason)) {
            log_info(
                tt::LogTest,
                "Skipping benchmark registration for {}: {}",
                benchmark_name_for_case(benchmark_case),
                rejection_reason);
            continue;
        }

        const auto benchmark_name = benchmark_name_for_case(benchmark_case);
        benchmark::RegisterBenchmark(
            benchmark_name,
            [&context, benchmark_case](benchmark::State& state) {
                BM_StandaloneMuxV2Throughput(state, &context, benchmark_case);
            })
            ->UseManualTime()
            ->Iterations(1)
            ->ReportAggregatesOnly(true);
        registered_case_count += 1;
    }

    if (registered_case_count == 0) {
        log_info(tt::LogTest, "No standalone mux-v2 throughput benchmark cases were registered on this device");
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}

}  // namespace tt::tt_fabric::bench

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    tt::tt_fabric::bench::FabricMuxV2BenchmarkContext context;
    context.initialize();
    benchmark::AddCustomContext("arch", tt::arch_to_str(tt::tt_metal::MetalContext::instance().get_cluster().arch()));

    tt::tt_fabric::bench::register_and_run_standalone_mux_v2_throughput_benchmarks(context);

    context.shutdown();
    return 0;
}
