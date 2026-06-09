// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include "fabric_mux_saturation_program.hpp"

namespace tt::tt_fabric::bench {

namespace {

std::string implementation_name(SaturationImplementation implementation) {
    switch (implementation) {
        case SaturationImplementation::V1: return "v1";
        case SaturationImplementation::V2: return "v2";
    }
    return "unknown";
}

std::string topology_name(SaturationTopology topology) {
    switch (topology) {
        case SaturationTopology::Fabric1D: return "1d";
        case SaturationTopology::Fabric2D: return "2d";
    }
    return "unknown";
}

std::string benchmark_name_for_case(const SaturationVariant& variant, const SaturationCase& benchmark_case) {
    return "mux_saturation/" + implementation_name(variant.implementation) +
           "/topo=" + topology_name(variant.topology) + "/" + benchmark_case.name_suffix;
}

std::vector<SaturationCase> get_saturation_cases() {
    return {
        SaturationCase{
            .name_suffix = "clients=2/bufs=2/slot=2048/pkts=4096",
            .num_clients = 2,
            .num_buffers_per_channel = 2,
            .channel_buffer_size_bytes = 2048,
            .num_packets_per_sender = 4096,
        },
    };
}

std::vector<SaturationVariant> get_saturation_variants() {
    return {
        SaturationVariant{
            .implementation = SaturationImplementation::V2,
            .topology = SaturationTopology::Fabric1D,
        },
        SaturationVariant{
            .implementation = SaturationImplementation::V1,
            .topology = SaturationTopology::Fabric1D,
        },
        SaturationVariant{
            .implementation = SaturationImplementation::V2,
            .topology = SaturationTopology::Fabric2D,
        },
        SaturationVariant{
            .implementation = SaturationImplementation::V1,
            .topology = SaturationTopology::Fabric2D,
        },
    };
}

class SaturationContextManager {
public:
    FabricMuxSaturationBenchmarkContext& get_context(SaturationTopology topology) {
        if (!active_topology_.has_value() || active_topology_.value() != topology) {
            context_.shutdown();
            context_.initialize(topology);
            active_topology_ = topology;
        }
        return context_;
    }

    void shutdown() {
        context_.shutdown();
        active_topology_.reset();
    }

private:
    FabricMuxSaturationBenchmarkContext context_;
    std::optional<SaturationTopology> active_topology_;
};

void BM_FabricMuxSaturation(
    benchmark::State& state,
    SaturationContextManager* context_manager,
    const SaturationVariant& variant,
    const SaturationCase& benchmark_case) {
    auto& context = context_manager->get_context(variant.topology);
    state.counters["clients"] = benchmark::Counter(static_cast<double>(benchmark_case.num_clients));
    state.counters["buffers_per_channel"] =
        benchmark::Counter(static_cast<double>(benchmark_case.num_buffers_per_channel));
    state.counters["channel_buffer_size_bytes"] =
        benchmark::Counter(static_cast<double>(benchmark_case.channel_buffer_size_bytes));
    state.counters["num_packets_per_sender"] =
        benchmark::Counter(static_cast<double>(benchmark_case.num_packets_per_sender));

    if (context.get_devices().empty()) {
        state.SkipWithError("benchmark context did not expose any devices");
        return;
    }

    auto* primary_device = context.get_devices().front()->get_devices()[0];
    state.counters["clock_mhz"] = benchmark::Counter(static_cast<double>(get_tt_npu_clock(primary_device)));

    std::string rejection_reason;
    if (!context.can_support_case(variant, benchmark_case, &rejection_reason)) {
        state.SkipWithError(rejection_reason.c_str());
        return;
    }

    const double device_clock_hz = static_cast<double>(get_tt_npu_clock(primary_device)) * 1.0e6;
    for ([[maybe_unused]] auto _ : state) {
        const auto run_result = run_mux_saturation_once(context, variant, benchmark_case);
        if (!run_result.success) {
            state.SkipWithError(run_result.error_message.c_str());
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

}  // namespace tt::tt_fabric::bench

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    auto benchmark_cases = tt::tt_fabric::bench::get_saturation_cases();
    auto benchmark_variants = tt::tt_fabric::bench::get_saturation_variants();
    tt::tt_fabric::bench::SaturationContextManager runtime_context_manager;
    std::size_t registered_case_count = 0;
    const std::array<tt::tt_fabric::bench::SaturationTopology, 2> topology_registration_order = {
        tt::tt_fabric::bench::SaturationTopology::Fabric1D,
        tt::tt_fabric::bench::SaturationTopology::Fabric2D,
    };
    for (const auto topology : topology_registration_order) {
        tt::tt_fabric::bench::FabricMuxSaturationBenchmarkContext registration_context;
        registration_context.initialize(topology);
        for (const auto& variant : benchmark_variants) {
            if (variant.topology != topology) {
                continue;
            }

            for (const auto& benchmark_case : benchmark_cases) {
                std::string rejection_reason;
                if (!registration_context.can_support_case(variant, benchmark_case, &rejection_reason)) {
                    log_info(
                        tt::LogTest,
                        "Skipping benchmark registration for {}: {}",
                        tt::tt_fabric::bench::benchmark_name_for_case(variant, benchmark_case),
                        rejection_reason);
                    continue;
                }

                const auto benchmark_name = tt::tt_fabric::bench::benchmark_name_for_case(variant, benchmark_case);
                benchmark::RegisterBenchmark(
                    benchmark_name.c_str(),
                    [&runtime_context_manager, variant, benchmark_case](benchmark::State& state) {
                        tt::tt_fabric::bench::BM_FabricMuxSaturation(
                            state, &runtime_context_manager, variant, benchmark_case);
                    })
                    ->UseManualTime()
                    ->Iterations(1)
                    ->ReportAggregatesOnly(true);
                registered_case_count += 1;
            }
        }
        registration_context.shutdown();
    }

    if (registered_case_count == 0) {
        log_info(tt::LogTest, "No mux saturation benchmark cases were registered on this system");
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    runtime_context_manager.shutdown();
    return 0;
}
