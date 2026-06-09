// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <sstream>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include "fabric_mux_benchmark_program_utils.hpp"
#include "fabric_mux_v2_benchmark_program.hpp"

namespace tt::tt_fabric::bench {

namespace {

std::string benchmark_name_for_case(const MuxV2ThroughputCase& benchmark_case) {
    return "mux_v2/standalone_mux_throughput/" + benchmark_case.name_suffix;
}

std::vector<MuxV2ThroughputCase> get_standalone_mux_v2_throughput_cases() {
    return {
        MuxV2ThroughputCase{
            .name_suffix = "baseline_1s_max_buf1_r0",
        },
        MuxV2ThroughputCase{
            .name_suffix = "payload_1s_256B_buf1_r0",
            .packet_payload_size_bytes = 256,
        },
        MuxV2ThroughputCase{
            .name_suffix = "payload_1s_64B_buf1_r0",
            .packet_payload_size_bytes = 64,
        },
        MuxV2ThroughputCase{
            .name_suffix = "buffers_1s_max_buf4_r0",
            .num_buffers_per_channel = 4,
        },
        MuxV2ThroughputCase{
            .name_suffix = "noc_1s_max_buf4_r1",
            .num_buffers_per_channel = 4,
            .forwarder_noc = tt::tt_metal::NOC::RISCV_1_default,
        },
        MuxV2ThroughputCase{
            .name_suffix = "scale_4s_max_buf4_r0",
            .num_senders = 4,
            .num_buffers_per_channel = 4,
        },
        MuxV2ThroughputCase{
            .name_suffix = "scale_16s_64B_buf4_r0",
            .num_senders = 16,
            .packet_payload_size_bytes = 64,
            .num_buffers_per_channel = 4,
        },
        MuxV2ThroughputCase{
            .name_suffix = "tune_4s_256B_buf4_sb1_trid1",
            .num_senders = 4,
            .packet_payload_size_bytes = 256,
            .num_buffers_per_channel = 4,
            .service_burst_size = 1,
            .max_in_flight_trids = 1,
        },
        MuxV2ThroughputCase{
            .name_suffix = "tune_4s_256B_buf4_sb8_trid8",
            .num_senders = 4,
            .packet_payload_size_bytes = 256,
            .num_buffers_per_channel = 4,
        },
    };
}

}  // namespace

void FabricMuxV2BenchmarkContext::initialize() {
    auto available_device_ids = tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids();
    TT_FATAL(available_device_ids.contains(0), "Device 0 not available for mux-v2 standalone benchmark");

    auto unit_mesh_devices = tt::tt_metal::distributed::MeshDevice::create_unit_meshes({0});
    TT_FATAL(unit_mesh_devices.size() == 1, "Expected exactly one unit mesh device for device 0");

    mesh_device_ = unit_mesh_devices.begin()->second;
    TT_FATAL(mesh_device_ != nullptr, "Failed to create unit mesh device for device 0");

    device_ = mesh_device_->get_devices()[0];
    TT_FATAL(device_ != nullptr, "Unit mesh device did not expose an underlying device handle");

    worker_cores_ = enumerate_worker_cores(mesh_device_);
    TT_FATAL(
        worker_cores_.size() >= 3,
        "Standalone mux-v2 throughput benchmark requires at least mux + drainer + one sender core");
}

void FabricMuxV2BenchmarkContext::shutdown() {
    if (mesh_device_ != nullptr) {
        [[maybe_unused]] const bool closed = mesh_device_->close();
    }
    mesh_device_.reset();
    device_ = nullptr;
    worker_cores_.clear();
}

bool FabricMuxV2BenchmarkContext::can_support_case(
    const MuxV2ThroughputCase& benchmark_case, std::string* rejection_reason) const {
    if (mesh_device_ == nullptr || device_ == nullptr) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "benchmark context is not initialized";
        }
        return false;
    }

    constexpr std::size_t kReservedStandaloneCores = 2;  // one mux core + one drainer core
    const std::size_t required_worker_cores = kReservedStandaloneCores + benchmark_case.num_senders;
    if (worker_cores_.size() < required_worker_cores) {
        if (rejection_reason != nullptr) {
            std::ostringstream message;
            message << "requires " << required_worker_cores << " worker cores but only " << worker_cores_.size()
                    << " are available on device " << device_->id();
            *rejection_reason = message.str();
        }
        return false;
    }

    return true;
}

CoreCoord FabricMuxV2BenchmarkContext::get_mux_logical_core() const {
    TT_FATAL(worker_cores_.size() >= 1, "No worker cores are available for the standalone mux benchmark");
    return worker_cores_.at(0);
}

CoreCoord FabricMuxV2BenchmarkContext::get_drainer_logical_core() const {
    TT_FATAL(worker_cores_.size() >= 2, "No drainer core is available for the standalone mux benchmark");
    return worker_cores_.at(1);
}

std::vector<CoreCoord> FabricMuxV2BenchmarkContext::get_sender_logical_cores(
    const MuxV2ThroughputCase& benchmark_case) const {
    TT_FATAL(
        worker_cores_.size() >= static_cast<std::size_t>(benchmark_case.num_senders + 2),
        "Not enough worker cores to satisfy sender-core request");

    return std::vector<CoreCoord>(
        worker_cores_.begin() + 2, worker_cores_.begin() + 2 + static_cast<std::ptrdiff_t>(benchmark_case.num_senders));
}

void BM_StandaloneMuxV2Throughput(
    benchmark::State& state, FabricMuxV2BenchmarkContext* context, const MuxV2ThroughputCase& benchmark_case) {
    const auto resolved_payload_size_bytes = resolve_packet_payload_size_bytes(benchmark_case);
    const auto num_packets = derive_num_packets(benchmark_case);

    state.counters["senders"] = benchmark::Counter(static_cast<double>(benchmark_case.num_senders));
    state.counters["payload_bytes"] = benchmark::Counter(static_cast<double>(resolved_payload_size_bytes));
    state.counters["num_packets"] = benchmark::Counter(static_cast<double>(num_packets));
    state.counters["buffers_per_channel"] =
        benchmark::Counter(static_cast<double>(benchmark_case.num_buffers_per_channel));
    state.counters["service_burst_size"] = benchmark::Counter(static_cast<double>(benchmark_case.service_burst_size));
    state.counters["max_in_flight_trids"] = benchmark::Counter(static_cast<double>(benchmark_case.max_in_flight_trids));
    state.counters["target_payload_bytes"] =
        benchmark::Counter(static_cast<double>(benchmark_case.target_aggregate_payload_bytes));
    state.counters["clock_mhz"] = benchmark::Counter(static_cast<double>(get_tt_npu_clock(context->get_device())));

    std::string rejection_reason;
    if (!context->can_support_case(benchmark_case, &rejection_reason)) {
        state.SkipWithError(rejection_reason.c_str());
        return;
    }

    const double device_clock_hz = static_cast<double>(get_tt_npu_clock(context->get_device())) * 1.0e6;
    for ([[maybe_unused]] auto _ : state) {
        const auto run_result = run_standalone_mux_v2_benchmark_once(*context, benchmark_case);
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

}  // namespace tt::tt_fabric::bench

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    tt::tt_fabric::bench::FabricMuxV2BenchmarkContext context;
    context.initialize();

    auto benchmark_cases = tt::tt_fabric::bench::get_standalone_mux_v2_throughput_cases();
    std::size_t registered_case_count = 0;
    for (const auto& benchmark_case : benchmark_cases) {
        std::string rejection_reason;
        if (!context.can_support_case(benchmark_case, &rejection_reason)) {
            log_info(
                tt::LogTest,
                "Skipping benchmark registration for {}: {}",
                tt::tt_fabric::bench::benchmark_name_for_case(benchmark_case),
                rejection_reason);
            continue;
        }

        const auto benchmark_name = tt::tt_fabric::bench::benchmark_name_for_case(benchmark_case);
        benchmark::RegisterBenchmark(
            benchmark_name.c_str(),
            [&context, benchmark_case](benchmark::State& state) {
                tt::tt_fabric::bench::BM_StandaloneMuxV2Throughput(state, &context, benchmark_case);
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

    context.shutdown();
    return 0;
}
