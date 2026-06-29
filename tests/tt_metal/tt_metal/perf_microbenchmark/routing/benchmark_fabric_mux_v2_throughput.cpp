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
#include "fabric_mux_benchmark_program_utils.hpp"
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
    constexpr uint32_t kTuningSenderCount = 8;
    constexpr std::array<ForwarderNocConfig, 2> kForwarderNocSweep = {{
        ForwarderNocConfig{.name = "r0", .noc = tt::tt_metal::NOC::RISCV_0_default},
        ForwarderNocConfig{.name = "r1", .noc = tt::tt_metal::NOC::RISCV_1_default},
    }};
    constexpr std::array<uint8_t, 5> kBufferSweep = {1, 2, 4, 8, 16};
    constexpr std::array<uint32_t, 5> kPayloadSweep = {64, 1024, 2048, 4096, 0};
    constexpr std::array<uint32_t, 5> kSenderSweep = {1, 2, 4, 8, 16};
    constexpr std::array<uint32_t, 4> kTridRingCapacitySweep = {1, 2, 4, 8};
    constexpr std::array<uint32_t, 5> kServiceBurstSweep = {1, 2, 4, 8, 16};
    constexpr std::array<uint32_t, 6> kDrainerSlotsSweep = {1, 2, 4, 8, 16, 32};

    std::vector<MuxV2ThroughputCase> cases;
    cases.reserve(
        kForwarderNocSweep.size() *
        (kBufferSweep.size() + kPayloadSweep.size() + kSenderSweep.size() + kTridRingCapacitySweep.size() +
         kServiceBurstSweep.size() + kDrainerSlotsSweep.size()));

    for (const auto& noc_config : kForwarderNocSweep) {
        for (const auto buffer_count : kBufferSweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix =
                    "buffer_sweep_1s_max_buf" + std::to_string(buffer_count) + "_" + noc_config.name + "_sb8_trid8",
                .num_buffers_per_channel = buffer_count,
                .forwarder_noc = noc_config.noc,
            });
        }

        for (const auto payload_bytes : kPayloadSweep) {
            const auto payload_name = payload_bytes == 0 ? std::string("max") : std::to_string(payload_bytes) + "B";
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "payload_sweep_1s_" + payload_name + "_buf8_" + noc_config.name + "_sb8_trid8",
                .packet_payload_size_bytes = payload_bytes,
                .num_buffers_per_channel = kDefaultBufferCount,
                .forwarder_noc = noc_config.noc,
            });
        }

        for (const auto sender_count : kSenderSweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix =
                    "sender_sweep_" + std::to_string(sender_count) + "s_max_buf8_" + noc_config.name + "_sb8_trid8",
                .num_senders = sender_count,
                .num_buffers_per_channel = kDefaultBufferCount,
                .forwarder_noc = noc_config.noc,
            });
        }

        for (const auto trid_ring_capacity : kTridRingCapacitySweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "trid_sweep_8s_max_buf8_" + std::string(noc_config.name) + "_sb8_trid" +
                               std::to_string(trid_ring_capacity),
                .num_senders = kTuningSenderCount,
                .num_buffers_per_channel = kDefaultBufferCount,
                .forwarder_noc = noc_config.noc,
                .trid_ring_capacity = trid_ring_capacity,
            });
        }

        for (const auto service_burst_size : kServiceBurstSweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "service_sweep_8s_max_buf8_" + std::string(noc_config.name) + "_sb" +
                               std::to_string(service_burst_size) + "_trid8",
                .num_senders = kTuningSenderCount,
                .num_buffers_per_channel = kDefaultBufferCount,
                .forwarder_noc = noc_config.noc,
                .service_burst_size = service_burst_size,
            });
        }

        for (const auto num_drainer_buffers : kDrainerSlotsSweep) {
            cases.push_back(MuxV2ThroughputCase{
                .name_suffix = "drainer_sweep_8s_max_buf8_" + std::string(noc_config.name) + "_sb8_trid8_dr" +
                               std::string(2 - std::to_string(num_drainer_buffers).size(), '0') +
                               std::to_string(num_drainer_buffers),
                .num_senders = kTuningSenderCount,
                .num_buffers_per_channel = kDefaultBufferCount,
                .forwarder_noc = noc_config.noc,
                .num_drainer_buffers = num_drainer_buffers,
            });
        }
    }

    return cases;
}

}  // namespace

void FabricMuxV2BenchmarkContext::initialize() {
    shutdown();

    tt::tt_fabric::SetFabricConfig(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

    auto available_device_ids = tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids();
    TT_FATAL(available_device_ids.contains(0), "Device 0 not available for mux-v2 standalone benchmark");

    const auto system_mesh_shape = tt::tt_metal::MetalContext::instance().get_system_mesh().shape();
    mesh_device_ =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(system_mesh_shape));
    TT_FATAL(mesh_device_ != nullptr, "Failed to create full mesh device for mux-v2 standalone benchmark");

    device_ = nullptr;
    for (auto* device : mesh_device_->get_devices()) {
        if (device != nullptr && device->id() == 0) {
            device_ = device;
            break;
        }
    }
    TT_FATAL(device_ != nullptr, "Full mesh device did not expose device 0");

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
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
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
    state.counters["trid_ring_capacity"] = benchmark::Counter(static_cast<double>(benchmark_case.trid_ring_capacity));
    state.counters["drainer_buffers"] = benchmark::Counter(static_cast<double>(benchmark_case.num_drainer_buffers));
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
    benchmark::AddCustomContext("arch", tt::arch_to_str(tt::tt_metal::MetalContext::instance().get_cluster().arch()));

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
