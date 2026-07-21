// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_configs.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshDevice;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ::tt::tt_metal::distributed::MeshWorkload;
using ttnn::Tensor;

constexpr uint32_t kMinWarmupIters = 4;
constexpr uint32_t kWarmupSettlingIters = 2;
constexpr uint32_t kPerfIters = 300;
constexpr uint32_t kLatencyIters = 50;

constexpr uint32_t kElemBytes = sizeof(uint32_t);
constexpr uint32_t kGenericTensorPageBytes = 4 * 1024;
constexpr uint32_t kLargeTunePerDeviceBytes = 2 * 1024 * 1024;
constexpr uint32_t kAutoFifoSocketPages = 8;  // Mirrors H2DStreamService's current auto-FIFO policy.

// One fixed, representative worker grid for the whole sweep (4x4 = 16 drain cores).
// Worker-grid size barely moved throughput, so it is held constant rather than swept.
const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{3, 3}};

std::shared_ptr<MeshDevice> g_mesh_device;

enum class PlacementPattern {
    FullShard2D,
};

struct BenchmarkCase {
    std::string label;   // "<payload_regime>/<mode>/<host...|threads...>/bytes<...>/max_coalesce<...>/fifo<...>"
    std::string regime;  // small_payload / medium_payload / large_payload
    std::string mode;    // size / tune
    PlacementPattern placement = PlacementPattern::FullShard2D;
    uint32_t per_device_pages = 1;
    uint32_t tensor_page_bytes = 0;
    uint32_t max_socket_page_size_bytes = 0;  // 0 = service default.
    uint32_t fifo_size_bytes = 0;             // 0 = service auto.
    uint32_t metadata_size_bytes = 0;
    bool measure_latency = false;
    bool parallel_host_push = true;
    uint32_t host_push_thread_count = 0;
};

struct WarmupPlan {
    uint32_t warmup_iters;
    uint64_t host_fifo_depth_transfers;
    uint64_t device_cb_depth_transfers;
    uint64_t pipeline_depth_transfers;
};

struct LatencyStats {
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p90_us = 0.0;
    double max_us = 0.0;
};

template <typename T>
std::string stream_string(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

uint32_t worker_count(const CoreRange& worker_cores) {
    return (worker_cores.end_coord.x - worker_cores.start_coord.x + 1) *
           (worker_cores.end_coord.y - worker_cores.start_coord.y + 1);
}

uint32_t effective_host_push_thread_count(
    bool parallel_host_push, uint32_t host_push_thread_count, size_t num_sockets) {
    if (!parallel_host_push || host_push_thread_count == 1 || num_sockets <= 1) {
        return 1;
    }
    if (host_push_thread_count == 0) {
        return static_cast<uint32_t>(
            std::min<size_t>(tt::tt_metal::H2DStreamService::kAutoHostPushThreadCount, num_sockets));
    }
    return static_cast<uint32_t>(std::min<size_t>(host_push_thread_count, num_sockets));
}

ttsl::SmallVector<MeshMapperConfig::Placement> full_shard_2d_placements() {
    // Shard tensor dim 2 (height) across mesh rows, dim 3 (width) across mesh cols.
    return {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}};
}

MeshWorkload build_drain_workload(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const tt::tt_metal::H2DStreamService& service,
    const CoreRange& worker_cores,
    uint32_t total_iters) {
    const Tensor& input_tensor = service.get_backing_tensor();
    const uint32_t num_workers = worker_count(worker_cores);
    TT_FATAL(num_workers > 0, "build_drain_workload: worker_cores must contain at least one core");

    const uint32_t data_ready_sem_addr = static_cast<uint32_t>(service.get_data_ready_sem_addr());

    const auto& coords = input_tensor.tensor_topology().mesh_coords();
    TT_FATAL(!coords.empty(), "build_drain_workload: tensor topology has no coords");

    MeshWorkload worker_workload;
    for (const auto& coord : coords) {
        auto* device = mesh_device->get_device(coord);
        const CoreCoord service_logical = service.get_service_core(coord);
        const CoreCoord service_phys = device->worker_core_from_logical_core(service_logical);
        const uint32_t consumed_counter_addr = static_cast<uint32_t>(service.get_consumed_counter_addr(coord));

        auto program = tt::tt_metal::CreateProgram();

        auto kernel_handle = tt::tt_metal::CreateKernel(
            program,
            "tests/ttnn/benchmark/cpp/kernels/persistent_h2d_drain_benchmark.cpp",
            worker_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {data_ready_sem_addr},
            });

        for (uint32_t y = worker_cores.start_coord.y; y <= worker_cores.end_coord.y; ++y) {
            for (uint32_t x = worker_cores.start_coord.x; x <= worker_cores.end_coord.x; ++x) {
                const CoreCoord core{x, y};
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    kernel_handle,
                    core,
                    {
                        consumed_counter_addr,
                        static_cast<uint32_t>(service_phys.x),
                        static_cast<uint32_t>(service_phys.y),
                        total_iters,
                    });
            }
        }

        worker_workload.add_program(MeshCoordinateRange(coord, coord), std::move(program));
    }

    return worker_workload;
}

uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
    TT_FATAL(denominator > 0, "ceil_div denominator must be > 0");
    return (numerator / denominator) + (numerator % denominator != 0 ? 1 : 0);
}

WarmupPlan compute_warmup_plan(
    uint32_t fifo_size_bytes, uint64_t per_shard_payload_bytes, uint32_t slot_count, uint32_t num_socket_pages) {
    TT_FATAL(per_shard_payload_bytes > 0, "per_shard_payload_bytes must be > 0");
    TT_FATAL(slot_count > 0, "slot_count must be > 0");
    TT_FATAL(num_socket_pages > 0, "num_socket_pages must be > 0");

    const uint64_t host_fifo_depth_transfers = ceil_div(fifo_size_bytes, per_shard_payload_bytes);
    const uint64_t device_cb_depth_transfers = ceil_div(slot_count, num_socket_pages);
    const uint64_t pipeline_depth_transfers = host_fifo_depth_transfers + device_cb_depth_transfers;
    const uint64_t warmup_iters = std::max<uint64_t>(kMinWarmupIters, pipeline_depth_transfers + kWarmupSettlingIters);
    TT_FATAL(
        warmup_iters <= std::numeric_limits<uint32_t>::max(), "warmup_iters ({}) exceeds uint32_t range", warmup_iters);

    return WarmupPlan{
        .warmup_iters = static_cast<uint32_t>(warmup_iters),
        .host_fifo_depth_transfers = host_fifo_depth_transfers,
        .device_cb_depth_transfers = device_cb_depth_transfers,
        .pipeline_depth_transfers = pipeline_depth_transfers,
    };
}

LatencyStats summarize_latency_us(std::vector<double> latencies_us) {
    TT_FATAL(!latencies_us.empty(), "latencies_us must not be empty");
    std::sort(latencies_us.begin(), latencies_us.end());
    const auto percentile = [&](double fraction) {
        const auto idx = static_cast<std::size_t>(std::lround(static_cast<double>(latencies_us.size() - 1) * fraction));
        return latencies_us[idx];
    };
    const double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    return LatencyStats{
        .avg_us = sum / static_cast<double>(latencies_us.size()),
        .p50_us = percentile(0.50),
        .p90_us = percentile(0.90),
        .max_us = latencies_us.back(),
    };
}

bool benchmark_supported(benchmark::State& state) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy()) {
        state.SkipWithMessage("H2DStreamService kernels are only available on UBB Galaxy systems");
        return false;
    }
    if (!tt::tt_metal::experimental::GetMemoryPinningParameters(*g_mesh_device).can_map_to_noc) {
        state.SkipWithMessage("Mapping host memory to NOC is not supported on this system");
        return false;
    }
    return true;
}

uint32_t per_device_payload_bytes(const BenchmarkCase& cs) { return cs.per_device_pages * cs.tensor_page_bytes; }

std::string host_push_mode_name(bool parallel_host_push) { return parallel_host_push ? "parallel" : "serial"; }

std::string host_push_label(bool parallel_host_push, uint32_t host_push_thread_count) {
    if (host_push_thread_count > 0) {
        return "threads" + std::to_string(host_push_thread_count);
    }
    return "host" + host_push_mode_name(parallel_host_push);
}

uint32_t tensor_page_elems(const BenchmarkCase& cs) {
    TT_FATAL(cs.tensor_page_bytes > 0, "tensor_page_bytes must be > 0");
    TT_FATAL(
        cs.tensor_page_bytes % kElemBytes == 0,
        "tensor_page_bytes ({}) must be divisible by element size ({})",
        cs.tensor_page_bytes,
        kElemBytes);
    return cs.tensor_page_bytes / kElemBytes;
}

ttsl::SmallVector<MeshMapperConfig::Placement> placements_for(const BenchmarkCase& cs) {
    switch (cs.placement) {
        case PlacementPattern::FullShard2D: return full_shard_2d_placements();
    }
    TT_FATAL(false, "Unhandled placement pattern");
    return {};
}

ttnn::Shape global_shape_for(const BenchmarkCase& cs, uint32_t mesh_rows, uint32_t mesh_cols) {
    const uint32_t elems_per_page = tensor_page_elems(cs);
    switch (cs.placement) {
        case PlacementPattern::FullShard2D:
            return ttnn::Shape({1, 1, cs.per_device_pages * mesh_rows, elems_per_page * mesh_cols});
    }
    TT_FATAL(false, "Unhandled placement pattern");
    return ttnn::Shape({});
}

void run_h2d_stream_service_benchmark(benchmark::State& state, const BenchmarkCase& cs) {
    // The drain kernel is sized to exactly one warmup+perf pass, so the benchmark body must run a
    // single iteration; more would push past the kernel's bounded loop and deadlock the service's
    // worker-sync wait. This guards against benchmark misconfiguration (it is set via Iterations(1)).
    TT_FATAL(
        state.max_iterations == 1,
        "benchmark_h2d_stream_service must run exactly one iteration per case; got max_iterations={}",
        state.max_iterations);
    if (!benchmark_supported(state)) {
        return;
    }
    const std::string& case_name = cs.label;

    // The benchmark covers both FullShard2D and SP-shard/TP-replicate cases, so it needs a 2D mesh.
    const auto& mesh_shape = g_mesh_device->shape();
    if (mesh_shape.dims() != 2 || mesh_shape[0] < 2 || mesh_shape[1] < 2) {
        state.SkipWithMessage("H2DStreamService benchmark requires a 2D mesh with >= 2 devices on each axis");
        return;
    }
    const uint32_t mesh_rows = mesh_shape[0];
    const uint32_t mesh_cols = mesh_shape[1];

    const ttnn::Shape global_shape = global_shape_for(cs, mesh_rows, mesh_cols);
    const auto placements = placements_for(cs);
    const uint32_t max_coalesce_pages =
        cs.max_socket_page_size_bytes == 0 ? 0 : cs.max_socket_page_size_bytes / cs.tensor_page_bytes;
    const uint32_t fifo_pages = cs.fifo_size_bytes == 0 ? 0 : cs.fifo_size_bytes / cs.tensor_page_bytes;
    const uint32_t num_workers = worker_count(kWorkerCores);

    log_info(
        tt::LogTest,
        "[{}] Starting: global_shape={}, per_device_bytes={}, tensor_page_bytes={}, fifo_size_bytes={}, "
        "max_socket_page_size_bytes={}, metadata_size_bytes={}, host_push={}, workers={}, perf_iters={}",
        case_name,
        stream_string(global_shape),
        per_device_payload_bytes(cs),
        cs.tensor_page_bytes,
        cs.fifo_size_bytes,
        cs.max_socket_page_size_bytes,
        cs.metadata_size_bytes,
        host_push_mode_name(cs.parallel_host_push),
        num_workers,
        kPerfIters);

    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(global_shape, tensor_layout);

    tt::tt_metal::H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = ttnn::distributed::create_mesh_mapper(*g_mesh_device, MeshMapperConfig{.placements = placements}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = cs.fifo_size_bytes,
        .max_socket_page_size_bytes = cs.max_socket_page_size_bytes,
        .worker_cores = kWorkerCores,
        .metadata_size_bytes = cs.metadata_size_bytes,
        .parallel_host_push = cs.parallel_host_push,
        .host_push_thread_count = cs.host_push_thread_count,
    };

    tt::tt_metal::H2DStreamService service(g_mesh_device, std::move(cfg));
    const auto sockets = service.get_sockets();
    TT_FATAL(!sockets.empty(), "benchmark requires at least one socket");

    const Tensor& backing = service.get_backing_tensor();
    auto* backing_buf = backing.buffer();
    TT_FATAL(backing_buf != nullptr, "benchmark backing tensor has no buffer");

    const auto& coords = backing.tensor_topology().mesh_coords();
    const uint64_t per_shard_payload_bytes =
        static_cast<uint64_t>(backing_buf->page_size()) * static_cast<uint64_t>(backing_buf->num_pages());
    const uint32_t socket_page_size = sockets.front()->get_page_size();
    TT_FATAL(socket_page_size > 0, "socket page size must be initialized");
    TT_FATAL(per_shard_payload_bytes % socket_page_size == 0, "socket page size must divide per-shard payload bytes");
    const uint32_t num_socket_pages = static_cast<uint32_t>(per_shard_payload_bytes / socket_page_size);
    const uint32_t pages_per_chunk = socket_page_size / backing_buf->page_size();
    const uint32_t effective_host_push_threads =
        effective_host_push_thread_count(cs.parallel_host_push, cs.host_push_thread_count, sockets.size());
    // The service auto-sizes slot depth from service-core L1 (no longer max_socket_page_size_bytes /
    // socket_page_size), so read the actual derived value rather than recomputing it.
    const uint32_t slot_count = service.get_slot_count();
    const uint32_t effective_fifo_size_bytes =
        cs.fifo_size_bytes > 0 ? cs.fifo_size_bytes : kAutoFifoSocketPages * socket_page_size;
    const uint64_t fifo_socket_pages = effective_fifo_size_bytes / socket_page_size;
    const double fifo_transfer_depth = static_cast<double>(fifo_socket_pages) / static_cast<double>(num_socket_pages);
    const WarmupPlan warmup_plan =
        compute_warmup_plan(effective_fifo_size_bytes, per_shard_payload_bytes, slot_count, num_socket_pages);
    const uint32_t warmup_iters = warmup_plan.warmup_iters;
    const bool measure_latency = cs.measure_latency;
    const uint32_t latency_iters = measure_latency ? kLatencyIters : 0;
    log_info(
        tt::LogTest,
        "[{}] Geometry: sockets={}, per_shard_payload_bytes={}, socket_page_size={}, num_socket_pages={}, "
        "pages_per_chunk={}, slot_count={}, fifo_socket_pages={}, fifo_transfer_depth={:.3f}, "
        "host_fifo_depth_transfers={}, device_cb_depth_transfers={}, warmup_iters={}",
        case_name,
        sockets.size(),
        per_shard_payload_bytes,
        socket_page_size,
        num_socket_pages,
        pages_per_chunk,
        slot_count,
        fifo_socket_pages,
        fifo_transfer_depth,
        warmup_plan.host_fifo_depth_transfers,
        warmup_plan.device_cb_depth_transfers,
        warmup_iters);

    // The drain kernel runs a bounded loop of exactly this many iterations, so the host must push
    // exactly total_iters transfers (warmup + perf + optional latency) below; a mismatch deadlocks the service's
    // worker-sync wait. This is why the benchmark must run a single state iteration (guard above).
    const uint32_t total_iters = warmup_iters + kPerfIters + latency_iters;
    auto drain_workload = build_drain_workload(g_mesh_device, service, kWorkerCores, total_iters);
    log_info(
        tt::LogTest,
        "[{}] Enqueuing bounded persistent drain workload for {} total iterations across {} coords",
        case_name,
        total_iters,
        coords.size());
    tt::tt_metal::distributed::EnqueueMeshWorkload(
        g_mesh_device->mesh_command_queue(), drain_workload, /*blocking=*/false);

    // Distributed-tensor input path. Build the distributed host tensor once and reuse it every push;
    // the values are irrelevant since the benchmark does not verify contents.
    std::vector<uint32_t> source_data(global_shape.volume());
    std::iota(source_data.begin(), source_data.end(), 0u);
    auto host_mapper =
        ttnn::distributed::create_mesh_mapper(*g_mesh_device, MeshMapperConfig{.placements = placements});
    const Tensor tensor_input =
        ttnn::distributed::distribute_tensor(Tensor::from_vector<uint32_t>(source_data, global_spec), *host_mapper);
    std::vector<std::byte> metadata(cs.metadata_size_bytes);

    auto push_once = [&]() {
        service.forward_to_tensor(tensor_input, ttsl::Span<const std::byte>(metadata.data(), metadata.size()));
    };

    for ([[maybe_unused]] auto _ : state) {
        log_info(tt::LogTest, "[{}] Starting warmup phase with {} iterations", case_name, warmup_iters);
        for (uint32_t iter = 0; iter < warmup_iters; ++iter) {
            push_once();
        }
        log_info(tt::LogTest, "[{}] Warmup phase complete", case_name);

        // Time only the steady-state feeder loop. After the reader/writer split, socket ACKs mean
        // "reader staged the page into L1", not "writer committed the transfer to DRAM" and not
        // "workers drained it". The primary metric therefore remains aggregate bytes accepted by the
        // service under its real backpressure. The barrier tail below shows how much DRAM-completion
        // cleanup remained after the timed push window before the optional serialized latency loop.
        log_info(tt::LogTest, "[{}] Starting timed phase with {} iterations", case_name, kPerfIters);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t iter = 0; iter < kPerfIters; ++iter) {
            push_once();
        }
        const auto t1 = std::chrono::steady_clock::now();

        log_info(tt::LogTest, "[{}] Starting untimed service.barrier() tail", case_name);
        const auto barrier_t0 = std::chrono::steady_clock::now();
        service.barrier();
        const auto barrier_t1 = std::chrono::steady_clock::now();

        LatencyStats latency_stats;
        if (measure_latency) {
            log_info(
                tt::LogTest, "[{}] Starting serialized latency phase with {} iterations", case_name, latency_iters);
            std::vector<double> latency_us;
            latency_us.reserve(latency_iters);
            for (uint32_t iter = 0; iter < latency_iters; ++iter) {
                const auto latency_t0 = std::chrono::steady_clock::now();
                push_once();
                service.barrier();
                const auto latency_t1 = std::chrono::steady_clock::now();
                latency_us.push_back(std::chrono::duration<double, std::micro>(latency_t1 - latency_t0).count());
            }
            latency_stats = summarize_latency_us(std::move(latency_us));
            log_info(
                tt::LogTest,
                "[{}] Serialized latency phase complete: avg_us={:.3f}, p50_us={:.3f}, p90_us={:.3f}, max_us={:.3f}",
                case_name,
                latency_stats.avg_us,
                latency_stats.p50_us,
                latency_stats.p90_us,
                latency_stats.max_us);
        }

        log_info(tt::LogTest, "[{}] Starting final untimed drain Finish() tail", case_name);
        const auto finish_t0 = std::chrono::steady_clock::now();
        tt::tt_metal::distributed::Finish(g_mesh_device->mesh_command_queue());
        const auto finish_t1 = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double barrier_tail_ms = std::chrono::duration<double, std::milli>(barrier_t1 - barrier_t0).count();
        const double drain_finish_tail_ms = std::chrono::duration<double, std::milli>(finish_t1 - finish_t0).count();
        const double aggregate_bytes = static_cast<double>(sockets.size()) *
                                       static_cast<double>(per_shard_payload_bytes) * static_cast<double>(kPerfIters);
        const double aggregate_gbps = aggregate_bytes / elapsed_s / 1.0e9;
        const double global_gbps =
            static_cast<double>(service.payload_size_bytes()) * static_cast<double>(kPerfIters) / elapsed_s / 1.0e9;

        state.SetIterationTime(elapsed_s);
        state.SetLabel(case_name);
        state.counters["aggregate_gbps"] = aggregate_gbps;
        state.counters["global_payload_gbps"] = global_gbps;
        state.counters["total_aggregate_bytes"] = aggregate_bytes;
        state.counters["num_sockets"] = static_cast<double>(sockets.size());
        state.counters["warmup_iters"] = static_cast<double>(warmup_iters);
        state.counters["perf_iters"] = static_cast<double>(kPerfIters);
        state.counters["latency_iters"] = static_cast<double>(latency_iters);
        state.counters["per_device_bytes"] = static_cast<double>(per_device_payload_bytes(cs));
        state.counters["per_device_pages"] = static_cast<double>(cs.per_device_pages);
        state.counters["tensor_page_bytes"] = static_cast<double>(cs.tensor_page_bytes);
        state.counters["max_coalesce_pages"] = static_cast<double>(max_coalesce_pages);
        state.counters["fifo_pages"] = static_cast<double>(fifo_pages);
        state.counters["fifo_size_bytes"] = static_cast<double>(effective_fifo_size_bytes);
        state.counters["configured_fifo_size_bytes"] = static_cast<double>(cs.fifo_size_bytes);
        state.counters["max_socket_page_size_bytes"] = static_cast<double>(cs.max_socket_page_size_bytes);
        state.counters["metadata_size_bytes"] = static_cast<double>(cs.metadata_size_bytes);
        state.counters["worker_count"] = static_cast<double>(num_workers);
        state.counters["parallel_host_push"] = cs.parallel_host_push ? 1.0 : 0.0;
        state.counters["host_push_thread_count"] = static_cast<double>(cs.host_push_thread_count);
        state.counters["effective_host_push_thread_count"] = static_cast<double>(effective_host_push_threads);
        state.counters["per_shard_bytes"] = static_cast<double>(per_shard_payload_bytes);
        state.counters["socket_page_size"] = static_cast<double>(socket_page_size);
        state.counters["num_socket_pages"] = static_cast<double>(num_socket_pages);
        state.counters["pages_per_chunk"] = static_cast<double>(pages_per_chunk);
        state.counters["slot_count"] = static_cast<double>(slot_count);
        state.counters["fifo_socket_pages"] = static_cast<double>(fifo_socket_pages);
        state.counters["fifo_transfer_depth"] = fifo_transfer_depth;
        state.counters["host_fifo_depth_transfers"] = static_cast<double>(warmup_plan.host_fifo_depth_transfers);
        state.counters["device_cb_depth_transfers"] = static_cast<double>(warmup_plan.device_cb_depth_transfers);
        state.counters["pipeline_depth_transfers"] = static_cast<double>(warmup_plan.pipeline_depth_transfers);
        state.counters["barrier_tail_ms"] = barrier_tail_ms;
        state.counters["drain_finish_tail_ms"] = drain_finish_tail_ms;
        if (measure_latency) {
            state.counters["latency_avg_us"] = latency_stats.avg_us;
            state.counters["latency_p50_us"] = latency_stats.p50_us;
            state.counters["latency_p90_us"] = latency_stats.p90_us;
            state.counters["latency_max_us"] = latency_stats.max_us;
        }
        log_info(
            tt::LogTest,
            "[{}] Timed phase complete: elapsed_s={:.6f}, aggregate_gbps={:.6f}, "
            "global_payload_gbps={:.6f}, barrier_tail_ms={:.6f}, drain_finish_tail_ms={:.6f}",
            case_name,
            elapsed_s,
            aggregate_gbps,
            global_gbps,
            barrier_tail_ms,
            drain_finish_tail_ms);
    }

    // service.barrier(), optional serialized latency, and Finish() are intentionally outside the
    // primary timed region. They are diagnostics/side measurements, not part of aggregate throughput.
    log_info(tt::LogTest, "[{}] Benchmark case complete", case_name);
}

std::vector<BenchmarkCase> make_benchmark_cases(const MeshDevice& mesh_device) {
    const auto& mesh_shape = mesh_device.shape();
    if (mesh_shape.dims() != 2 || mesh_shape[0] < 2 || mesh_shape[1] < 2) {
        log_warning(
            tt::LogTest,
            "H2DStreamService benchmark needs a 2D mesh with >= 2 devices per axis; "
            "no cases generated");
        return {};
    }

    std::vector<BenchmarkCase> cases;
    auto add_case = [&](const std::string& regime,
                        const std::string& mode,
                        PlacementPattern placement,
                        uint32_t per_device_bytes,
                        uint32_t tensor_page_bytes,
                        uint32_t max_coalesce_pages,
                        uint32_t fifo_pages,
                        uint32_t metadata_size_bytes = 0,
                        bool measure_latency = false,
                        bool parallel_host_push = true,
                        uint32_t host_push_thread_count = 0) {
        TT_FATAL(tensor_page_bytes > 0, "tensor_page_bytes must be > 0");
        TT_FATAL(
            per_device_bytes % tensor_page_bytes == 0,
            "per_device_bytes ({}) must be an integer number of tensor pages ({})",
            per_device_bytes,
            tensor_page_bytes);
        if (fifo_pages != 0 && max_coalesce_pages != 0 && fifo_pages < max_coalesce_pages) {
            return;
        }
        const uint32_t per_device_pages = per_device_bytes / tensor_page_bytes;
        const uint32_t max_socket_page_size_bytes = max_coalesce_pages * tensor_page_bytes;
        const uint32_t fifo_size_bytes = fifo_pages * tensor_page_bytes;
        cases.push_back(BenchmarkCase{
            .label = regime + "/" + mode + "/" + host_push_label(parallel_host_push, host_push_thread_count) +
                     "/bytes" + std::to_string(per_device_bytes) + "/max_coalesce" +
                     (max_coalesce_pages == 0 ? std::string("auto") : std::to_string(max_coalesce_pages)) + "/fifo" +
                     (fifo_pages == 0 ? std::string("auto") : std::to_string(fifo_pages)),
            .regime = regime,
            .mode = mode,
            .placement = placement,
            .per_device_pages = per_device_pages,
            .tensor_page_bytes = tensor_page_bytes,
            .max_socket_page_size_bytes = max_socket_page_size_bytes,
            .fifo_size_bytes = fifo_size_bytes,
            .metadata_size_bytes = metadata_size_bytes,
            .measure_latency = measure_latency,
            .parallel_host_push = parallel_host_push,
            .host_push_thread_count = host_push_thread_count,
        });
    };
    auto add_size_case = [&](const std::string& regime, uint32_t per_device_bytes) {
        for (bool parallel_host_push : {false, true}) {
            add_case(
                regime,
                "size",
                PlacementPattern::FullShard2D,
                per_device_bytes,
                kGenericTensorPageBytes,
                0,
                0,
                /*metadata_size_bytes=*/0,
                /*measure_latency=*/true,
                parallel_host_push);
        }
    };
    auto add_host_thread_sweep_cases = [&](const std::string& regime, uint32_t per_device_bytes) {
        for (uint32_t host_threads : {1u, 2u, 4u, 8u, 16u, 32u}) {
            add_case(
                regime,
                "host_threads",
                PlacementPattern::FullShard2D,
                per_device_bytes,
                kGenericTensorPageBytes,
                0,
                0,
                /*metadata_size_bytes=*/0,
                /*measure_latency=*/true,
                /*parallel_host_push=*/true,
                host_threads);
        }
    };

    // Size anchors characterize payload regimes without tying the benchmark to one use case.
    for (uint32_t bytes : {4u * 1024, 8u * 1024, 16u * 1024, 32u * 1024}) {
        add_size_case("small_payload", bytes);
        add_host_thread_sweep_cases("small_payload", bytes);
    }
    for (uint32_t bytes : {64u * 1024, 128u * 1024, 256u * 1024, 512u * 1024}) {
        add_size_case("medium_payload", bytes);
        add_host_thread_sweep_cases("medium_payload", bytes);
    }
    for (uint32_t bytes : {1u * 1024 * 1024, 2u * 1024 * 1024, 4u * 1024 * 1024, 8u * 1024 * 1024}) {
        add_size_case("large_payload", bytes);
        add_host_thread_sweep_cases("large_payload", bytes);
    }

    // Tune rows perturb socket-page cap and FIFO depth within each payload regime, and stay parallel-only
    // to keep benchmark runtime bounded.
    for (uint32_t page_pages : {1u, 2u, 4u, 0u}) {
        for (uint32_t fifo_pages : {0u, 8u, 32u}) {
            add_case(
                "small_payload",
                "tune",
                PlacementPattern::FullShard2D,
                32 * 1024,
                kGenericTensorPageBytes,
                page_pages,
                fifo_pages);
        }
    }
    for (uint32_t page_pages : {1u, 4u, 16u, 0u}) {
        for (uint32_t fifo_pages : {0u, 16u, 64u}) {
            add_case(
                "medium_payload",
                "tune",
                PlacementPattern::FullShard2D,
                256 * 1024,
                kGenericTensorPageBytes,
                page_pages,
                fifo_pages);
        }
    }
    for (uint32_t page_pages : {8u, 16u, 32u, 64u, 128u, 0u}) {
        for (uint32_t fifo_pages : {0u, 64u, 128u, 256u}) {
            add_case(
                "large_payload",
                "tune",
                PlacementPattern::FullShard2D,
                kLargeTunePerDeviceBytes,
                kGenericTensorPageBytes,
                page_pages,
                fifo_pages);
        }
    }

    return cases;
}

void register_benchmarks() {
    for (const auto& cs : make_benchmark_cases(*g_mesh_device)) {
        const std::string benchmark_name = "BM_H2DStreamService/" + cs.label;
        benchmark::RegisterBenchmark(
            benchmark_name, [cs](benchmark::State& state) { run_h2d_stream_service_benchmark(state, cs); })
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    g_mesh_device = MeshDevice::create(
        tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1,
        tt::tt_metal::DispatchCoreType::WORKER);

    ::benchmark::Initialize(&argc, argv);
    register_benchmarks();
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    if (g_mesh_device) {
        ttnn::distributed::close_mesh_device(g_mesh_device);
        g_mesh_device.reset();
    }

    return 0;
}
