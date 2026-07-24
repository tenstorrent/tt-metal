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
#include <sstream>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_configs.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/services/d2h_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
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
constexpr uint32_t kDefaultTargetSocketPageBytes = 128 * 1024;
constexpr uint32_t kDefaultFifoSocketPages = 8;

// One ack worker is enough to preserve the service's worker-sync handshake without adding
// producer-side page writes to the benchmark's page-granularity curve.
const CoreRange kProducerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};

std::shared_ptr<MeshDevice> g_mesh_device;

enum class PlacementPattern {
    FullShard2D,
};

struct BenchmarkCase {
    std::string label;   // "<payload_regime>/<mode>/bytes<...>/pages<...>/fifo_socket_pages<...>"
    std::string regime;  // small_payload / medium_payload / large_payload
    std::string mode;    // size / page_granularity / host_threads
    PlacementPattern placement = PlacementPattern::FullShard2D;
    uint32_t per_device_bytes = 0;
    uint32_t tensor_num_pages = 0;
    uint32_t fifo_socket_pages = kDefaultFifoSocketPages;
    uint32_t target_socket_page_bytes = 0;  // 0 = use kDefaultTargetSocketPageBytes
    bool parallel_host_read = true;
    uint32_t host_read_thread_count = 0;
};

struct ServiceGeometryConfig {
    uint32_t tensor_page_bytes = 0;
    uint32_t max_socket_page_size_bytes = 0;
    uint32_t fifo_size_bytes = 0;
    uint32_t target_socket_page_bytes = kDefaultTargetSocketPageBytes;
};

struct WarmupPlan {
    uint32_t warmup_iters = 0;
    uint64_t host_fifo_depth_transfers = 0;
    uint64_t pipeline_depth_transfers = 0;
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

uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
    TT_FATAL(denominator > 0, "ceil_div denominator must be > 0");
    return (numerator / denominator) + (numerator % denominator != 0 ? 1 : 0);
}

uint32_t core_range_size(const CoreRange& core_range) {
    return (core_range.end_coord.x - core_range.start_coord.x + 1) *
           (core_range.end_coord.y - core_range.start_coord.y + 1);
}

uint32_t effective_host_read_thread_count(
    bool parallel_host_read, uint32_t host_read_thread_count, size_t num_sockets) {
    if (!parallel_host_read || host_read_thread_count == 1 || num_sockets <= 1) {
        return 1;
    }
    if (host_read_thread_count == 0) {
        return static_cast<uint32_t>(
            std::min<size_t>(tt::tt_metal::D2HStreamService::kAutoHostReadThreadCount, num_sockets));
    }
    return static_cast<uint32_t>(std::min<size_t>(host_read_thread_count, num_sockets));
}

std::string host_read_mode_name(bool parallel_host_read) { return parallel_host_read ? "parallel" : "serial"; }

std::string host_read_label(bool parallel_host_read, uint32_t host_read_thread_count) {
    if (host_read_thread_count > 0) {
        return "threads" + std::to_string(host_read_thread_count);
    }
    return "host" + host_read_mode_name(parallel_host_read);
}

ttsl::SmallVector<MeshMapperConfig::Placement> full_shard_2d_placements() {
    // Shard tensor dim 2 (height) across mesh rows, dim 3 (width) across mesh cols.
    return {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}};
}

ttsl::SmallVector<MeshMapperConfig::Placement> placements_for(const BenchmarkCase& cs) {
    switch (cs.placement) {
        case PlacementPattern::FullShard2D: return full_shard_2d_placements();
    }
    TT_FATAL(false, "Unhandled placement pattern");
    return {};
}

uint32_t tensor_page_elems(const ServiceGeometryConfig& geometry) {
    TT_FATAL(geometry.tensor_page_bytes > 0, "tensor_page_bytes must be > 0");
    TT_FATAL(
        geometry.tensor_page_bytes % kElemBytes == 0,
        "tensor_page_bytes ({}) must be divisible by element size ({})",
        geometry.tensor_page_bytes,
        kElemBytes);
    return geometry.tensor_page_bytes / kElemBytes;
}

ttnn::Shape global_shape_for(
    const BenchmarkCase& cs, const ServiceGeometryConfig& geometry, uint32_t mesh_rows, uint32_t mesh_cols) {
    const uint32_t elems_per_page = tensor_page_elems(geometry);
    switch (cs.placement) {
        case PlacementPattern::FullShard2D:
            return ttnn::Shape({1, 1, cs.tensor_num_pages * mesh_rows, elems_per_page * mesh_cols});
    }
    TT_FATAL(false, "Unhandled placement pattern");
    return ttnn::Shape({});
}

ServiceGeometryConfig service_geometry_for(const BenchmarkCase& cs) {
    TT_FATAL(cs.per_device_bytes > 0, "per_device_bytes must be > 0");
    TT_FATAL(cs.tensor_num_pages > 0, "tensor_num_pages must be > 0");
    TT_FATAL(
        cs.per_device_bytes % cs.tensor_num_pages == 0,
        "per_device_bytes ({}) must divide evenly into tensor_num_pages ({})",
        cs.per_device_bytes,
        cs.tensor_num_pages);

    ServiceGeometryConfig geometry;
    geometry.tensor_page_bytes = cs.per_device_bytes / cs.tensor_num_pages;
    geometry.target_socket_page_bytes =
        cs.target_socket_page_bytes > 0 ? cs.target_socket_page_bytes : kDefaultTargetSocketPageBytes;
    TT_FATAL(
        geometry.tensor_page_bytes % kElemBytes == 0,
        "derived tensor_page_bytes ({}) must be divisible by element size ({})",
        geometry.tensor_page_bytes,
        kElemBytes);

    const uint32_t target_pages_per_chunk =
        std::max<uint32_t>(1, geometry.target_socket_page_bytes / geometry.tensor_page_bytes);
    const uint32_t pages_per_chunk = std::min(cs.tensor_num_pages, target_pages_per_chunk);

    // Translate the benchmark's target socket-page size into max_socket_page_size_bytes. Keep this
    // translation local so benchmark case names do not depend on the service's transient API.
    geometry.max_socket_page_size_bytes = pages_per_chunk * geometry.tensor_page_bytes;
    geometry.fifo_size_bytes = cs.fifo_socket_pages * geometry.max_socket_page_size_bytes;
    return geometry;
}

WarmupPlan compute_warmup_plan(uint32_t fifo_size_bytes, uint64_t per_shard_payload_bytes) {
    TT_FATAL(per_shard_payload_bytes > 0, "per_shard_payload_bytes must be > 0");
    const uint64_t host_fifo_depth_transfers = ceil_div(fifo_size_bytes, per_shard_payload_bytes);
    const uint64_t pipeline_depth_transfers = host_fifo_depth_transfers;
    const uint64_t warmup_iters = std::max<uint64_t>(kMinWarmupIters, pipeline_depth_transfers + kWarmupSettlingIters);
    TT_FATAL(
        warmup_iters <= std::numeric_limits<uint32_t>::max(), "warmup_iters ({}) exceeds uint32_t range", warmup_iters);
    return WarmupPlan{
        .warmup_iters = static_cast<uint32_t>(warmup_iters),
        .host_fifo_depth_transfers = host_fifo_depth_transfers,
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
        state.SkipWithMessage("D2HStreamService kernels are only available on UBB Galaxy systems");
        return false;
    }
    if (!tt::tt_metal::experimental::GetMemoryPinningParameters(*g_mesh_device).can_map_to_noc) {
        state.SkipWithMessage("Mapping host memory to NOC is not supported on this system");
        return false;
    }
    return true;
}

MeshWorkload build_producer_workload(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const tt::tt_metal::D2HStreamService& service,
    const CoreRange& producer_cores,
    uint32_t total_iters,
    uint32_t ungated_iters,
    uint32_t latency_gate_sem_addr) {
    const Tensor& backing = service.get_backing_tensor();
    auto* backing_buf = backing.buffer();
    TT_FATAL(backing_buf != nullptr, "build_producer_workload: backing tensor has no buffer");

    const uint32_t num_workers = core_range_size(producer_cores);
    TT_FATAL(num_workers > 0, "build_producer_workload: producer_cores must contain at least one core");

    const uint32_t transfer_done_sem_addr = static_cast<uint32_t>(service.get_transfer_done_sem_addr());
    const auto& coords = backing.tensor_topology().mesh_coords();
    TT_FATAL(!coords.empty(), "build_producer_workload: tensor topology has no coords");

    MeshWorkload producer_workload;
    for (const auto& coord : coords) {
        auto* device = mesh_device->get_device(coord);
        const CoreCoord service_logical = service.get_service_core(coord);
        const CoreCoord service_phys = device->worker_core_from_logical_core(service_logical);
        const uint32_t write_ack_counter_addr = static_cast<uint32_t>(service.get_write_ack_counter_addr(coord));

        auto program = tt::tt_metal::CreateProgram();

        auto producer_kernel = tt::tt_metal::CreateKernel(
            program,
            "tests/ttnn/benchmark/cpp/kernels/persistent_d2h_producer_benchmark.cpp",
            producer_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {transfer_done_sem_addr, total_iters, ungated_iters, latency_gate_sem_addr},
            });

        for (uint32_t y = producer_cores.start_coord.y; y <= producer_cores.end_coord.y; ++y) {
            for (uint32_t x = producer_cores.start_coord.x; x <= producer_cores.end_coord.x; ++x) {
                const CoreCoord core{x, y};
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    producer_kernel,
                    core,
                    {
                        static_cast<uint32_t>(service_phys.x),
                        static_cast<uint32_t>(service_phys.y),
                        write_ack_counter_addr,
                    });
            }
        }

        producer_workload.add_program(MeshCoordinateRange(coord, coord), std::move(program));
    }

    return producer_workload;
}

void run_d2h_stream_service_benchmark(benchmark::State& state, const BenchmarkCase& cs) {
    TT_FATAL(
        state.max_iterations == 1,
        "benchmark_d2h_stream_service must run exactly one iteration per case; got max_iterations={}",
        state.max_iterations);
    if (!benchmark_supported(state)) {
        return;
    }

    const auto& mesh_shape = g_mesh_device->shape();
    if (mesh_shape.dims() != 2 || mesh_shape[0] < 2 || mesh_shape[1] < 2) {
        state.SkipWithMessage("D2HStreamService benchmark requires a 2D mesh with >= 2 devices on each axis");
        return;
    }
    const uint32_t mesh_rows = mesh_shape[0];
    const uint32_t mesh_cols = mesh_shape[1];

    const ServiceGeometryConfig geometry = service_geometry_for(cs);
    const ttnn::Shape global_shape = global_shape_for(cs, geometry, mesh_rows, mesh_cols);
    const auto placements = placements_for(cs);
    const uint32_t ack_worker_count = core_range_size(kProducerCores);

    log_info(
        tt::LogTest,
        "[{}] Starting: global_shape={}, per_device_bytes={}, tensor_num_pages={}, tensor_page_bytes={}, "
        "max_socket_page_size_bytes={}, fifo_size_bytes={}, ack_worker_count={}, parallel_host_read={}, "
        "host_read_thread_count={}, perf_iters={}",
        cs.label,
        stream_string(global_shape),
        cs.per_device_bytes,
        cs.tensor_num_pages,
        geometry.tensor_page_bytes,
        geometry.max_socket_page_size_bytes,
        geometry.fifo_size_bytes,
        ack_worker_count,
        cs.parallel_host_read,
        cs.host_read_thread_count,
        kPerfIters);

    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = tt::tt_metal::TensorSpec(global_shape, tensor_layout);

    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = ttnn::distributed::create_mesh_mapper(*g_mesh_device, MeshMapperConfig{.placements = placements}),
        .fifo_size_bytes = geometry.fifo_size_bytes,
        .max_socket_page_size_bytes = geometry.max_socket_page_size_bytes,
        .worker_cores = kProducerCores,
        .parallel_host_read = cs.parallel_host_read,
        .host_read_thread_count = cs.host_read_thread_count,
    };

    tt::tt_metal::D2HStreamService service(g_mesh_device, std::move(cfg));
    auto latency_gate_sem = ttnn::global_semaphore::create_global_semaphore(
        g_mesh_device.get(), CoreRangeSet(kProducerCores), /*initial_value=*/0, BufferType::L1);
    const uint32_t latency_gate_sem_addr = static_cast<uint32_t>(latency_gate_sem.address());
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
    const uint64_t fifo_socket_pages = geometry.fifo_size_bytes / socket_page_size;
    const double fifo_transfer_depth = static_cast<double>(fifo_socket_pages) / static_cast<double>(num_socket_pages);
    const uint32_t effective_host_read_threads =
        effective_host_read_thread_count(cs.parallel_host_read, cs.host_read_thread_count, sockets.size());
    const WarmupPlan warmup_plan = compute_warmup_plan(geometry.fifo_size_bytes, per_shard_payload_bytes);
    const uint32_t warmup_iters = warmup_plan.warmup_iters;
    const uint32_t latency_iters = kLatencyIters;
    const uint32_t ungated_iters = warmup_iters + kPerfIters;
    const uint32_t total_iters = ungated_iters + latency_iters;

    log_info(
        tt::LogTest,
        "[{}] Geometry: sockets={}, per_shard_payload_bytes={}, socket_page_size={}, num_socket_pages={}, "
        "pages_per_chunk={}, fifo_socket_pages={}, fifo_transfer_depth={:.3f}, "
        "effective_host_read_thread_count={}, host_fifo_depth_transfers={}, warmup_iters={}",
        cs.label,
        sockets.size(),
        per_shard_payload_bytes,
        socket_page_size,
        num_socket_pages,
        pages_per_chunk,
        fifo_socket_pages,
        fifo_transfer_depth,
        effective_host_read_threads,
        warmup_plan.host_fifo_depth_transfers,
        warmup_iters);

    auto producer_workload = build_producer_workload(
        g_mesh_device, service, kProducerCores, total_iters, ungated_iters, latency_gate_sem_addr);
    log_info(
        tt::LogTest,
        "[{}] Enqueuing bounded persistent D2H ready/ack workload for {} iterations across {} coords",
        cs.label,
        total_iters,
        coords.size());
    tt::tt_metal::distributed::EnqueueMeshWorkload(
        g_mesh_device->mesh_command_queue(), producer_workload, /*blocking=*/false);

    std::vector<uint32_t> host_storage(global_shape.volume(), 0u);
    auto host_mapper =
        ttnn::distributed::create_mesh_mapper(*g_mesh_device, MeshMapperConfig{.placements = placements});
    Tensor host_output =
        ttnn::distributed::distribute_tensor(Tensor::from_vector<uint32_t>(host_storage, global_spec), *host_mapper);

    auto read_once = [&]() { service.read_from_tensor(host_output); };
    std::vector<uint32_t> latency_gate_word{1};
    auto release_latency_gate = [&]() {
        for (const auto& coord : coords) {
            auto* device = g_mesh_device->get_device(coord);
            for (uint32_t y = kProducerCores.start_coord.y; y <= kProducerCores.end_coord.y; ++y) {
                for (uint32_t x = kProducerCores.start_coord.x; x <= kProducerCores.end_coord.x; ++x) {
                    tt::tt_metal::detail::WriteToDeviceL1(
                        device, CoreCoord{x, y}, latency_gate_sem_addr, latency_gate_word);
                }
            }
        }
    };

    for ([[maybe_unused]] auto _ : state) {
        log_info(tt::LogTest, "[{}] Starting warmup phase with {} iterations", cs.label, warmup_iters);
        for (uint32_t iter = 0; iter < warmup_iters; ++iter) {
            read_once();
        }
        log_info(tt::LogTest, "[{}] Warmup phase complete", cs.label);

        log_info(tt::LogTest, "[{}] Starting timed phase with {} iterations", cs.label, kPerfIters);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t iter = 0; iter < kPerfIters; ++iter) {
            read_once();
        }
        const auto t1 = std::chrono::steady_clock::now();

        log_info(tt::LogTest, "[{}] Starting untimed service.barrier() tail", cs.label);
        const auto barrier_t0 = std::chrono::steady_clock::now();
        service.barrier();
        const auto barrier_t1 = std::chrono::steady_clock::now();

        LatencyStats latency_stats;
        if (latency_iters > 0) {
            log_info(tt::LogTest, "[{}] Starting serialized latency phase with {} iterations", cs.label, latency_iters);
            std::vector<double> latency_us;
            latency_us.reserve(latency_iters);
            for (uint32_t iter = 0; iter < latency_iters; ++iter) {
                // Host-side gate write is deliberately outside the timed window. The latency metric starts
                // immediately after release and measures serialized D2H release-to-read completion.
                release_latency_gate();
                const auto latency_t0 = std::chrono::steady_clock::now();
                read_once();
                service.barrier();
                const auto latency_t1 = std::chrono::steady_clock::now();
                latency_us.push_back(std::chrono::duration<double, std::micro>(latency_t1 - latency_t0).count());
            }
            latency_stats = summarize_latency_us(std::move(latency_us));
            log_info(
                tt::LogTest,
                "[{}] Serialized latency phase complete: avg_us={:.3f}, p50_us={:.3f}, p90_us={:.3f}, "
                "max_us={:.3f}",
                cs.label,
                latency_stats.avg_us,
                latency_stats.p50_us,
                latency_stats.p90_us,
                latency_stats.max_us);
        }

        log_info(tt::LogTest, "[{}] Starting final untimed producer Finish() tail", cs.label);
        const auto finish_t0 = std::chrono::steady_clock::now();
        tt::tt_metal::distributed::Finish(g_mesh_device->mesh_command_queue());
        const auto finish_t1 = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double barrier_tail_ms = std::chrono::duration<double, std::milli>(barrier_t1 - barrier_t0).count();
        const double producer_finish_tail_ms = std::chrono::duration<double, std::milli>(finish_t1 - finish_t0).count();
        const double aggregate_bytes = static_cast<double>(sockets.size()) *
                                       static_cast<double>(per_shard_payload_bytes) * static_cast<double>(kPerfIters);
        const double aggregate_gbps = aggregate_bytes / elapsed_s / 1.0e9;
        const double global_gbps =
            static_cast<double>(service.payload_size_bytes()) * static_cast<double>(kPerfIters) / elapsed_s / 1.0e9;

        state.SetIterationTime(elapsed_s);
        state.SetLabel(cs.label);
        state.counters["aggregate_gbps"] = aggregate_gbps;
        state.counters["global_payload_gbps"] = global_gbps;
        state.counters["total_aggregate_bytes"] = aggregate_bytes;
        state.counters["num_sockets"] = static_cast<double>(sockets.size());
        state.counters["warmup_iters"] = static_cast<double>(warmup_iters);
        state.counters["perf_iters"] = static_cast<double>(kPerfIters);
        state.counters["latency_iters"] = static_cast<double>(latency_iters);
        state.counters["per_device_bytes"] = static_cast<double>(cs.per_device_bytes);
        state.counters["tensor_num_pages"] = static_cast<double>(cs.tensor_num_pages);
        state.counters["tensor_page_bytes"] = static_cast<double>(geometry.tensor_page_bytes);
        state.counters["target_socket_page_bytes"] = static_cast<double>(geometry.target_socket_page_bytes);
        state.counters["max_socket_page_size_bytes"] = static_cast<double>(geometry.max_socket_page_size_bytes);
        state.counters["fifo_socket_pages_configured"] = static_cast<double>(cs.fifo_socket_pages);
        state.counters["fifo_size_bytes"] = static_cast<double>(geometry.fifo_size_bytes);
        state.counters["ack_worker_count"] = static_cast<double>(ack_worker_count);
        state.counters["parallel_host_read"] = cs.parallel_host_read ? 1.0 : 0.0;
        state.counters["host_read_thread_count"] = static_cast<double>(cs.host_read_thread_count);
        state.counters["effective_host_read_thread_count"] = static_cast<double>(effective_host_read_threads);
        state.counters["per_shard_bytes"] = static_cast<double>(per_shard_payload_bytes);
        state.counters["socket_page_size"] = static_cast<double>(socket_page_size);
        state.counters["num_socket_pages"] = static_cast<double>(num_socket_pages);
        state.counters["pages_per_chunk"] = static_cast<double>(pages_per_chunk);
        state.counters["slot_count"] = static_cast<double>(service.get_slot_count());
        state.counters["fifo_socket_pages"] = static_cast<double>(fifo_socket_pages);
        state.counters["fifo_transfer_depth"] = fifo_transfer_depth;
        state.counters["host_fifo_depth_transfers"] = static_cast<double>(warmup_plan.host_fifo_depth_transfers);
        state.counters["pipeline_depth_transfers"] = static_cast<double>(warmup_plan.pipeline_depth_transfers);
        state.counters["device_cb_depth_transfers"] =
            static_cast<double>(service.get_slot_count()) / static_cast<double>(num_socket_pages);
        state.counters["barrier_tail_ms"] = barrier_tail_ms;
        state.counters["producer_finish_tail_ms"] = producer_finish_tail_ms;
        if (latency_iters > 0) {
            state.counters["latency_avg_us"] = latency_stats.avg_us;
            state.counters["latency_p50_us"] = latency_stats.p50_us;
            state.counters["latency_p90_us"] = latency_stats.p90_us;
            state.counters["latency_max_us"] = latency_stats.max_us;
        }

        log_info(
            tt::LogTest,
            "[{}] Timed phase complete: elapsed_s={:.6f}, aggregate_gbps={:.6f}, "
            "global_payload_gbps={:.6f}, barrier_tail_ms={:.6f}, producer_finish_tail_ms={:.6f}",
            cs.label,
            elapsed_s,
            aggregate_gbps,
            global_gbps,
            barrier_tail_ms,
            producer_finish_tail_ms);
    }

    log_info(tt::LogTest, "[{}] Benchmark case complete", cs.label);
}

std::vector<BenchmarkCase> make_benchmark_cases(const MeshDevice& mesh_device) {
    const auto& mesh_shape = mesh_device.shape();
    if (mesh_shape.dims() != 2 || mesh_shape[0] < 2 || mesh_shape[1] < 2) {
        log_warning(
            tt::LogTest, "D2HStreamService benchmark needs a 2D mesh with >= 2 devices per axis; no cases generated");
        return {};
    }

    std::vector<BenchmarkCase> cases;
    auto add_case = [&](const std::string& regime,
                        const std::string& mode,
                        uint32_t per_device_bytes,
                        uint32_t tensor_num_pages,
                        uint32_t fifo_socket_pages = kDefaultFifoSocketPages,
                        bool parallel_host_read = true,
                        uint32_t host_read_thread_count = 0,
                        uint32_t target_socket_page_bytes = 0) {
        TT_FATAL(per_device_bytes > 0, "per_device_bytes must be > 0");
        TT_FATAL(tensor_num_pages > 0, "tensor_num_pages must be > 0");
        if (per_device_bytes % tensor_num_pages != 0) {
            return;
        }
        const uint32_t tensor_page_bytes = per_device_bytes / tensor_num_pages;
        if (tensor_page_bytes == 0 || tensor_page_bytes % kElemBytes != 0) {
            return;
        }
        // When an explicit socket page target is set, verify it divides evenly into the payload.
        if (target_socket_page_bytes > 0 && target_socket_page_bytes % tensor_page_bytes != 0) {
            return;
        }
        std::string label = regime + "/" + mode + "/" + host_read_label(parallel_host_read, host_read_thread_count) +
                            "/bytes" + std::to_string(per_device_bytes) + "/pages" + std::to_string(tensor_num_pages);
        if (target_socket_page_bytes > 0) {
            label += "/socket_page" + std::to_string(target_socket_page_bytes);
        }
        label += "/fifo_socket_pages" + std::to_string(fifo_socket_pages);
        cases.push_back(BenchmarkCase{
            .label = label,
            .regime = regime,
            .mode = mode,
            .placement = PlacementPattern::FullShard2D,
            .per_device_bytes = per_device_bytes,
            .tensor_num_pages = tensor_num_pages,
            .fifo_socket_pages = fifo_socket_pages,
            .target_socket_page_bytes = target_socket_page_bytes,
            .parallel_host_read = parallel_host_read,
            .host_read_thread_count = host_read_thread_count,
        });
    };

    auto add_size_case = [&](const std::string& regime, uint32_t per_device_bytes) {
        const uint32_t tensor_num_pages = std::max<uint32_t>(1, per_device_bytes / (4 * 1024));
        for (bool parallel_host_read : {false, true}) {
            add_case(regime, "size", per_device_bytes, tensor_num_pages, kDefaultFifoSocketPages, parallel_host_read);
        }
    };

    auto add_host_thread_sweep_cases = [&](const std::string& regime, uint32_t per_device_bytes) {
        const uint32_t tensor_num_pages = std::max<uint32_t>(1, per_device_bytes / (4 * 1024));
        for (uint32_t host_threads : {1u, 2u, 4u, 8u, 16u, 32u}) {
            add_case(
                regime,
                "host_threads",
                per_device_bytes,
                tensor_num_pages,
                kDefaultFifoSocketPages,
                /*parallel_host_read=*/true,
                host_threads);
        }
    };

    auto add_page_granularity_cases = [&](const std::string& regime,
                                          const std::vector<uint32_t>& per_device_bytes_values,
                                          const std::vector<uint32_t>& page_counts) {
        for (uint32_t per_device_bytes : per_device_bytes_values) {
            for (uint32_t tensor_num_pages : page_counts) {
                add_case(regime, "page_granularity", per_device_bytes, tensor_num_pages);
            }
        }
    };

    const std::vector<uint32_t> small_payload_bytes = {4u * 1024, 8u * 1024, 16u * 1024, 32u * 1024};
    const std::vector<uint32_t> medium_payload_bytes = {64u * 1024, 128u * 1024, 256u * 1024, 512u * 1024};
    const std::vector<uint32_t> large_payload_bytes = {
        1u * 1024 * 1024, 2u * 1024 * 1024, 4u * 1024 * 1024, 8u * 1024 * 1024};

    for (uint32_t bytes : small_payload_bytes) {
        add_size_case("small_payload", bytes);
        add_host_thread_sweep_cases("small_payload", bytes);
    }
    for (uint32_t bytes : medium_payload_bytes) {
        add_size_case("medium_payload", bytes);
        add_host_thread_sweep_cases("medium_payload", bytes);
    }
    for (uint32_t bytes : large_payload_bytes) {
        add_size_case("large_payload", bytes);
        add_host_thread_sweep_cases("large_payload", bytes);
    }

    add_page_granularity_cases("small_payload", small_payload_bytes, {1u, 2u, 4u, 8u, 16u});
    add_page_granularity_cases("medium_payload", medium_payload_bytes, {32u, 64u, 128u, 256u});
    add_page_granularity_cases("large_payload", large_payload_bytes, {512u, 1024u, 2048u, 4096u});

    return cases;
}

void register_benchmarks() {
    for (const auto& cs : make_benchmark_cases(*g_mesh_device)) {
        const std::string benchmark_name = "BM_D2HStreamService/" + cs.label;
        benchmark::RegisterBenchmark(
            benchmark_name, [cs](benchmark::State& state) { run_d2h_stream_service_benchmark(state, cs); })
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
