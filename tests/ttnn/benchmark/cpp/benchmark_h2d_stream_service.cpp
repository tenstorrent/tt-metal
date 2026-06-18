// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_configs.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/socket_services.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::DeviceAddr;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::Tensor;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::H2DMode;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshDevice;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ::tt::tt_metal::distributed::MeshWorkload;

constexpr uint32_t kMinWarmupIters = 4;
constexpr uint32_t kWarmupSettlingIters = 2;
constexpr uint32_t kPerfIters = 100;

// Tensor element size. The H2D service streams raw bytes, so the transfer is
// dtype-agnostic: a production per-shard page of 1792 bf16 elems is 3584 B, which
// equals 896 UINT32 elems. We use UINT32 with per_row=896 so each page is
// byte-for-byte the production [.,1792] bf16 page (identical page size, chunk plan,
// and throughput). Switch to bf16/1792 only if you want the literal shape labels.
constexpr uint32_t kElemBytes = sizeof(uint32_t);
constexpr uint32_t kProdPerRow = 896;          // tensor page = 896 * 4 = 3584 B
constexpr uint32_t kProdPerDevicePages = 640;  // per-device page count at production size

// One fixed, representative worker grid for the whole sweep (4x4 = 16 drain cores).
// Worker-grid size barely moved throughput, so it is held constant rather than swept.
const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{3, 3}};

// Anchor chunk plan: the baseline (cb, fifo) that each chart's extra lines perturb around, and the
// held value for whichever chunk-plan axis a family is not sweeping. Kept small so it never
// constrains the other sweeps (a large cb raises the FIFO floor via fifo >= cb and drops small sizes).
constexpr uint32_t kDefaultCbPages = 4;
constexpr uint32_t kDefaultFifoPages = 64;

std::shared_ptr<MeshDevice> g_mesh_device;

// One benchmark case. Pattern is always FullShard2D, input is always the
// distributed-tensor path, and the worker grid is fixed (kWorkerCores); the only
// dimensions that vary are the per-device page count and the chunk plan (cb/fifo).
struct BenchmarkCase {
    std::string label;          // "<family>/p<pages>/cb<cb>/fifo<fifo>"
    uint32_t per_device_pages;  // tensor pages per device (the "size" axis)
    uint32_t cb_pages;          // scratch-CB size in tensor pages (read granularity)
    uint32_t fifo_pages;        // socket FIFO size in tensor pages (buffering depth)
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
    auto* input_buf = input_tensor.buffer();
    TT_FATAL(input_buf != nullptr, "build_drain_workload: input tensor has no buffer");

    const uint32_t page_size = input_buf->page_size();
    const uint32_t num_pages = input_buf->num_pages();
    const uint32_t num_workers = worker_count(worker_cores);
    TT_FATAL(num_workers > 0, "build_drain_workload: worker_cores must contain at least one core");
    TT_FATAL(
        num_pages % num_workers == 0,
        "build_drain_workload: tensor page count ({}) must be divisible by num_workers ({})",
        num_pages,
        num_workers);
    const uint32_t pages_per_worker = num_pages / num_workers;

    const uint32_t data_ready_sem_addr = static_cast<uint32_t>(service.get_data_ready_sem_addr());
    const uint32_t input_tensor_addr = static_cast<uint32_t>(input_buf->address());

    const auto& coords = input_tensor.tensor_topology().mesh_coords();
    TT_FATAL(!coords.empty(), "build_drain_workload: tensor topology has no coords");
    const tt::tt_metal::Buffer* sample_dbuf = input_tensor.mesh_buffer().get_device_buffer(coords.front());
    TT_FATAL(sample_dbuf != nullptr, "build_drain_workload: missing device buffer for sample coord");
    auto accessor_args = tt::tt_metal::TensorAccessorArgs(*sample_dbuf);
    auto accessor_compile_args = accessor_args.get_compile_time_args();

    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;

    MeshWorkload worker_workload;
    for (const auto& coord : coords) {
        auto* device = mesh_device->get_device(coord);
        const CoreCoord service_logical = service.get_service_core(coord);
        const CoreCoord service_phys = device->worker_core_from_logical_core(service_logical);
        const uint32_t consumed_counter_addr = static_cast<uint32_t>(service.get_consumed_counter_addr(coord));

        auto program = tt::tt_metal::CreateProgram();

        auto cb_cfg = tt::tt_metal::CircularBufferConfig(page_size, {{scratch_cb_index, tt::DataFormat::UInt32}})
                          .set_page_size(scratch_cb_index, page_size);
        tt::tt_metal::CreateCircularBuffer(program, worker_cores, cb_cfg);

        std::vector<uint32_t> ct_args = {
            data_ready_sem_addr,
            input_tensor_addr,
            page_size,
            static_cast<uint32_t>(scratch_cb_index),
        };
        ct_args.insert(ct_args.end(), accessor_compile_args.begin(), accessor_compile_args.end());

        auto kernel_handle = tt::tt_metal::CreateKernel(
            program,
            "tests/ttnn/benchmark/cpp/kernels/persistent_h2d_drain_benchmark.cpp",
            worker_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = ct_args,
            });

        uint32_t worker_idx = 0;
        for (uint32_t y = worker_cores.start_coord.y; y <= worker_cores.end_coord.y; ++y) {
            for (uint32_t x = worker_cores.start_coord.x; x <= worker_cores.end_coord.x; ++x) {
                const CoreCoord core{x, y};
                const uint32_t start_page = worker_idx * pages_per_worker;
                const uint32_t end_page = start_page + pages_per_worker;
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    kernel_handle,
                    core,
                    {
                        start_page,
                        end_page,
                        consumed_counter_addr,
                        static_cast<uint32_t>(service_phys.x),
                        static_cast<uint32_t>(service_phys.y),
                        total_iters,
                    });
                ++worker_idx;
            }
        }

        worker_workload.add_program(MeshCoordinateRange(coord, coord), std::move(program));
    }

    return worker_workload;
}

uint32_t compute_warmup_iters(uint32_t fifo_size_bytes, uint64_t per_shard_payload_bytes) {
    TT_FATAL(per_shard_payload_bytes > 0, "per_shard_payload_bytes must be > 0");
    const uint64_t depth_in_transfers = (fifo_size_bytes + per_shard_payload_bytes - 1) / per_shard_payload_bytes;
    return std::max<uint32_t>(kMinWarmupIters, static_cast<uint32_t>(depth_in_transfers) + kWarmupSettlingIters);
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

void run_h2d_stream_service_benchmark(benchmark::State& state, const BenchmarkCase& cs) {
    // The drain kernel is sized to exactly one warmup+perf pass, so the benchmark body must run a
    // single iteration; more would push past the kernel's bounded loop and deadlock the receiver's
    // worker-sync wait. This guards against benchmark misconfiguration (it is set via Iterations(1)).
    TT_FATAL(
        state.max_iterations == 1,
        "benchmark_h2d_stream_service must run exactly one iteration per case; got max_iterations={}",
        state.max_iterations);
    if (!benchmark_supported(state)) {
        return;
    }
    const std::string& case_name = cs.label;

    // FullShard2D needs a 2D mesh with >= 2 devices on each axis (e.g. an 8x4 Galaxy).
    const auto mesh_shape = g_mesh_device->shape();
    if (mesh_shape.dims() != 2 || mesh_shape[0] < 2 || mesh_shape[1] < 2) {
        state.SkipWithMessage("FullShard2D requires a 2D mesh with >= 2 devices on each axis");
        return;
    }
    const uint32_t mesh_rows = mesh_shape[0];
    const uint32_t mesh_cols = mesh_shape[1];

    // Per-device footprint is [1,1,per_device_pages,kProdPerRow]; FullShard2D scales the global
    // tensor up by the mesh dims so each device lands on exactly that footprint after sharding.
    const ttnn::Shape global_shape({1, 1, cs.per_device_pages * mesh_rows, kProdPerRow * mesh_cols});
    const auto placements = full_shard_2d_placements();

    const uint32_t tensor_page_bytes = kProdPerRow * kElemBytes;
    const uint32_t scratch_cb_size_bytes = cs.cb_pages * tensor_page_bytes;
    const uint32_t fifo_size_bytes = cs.fifo_pages * tensor_page_bytes;
    const uint32_t num_workers = worker_count(kWorkerCores);

    log_info(
        tt::LogTest,
        "[{}] Starting: global_shape={}, fifo_size_bytes={}, scratch_cb_size_bytes={}, workers={}, perf_iters={}",
        case_name,
        stream_string(global_shape),
        fifo_size_bytes,
        scratch_cb_size_bytes,
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
        .fifo_size_bytes = fifo_size_bytes,
        .scratch_cb_size_bytes = scratch_cb_size_bytes,
        .socket_mode = H2DMode::DEVICE_PULL,
        .worker_cores = kWorkerCores,
        .metadata_size_bytes = 0,
    };

    tt::tt_metal::H2DStreamService service(g_mesh_device, std::move(cfg));
    const auto sockets = service.get_sockets();
    TT_FATAL(!sockets.empty(), "benchmark requires at least one socket");

    const Tensor& backing = service.get_backing_tensor();
    auto* backing_buf = backing.buffer();
    TT_FATAL(backing_buf != nullptr, "benchmark backing tensor has no buffer");

    if (backing_buf->num_pages() % num_workers != 0) {
        state.SkipWithMessage("backing tensor page count is not divisible by worker count");
        return;
    }

    const auto& coords = backing.tensor_topology().mesh_coords();
    const uint64_t per_shard_payload_bytes =
        static_cast<uint64_t>(backing_buf->page_size()) * static_cast<uint64_t>(backing_buf->num_pages());
    const uint32_t socket_page_size = sockets.front()->get_page_size();
    TT_FATAL(socket_page_size > 0, "socket page size must be initialized");
    TT_FATAL(per_shard_payload_bytes % socket_page_size == 0, "per-shard payload bytes must divide socket page size");
    const uint32_t num_socket_pages = static_cast<uint32_t>(per_shard_payload_bytes / socket_page_size);
    const uint32_t pages_per_chunk = socket_page_size / backing_buf->page_size();
    const uint32_t warmup_iters = compute_warmup_iters(fifo_size_bytes, per_shard_payload_bytes);
    log_info(
        tt::LogTest,
        "[{}] Geometry: sockets={}, per_shard_payload_bytes={}, socket_page_size={}, num_socket_pages={}, "
        "pages_per_chunk={}, warmup_iters={}",
        case_name,
        sockets.size(),
        per_shard_payload_bytes,
        socket_page_size,
        num_socket_pages,
        pages_per_chunk,
        warmup_iters);

    // The drain kernel runs a bounded loop of exactly this many iterations, so the host must push
    // exactly total_iters transfers (warmup + perf) below; a mismatch deadlocks the receiver's
    // worker-sync wait. This is why the benchmark must run a single state iteration (guard above).
    const uint32_t total_iters = warmup_iters + kPerfIters;
    auto drain_workload = build_drain_workload(g_mesh_device, service, kWorkerCores, total_iters);
    log_info(
        tt::LogTest,
        "[{}] Enqueuing bounded persistent drain workload for {} total iterations across {} coords",
        case_name,
        total_iters,
        coords.size());
    tt::tt_metal::distributed::EnqueueMeshWorkload(
        g_mesh_device->mesh_command_queue(), drain_workload, /*blocking=*/false);

    // Distributed-tensor input path (the production path). Build the distributed host tensor once
    // and reuse it every push; the values are irrelevant since the benchmark does not verify.
    std::vector<uint32_t> source_data(global_shape.volume());
    std::iota(source_data.begin(), source_data.end(), 0u);
    auto host_mapper =
        ttnn::distributed::create_mesh_mapper(*g_mesh_device, MeshMapperConfig{.placements = placements});
    const Tensor tensor_input =
        ttnn::distributed::distribute_tensor(Tensor::from_vector<uint32_t>(source_data, global_spec), *host_mapper);

    auto push_once = [&]() { service.forward_to_tensor(tensor_input); };

    for ([[maybe_unused]] auto _ : state) {
        log_info(tt::LogTest, "[{}] Starting warmup phase with {} iterations", case_name, warmup_iters);
        for (uint32_t iter = 0; iter < warmup_iters; ++iter) {
            push_once();
        }
        log_info(tt::LogTest, "[{}] Warmup phase complete", case_name);

        // Time only the perf-push loop. After warmup the FIFO is full, so every push blocks until
        // the receiver pops a page, which it can only do after the previous transfer's worker drain
        // (the receiver is single-threaded: the next read loop follows the worker-sync wait). So the
        // push rate equals the steady-state end-to-end drain rate. The in-flight depth is identical
        // at t0 and t1 (FIFO full both times), so exactly perf_iters transfers drain inside the
        // window -- no warmup-backlog tail, and no need to drain the pipeline before stopping.
        log_info(tt::LogTest, "[{}] Starting timed phase with {} iterations", case_name, kPerfIters);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t iter = 0; iter < kPerfIters; ++iter) {
            push_once();
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
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
        state.counters["per_device_pages"] = static_cast<double>(cs.per_device_pages);
        state.counters["cb_pages"] = static_cast<double>(cs.cb_pages);
        state.counters["fifo_pages"] = static_cast<double>(cs.fifo_pages);
        state.counters["fifo_size_bytes"] = static_cast<double>(fifo_size_bytes);
        state.counters["scratch_cb_size_bytes"] = static_cast<double>(scratch_cb_size_bytes);
        state.counters["worker_count"] = static_cast<double>(num_workers);
        state.counters["per_shard_bytes"] = static_cast<double>(per_shard_payload_bytes);
        state.counters["socket_page_size"] = static_cast<double>(socket_page_size);
        state.counters["num_socket_pages"] = static_cast<double>(num_socket_pages);
        state.counters["pages_per_chunk"] = static_cast<double>(pages_per_chunk);
        log_info(
            tt::LogTest,
            "[{}] Timed phase complete: elapsed_s={:.6f}, aggregate_gbps={:.6f}, global_payload_gbps={:.6f}",
            case_name,
            elapsed_s,
            aggregate_gbps,
            global_gbps);
    }

    // The bounded drain kernel exits only after consuming all total_iters transfers, and consuming
    // transfer N requires the receiver to have acked N's pages and the workers to have drained it.
    // So Finish blocking on the drain workload's completion already guarantees every pushed transfer
    // was acked and fully drained -- no explicit barrier() or consumed-counter wait is needed.
    tt::tt_metal::distributed::Finish(g_mesh_device->mesh_command_queue());
    log_info(tt::LogTest, "[{}] Benchmark case complete", case_name);
}

std::vector<BenchmarkCase> make_benchmark_cases(const MeshDevice& mesh_device) {
    const auto mesh_shape = mesh_device.shape();
    if (mesh_shape.dims() != 2 || mesh_shape[0] < 2 || mesh_shape[1] < 2) {
        log_warning(
            tt::LogTest,
            "H2DStreamService benchmark needs a 2D mesh with >= 2 devices per axis for FullShard2D; "
            "no cases generated");
        return {};
    }
    const uint32_t num_workers = worker_count(kWorkerCores);

    std::vector<BenchmarkCase> cases;
    auto add_case = [&](const std::string& family, uint32_t pages, uint32_t cb, uint32_t fifo) {
        // pages must split evenly across the drain workers and the scratch-CB chunk; the socket
        // page (cb) must fit in the FIFO.
        if (pages % num_workers != 0 || pages % cb != 0 || fifo < cb) {
            return;
        }
        cases.push_back(BenchmarkCase{
            .label =
                family + "/p" + std::to_string(pages) + "/cb" + std::to_string(cb) + "/fifo" + std::to_string(fifo),
            .per_device_pages = pages,
            .cb_pages = cb,
            .fifo_pages = fifo,
        });
    };

    // Each family fixes one axis as the x-axis and draws a few lines that perturb the other knobs,
    // so a chart shows how that axis's trend shifts with the chunk plan. add_case() drops invalid
    // combos (pages not divisible by workers/cb, or fifo < cb), so a line simply starts/ends where
    // the constraints allow.

    // size -- throughput vs per-device pages, at three chunk plans (anchor, coarser reads, shallower
    // buffer). The chunk-plan effect at a fixed size is studied by the cb/fifo families below.
    const uint32_t size_pages[] = {16, 32, 64, 128, 256, 512, kProdPerDevicePages, 1024};
    const uint32_t size_lines[][2] = {
        {kDefaultCbPages, kDefaultFifoPages},  // anchor
        {16, kDefaultFifoPages},               // coarser reads (higher cb)
        {kDefaultCbPages, 16},                 // shallower buffer (lower fifo)
    };
    for (const auto& line : size_lines) {
        for (uint32_t pages : size_pages) {
            add_case("size", pages, line[0], line[1]);
        }
    }

    // cb -- throughput vs scratch-CB pages (read granularity) at the production size, at three FIFO
    // depths: shows how coalescing interacts with buffering.
    const uint32_t cb_sweep[] = {1, 2, 4, 8, 16, 32, 64};
    const uint32_t cb_line_fifo[] = {16, kDefaultFifoPages, 128};
    for (uint32_t fifo : cb_line_fifo) {
        for (uint32_t cb : cb_sweep) {
            add_case("cb", kProdPerDevicePages, cb, fifo);
        }
    }

    // fifo -- throughput vs FIFO pages (buffering depth) at the production size, at three read
    // granularities: shows how buffering interacts with the socket-page size.
    const uint32_t fifo_sweep[] = {4, 8, 16, 32, 64, 128};
    const uint32_t fifo_line_cb[] = {1, kDefaultCbPages, 16};
    for (uint32_t cb : fifo_line_cb) {
        for (uint32_t fifo : fifo_sweep) {
            add_case("fifo", kProdPerDevicePages, cb, fifo);
        }
    }

    return cases;
}

void register_benchmarks() {
    for (const auto& cs : make_benchmark_cases(*g_mesh_device)) {
        const std::string benchmark_name = "BM_H2DStreamService/" + cs.label;
        benchmark::RegisterBenchmark(
            benchmark_name.c_str(), [cs](benchmark::State& state) { run_h2d_stream_service_benchmark(state, cs); })
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
