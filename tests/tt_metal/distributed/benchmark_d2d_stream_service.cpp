// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbenchmarks for the D2D stream service (ttnn/core/tensor/d2d_stream_service.cpp).
// Host wall-clock (std::chrono) only — no device timestamps — modeled on the H2D path
// in tests/tt_metal/distributed/benchmark_hd_sockets.cpp. The service is built ONCE per
// benchmark row and excluded from timing (we measure a pre-initialized service).
//
//   BM_D2DStreamLatency    — N-stage streaming pipeline (Host -> H2D -> [D2D]* -> output),
//                            nearly identical to StreamPipelineTest.FourStageAllCores. Times
//                            the per-iteration stage walk; reports total_us and a per-hop
//                            figure. Sweep num_stages {2,4,8} so the per-hop DIFFERENTIAL
//                            (slope across N) cancels the fixed H2D + host-dispatch overhead
//                            and isolates marginal D2D hop latency.
//   BM_D2DStreamThroughput — single D2D hop, large DRAM-backed tensor, all worker cores.
//                            The sender backing is left resident in DRAM and the persistent
//                            sender service re-reads it over fabric each transfer; a
//                            signal-only worker (bench_d2d_signal_sender_worker.cpp) drives
//                            the handshake WITHOUT re-filling the backing, so the timed loop
//                            measures fabric-transfer GB/s rather than redundant DRAM fills.
//
// Both are swept over metadata (disabled/enabled) and LEASE vs OWN fabric-link mode.
//
// Hardware: FABRIC_2D, service cores (Blackhole / UBB Galaxy under Fast Dispatch), H2D
// host-DMA pinning (latency only), and >= num_stages devices to carve one 1x1 submesh per
// stage. Rows that don't apply SkipWithMessage cleanly.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"  // create_device_tensor
#include "ttnn/tensor/types.hpp"

#include "tt_metal/llrt/tt_cluster.hpp"  // full tt::Cluster (is_ubb_galaxy / is_iommu_enabled in the utils header)

#include "tests/ttnn/unit_tests/gtests/tensor/stream_service_test_utils.hpp"

namespace tt::tt_metal::distributed {
namespace {

using ::tt::CBIndex;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CircularBufferConfig;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::create_device_tensor;
using ::tt::tt_metal::CreateCircularBuffer;
using ::tt::tt_metal::CreateKernel;
using ::tt::tt_metal::CreateProgram;
using ::tt::tt_metal::D2DStreamConfig;
using ::tt::tt_metal::D2DStreamService;
using ::tt::tt_metal::D2DStreamServiceReceiver;
using ::tt::tt_metal::D2DStreamServiceSender;
using ::tt::tt_metal::DataMovementConfig;
using ::tt::tt_metal::DataMovementProcessor;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::DeviceAddr;
using ::tt::tt_metal::H2DStreamService;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::NOC;
using ::tt::tt_metal::SetRuntimeArgs;
using ::tt::tt_metal::Tensor;
using ::tt::tt_metal::TensorAccessorArgs;
using ::tt::tt_metal::TensorSpec;
using ::ttnn::distributed::create_mesh_mapper;
using ::ttnn::distributed::MeshMapperConfig;

// Shared helpers (replicate_all / service_cores_supported / h2d_host_pinning_supported /
// make_spec / all_cores_for / worker_index / worker_page_range / fifo_bytes_for /
// core_range_volume) live in stream_service_test_utils.hpp.
using namespace ttnn::distributed::test;  // NOLINT(google-build-using-namespace)

// ── Kernel paths (repo-relative; resolved at CreateKernel time) ──────────────────
constexpr const char* kRelayKernel = "tests/ttnn/unit_tests/gtests/tensor/kernels/pipeline_relay_worker.cpp";
constexpr const char* kSignalSenderKernel = "tests/tt_metal/distributed/kernels/bench_d2d_signal_sender_worker.cpp";
constexpr const char* kReceiverKernel =
    "tests/ttnn/unit_tests/gtests/tensor/kernels/placeholder_d2d_receiver_worker.cpp";

// ── Sweep / iteration constants ──────────────────────────────────────────────────
// Iteration counts default to these but can be overridden at runtime via env vars
// (D2D_BENCH_WARMUP / D2D_BENCH_LAT_ITERS / D2D_BENCH_TPUT_ITERS) so a hang can be
// bisected by shrinking the loop without a rebuild.
constexpr uint32_t kWarmupItersDefault = 5;
constexpr uint32_t kLatencyItersDefault = 50;
constexpr uint32_t kThroughputItersDefault = 100;
constexpr uint32_t kMetadataTripleBytes = 3u * static_cast<uint32_t>(sizeof(uint32_t));

// Read an unsigned env override, falling back to `fallback` if unset/unparseable.
uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 10);
    return (end == v) ? fallback : static_cast<uint32_t>(parsed);
}

uint32_t warmup_iters() { return env_u32("D2D_BENCH_WARMUP", kWarmupItersDefault); }
uint32_t latency_iters() { return env_u32("D2D_BENCH_LAT_ITERS", kLatencyItersDefault); }
uint32_t throughput_iters() { return env_u32("D2D_BENCH_TPUT_ITERS", kThroughputItersDefault); }

// Host-side progress logging, gated on D2D_BENCH_VERBOSE (any non-empty value). Goes
// to stderr and is flushed immediately so the LAST line printed before a hang pins the
// phase that wedged. Use plain printf-style formatting (no benchmark/log deps).
bool bench_verbose() {
    static const bool on = [] {
        const char* v = std::getenv("D2D_BENCH_VERBOSE");
        return v != nullptr && *v != '\0';
    }();
    return on;
}

#define BENCH_LOG(...)                                        \
    do {                                                      \
        if (bench_verbose()) {                                \
            std::fprintf(stderr, "[d2d-bench] " __VA_ARGS__); \
            std::fprintf(stderr, "\n");                       \
            std::fflush(stderr);                              \
        }                                                     \
    } while (0)

// Latency uses a small fixed tensor (the FourStageAllCores shape).
const ttnn::Shape kLatencyShape({1, 1, 32, 64});

// Throughput payload sweep (uint32, DRAM-backed): ~0.5 MB -> ~512 MB.
const std::vector<ttnn::Shape> kThroughputShapes = {
    ttnn::Shape({1, 1, 256, 512}),    // 0.5 MB
    ttnn::Shape({1, 1, 1024, 1024}),  // 4 MB
    ttnn::Shape({1, 1, 4096, 1024}),  // 16 MB
    ttnn::Shape({1, 1, 4096, 4096}),  // 64 MB
    ttnn::Shape({1, 8, 4096, 4096}),  // 512 MB (ShapeLarge)
};

// ===========================================================================
// FABRIC_2D mesh fixture (singleton, like benchmark_hd_sockets' DeviceFixture).
// The plain DeviceFixture there does NOT enable fabric; D2D needs FABRIC_2D set
// before MeshDevice::create (and reset on teardown) — see GenericMeshDeviceFabric2DFixture.
// ===========================================================================
struct FabricMeshFixture {
    std::shared_ptr<MeshDevice> mesh_device;
    // Persistent 1x1 submesh pair reused across throughput cycles (created lazily). The
    // per-cycle carve+destroy pattern (carve_stages) is what triggers the cross-cycle hang
    // — see notes/d2d_throughput_sweep_hang.md — so the throughput path reuses these.
    std::vector<std::shared_ptr<MeshDevice>> persistent_pair;

    FabricMeshFixture() {
        tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::FABRIC_2D);
        mesh_device = MeshDevice::create(
            MeshDeviceConfig(std::nullopt),
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            /*num_command_queues=*/1,
            DispatchCoreType::WORKER);
    }
    ~FabricMeshFixture() {
        if (mesh_device) {
            persistent_pair.clear();  // drop submeshes before closing the parent
            persistent_latency.clear();
            mesh_device->close();
            mesh_device.reset();
            tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
        }
    }

    // Cached pair of 1x1 submeshes at (0,0) and (0,1) (matches carve_stages(parent, 2)).
    std::vector<std::shared_ptr<MeshDevice>>& throughput_pair() {
        if (persistent_pair.empty()) {
            const auto shape = mesh_device->shape();
            for (uint32_t i = 0; i < 2; ++i) {
                persistent_pair.push_back(
                    mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(i / shape[1], i % shape[1])));
            }
        }
        return persistent_pair;
    }

    // Persistent 1x1 submeshes reused across latency cycles. carve_stages(parent, n) always
    // places submesh i at the SAME row-major coord regardless of n, so we grow one cached
    // vector and hand back its first n entries — different num_stages rows share the lower
    // submeshes instead of recreating them (the recreate-at-same-coord hang trigger).
    std::vector<std::shared_ptr<MeshDevice>> persistent_latency;
    std::vector<std::shared_ptr<MeshDevice>> latency_stages(uint32_t n) {
        const auto shape = mesh_device->shape();
        while (persistent_latency.size() < n) {
            const uint32_t i = static_cast<uint32_t>(persistent_latency.size());
            persistent_latency.push_back(
                mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(i / shape[1], i % shape[1])));
        }
        return std::vector<std::shared_ptr<MeshDevice>>(persistent_latency.begin(), persistent_latency.begin() + n);
    }
};

FabricMeshFixture& get_fixture() {
    static FabricMeshFixture fixture;
    return fixture;
}

bool enough_devices(const MeshDevice& mesh, uint32_t num_stages) {
    return mesh.shape().dims() == 2 && mesh.num_devices() >= num_stages;
}

// Carve `n` distinct 1x1 submeshes (one per stage), row-major (copy of carve_stages
// from test_stream_pipeline.cpp).
std::vector<std::shared_ptr<MeshDevice>> carve_stages(MeshDevice& parent, uint32_t n) {
    const auto shape = parent.shape();
    std::vector<std::shared_ptr<MeshDevice>> stages;
    stages.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        stages.push_back(parent.create_submesh(MeshShape(1, 1), MeshCoordinate(i / shape[1], i % shape[1])));
    }
    return stages;
}

// ── Latency run result ────────────────────────────────────────────────────────────
struct LatencyRunResult {
    std::vector<double> per_iter_us;
    uint32_t num_workers = 0;
};

// ── Latency summary (host-side microseconds) ──────────────────────────────────────
struct LatencySummary {
    double avg_us;
    double min_us;
    double max_us;
    double p50_us;
    double p99_us;
};

LatencySummary summarize_us(std::vector<double> us) {
    std::sort(us.begin(), us.end());
    const double avg = std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(us.size());
    return LatencySummary{
        .avg_us = avg,
        .min_us = us.front(),
        .max_us = us.back(),
        .p50_us = us[us.size() / 2],
        .p99_us = us[(us.size() * 99) / 100],
    };
}

// ===========================================================================
// Relay-pipeline plumbing (ported verbatim from test_stream_pipeline.cpp: the
// Upstream/Downstream handles + build_relay_workload). The benchmark reuses the
// exact production relay kernel, so the timing reflects the real handshake.
// ===========================================================================
struct UpstreamHandle {
    const Tensor* backing = nullptr;
    DeviceAddr data_ready_sem_addr = 0;
    std::function<DeviceAddr(const MeshCoordinate&)> consumed_counter_addr;
    std::function<CoreCoord(const MeshCoordinate&)> service_core;
    std::function<DeviceAddr()> metadata_addr;
};

template <typename Svc>
UpstreamHandle make_upstream(Svc& svc) {
    return UpstreamHandle{
        .backing = &svc.get_backing_tensor(),
        .data_ready_sem_addr = svc.get_data_ready_sem_addr(),
        .consumed_counter_addr = [&svc](const MeshCoordinate& c) { return svc.get_consumed_counter_addr(c); },
        .service_core = [&svc](const MeshCoordinate& c) { return svc.get_service_core(c); },
        .metadata_addr = [&svc] { return svc.get_metadata_addr(); },
    };
}

struct DownstreamHandle {
    const Tensor* dest = nullptr;
    bool produce = false;
    std::function<DeviceAddr(const MeshCoordinate&)> data_ready_counter_addr;
    std::function<CoreCoord(const MeshCoordinate&)> service_core;
    std::function<DeviceAddr(const MeshCoordinate&)> metadata_addr;
};

DownstreamHandle make_downstream_producer(D2DStreamServiceSender& s) {
    return DownstreamHandle{
        .dest = &s.get_backing_tensor(),
        .produce = true,
        .data_ready_counter_addr = [&s](const MeshCoordinate& c) { return s.get_data_ready_counter_addr(c); },
        .service_core = [&s](const MeshCoordinate& c) { return s.get_service_core(c); },
        .metadata_addr = [&s](const MeshCoordinate& c) { return s.get_metadata_addr(c); },
    };
}

DownstreamHandle make_downstream_terminal(const Tensor& output) {
    return DownstreamHandle{.dest = &output, .produce = false};
}

MeshWorkload build_relay_workload(
    MeshDevice& stage,
    const CoreRange& workers,
    uint32_t num_iters,
    const UpstreamHandle& up,
    const DownstreamHandle& down,
    uint32_t metadata_size_bytes) {
    const auto* up_buf = up.backing->buffer();
    const uint32_t page_size = up_buf->aligned_page_size();
    const uint32_t num_pages = up_buf->num_pages();
    const bool metadata_enabled = metadata_size_bytes > 0;
    const bool forwards_metadata = metadata_enabled && down.produce;
    constexpr auto kScratchCb = CBIndex::c_0;

    MeshWorkload workload;
    for (const auto& coord : up.backing->tensor_topology().mesh_coords()) {
        auto program = CreateProgram();

        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, workers, cb_cfg);

        const auto* up_dbuf = up.backing->mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*up_dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(up.data_ready_sem_addr),
            static_cast<uint32_t>(up.backing->buffer()->address()),
            static_cast<uint32_t>(down.dest->buffer()->address()),
            page_size,
            num_iters,
            static_cast<uint32_t>(kScratchCb),
            down.produce ? 1u : 0u,
            metadata_enabled ? 1u : 0u,
            metadata_size_bytes,
            metadata_enabled ? static_cast<uint32_t>(up.metadata_addr()) : 0u,
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            kRelayKernel,
            workers,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = stage.get_device(coord);
        const auto up_svc_phys = device->worker_core_from_logical_core(up.service_core(coord));
        CoreCoord down_svc_phys{0, 0};
        uint32_t down_counter_addr = 0;
        uint32_t down_metadata_addr = 0;
        if (down.produce) {
            down_svc_phys = device->worker_core_from_logical_core(down.service_core(coord));
            down_counter_addr = static_cast<uint32_t>(down.data_ready_counter_addr(coord));
            if (metadata_enabled) {
                down_metadata_addr = static_cast<uint32_t>(down.metadata_addr(coord));
            }
        }

        const uint32_t num_workers = core_range_volume(workers);
        for (const auto& wc : workers) {
            const auto [start_page, end_page] = worker_page_range(worker_index(wc, workers), num_workers, num_pages);
            const uint32_t is_metadata_writer = (forwards_metadata && wc == workers.end_coord) ? 1u : 0u;
            const std::vector<uint32_t> rt_args = {
                start_page,
                end_page,
                static_cast<uint32_t>(up.consumed_counter_addr(coord)),
                static_cast<uint32_t>(up_svc_phys.x),
                static_cast<uint32_t>(up_svc_phys.y),
                down_counter_addr,
                static_cast<uint32_t>(down_svc_phys.x),
                static_cast<uint32_t>(down_svc_phys.y),
                is_metadata_writer,
                down_metadata_addr,
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// ===========================================================================
// LATENCY: build the N-stage pipeline once, then time the per-iter stage walk.
// Returns per-measured-iteration end-to-end microseconds.
// ===========================================================================
LatencyRunResult run_latency(MeshDevice& parent, uint32_t num_stages, uint32_t metadata_size_bytes, bool lease) {
    const bool metadata_enabled = metadata_size_bytes > 0;
    // Reuse persistent submeshes by default (D2D_BENCH_REUSE_SUBMESH=1) instead of carving
    // per row — per-cycle submesh recreation is the cross-row hang trigger
    // (notes/d2d_throughput_sweep_hang.md). Set 0 to restore per-row carve for A/B.
    const bool reuse_submesh = env_u32("D2D_BENCH_REUSE_SUBMESH", 1) != 0;
    auto stages = reuse_submesh ? get_fixture().latency_stages(num_stages) : carve_stages(parent, num_stages);
    const TensorSpec global_spec = make_spec(kLatencyShape);
    const CoreRange workers = all_cores_for(*stages[0]);
    const uint32_t fifo_bytes = fifo_bytes_for(global_spec);

    H2DStreamService h2d(
        stages[0],
        H2DStreamService::Config{
            .global_spec = global_spec,
            .mapper = create_mesh_mapper(*stages[0], MeshMapperConfig{.placements = replicate_all(*stages[0])}),
            .socket_buffer_type = BufferType::L1,
            .fifo_size_bytes = fifo_bytes,
            .scratch_cb_size_bytes = fifo_bytes,
            .worker_cores = workers,
            .metadata_size_bytes = metadata_size_bytes,
        });

    std::vector<std::pair<std::unique_ptr<D2DStreamServiceSender>, std::unique_ptr<D2DStreamServiceReceiver>>> d2d;
    for (uint32_t i = 0; i + 1 < num_stages; ++i) {
        d2d.push_back(D2DStreamService::create_pair(
            stages[i],
            stages[i + 1],
            D2DStreamConfig{
                .global_spec = global_spec,
                .mapper = create_mesh_mapper(*stages[i], MeshMapperConfig{.placements = replicate_all(*stages[i])}),
                .socket_mem_config = SocketMemoryConfig{BufferType::L1, fifo_bytes},
                .sender_worker_cores = workers,
                .receiver_worker_cores = workers,
                .metadata_size_bytes = metadata_size_bytes,
                .share_fabric_links = lease,
            }));
    }

    D2DStreamServiceReceiver& last_recv = *d2d.back().second;
    Tensor output_tensor = create_device_tensor(
        last_recv.get_per_shard_spec(), stages[num_stages - 1].get(), last_recv.get_backing_tensor().tensor_topology());

    // Pre-build each stage's relay workload ONCE (addresses are fixed for the service
    // lifetime), so the timed loop only re-enqueues — no host program-build cost.
    std::vector<MeshWorkload> relays;
    relays.reserve(num_stages);
    for (uint32_t i = 0; i < num_stages; ++i) {
        const bool has_inbound = (i > 0);
        const bool has_outbound = (i + 1 < num_stages);
        UpstreamHandle up = has_inbound ? make_upstream(*d2d[i - 1].second) : make_upstream(h2d);
        DownstreamHandle down =
            has_outbound ? make_downstream_producer(*d2d[i].first) : make_downstream_terminal(output_tensor);
        relays.push_back(build_relay_workload(*stages[i], workers, /*num_iters=*/1, up, down, metadata_size_bytes));
    }

    // Fixed host source + metadata, distributed once (content is irrelevant to timing).
    const uint32_t num_elems = static_cast<uint32_t>(kLatencyShape.volume());
    std::vector<uint32_t> src(num_elems);
    std::iota(src.begin(), src.end(), 1u);
    const Tensor host_src = Tensor::from_vector<uint32_t>(src, global_spec);
    auto mapper = create_mesh_mapper(*stages[0], MeshMapperConfig{.placements = replicate_all(*stages[0])});
    const Tensor distributed_src = ttnn::distributed::distribute_tensor(host_src, *mapper);
    std::vector<uint32_t> md(metadata_enabled ? metadata_size_bytes / sizeof(uint32_t) : 0u);
    if (metadata_enabled) {
        std::iota(md.begin(), md.end(), 0u);
    }
    auto md_span = metadata_enabled
                       ? ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(md.data()), metadata_size_bytes)
                       : ttsl::Span<const std::byte>{};

    auto run_one_iter = [&]() {
        h2d.forward_to_tensor(distributed_src, md_span);
        for (uint32_t i = 0; i < num_stages; ++i) {
            const bool has_inbound = (i > 0);
            const bool has_outbound = (i + 1 < num_stages);
            if (lease) {
                if (has_inbound) {
                    d2d[i - 1].second->wait_for_fabric_links();
                }
                if (has_outbound) {
                    d2d[i].first->wait_for_fabric_links();
                }
                if (has_inbound) {
                    d2d[i - 1].second->release_fabric_links();
                }
            }
            EnqueueMeshWorkload(stages[i]->mesh_command_queue(), relays[i], /*blocking=*/false);
            Finish(stages[i]->mesh_command_queue());
            if (lease && has_outbound) {
                d2d[i].first->release_fabric_links();
            }
        }
    };

    const uint32_t n_warmup = warmup_iters();
    const uint32_t n_iters = latency_iters();
    for (uint32_t w = 0; w < n_warmup; ++w) {
        run_one_iter();
    }
    std::vector<double> per_iter_us(n_iters);
    for (uint32_t it = 0; it < n_iters; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_one_iter();
        auto t1 = std::chrono::high_resolution_clock::now();
        per_iter_us[it] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return {std::move(per_iter_us), core_range_volume(workers)};
}

// ===========================================================================
// THROUGHPUT worker workloads (single D2D hop).
// ===========================================================================
MeshWorkload build_signal_sender_workload(
    D2DStreamServiceSender& sender, MeshDevice& mesh, uint32_t num_iters, uint32_t metadata_size_bytes) {
    const CoreRange workers = sender.get_worker_cores();
    const bool metadata_enabled = metadata_size_bytes > 0;
    constexpr auto kScratchCb = CBIndex::c_0;
    constexpr uint32_t kScratchBytes = 4096u;

    MeshWorkload workload;
    for (const auto& coord : sender.get_backing_tensor().tensor_topology().mesh_coords()) {
        auto program = CreateProgram();
        auto cb_cfg = CircularBufferConfig(kScratchBytes, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, kScratchBytes);
        CreateCircularBuffer(program, workers, cb_cfg);

        const std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(sender.get_consumed_sem_addr()),
            num_iters,
            static_cast<uint32_t>(kScratchCb),
            /*fill_base=*/1u,
            metadata_enabled ? 1u : 0u,
            metadata_size_bytes,
        };
        auto kernel = CreateKernel(
            program,
            kSignalSenderKernel,
            workers,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh.get_device(coord);
        const auto service_phys = device->worker_core_from_logical_core(sender.get_service_core(coord));
        const uint32_t md_addr = metadata_enabled ? static_cast<uint32_t>(sender.get_metadata_addr(coord)) : 0u;
        for (const auto& wc : workers) {
            const uint32_t is_metadata_writer = (metadata_enabled && wc == workers.end_coord) ? 1u : 0u;
            const std::vector<uint32_t> rt_args = {
                static_cast<uint32_t>(sender.get_data_ready_counter_addr(coord)),
                static_cast<uint32_t>(service_phys.x),
                static_cast<uint32_t>(service_phys.y),
                is_metadata_writer,
                md_addr,
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

MeshWorkload build_receiver_workload(D2DStreamServiceReceiver& receiver, MeshDevice& mesh, uint32_t num_iters) {
    const CoreRange workers = receiver.get_worker_cores();
    MeshWorkload workload;
    for (const auto& coord : receiver.get_backing_tensor().tensor_topology().mesh_coords()) {
        auto program = CreateProgram();
        const std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(receiver.get_data_ready_sem_addr()),
            num_iters,
        };
        auto kernel = CreateKernel(
            program,
            kReceiverKernel,
            workers,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh.get_device(coord);
        const auto service_phys = device->worker_core_from_logical_core(receiver.get_service_core(coord));
        const std::vector<uint32_t> rt_args = {
            static_cast<uint32_t>(receiver.get_consumed_counter_addr(coord)),
            static_cast<uint32_t>(service_phys.x),
            static_cast<uint32_t>(service_phys.y),
        };
        for (const auto& wc : workers) {
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

struct ThroughputResult {
    double tensor_bytes = 0.0;
    double gbps = 0.0;
    double transfer_ms = 0.0;  // chrono-measured time of the timed transfer region (all n_iters)
    int data_ok = -1;          // -1 = unchecked, 0 = FAIL, 1 = PASS (D2D_BENCH_CHECK_DATA=1)
    uint32_t num_workers = 0;
};

// Returns {tensor_bytes_per_transfer, sustained_GBps, transfer_ms, data_ok}.
ThroughputResult run_throughput(
    MeshDevice& parent, const ttnn::Shape& shape, uint32_t metadata_size_bytes, bool lease) {
    const uint32_t n_warmup = warmup_iters();
    const uint32_t n_iters = throughput_iters();
    // Optional metadata-size override (e.g. D2D_BENCH_MD_OVERRIDE=16 for a 16 B blob /
    // better alignment than the default 12 B), applied only to metadata-enabled rows.
    if (metadata_size_bytes > 0) {
        const uint32_t md_over = env_u32("D2D_BENCH_MD_OVERRIDE", 0);
        if (md_over > 0) {
            metadata_size_bytes = md_over;
        }
    }
    // D2D_BENCH_SINGLE_CORE=1 runs a 1x1 worker grid instead of the full compute grid.
    const bool single_core = env_u32("D2D_BENCH_SINGLE_CORE", 0) != 0;
    // D2D_BENCH_STEP drives each transfer one-at-a-time from the host (even in OWN mode)
    // so the last "iter k/N" line before a hang pins which transfer wedged. This
    // host-serializes the loop AND replaces the single on-device num_iters loop with N
    // separate num_iters=1 workloads, so the reported GB/s is NOT representative and the
    // on-device-loop code path is no longer exercised — use it only to localize a hang.
    // Kept SEPARATE from D2D_BENCH_VERBOSE so verbose logging can trace the real batched
    // (single on-device loop) path too.
    const bool step = env_u32("D2D_BENCH_STEP", 0) != 0;
    // D2D_BENCH_REUSE_SUBMESH (default ON): reuse one persistent 1x1 submesh pair across
    // all cycles instead of carving + destroying per cycle. Per-cycle submesh recreation
    // is the cross-cycle-hang trigger (notes/d2d_throughput_sweep_hang.md); set to 0 to
    // restore the old per-cycle carve for A/B comparison.
    const bool reuse_submesh = env_u32("D2D_BENCH_REUSE_SUBMESH", 1) != 0;
    // D2D_BENCH_CHECK_DATA=1: fill sender backing with a known pattern, zero the receiver
    // backing, and after the transfers verify the receiver backing matches. The throughput
    // path normally never checks data (signal-only workers), so this guards correctness.
    const bool check_data = env_u32("D2D_BENCH_CHECK_DATA", 0) != 0;
    BENCH_LOG(
        "run_throughput ENTER vol=%llu mode=%s md=%u cores=%s submesh=%s check_data=%d warmup=%u iters=%u%s",
        static_cast<unsigned long long>(shape.volume()),
        lease ? "lease" : "own",
        metadata_size_bytes,
        single_core ? "1x1" : "grid",
        reuse_submesh ? "reused" : "per-cycle",
        check_data ? 1 : 0,
        n_warmup,
        n_iters,
        step ? " [STEPPED: gbps not representative]" : "");

    std::vector<std::shared_ptr<MeshDevice>> local_stages;
    if (!reuse_submesh) {
        local_stages = carve_stages(parent, 2);
    }
    std::vector<std::shared_ptr<MeshDevice>>& stages = reuse_submesh ? get_fixture().throughput_pair() : local_stages;
    BENCH_LOG("stages ready (%s)", reuse_submesh ? "reused persistent pair" : "carved per-cycle");
    const TensorSpec global_spec = make_spec(shape);
    const CoreRange workers = single_core ? CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}} : all_cores_for(*stages[0]);
    const uint32_t fifo_bytes = fifo_bytes_for(global_spec);

    auto [sender, receiver] = D2DStreamService::create_pair(
        stages[0],
        stages[1],
        D2DStreamConfig{
            .global_spec = global_spec,
            .mapper = create_mesh_mapper(*stages[0], MeshMapperConfig{.placements = replicate_all(*stages[0])}),
            .socket_mem_config = SocketMemoryConfig{BufferType::L1, fifo_bytes},
            .sender_worker_cores = workers,
            .receiver_worker_cores = workers,
            .metadata_size_bytes = metadata_size_bytes,
            .share_fabric_links = lease,
        });
    BENCH_LOG("create_pair done (fifo_bytes=%u)", fifo_bytes);

    const auto* backing = sender->get_backing_tensor().buffer();
    const double tensor_bytes = static_cast<double>(backing->num_pages()) * backing->aligned_page_size();
    BENCH_LOG(
        "backing num_pages=%llu aligned_page=%llu tensor_bytes=%.0f",
        static_cast<unsigned long long>(backing->num_pages()),
        static_cast<unsigned long long>(backing->aligned_page_size()),
        tensor_bytes);

    auto sender_cq = [&]() -> auto& { return stages[0]->mesh_command_queue(); };
    auto recv_cq = [&]() -> auto& { return stages[1]->mesh_command_queue(); };

    // ── Optional data-correctness check ──────────────────────────────────────────
    // Fill the sender backing with a known iota pattern + zero the receiver backing now;
    // verify_data() (called after the transfers) reads the receiver backing back and
    // compares. The persistent sender service re-reads its DRAM backing every transfer,
    // so a single fill suffices.
    const uint32_t check_words = static_cast<uint32_t>(shape.volume());
    constexpr uint32_t kPatternBase = 0x1000u;
    auto verify_data = [&]() -> int {
        if (!check_data) {
            return -1;
        }
        Finish(sender_cq());
        Finish(recv_cq());
        auto recv_mb = mesh_buffer_view(receiver->get_backing_tensor());
        std::vector<uint32_t> expected(check_words);
        std::iota(expected.begin(), expected.end(), kPatternBase);
        int ok = 1;
        uint32_t coord_idx = 0;
        for (const auto& coord : receiver->get_backing_tensor().tensor_topology().mesh_coords()) {
            std::vector<uint32_t> got;
            ReadShard(recv_cq(), got, recv_mb, coord, /*blocking=*/true);
            if (got.size() < check_words || !std::equal(expected.begin(), expected.end(), got.begin())) {
                ok = 0;
                uint32_t first_bad = 0;
                for (; first_bad < check_words && first_bad < got.size(); ++first_bad) {
                    if (got[first_bad] != expected[first_bad]) {
                        break;
                    }
                }
                BENCH_LOG(
                    "DATA CHECK FAIL coord#%u first_mismatch idx=%u expected=%u got=%u",
                    coord_idx,
                    first_bad,
                    first_bad < check_words ? expected[first_bad] : 0u,
                    first_bad < got.size() ? got[first_bad] : 0u);
            }
            ++coord_idx;
        }
        BENCH_LOG("DATA CHECK %s", ok ? "PASS" : "FAIL");
        return ok;
    };
    if (check_data) {
        auto sender_mb = mesh_buffer_view(sender->get_backing_tensor());
        auto recv_mb = mesh_buffer_view(receiver->get_backing_tensor());
        std::vector<uint32_t> pattern(check_words);
        std::iota(pattern.begin(), pattern.end(), kPatternBase);
        std::vector<uint32_t> zeros(check_words, 0u);
        for (const auto& coord : sender->get_backing_tensor().tensor_topology().mesh_coords()) {
            WriteShard(sender_cq(), sender_mb, pattern, coord, /*blocking=*/true);
        }
        for (const auto& coord : receiver->get_backing_tensor().tensor_topology().mesh_coords()) {
            WriteShard(recv_cq(), recv_mb, zeros, coord, /*blocking=*/true);
        }
        BENCH_LOG("DATA CHECK: filled sender backing (iota) + zeroed receiver backing");
    }

    if (lease) {
        for (uint32_t w = 0; w < n_warmup; ++w) {
            BENCH_LOG("lease warmup %u/%u start", w + 1, n_warmup);

            receiver->release_fabric_links();
            sender->release_fabric_links();
            auto recv_wl = build_receiver_workload(*receiver, *stages[1], /*num_iters=*/1);
            auto send_wl = build_signal_sender_workload(*sender, *stages[0], /*num_iters=*/1, metadata_size_bytes);

            // One grant -> one transfer per host iteration (mirrors the lease stress test).
            EnqueueMeshWorkload(recv_cq(), recv_wl, /*blocking=*/false);
            EnqueueMeshWorkload(sender_cq(), send_wl, /*blocking=*/false);
            sender->wait_for_fabric_links();
            receiver->wait_for_fabric_links();
            Finish(sender_cq());
            Finish(recv_cq());
        }
        BENCH_LOG("lease warmup done; starting measured loop");
        auto delta = std::chrono::duration<double>::zero();
        for (uint32_t it = 0; it < n_iters; ++it) {
            receiver->release_fabric_links();
            sender->release_fabric_links();
            auto recv_wl = build_receiver_workload(*receiver, *stages[1], /*num_iters=*/1);
            auto send_wl = build_signal_sender_workload(*sender, *stages[0], /*num_iters=*/1, metadata_size_bytes);

            BENCH_LOG("lease iter %u/%u start", it + 1, n_iters);

            auto t0 = std::chrono::high_resolution_clock::now();
            // One grant -> one transfer per host iteration (mirrors the lease stress test).
            EnqueueMeshWorkload(recv_cq(), recv_wl, /*blocking=*/false);
            EnqueueMeshWorkload(sender_cq(), send_wl, /*blocking=*/false);
            sender->wait_for_fabric_links();
            receiver->wait_for_fabric_links();
            Finish(sender_cq());
            Finish(recv_cq());

            auto t1 = std::chrono::high_resolution_clock::now();
            delta += std::chrono::duration<double>(t1 - t0);
        }
        BENCH_LOG("lease measured loop done");
        const double secs = delta.count();
        const double gbps = (static_cast<double>(n_iters) * tensor_bytes) / (secs * 1e9);
        const int data_ok = verify_data();
        return {tensor_bytes, gbps, secs * 1e3, data_ok, core_range_volume(workers)};
    }

    // OWN mode: the kernels loop num_iters autonomously, so a single enqueue per side drives
    // the whole measured block. To keep host dispatch (workload build + enqueue) OUT of the
    // timed region, capture each side's enqueue into a mesh trace ONCE, then time only the
    // replay (mirrors test_pgm_dispatch.cpp's BeginTraceCapture / replay_mesh_trace). The
    // worker enqueues target two different submeshes (sender=stages[0], receiver=stages[1]),
    // so we capture and replay one trace PER submesh.
    //
    // D2D_BENCH_STEP no longer applies here: OWN+trace is a single replay, not a host loop,
    // so there is nothing to step transfer-by-transfer.
    if (step) {
        BENCH_LOG("own: D2D_BENCH_STEP is inert in trace-replay mode (single replay drives all iters)");
    }
    // Build the worker workloads ONCE (num_iters=n_iters baked in) and keep them alive for the
    // lifetime of the captured traces.
    auto recv_wl = build_receiver_workload(*receiver, *stages[1], n_iters);
    auto send_wl = build_signal_sender_workload(*sender, *stages[0], n_iters, metadata_size_bytes);

    // Warm up with normal enqueues so kernels are compiled / binaries cached before capture.
    BENCH_LOG("own warmup start (normal enqueue)");
    for (uint32_t w = 0; w < n_warmup; ++w) {
        EnqueueMeshWorkload(recv_cq(), recv_wl, /*blocking=*/false);
        EnqueueMeshWorkload(sender_cq(), send_wl, /*blocking=*/false);
        Finish(sender_cq());
        Finish(recv_cq());
    }

    // Capture each side's enqueue into its submesh's trace (outside the timed region).
    BENCH_LOG("own warmup done; capturing traces");
    MeshTraceId send_tid = BeginTraceCapture(stages[0].get(), /*cq_id=*/0);
    EnqueueMeshWorkload(sender_cq(), send_wl, /*blocking=*/false);
    stages[0]->end_mesh_trace(/*cq_id=*/0, send_tid);
    MeshTraceId recv_tid = BeginTraceCapture(stages[1].get(), /*cq_id=*/0);
    EnqueueMeshWorkload(recv_cq(), recv_wl, /*blocking=*/false);
    stages[1]->end_mesh_trace(/*cq_id=*/0, recv_tid);
    Finish(sender_cq());
    Finish(recv_cq());

    // Timed region: replay only (one replay = the full n_iters on-device loop per side).
    BENCH_LOG("traces captured; starting measured replay");
    auto t0 = std::chrono::high_resolution_clock::now();
    stages[1]->replay_mesh_trace(/*cq_id=*/0, recv_tid, /*blocking=*/false);
    stages[0]->replay_mesh_trace(/*cq_id=*/0, send_tid, /*blocking=*/false);
    Finish(sender_cq());
    Finish(recv_cq());
    auto t1 = std::chrono::high_resolution_clock::now();
    BENCH_LOG("own measured replay done");

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double gbps = (static_cast<double>(n_iters) * tensor_bytes) / (secs * 1e9);
    const int data_ok = verify_data();
    const uint32_t num_workers = core_range_volume(workers);

    // Free the per-cycle traces (submeshes are persistent across rows, so leaking traces
    // would accumulate trace buffers).
    stages[0]->release_mesh_trace(send_tid);
    stages[1]->release_mesh_trace(recv_tid);

    BENCH_LOG(
        "run_throughput RETURN gbps=%.4f data_ok=%d; tearing down service%s",
        gbps,
        data_ok,
        reuse_submesh ? " (submesh reused)" : " + submeshes");
    return {tensor_bytes, gbps, secs * 1e3, data_ok, num_workers};
}

// ===========================================================================
// Google Benchmark entry points
// ===========================================================================
void init_latency_counters(benchmark::State& state) {
    state.counters["num_stages"] = 0;
    state.counters["metadata_bytes"] = 0;
    state.counters["lease"] = 0;
    state.counters["num_workers"] = 0;
    state.counters["total_avg_us"] = 0;
    state.counters["total_p50_us"] = 0;
    state.counters["total_p99_us"] = 0;
    state.counters["per_hop_simple_us"] = 0;  // total_avg / (num_stages - 1)
}

void init_throughput_counters(benchmark::State& state) {
    state.counters["payload_bytes"] = 0;
    state.counters["metadata_bytes"] = 0;
    state.counters["lease"] = 0;
    state.counters["num_workers"] = 0;
    state.counters["throughput_gbps"] = 0;
    state.counters["transfer_ms"] = 0;  // chrono-measured timed transfer region (all n_iters)
}

void BM_D2DStreamLatency(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_latency_counters(state);

    const auto num_stages = static_cast<uint32_t>(state.range(0));
    const auto metadata_bytes = static_cast<uint32_t>(state.range(1));
    const bool lease = state.range(2) != 0;
    auto& fx = get_fixture();

    if (!service_cores_supported()) {
        state.SkipWithMessage("Service cores require Blackhole or UBB Galaxy");
        return;
    }
    if (!h2d_host_pinning_supported()) {
        state.SkipWithMessage("H2D front-end host-DMA pinning requires a DMA-translation IOMMU");
        return;
    }
    if (!enough_devices(*fx.mesh_device, num_stages)) {
        state.SkipWithMessage("Not enough devices to carve one 1x1 submesh per stage");
        return;
    }

    for ([[maybe_unused]] auto _ : state) {
        auto result = run_latency(*fx.mesh_device, num_stages, metadata_bytes, lease);
        auto s = summarize_us(std::move(result.per_iter_us));
        state.counters["num_stages"] = num_stages;
        state.counters["metadata_bytes"] = metadata_bytes;
        state.counters["lease"] = lease ? 1 : 0;
        state.counters["num_workers"] = result.num_workers;
        state.counters["total_avg_us"] = s.avg_us;
        state.counters["total_p50_us"] = s.p50_us;
        state.counters["total_p99_us"] = s.p99_us;
        state.counters["per_hop_simple_us"] = s.avg_us / static_cast<double>(num_stages - 1);
        state.SetLabel(
            std::to_string(num_stages) + "stage md=" + std::to_string(metadata_bytes) + (lease ? " lease" : " own"));
    }
}

void BM_D2DStreamThroughput(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_throughput_counters(state);

    const auto size_index = static_cast<std::size_t>(state.range(0));
    const auto metadata_bytes = static_cast<uint32_t>(state.range(1));
    const bool lease = state.range(2) != 0;
    auto& fx = get_fixture();

    if (!service_cores_supported()) {
        state.SkipWithMessage("Service cores require Blackhole or UBB Galaxy");
        return;
    }
    if (!enough_devices(*fx.mesh_device, 2)) {
        state.SkipWithMessage("Need a 2D mesh with >= 2 devices");
        return;
    }
    if (size_index >= kThroughputShapes.size()) {
        state.SkipWithMessage("size_index out of range");
        return;
    }

    BENCH_LOG("ROW BEGIN size_index=%zu metadata_bytes=%u lease=%d", size_index, metadata_bytes, lease ? 1 : 0);
    for ([[maybe_unused]] auto _ : state) {
        auto r = run_throughput(*fx.mesh_device, kThroughputShapes[size_index], metadata_bytes, lease);
        state.counters["payload_bytes"] = r.tensor_bytes;
        state.counters["metadata_bytes"] = metadata_bytes;
        state.counters["lease"] = lease ? 1 : 0;
        state.counters["num_workers"] = r.num_workers;
        state.counters["throughput_gbps"] = r.gbps;
        state.counters["transfer_ms"] = r.transfer_ms;  // chrono timed transfer region (all n_iters)
        state.counters["data_ok"] = r.data_ok;          // -1 unchecked, 0 FAIL, 1 PASS
        state.SetLabel(
            std::to_string(static_cast<uint64_t>(r.tensor_bytes)) + "B md=" + std::to_string(metadata_bytes) +
            (lease ? " lease" : " own") + (r.data_ok == 0 ? " DATA_FAIL" : ""));
    }
}

// Env-driven SCENARIO runner: execute a controlled SEQUENCE of D2D create_pair ->
// transfer -> teardown cycles in ONE process, to isolate the cross-row hang and build a
// minimum repro. Inert (skips) unless D2D_BENCH_SCENARIO is set.
//   D2D_BENCH_SCENARIO      comma list of metadata_bytes, one per op, e.g. "0,12" or
//                           "0,0,0,12" or "0,12,12,12". Each entry is one service cycle.
//   D2D_BENCH_SCENARIO_SIZE size_index into kThroughputShapes for ALL ops (default 3).
//   D2D_BENCH_SCENARIO_LEASE 0/1 fabric-link mode (default 0 = OWN).
// Watch the [d2d-bench] "SCENARIO op i/N md=M BEGIN/DONE" trace for which op wedges.
// D2D_BENCH_WARMUP / D2D_BENCH_TPUT_ITERS still control per-op iteration counts.
void BM_D2DStreamScenario(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    const char* scenario = std::getenv("D2D_BENCH_SCENARIO");
    if (scenario == nullptr || *scenario == '\0') {
        state.SkipWithMessage("Set D2D_BENCH_SCENARIO=m0,m1,... (per-op metadata_bytes) to run");
        return;
    }
    auto& fx = get_fixture();
    if (!service_cores_supported() || !enough_devices(*fx.mesh_device, 2)) {
        state.SkipWithMessage("Needs Blackhole / UBB Galaxy with >= 2 devices");
        return;
    }

    std::vector<uint32_t> md_seq;
    for (const char* p = scenario; *p != '\0';) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(p, &end, 10);
        if (end == p) {
            ++p;  // skip a stray separator
            continue;
        }
        md_seq.push_back(static_cast<uint32_t>(v));
        p = end;
        while (*p == ',' || *p == ' ') {
            ++p;
        }
    }
    const auto size_index = static_cast<std::size_t>(env_u32("D2D_BENCH_SCENARIO_SIZE", 3));
    const bool lease = env_u32("D2D_BENCH_SCENARIO_LEASE", 0) != 0;
    if (size_index >= kThroughputShapes.size()) {
        state.SkipWithMessage("D2D_BENCH_SCENARIO_SIZE out of range");
        return;
    }
    // Optional per-op size_index list (same length as the md list); else size_index for all.
    std::vector<std::size_t> size_seq;
    if (const char* sizes = std::getenv("D2D_BENCH_SCENARIO_SIZES"); sizes != nullptr && *sizes != '\0') {
        for (const char* p = sizes; *p != '\0';) {
            char* end = nullptr;
            const unsigned long v = std::strtoul(p, &end, 10);
            if (end == p) {
                ++p;
                continue;
            }
            size_seq.push_back(static_cast<std::size_t>(v));
            p = end;
            while (*p == ',' || *p == ' ') {
                ++p;
            }
        }
        if (size_seq.size() != md_seq.size()) {
            state.SkipWithMessage("D2D_BENCH_SCENARIO_SIZES length must match D2D_BENCH_SCENARIO");
            return;
        }
        for (auto s : size_seq) {
            if (s >= kThroughputShapes.size()) {
                state.SkipWithMessage("D2D_BENCH_SCENARIO_SIZES entry out of range");
                return;
            }
        }
    }

    for ([[maybe_unused]] auto _ : state) {
        BENCH_LOG("SCENARIO start: %zu ops, default size_index=%zu lease=%d", md_seq.size(), size_index, lease ? 1 : 0);
        for (std::size_t i = 0; i < md_seq.size(); ++i) {
            const std::size_t si = size_seq.empty() ? size_index : size_seq[i];
            BENCH_LOG("SCENARIO op %zu/%zu md=%u size_index=%zu BEGIN", i + 1, md_seq.size(), md_seq[i], si);
            auto r = run_throughput(*fx.mesh_device, kThroughputShapes[si], md_seq[i], lease);
            BENCH_LOG(
                "SCENARIO op %zu/%zu md=%u size_index=%zu DONE gbps=%.3f data_ok=%d",
                i + 1,
                md_seq.size(),
                md_seq[i],
                si,
                r.gbps,
                r.data_ok);
        }
        BENCH_LOG("SCENARIO complete (%zu ops)", md_seq.size());
        state.counters["ops"] = static_cast<double>(md_seq.size());
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed

using tt::tt_metal::distributed::BM_D2DStreamLatency;
using tt::tt_metal::distributed::BM_D2DStreamScenario;
using tt::tt_metal::distributed::BM_D2DStreamThroughput;
using tt::tt_metal::distributed::kMetadataTripleBytes;
using tt::tt_metal::distributed::kThroughputShapes;

BENCHMARK(BM_D2DStreamLatency)
    ->ArgsProduct({
        {2, 4, 8},                                        // num_stages
        {0, static_cast<int64_t>(kMetadataTripleBytes)},  // metadata_bytes
        {0, 1},                                           // lease (0 = OWN, 1 = LEASE)
    })
    ->ArgNames({"num_stages", "metadata_bytes", "lease"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_D2DStreamThroughput)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, 4, 1),             // size_index into kThroughputShapes
        {0, static_cast<int64_t>(kMetadataTripleBytes)},  // metadata_bytes
        {0, 1},                                           // lease (0 = OWN, 1 = LEASE)
    })
    ->ArgNames({"size_index", "metadata_bytes", "lease"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_D2DStreamScenario)->Arg(0)->UseRealTime()->Iterations(1)->Unit(benchmark::kMillisecond);
