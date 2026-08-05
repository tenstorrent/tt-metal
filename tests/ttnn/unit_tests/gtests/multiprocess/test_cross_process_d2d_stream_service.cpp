// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-process (multi-host) D2DStreamService forward-pipeline test.
//
// A forward pipeline of stages, one process per stage, derived entirely from the
// DistributedContext with no hardcoded rank count and no assumptions about mesh
// shape/size. The launch (tt-run --rank-binding + mesh-graph descriptor) decides how
// many meshes/processes exist and how they are fabric-chained; the test just adapts
// (GTEST_SKIP if < 2 ranks).
//
//   rank 0 (stage 0)        rank i (0<i<N-1)        rank N-1
//   ┌──────────────┐      ┌────────────────┐      ┌──────────┐
//   │ H2D feed ->  │ D2D  │ relay (recv +  │ D2D  │ consumer │
//   │ relay        │ ───► │  forward)      │─ … ─►│ + verify │
//   └──────────────┘      └────────────────┘      └──────────┘
//
// Stage 0 (rank 0) owns the H2DStreamService and feeds it from a SEPARATE THREAD that
// streams host tokens concurrently with its device loop. That decouples the push from
// the per-iter device loop (no in-loop push, no host<->device lock-step), so stage 0's
// device loop is uniform with every other stage.
//
// In case a black-box op can't spin a semaphore before it runs, the
// overwrite-gate is a SEPARATE op — `d2d_sync` — that waits the sender's
// consumed_sem. So each stage's per-iter sequence is the production cascade shape:
//
//     [d2d_sync GATE: wait outbound consumed_sem]      // separate op; skip iter 0
//     op  (reads inbound, produces outbound; owns data_ready / consumed_counter incs)
//     wait both links → (ccl) → release both links     // outbound grant = this fwd,
//                                                       // inbound grant  = next recv
//
// The lease (wait/release_fabric_links) is the service↔graph fabric-link arbitration,
// NOT the data handshake. Both endpoints are released TOGETHER after the op, mirroring
// the steady-state pipeline: the outbound grant forwards THIS iter's output while the
// inbound grant sets up the NEXT iter's receive (one phase ahead). That pipelining needs
// a priming inbound release before the loop (fill) and skips the final inbound release
// (drain). (A CCL is just another op; it syncs its own fabric use.)
//
// Stage 0 drains its H2D backing into the outbound D2D sender backing. Every stage,
// including stage 0, runs the SAME gate-free `pipeline_relay_worker`: its consumer half
// is upstream-agnostic (an H2DStreamService and a D2DStreamServiceReceiver expose the
// identical data_ready-sem + consumed-counter handshake), so stage 0 just points it at
// the H2D service instead of a D2D receiver. The relay bundles only the natural inbound
// input-wait + both signals, no outbound gate; the outbound overwrite-gate is the
// standalone `d2d_sync`, uniform across all producing stages. All kernels by repo-
// relative path.
//
// Construction ordering is load-bearing: every rank builds its INBOUND receiver
// before its OUTBOUND sender, so the per-endpoint MeshSocket handshakes cascade
// 0↔1, 1↔2, … without deadlock.
//
// Fixture: MeshDeviceExaboxFixture — each rank opens its local mesh_device_ on a
// UBB Galaxy where world_size == number of meshes in the mesh-graph descriptor.
//
// Launch (no CI yet):
//   tt-run --rank-binding <rank_binding.yaml> --mpi-args "--allow-run-as-root --tag-output"
//       ./build/test/ttnn/multiprocess/unit_tests_cross_process_d2d

#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"

#include <ttnn/api/ttnn/distributed/distributed_configs.hpp>
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/services/d2h_socket_service.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::distributed::test {
namespace {

using ::tt::CBIndex;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CircularBufferConfig;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::CreateCircularBuffer;
using ::tt::tt_metal::CreateKernel;
using ::tt::tt_metal::CreateProgram;
using ::tt::tt_metal::D2HStreamService;
using ::tt::tt_metal::DataMovementConfig;
using ::tt::tt_metal::DataMovementProcessor;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::GlobalSemaphore;
using ::tt::tt_metal::H2DStreamService;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::NOC;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::SetRuntimeArgs;
using ::tt::tt_metal::TensorAccessorArgs;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::distributed::EnqueueMeshWorkload;
using ::tt::tt_metal::distributed::Finish;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshDevice;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ::tt::tt_metal::distributed::MeshWorkload;
using ::tt::tt_metal::distributed::SocketMemoryConfig;
using ::tt::tt_metal::distributed::Synchronize;
using ::tt::tt_metal::distributed::multihost::DistributedContext;
using ::tt::tt_metal::distributed::multihost::Rank;
using ttnn::D2DEndpointConfig;
using ttnn::D2DStreamConfig;
using ttnn::D2DStreamService;
using ttnn::D2DStreamServiceReceiver;
using ttnn::D2DStreamServiceSender;

// Each rank owns its local mesh; world_size == number of meshes in the descriptor.
using CrossProcessD2DFixture = tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture;

const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};  // single worker core
// ForwardChainStress uses a multi-core grid so the num_workers machinery is actually
// exercised (consumed_sem mcast to N cores, data_ready reaching N acks, page split).
const CoreRange kStressWorkerCores{CoreCoord{0, 0}, CoreCoord{1, 1}};  // 2x2 = 4 worker cores
constexpr uint32_t kStressIncsPerConn = 4u;  // atomic-inc packets sent per fabric connection per iter
constexpr uint32_t kFillBase = 1u;  // stage-0 source iota base; end stage verifies kFillBase + iter
// ForwardChainStress carries a per-ITERATION random increment THROUGH the metadata
// path instead of a kernel arg: stage 0 injects metadata[i] as H2D metadata, each
// producing stage reads it as its mutation delta + propagates it to its outbound D2D
// sender, and the end stage (a) verifies the iota shifted by world_size * delta and
// (b) reads the metadata blob back from its worker L1 and checks the exact value. The
// increments are generated deterministically from a fixed seed, so every rank (the
// stage-0 feeder thread and the end-stage verifier on another rank) reproduces the
// SAME array with no inter-rank exchange. metadata_size_bytes == 0 disables it (the CT
// fill_delta fallback applies; used by the non-stress ForwardChain test).
constexpr uint32_t kFillDelta = 1u;  // CT fallback increment when metadata disabled
constexpr uint32_t kStressMetadataBytes = sizeof(uint32_t);
constexpr uint32_t kStressMetadataSeed = 0x5eed1234u;

// --- small helpers (mirrors test_d2d_stream_service.cpp / test_stream_pipeline.cpp) ---

ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(const MeshDevice& mesh) {
    return ttsl::SmallVector<MeshMapperConfig::Placement>(mesh.shape().dims(), MeshMapperConfig::Replicate{});
}

uint32_t core_range_volume(const CoreRange& cr) {
    return (cr.end_coord.x - cr.start_coord.x + 1) * (cr.end_coord.y - cr.start_coord.y + 1);
}

uint32_t worker_index(const CoreCoord& wc, const CoreRange& worker_cores) {
    const uint32_t width = worker_cores.end_coord.x - worker_cores.start_coord.x + 1;
    return (wc.y - worker_cores.start_coord.y) * width + (wc.x - worker_cores.start_coord.x);
}

std::pair<uint32_t, uint32_t> worker_page_range(uint32_t worker_idx, uint32_t num_workers, uint32_t num_pages) {
    const uint32_t base = num_pages / num_workers;
    const uint32_t rem = num_pages % num_workers;
    const uint32_t start = worker_idx * base + std::min(worker_idx, rem);
    const uint32_t end = start + base + (worker_idx < rem ? 1u : 0u);
    return {start, end};
}

std::vector<uint32_t> make_iota_u32(size_t n, uint32_t base) {
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = base + static_cast<uint32_t>(i);
    }
    return v;
}

// One pseudo-random increment per iteration in [1, 64], generated from a fixed seed by
// a pure-uint32 LCG (Numerical Recipes). Deterministic and platform-independent, so
// every rank reproduces the identical array: the stage-0 feeder thread pushes
// seeds[i] as iter i's metadata, and the end-stage verifier (a different rank) uses the
// same seeds[last] to predict the shifted iota + check the metadata readback — no
// cross-rank exchange. Range [1, 64] keeps world_size * delta well clear of overflow
// and avoids 0 (so every stage's mutation is observable).
std::vector<uint32_t> make_metadata_seeds(uint32_t num_iters, uint32_t seed) {
    std::vector<uint32_t> v(num_iters);
    uint32_t s = seed;
    for (uint32_t i = 0; i < num_iters; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = 1u + ((s >> 24) % 64u);
    }
    return v;
}

// Fresh config per call (the mapper is moved into each create_* call; a middle rank
// builds two services so it needs two mappers). UINT32 ROW_MAJOR DRAM, replicated on
// the local mesh, L1 socket FIFO, LEASE mode (the host drives the per-transfer grant).
D2DStreamConfig make_cfg(
    const std::shared_ptr<MeshDevice>& mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores = kWorkerCores,
    uint32_t metadata_size_bytes = 0) {
    const tt::tt_metal::TensorSpec global_spec(
        global_shape,
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}));
    return D2DStreamConfig{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh, MeshMapperConfig{.placements = replicate_all(*mesh)}),
        .socket_mem_config = SocketMemoryConfig{BufferType::L1, /*fifo_size=*/4096u},
        .sender_worker_cores = worker_cores,
        .receiver_worker_cores = worker_cores,
        .metadata_size_bytes = metadata_size_bytes,
        .share_fabric_links = true,  // LEASE mode (host-driven per-transfer grants)
    };
}

// --- stage-0 host feed (in-process H2DStreamService) ---

// H2D socket FIFO / scratch sizing: one tensor page rounded up to a 4 KB floor
// (mirrors fifo_bytes_for in test_d2d_stream_service.cpp; H2D streams page-by-page,
// so a page-sized FIFO is correct and keeps L1 use bounded).
uint32_t fifo_bytes_for(const tt::tt_metal::TensorSpec& spec) {
    constexpr uint32_t kMinFifo = 4096u;
    const uint32_t page = static_cast<uint32_t>(spec.compute_page_size_bytes());
    return std::max(kMinFifo, ((page + kMinFifo - 1u) / kMinFifo) * kMinFifo);
}

// Stage 0's real host->device source: an in-process H2DStreamService on the local
// mesh. Unlike the cross-process D2D MeshSocket, this needs NO rendezvous — the host
// owns the data and drives forward_to_tensor itself. Replicated on the mesh, L1
// socket FIFO, worker-sync handshake on kWorkerCores so the stage-0 relay can drain
// it (same data_ready-sem + consumed-counter shape as a D2D receiver). Metadata off.
std::unique_ptr<H2DStreamService> make_h2d_service(
    const std::shared_ptr<MeshDevice>& mesh,
    const tt::tt_metal::TensorSpec& global_spec,
    const CoreRange& worker_cores = kWorkerCores,
    uint32_t metadata_size_bytes = 0) {
    const uint32_t fifo_bytes = fifo_bytes_for(global_spec);
    H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh, MeshMapperConfig{.placements = replicate_all(*mesh)}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = fifo_bytes,
        .max_socket_page_size_bytes = fifo_bytes,
        .worker_cores = worker_cores,
        .metadata_size_bytes = metadata_size_bytes,
    };
    return std::make_unique<H2DStreamService>(mesh, std::move(cfg));
}

// Push one iter's iota token (global element i = base + i) into the H2D service. The
// mapper replicates it to every shard; the stage-0 relay then copies it verbatim into
// the outbound D2D backing. forward_to_tensor returns once the bytes are in the socket
// FIFOs; the caller's barrier() blocks until the device kernel drained them.
void h2d_push_token(
    H2DStreamService& h2d_service, uint32_t num_u32, uint32_t base, uint32_t metadata_size_bytes, uint32_t increment) {
    const std::vector<uint32_t> token = make_iota_u32(num_u32, base);
    const auto bytes =
        ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(token.data()), token.size() * sizeof(uint32_t));
    // Inject the per-stage increment as the metadata blob (1 word) when enabled; it
    // rides each token to stage 0's workers and is propagated down the chain.
    const std::vector<uint32_t> meta_words = {increment};
    const auto meta =
        ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(meta_words.data()), metadata_size_bytes);
    h2d_service.forward_to_tensor(bytes, meta);
}

// Body of stage 0's feeder THREAD: streams num_iters iota tokens into stage 0's own
// (in-process) H2D service, concurrently with the main thread's per-iter device loop.
// This is what decouples the host feed from the device loop — the push runs ahead,
// flow-controlled by the page-sized FIFO + the service<->relay handshake (it blocks when
// the FIFO fills; the relay drains one token per iter), so stage 0's device loop is
// uniform with every other stage (no in-loop push).
//
void run_h2d_feed_loop(
    H2DStreamService& h2d_service,
    const ttnn::Shape& global_shape,
    uint32_t num_iters,
    uint32_t metadata_size_bytes = 0,
    const std::vector<uint32_t>& increments = {}) {
    const uint32_t num_u32 = static_cast<uint32_t>(global_shape.volume());
    for (uint32_t i = 0; i < num_iters; ++i) {
        // Per-iter metadata increment (seeds[i]); 0 when metadata is disabled / unset.
        const uint32_t increment = i < increments.size() ? increments[i] : 0u;
        h2d_push_token(h2d_service, num_u32, kFillBase + i, metadata_size_bytes, increment);
    }
    h2d_service.barrier();  // flush every push to the device
}

// --- per-stage workloads (one program per coord; reuse kernels by repo path) ---

// Generic relay op (pipeline_relay_worker) for EVERY stage. Copies the upstream
// backing into a downstream dest, spinning the upstream data_ready_sem (natural
// input-wait, bundled) and incing the upstream consumed_counter + (if producing) the
// outbound data_ready_counter — the op OWNS those signals. NO outbound gate (that's
// d2d_sync). The upstream is templated because the consumer-side handshake is
// identical for an H2DStreamService (stage 0's host-fed source) and a
// D2DStreamServiceReceiver (middle/last stages' fabric inbound): both expose
// get_backing_tensor / get_worker_cores / get_data_ready_sem_addr /
// get_consumed_counter_addr / get_service_core. produce=true → dest is the outbound
// sender backing; produce=false → dest is `output`.
template <typename Upstream>
MeshWorkload make_relay_like_workload(
    Upstream* inbound,
    const std::shared_ptr<MeshDevice>& mesh,
    uint32_t dest_addr,
    bool produce,
    D2DStreamServiceSender* outbound /* null when !produce */) {
    const auto& coords = inbound->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* up_buf = inbound->get_backing_tensor().buffer();
    const uint32_t page_size = up_buf->aligned_page_size();
    const uint32_t num_pages = up_buf->num_pages();
    const CoreRange worker_cores = inbound->get_worker_cores();
    const uint32_t num_workers = core_range_volume(worker_cores);
    constexpr auto kScratchCb = CBIndex::c_0;

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        // Inbound backing and downstream dest share the per-shard spec.
        const auto* up_dbuf = inbound->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*up_dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(inbound->get_data_ready_sem_addr()),
            static_cast<uint32_t>(up_buf->address()),
            dest_addr,
            page_size,
            /*num_iters=*/1u,
            static_cast<uint32_t>(kScratchCb),
            produce ? 1u : 0u,
            0,  // metadata_enabled == false
            0,  // metadata_size_bytes == 0
            0,  // inbound_metadata_l1_addr == 0
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/pipeline_relay_worker.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh->get_device(coord);
        const auto up_svc_phys = device->worker_core_from_logical_core(inbound->get_service_core(coord));
        CoreCoord down_svc_phys{0, 0};
        uint32_t down_counter_addr = 0;
        if (produce) {
            down_svc_phys = device->worker_core_from_logical_core(outbound->get_service_core(coord));
            down_counter_addr = static_cast<uint32_t>(outbound->get_data_ready_counter_addr(coord));
        }
        for (const auto& wc : worker_cores) {
            const auto [start_page, end_page] =
                worker_page_range(worker_index(wc, worker_cores), num_workers, num_pages);
            const std::vector<uint32_t> rt_args = {
                start_page,
                end_page,
                static_cast<uint32_t>(inbound->get_consumed_counter_addr(coord)),
                static_cast<uint32_t>(up_svc_phys.x),
                static_cast<uint32_t>(up_svc_phys.y),
                down_counter_addr,
                static_cast<uint32_t>(down_svc_phys.x),
                static_cast<uint32_t>(down_svc_phys.y),
                0,  // is_metadata_writer == 0
                0,  // downstream_metadata_l1_addr == 0
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// Terminal-stage (D2H) relay op (pipeline_relay_d2h_worker) for the LAST stage only.
// The consumer half is identical to make_relay_like_workload — copies the upstream
// backing into a dest, spinning the upstream data_ready_sem and incing the upstream
// consumed_counter — but the producer half targets a D2HStreamService instead of a
// D2DStreamServiceSender: dest is the D2H backing tensor, and instead of bumping a
// downstream data_ready_counter the worker bumps the D2H sender's write_ack_counter,
// so the result streams to a host consumer over a PCIe socket (read_from_tensor)
// rather than forwarding over Fast Dispatch Command Queue. The handshake is the
// D2H-inverted analog of the D2D one: the persistent D2H sender mcasts
// transfer_done_sem (drained prev iter), the worker incs write_ack_counter
// (this iter staged) — vs the D2D data_ready / consumed pair. Upstream stays
// templated to mirror the relay helper, but in practice it's only ever a
// D2DStreamServiceReceiver (the terminal stage always has a fabric inbound).
template <typename Upstream>
MeshWorkload make_d2h_relay_workload(
    Upstream* inbound,
    const std::shared_ptr<MeshDevice>& mesh,
    D2HStreamService* d2h_service) {
    const auto& coords = inbound->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* up_buf = inbound->get_backing_tensor().buffer();
    const uint32_t page_size = up_buf->aligned_page_size();
    const uint32_t num_pages = up_buf->num_pages();
    const CoreRange worker_cores = inbound->get_worker_cores();
    const uint32_t num_workers = core_range_volume(worker_cores);
    constexpr auto kScratchCb = CBIndex::c_0;

    const uint32_t dest_addr = static_cast<uint32_t>(d2h_service->get_backing_tensor().buffer()->address());
    const uint32_t transfer_done_sem_addr = static_cast<uint32_t>(d2h_service->get_transfer_done_sem_addr());

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        // Inbound backing and downstream dest share the per-shard spec.
        const auto* up_dbuf = inbound->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*up_dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(inbound->get_data_ready_sem_addr()),
            static_cast<uint32_t>(up_buf->address()),
            dest_addr,
            page_size,
            /*num_iters=*/1u,
            static_cast<uint32_t>(kScratchCb),
            0,  // metadata_enabled == false
            0,  // metadata_size_bytes == 0
            0,  // inbound_metadata_l1_addr == 0,
            transfer_done_sem_addr};
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/pipeline_relay_d2h_worker.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh->get_device(coord);
        const auto up_svc_phys = device->worker_core_from_logical_core(inbound->get_service_core(coord));

        const auto d2h_svc_phys = device->worker_core_from_logical_core(d2h_service->get_service_core(coord));
        const uint32_t write_ack_addr = static_cast<uint32_t>(d2h_service->get_write_ack_counter_addr(coord));

        for (const auto& wc : worker_cores) {
            const auto [start_page, end_page] =
                worker_page_range(worker_index(wc, worker_cores), num_workers, num_pages);
            const std::vector<uint32_t> rt_args = {
                start_page,
                end_page,
                static_cast<uint32_t>(inbound->get_consumed_counter_addr(coord)),
                static_cast<uint32_t>(up_svc_phys.x),
                static_cast<uint32_t>(up_svc_phys.y),
                write_ack_addr,
                static_cast<uint32_t>(d2h_svc_phys.x),
                static_cast<uint32_t>(d2h_svc_phys.y),
                0,  // is_metadata_writer == 0
                0,  // d2h_metadata_input_addr == 0
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}
// ForwardChainStress COMPUTE pass (d2d_stress_relay, STRESS_MODE=0): like the relay above
// but (1) mutates every element by +1 so the end value tracks every hop, (2) FUSES the
// overwrite-gate (waits the outbound consumed_sem unless skip_gate, or no gate at all on
// the last stage), and (3) does NOT signal data_ready — that's the separate SIGNAL pass
// after the lease release. Multi-core: pages split across the worker grid. Upstream-
// agnostic (H2D on stage 0, D2D receiver elsewhere); outbound_for_gate is the sender
// whose consumed_sem gates the overwrite (null on the last stage).
template <typename Upstream>
MeshWorkload make_stress_compute_workload(
    Upstream* inbound,
    const std::shared_ptr<MeshDevice>& mesh,
    uint32_t dest_addr,
    D2DStreamServiceSender* outbound_for_gate,
    uint32_t skip_gate,
    uint32_t metadata_size_bytes = 0) {
    const auto& coords = inbound->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* up_buf = inbound->get_backing_tensor().buffer();
    const uint32_t page_size = up_buf->aligned_page_size();
    const uint32_t num_pages = up_buf->num_pages();
    const CoreRange worker_cores = inbound->get_worker_cores();
    const uint32_t num_workers = core_range_volume(worker_cores);
    constexpr auto kScratchCb = CBIndex::c_0;
    const uint32_t has_gate = outbound_for_gate != nullptr ? 1u : 0u;
    const uint32_t consumed_sem_addr =
        has_gate ? static_cast<uint32_t>(outbound_for_gate->get_consumed_sem_addr()) : 0u;
    // Metadata path: read the per-stage increment from the blob the inbound service
    // mcast into the worker grid (uniform L1 addr), and — when producing — the
    // designated core propagates it to the outbound D2D sender's metadata buffer.
    const bool metadata_enabled = metadata_size_bytes > 0;
    const uint32_t inbound_metadata_l1_addr =
        metadata_enabled ? static_cast<uint32_t>(inbound->get_metadata_addr()) : 0u;

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        const auto* up_dbuf = inbound->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*up_dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            has_gate,
            consumed_sem_addr,
            static_cast<uint32_t>(inbound->get_data_ready_sem_addr()),
            static_cast<uint32_t>(up_buf->address()),
            dest_addr,
            page_size,
            static_cast<uint32_t>(kScratchCb),
            kFillDelta,  // fallback increment when metadata disabled
            metadata_enabled ? 1u : 0u,
            metadata_size_bytes,
            inbound_metadata_l1_addr,
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_stress_relay.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = ct_args,
                .defines = {{"STRESS_MODE", "0"}}});

        auto* device = mesh->get_device(coord);
        const auto up_svc_phys = device->worker_core_from_logical_core(inbound->get_service_core(coord));
        // Outbound D2D sender's metadata buffer (service-core L1) + its physical NoC —
        // the propagate target, producing stages only.
        CoreCoord down_svc_phys{0, 0};
        uint32_t outbound_metadata_addr = 0;
        if (has_gate && metadata_enabled) {
            down_svc_phys = device->worker_core_from_logical_core(outbound_for_gate->get_service_core(coord));
            outbound_metadata_addr = static_cast<uint32_t>(outbound_for_gate->get_metadata_addr(coord));
        }
        for (const auto& wc : worker_cores) {
            const auto [start_page, end_page] =
                worker_page_range(worker_index(wc, worker_cores), num_workers, num_pages);
            // A single designated producing core forwards the metadata blob downstream.
            const uint32_t is_metadata_writer =
                (metadata_enabled && has_gate && wc == worker_cores.end_coord) ? 1u : 0u;
            const std::vector<uint32_t> rt_args = {
                start_page,
                end_page,
                static_cast<uint32_t>(inbound->get_consumed_counter_addr(coord)),
                static_cast<uint32_t>(up_svc_phys.x),
                static_cast<uint32_t>(up_svc_phys.y),
                skip_gate,
                is_metadata_writer,
                outbound_metadata_addr,
                static_cast<uint32_t>(down_svc_phys.x),
                static_cast<uint32_t>(down_svc_phys.y),
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// ForwardChainStress terminal-stage COMPUTE (d2d_stress_relay_d2h, STRESS_MODE=0): the
// last stage's producing op. Mirrors make_stress_compute_workload (per-stage +delta
// mutation, fused overwrite-gate, metadata forward) but streams to host via
// D2HStreamService instead of forwarding to a D2D sender: dest is the D2H backing, the
// gate waits the D2H transfer_done_sem (not an outbound consumed_sem), and the worker
// bumps the D2H write_ack_counter instead of a downstream data_ready — so there is no
// separate SIGNAL pass.
template <typename Upstream>
MeshWorkload make_stress_d2h_compute_workload(
    Upstream* inbound,
    const std::shared_ptr<MeshDevice>& mesh,
    D2HStreamService* d2h_service,
    uint32_t skip_gate,
    uint32_t metadata_size_bytes = 0) {
    const auto& coords = inbound->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* up_buf = inbound->get_backing_tensor().buffer();
    const uint32_t page_size = up_buf->aligned_page_size();
    const uint32_t num_pages = up_buf->num_pages();
    const CoreRange worker_cores = inbound->get_worker_cores();
    const uint32_t num_workers = core_range_volume(worker_cores);
    constexpr auto kScratchCb = CBIndex::c_0;

    const uint32_t dest_addr = static_cast<uint32_t>(d2h_service->get_backing_tensor().buffer()->address());
    const uint32_t transfer_done_sem_addr = static_cast<uint32_t>(d2h_service->get_transfer_done_sem_addr());

    // Metadata path: read the per-stage increment from the blob the inbound service
    // mcast into the worker grid (uniform L1 addr); the designated core forwards it to
    // the D2H service core's metadata input buffer, so the sender ships it to host inline.
    const bool metadata_enabled = metadata_size_bytes > 0;
    const uint32_t inbound_metadata_l1_addr =
        metadata_enabled ? static_cast<uint32_t>(inbound->get_metadata_addr()) : 0u;

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        const auto* up_dbuf = inbound->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*up_dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(inbound->get_data_ready_sem_addr()),
            static_cast<uint32_t>(up_buf->address()),
            dest_addr,
            page_size,
            static_cast<uint32_t>(kScratchCb),
            kFillDelta,  // fallback increment when metadata disabled
            metadata_enabled ? 1u : 0u,
            metadata_size_bytes,
            inbound_metadata_l1_addr,
            transfer_done_sem_addr,
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_stress_relay_d2h.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = ct_args,
                .defines = {{"STRESS_MODE", "0"}}});

        auto* device = mesh->get_device(coord);
        const auto up_svc_phys = device->worker_core_from_logical_core(inbound->get_service_core(coord));
        const auto d2h_svc_phys = device->worker_core_from_logical_core(d2h_service->get_service_core(coord));
        const uint32_t write_ack_addr = static_cast<uint32_t>(d2h_service->get_write_ack_counter_addr(coord));
        const uint32_t d2h_metadata_input_addr =
            metadata_enabled ? static_cast<uint32_t>(d2h_service->get_metadata_input_addr(coord)) : 0u;
        for (const auto& wc : worker_cores) {
            const auto [start_page, end_page] =
                worker_page_range(worker_index(wc, worker_cores), num_workers, num_pages);
            // Single designated core forwards the metadata blob to the D2H service core.
            const uint32_t is_metadata_writer = (metadata_enabled && wc == worker_cores.end_coord) ? 1u : 0u;
            const std::vector<uint32_t> rt_args = {
                start_page,
                end_page,
                static_cast<uint32_t>(inbound->get_consumed_counter_addr(coord)),
                static_cast<uint32_t>(up_svc_phys.x),
                static_cast<uint32_t>(up_svc_phys.y),
                skip_gate,
                is_metadata_writer,
                d2h_metadata_input_addr,
                static_cast<uint32_t>(d2h_svc_phys.x),
                static_cast<uint32_t>(d2h_svc_phys.y),
                write_ack_addr,
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// ForwardChainStress SIGNAL pass (d2d_stress_relay, STRESS_MODE=1): every worker core
// atomic-incs the outbound data_ready_counter -> num_workers acks -> the sender forwards
// this iter's output. Run AFTER the lease release so the forward fires last.
MeshWorkload make_stress_signal_workload(D2DStreamServiceSender* outbound, const std::shared_ptr<MeshDevice>& mesh) {
    const auto& coords = outbound->get_backing_tensor().tensor_topology().mesh_coords();
    const CoreRange worker_cores = outbound->get_worker_cores();

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_stress_relay.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = {{"STRESS_MODE", "1"}}});

        auto* device = mesh->get_device(coord);
        const auto down_svc_phys = device->worker_core_from_logical_core(outbound->get_service_core(coord));
        const uint32_t down_counter_addr = static_cast<uint32_t>(outbound->get_data_ready_counter_addr(coord));
        for (const auto& wc : worker_cores) {
            const std::vector<uint32_t> rt_args = {
                down_counter_addr,
                static_cast<uint32_t>(down_svc_phys.x),
                static_cast<uint32_t>(down_svc_phys.y),
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// The standalone outbound overwrite-gate (d2d_sync): runs on the sender's worker
// grid and waits the sender's consumed_sem (prev iter forwarded). Separate from the
// op — the unbundled split that mirrors a real graph.
MeshWorkload make_gate_workload(
    D2DStreamServiceSender* sender, [[maybe_unused]] const std::shared_ptr<MeshDevice>& mesh) {
    const auto& coords = sender->get_backing_tensor().tensor_topology().mesh_coords();
    const CoreRange worker_cores = sender->get_worker_cores();
    const std::vector<uint32_t> ct_args = {static_cast<uint32_t>(sender->get_consumed_sem_addr())};

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_sync.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// Build a stub competing fabric op for the outbound boundary: opens (and closes) a
// WorkerToFabricEdmSender on the SAME link the D2D sender service uses, so the lease
// arbitrates real contention on that EDM channel.
MeshWorkload make_stub_fabric_workload(
    D2DStreamServiceSender* sender, const std::shared_ptr<MeshDevice>& mesh, Rank downstream_rank) {
    const auto& coords = sender->get_backing_tensor().tensor_topology().mesh_coords();
    // The competing op models a SINGLE model-graph fabric op (e.g. one row CCL) contending
    // for the EDM sender channel, so it runs on exactly ONE worker core.
    const CoreCoord stub_core = sender->get_worker_cores().start_coord;
    const CoreRange stub_core_range{stub_core, stub_core};

    // Resolve the downstream mesh's (mesh_id, host_rank) from the global rank bindings,
    // mirroring MeshSocket's resolve_fabric_node_id_from_rank.
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& global_bindings = control_plane.get_global_logical_bindings();
    auto down_it = global_bindings.find(downstream_rank);
    TT_FATAL(
        down_it != global_bindings.end(), "stub fabric op: no global binding for downstream rank {}", *downstream_rank);
    const auto down_mesh_id = std::get<0>(down_it->second);
    const std::optional<tt::tt_fabric::MeshHostRankId> down_host_rank = std::get<1>(down_it->second);

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        const auto sender_node = mesh->get_fabric_node_id(coord);
        const tt::tt_fabric::FabricNodeId downstream_node(
            down_mesh_id, mesh_graph.coordinate_to_chip(down_mesh_id, coord, down_host_rank));
        const auto links = tt::tt_fabric::get_forwarding_link_indices(sender_node, downstream_node);
        TT_FATAL(!links.empty(), "stub fabric op: no fabric link sender->downstream at coord {}", coord);

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_stub_fabric_op.cpp",
            stub_core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> rt_args;
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_node, downstream_node, links.front(), program, stub_core, rt_args);
        SetRuntimeArgs(program, kernel, stub_core, rt_args);
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// ForwardChainStress competing fabric op (d2d_stress_fabric): on ONE core per chip, opens
// a connection on EVERY link toward EVERY intra-mesh neighbor (so the D2D sender's
// first-hop link is necessarily among them on interior chips -> the lease serializes the
// overlap) and sends incs_per_conn atomic-incs to each neighbor's fabric-test
// GlobalSemaphore. Each chip then spins (in-kernel) on its own GlobalSemaphore to the
// cumulative target -> a dropped inc hangs that launch's Finish. Intra-mesh only (every
// chip of the rank runs it in the same launch), so the exchange is deadlock-free. Runs on
// every stage; bracketed by the loop's wait/release of whatever leases the stage holds.
MeshWorkload make_stress_fabric_workload(
    const std::shared_ptr<MeshDevice>& mesh,
    const CoreRange& worker_cores,
    const GlobalSemaphore& fabric_sem,
    uint32_t incs_per_conn) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const CoreCoord fabric_core = worker_cores.start_coord;
    const CoreRange fabric_core_range{fabric_core, fabric_core};
    const uint32_t sem_addr = static_cast<uint32_t>(fabric_sem.address());

    MeshWorkload workload;
    for (const auto& coord : MeshCoordinateRange(mesh->shape())) {
        auto program = CreateProgram();
        auto* device = mesh->get_device(coord);
        const auto sender_node = mesh->get_fabric_node_id(coord);
        const auto fabric_core_phys = device->worker_core_from_logical_core(fabric_core);

        // Every link toward every intra-mesh neighbor, one connection each. Key subtlety:
        // get_intra_chip_neighbors returns the neighbor LOGICAL chip id once PER LINK in that
        // direction, so the span repeats the same chip (e.g. [D1, D1] for 2 links). Take it
        // once (front) and enumerate the links via get_forwarding_link_indices — iterating
        // BOTH the span and the link list double-counts (2 links * 2 span entries = 4 dupes
        // per direction), which is what blew the semaphore budget. neighbor FabricNodeId is
        // FabricNodeId(this mesh_id, chip) (mirrors fabric.cpp). Worst case is 4 dirs * 2
        // links = 8 connections; append_fabric_connection_rt_args allocates 2 WORKER
        // semaphores per connection, so 16 total = exactly the per-core cap. Opening both
        // links (not just link 0) means we hold whichever link the D2D sender forwards on,
        // so the lease contention is guaranteed regardless of the sender's link choice.
        std::vector<std::pair<tt::tt_fabric::FabricNodeId, uint32_t>> conns;
        for (auto dir :
             {tt::tt_fabric::RoutingDirection::N,
              tt::tt_fabric::RoutingDirection::E,
              tt::tt_fabric::RoutingDirection::S,
              tt::tt_fabric::RoutingDirection::W}) {
            const auto neighbors = control_plane.get_intra_chip_neighbors(sender_node, dir);
            if (neighbors.empty()) {
                continue;
            }
            const tt::tt_fabric::FabricNodeId neighbor_node(sender_node.mesh_id, neighbors.front());
            for (auto link : tt::tt_fabric::get_forwarding_link_indices(sender_node, neighbor_node)) {
                conns.emplace_back(neighbor_node, link);
            }
        }
        const uint32_t num_connections = static_cast<uint32_t>(conns.size());

        // Symmetric mesh: the neighbor relation is mutual (if A's E points to B over a link,
        // B's W points to A over a link), so a chip's incoming connection count equals its
        // outgoing one. Its per-iter spin target is therefore num_connections * incs_per_conn.
        // The kernel resets the sem at the end of each iter, so this is per-iter, not cumulative.
        const uint32_t expected_per_iter = num_connections * incs_per_conn;

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_stress_fabric.cpp",
            fabric_core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> rt = {
            num_connections,
            sem_addr,
            static_cast<uint32_t>(fabric_core_phys.x),
            static_cast<uint32_t>(fabric_core_phys.y),
            incs_per_conn,
            expected_per_iter,
        };
        for (const auto& [neighbor_node, link] : conns) {
            tt::tt_fabric::append_fabric_connection_rt_args(sender_node, neighbor_node, link, program, fabric_core, rt);
            rt.push_back(static_cast<uint32_t>(neighbor_node.chip_id));
            rt.push_back(static_cast<uint32_t>(*neighbor_node.mesh_id));
        }
        SetRuntimeArgs(program, kernel, fabric_core, rt);
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// World-size-derived forward pipeline (rank == stage), unbundled op + d2d_sync gate,
// LEASE mode with one release per iter after the op. Stage 0 (rank 0) owns the H2D feed
// and pushes it from a SEPARATE THREAD, concurrently with its device loop, so the push
// is decoupled and every stage's device loop is uniform. Hangs loudly (a gate spin, a
// MeshSocket handshake, or a Finish never returns) on any desync; the end stage's
// per-coord readback catches stale/dropped data.
TEST_F(CrossProcessD2DFixture, ForwardChain) {
    const auto& ctx = DistributedContext::get_current_world();
    // int (not uint32_t): multihost::Rank wraps int, so Rank{rank ± 1} must not narrow.
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();
    if (world_size < 2) {
        GTEST_SKIP() << "D2D forward chain needs >= 2 ranks; got " << world_size;
    }

    constexpr uint32_t kNumIters = 10;
    const ttnn::Shape global_shape({1, 1, 32, 64});  // shape is irrelevant (mapper replicates)

    const int stage = rank;                            // rank == stage; stage 0 (rank 0) owns the H2D feed
    const bool is_stage0 = stage == 0;                 // owns the H2D service (== !has_inbound)
    const bool has_inbound = stage > 0;                // D2D receiver of pair (rank-1, rank)
    const bool has_outbound = stage + 1 < world_size;  // D2D sender   of pair (rank, rank+1)

    // Construction ordering is load-bearing: inbound receiver BEFORE outbound sender,
    // so the per-endpoint handshakes cascade without deadlock. D2D endpoints use ACTUAL
    // ranks (rank±1) — which equal stage±1 only in the ws==2 (no-feeder) case.
    std::unique_ptr<D2DStreamServiceReceiver> inbound;
    std::unique_ptr<D2DStreamServiceSender> outbound;
    std::unique_ptr<H2DStreamService> h2d_service;  // stage 0 only: the host-fed source
    if (has_inbound) {
        inbound = D2DStreamService::create_receiver(
            mesh_device_,
            make_cfg(mesh_device_, global_shape),
            D2DEndpointConfig{.sender_rank = Rank{rank - 1}, .receiver_rank = Rank{rank}, .distributed_context = ctx});
    } else {
        // Stage 0 has no fabric inbound; its input is a real host->device H2D feed it
        // owns and pushes from a separate thread (spawned after [B1]). Built BEFORE the
        // outbound sender. Host-local — no peer handshake, so construction does not block
        // on other ranks.
        h2d_service = make_h2d_service(mesh_device_, make_cfg(mesh_device_, global_shape).global_spec);
    }
    if (has_outbound) {
        outbound = D2DStreamService::create_sender(
            mesh_device_,
            make_cfg(mesh_device_, global_shape),
            D2DEndpointConfig{.sender_rank = Rank{rank}, .receiver_rank = Rank{rank + 1}, .distributed_context = ctx});
    }

    std::unique_ptr<D2HStreamService> d2h_service;
    if (has_inbound && !has_outbound) {
        const auto spec = make_cfg(mesh_device_, global_shape).global_spec;
        D2HStreamService::Config cfg{
            .global_spec = spec,
            .mapper = create_mesh_mapper(*mesh_device_, MeshMapperConfig{.placements = replicate_all(*mesh_device_)}),
            .fifo_size_bytes = fifo_bytes_for(spec),
            .max_socket_page_size_bytes = fifo_bytes_for(spec),
            .worker_cores = kWorkerCores,
            .metadata_size_bytes = 0,
        };
        d2h_service = std::make_unique<D2HStreamService>(mesh_device_, std::move(cfg));
    }

    // [B1] All services resident across all ranks before any data flows.
    ctx->barrier();

    // Stage 0: spawn the feeder thread now (its H2D service is up). It streams tokens
    // concurrently with the device loop below — decoupling the push from the loop — and
    // is joined after the loop. Default-constructed (non-joinable) on every other stage.
    std::thread h2d_feeder;
    if (is_stage0) {
        h2d_feeder = std::thread([&] { run_h2d_feed_loop(*h2d_service, global_shape, kNumIters); });
    }

    // Op parameters are loop-invariant (backing/output addresses are fixed at allocation):
    // every stage produces downstream except the last; the dest is the outbound backing
    // while producing, else the end-stage output tensor.
    const bool produce = has_outbound;
    const uint32_t dest_addr = produce ? static_cast<uint32_t>(outbound->get_backing_tensor().buffer()->address())
                                       : static_cast<uint32_t>(d2h_service->get_backing_tensor().buffer()->address());
    D2DStreamServiceSender* const downstream = produce ? outbound.get() : nullptr;

    // FILL the pipeline: prime the inbound receiver so iter 0's input is received before the
    // first op reads it. Each iteration's release then grants the NEXT iter's receive
    // (steady-state pipelining), so this priming is the only "extra" release.
    if (has_inbound) {
        inbound->release_fabric_links();
    }

    std::vector<std::byte> host_buf;
    if (d2h_service) {
        host_buf.resize(d2h_service->payload_size_bytes());
    }

    auto& cq = mesh_device_->mesh_command_queue();
    for (uint32_t iter = 0; iter < kNumIters; ++iter) {
        // (a) Outbound overwrite-gate — the SEPARATE d2d_sync op, BEFORE the op: waits the
        //     prior forward to drain (sender consumed_sem) so the op doesn't overwrite the
        //     outbound backing too early. Skipped on iter 0. It's the SOLE overwrite
        //     protection (no host wait precedes the op), so the gate is load-bearing.
        if (has_outbound && iter > 0) {
            auto gate = make_gate_workload(outbound.get(), mesh_device_);
            EnqueueMeshWorkload(cq, gate, /*blocking=*/false);
            Finish(cq);
        }

        // (b) The op (one iter): waits its upstream's data (inbound data_ready_sem, granted
        //     last iter or by the priming release; the H2D feed on stage 0), copies it to
        //     the dest, and owns its handshake signals. Every stage runs the SAME gate-free
        //     relay; only the upstream differs (H2D service on stage 0, D2D receiver else).
        MeshWorkload op;
        if (is_stage0) {
            op = make_relay_like_workload(h2d_service.get(), mesh_device_, dest_addr, produce, downstream);
        } else if (has_outbound) {
            op = make_relay_like_workload(inbound.get(), mesh_device_, dest_addr, produce, downstream);
        } else {
            op = make_d2h_relay_workload(inbound.get(), mesh_device_, d2h_service.get());
        }
        EnqueueMeshWorkload(cq, op, /*blocking=*/false);
        Finish(cq);

        // (c) Wait BOTH services off their fabric links — inbound receiver done with this
        //     iter's receive, outbound sender done with the PREVIOUS forward — so the links
        //     are free for a model-graph fabric op.
        if (has_inbound) {
            inbound->wait_for_fabric_links();
        }
        if (has_outbound) {
            outbound->wait_for_fabric_links();
        }

        // (d) A competing model-graph fabric op (stub CCL) on the outbound link: stands in
        //     for a real graph op (e.g. a row CCL) and runs only because (c) confirmed the
        //     D2D sender vacated the EDM channel.
        if (has_outbound) {
            auto stub = make_stub_fabric_workload(outbound.get(), mesh_device_, Rank{rank + 1});
            EnqueueMeshWorkload(cq, stub, /*blocking=*/false);
            Finish(cq);
        }

        // (e) Hand the links back TOGETHER — the canonical cascade. The outbound grant
        //     forwards THIS iter's output; the inbound grant sets up the NEXT iter's receive
        //     (one phase ahead). DRAIN: skip the inbound grant on the final iter — there's
        //     no next receive, and a dangling grant would park the receiver forever.
        if (has_outbound) {
            outbound->release_fabric_links();
        }
        if (has_inbound && iter + 1 < kNumIters) {
            inbound->release_fabric_links();
        }

        if (d2h_service) {
            // ForwardChain: payload only
            d2h_service->read_from_tensor(host_buf);
            d2h_service->barrier();
            auto expected = make_iota_u32(host_buf.size() / sizeof(uint32_t), kFillBase + iter);
            std::vector<uint32_t> actual(host_buf.size() / sizeof(uint32_t));
            std::memcpy(actual.data(), host_buf.data(), host_buf.size());
            EXPECT_EQ(actual, expected) << "D2H readback mismatch at iter " << iter;
        }
    }

    // Stage 0: the feeder has pushed all tokens (its barrier() returns once the device
    // drained them, which the loop above did). Join before leaving the scope its lambda
    // captured by reference.
    if (h2d_feeder.joinable()) {
        h2d_feeder.join();
    }

    Synchronize(mesh_device_.get(), std::nullopt);
    ctx->barrier();  // [B2]
}

// One (grid, tensor-size, iters) point of the ForwardChainStress sweep.
struct StressCombo {
    CoreRange worker_cores;
    ttnn::Shape global_shape;
    uint32_t num_iters;
    const char* label;
};

// The sweep matrix. page = last_dim * 4 B (one row); pages = height; total = pages * page.
// All pages stay <= the 4 KB socket FIFO. The combos span: a single-worker grid, the
// multi-core baseline, an ODD page count (33 — indivisible by both the 2-lane split and
// the 4-worker grid, so it exercises the +1 remainder in lane 0's half and the worker
// page split), a wide grid with more pages, and a 4x4 grid whose 2 MB tensor EXCEEDS L1
// (1.5 MB) — proving the chain streams a > L1 transfer through the bounded FIFO
// page-by-page (the backing is DRAM, the relay's scratch CB is one page, so L1 use never
// scales with tensor size). The big combo runs fewer iters to bound HW time.
const std::vector<StressCombo> kStressCombos = {
    {kWorkerCores,
     ttnn::Shape({1, 1, 32, 64}),
     10u,
     "1x1 grid / 256B page / 32 pages / 8KB (single worker, multi-page)"},
    {kStressWorkerCores, ttnn::Shape({1, 1, 32, 64}), 10u, "2x2 grid / 256B page / 32 pages / 8KB (baseline)"},
    {kStressWorkerCores,
     ttnn::Shape({1, 1, 33, 64}),
     10u,
     "2x2 grid / 256B page / 33 pages / 8.25KB (odd: 17/16 lane split, 9/8/8/8 worker split)"},
    {CoreRange{CoreCoord{0, 0}, CoreCoord{3, 1}},
     ttnn::Shape({1, 1, 256, 128}),
     10u,
     "4x2 grid / 512B page / 256 pages / 128KB (wide grid)"},
    {CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}},
     ttnn::Shape({1, 1, 2048, 256}),
     4u,
     "4x4 grid / 1KB page / 2048 pages / 2MB (exceeds L1)"},
};

// Beefed-up regression variant of ForwardChain, one run of the sweep. For a given worker
// grid + tensor size: each stage's COMPUTE op mutates its slice (+1) so the end value is a
// function of every hop, the data_ready signal is a separate SIGNAL pass after the lease
// release, the overwrite-gate is fused into COMPUTE, and the competing fabric op opens
// every intra-mesh link (the lease + in-kernel drop test). The full-tensor element-wise
// verify catches a dropped/stale transfer or a stage/worker that skipped its op — any of
// those shifts the value off the expected iota + world_size. Extracted so the TEST_F can
// sweep the matrix; each call stands up its own services (the create_* rendezvous +
// [B1]/[B2] barriers keep all ranks in lockstep) and tears them down at scope exit.
void run_forward_chain_stress(
    const std::shared_ptr<MeshDevice>& mesh_device_,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_iters) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();

    const int stage = rank;
    const bool is_stage0 = stage == 0;
    const bool has_inbound = stage > 0;
    const bool has_outbound = stage + 1 < world_size;

    std::unique_ptr<D2DStreamServiceReceiver> inbound;
    std::unique_ptr<D2DStreamServiceSender> outbound;
    std::unique_ptr<H2DStreamService> h2d_service;
    if (has_inbound) {
        inbound = D2DStreamService::create_receiver(
            mesh_device_,
            make_cfg(mesh_device_, global_shape, worker_cores, kStressMetadataBytes),
            D2DEndpointConfig{.sender_rank = Rank{rank - 1}, .receiver_rank = Rank{rank}, .distributed_context = ctx});
    } else {
        h2d_service = make_h2d_service(
            mesh_device_,
            make_cfg(mesh_device_, global_shape, worker_cores).global_spec,
            worker_cores,
            kStressMetadataBytes);
    }
    if (has_outbound) {
        outbound = D2DStreamService::create_sender(
            mesh_device_,
            make_cfg(mesh_device_, global_shape, worker_cores, kStressMetadataBytes),
            D2DEndpointConfig{.sender_rank = Rank{rank}, .receiver_rank = Rank{rank + 1}, .distributed_context = ctx});
    }

    std::unique_ptr<D2HStreamService> d2h_service;
    if (has_inbound && !has_outbound) {
        const auto spec = make_cfg(mesh_device_, global_shape).global_spec;
        D2HStreamService::Config cfg{
            .global_spec = spec,
            .mapper = create_mesh_mapper(*mesh_device_, MeshMapperConfig{.placements = replicate_all(*mesh_device_)}),
            .fifo_size_bytes = fifo_bytes_for(spec),
            .max_socket_page_size_bytes = fifo_bytes_for(spec),
            .worker_cores = worker_cores,
            .metadata_master_core = worker_cores.end_coord,
            .metadata_size_bytes = kStressMetadataBytes,
        };
        d2h_service = std::make_unique<D2HStreamService>(mesh_device_, std::move(cfg));
    }

    ctx->barrier();  // [B1]

    // Per-iter random increments, generated identically on every rank from the fixed
    // seed (no cross-rank exchange): the stage-0 feeder injects seeds[i] as iter i's
    // metadata; the end stage uses seeds[last] to predict the shifted iota + check the
    // metadata readback.
    const std::vector<uint32_t> seeds = make_metadata_seeds(num_iters, kStressMetadataSeed);

    std::thread h2d_feeder;
    if (is_stage0) {
        h2d_feeder =
            std::thread([&] { run_h2d_feed_loop(*h2d_service, global_shape, num_iters, kStressMetadataBytes, seeds); });
    }

    const bool produce = has_outbound;
    const uint32_t dest_addr = produce ? static_cast<uint32_t>(outbound->get_backing_tensor().buffer()->address())
                                       : static_cast<uint32_t>(d2h_service->get_backing_tensor().buffer()->address());
    D2DStreamServiceSender* const gate_sender = has_outbound ? outbound.get() : nullptr;

    // Mesh-wide fabric-test GlobalSemaphore on the single fabric core; neighbors atomic-inc
    // it over fabric and each chip spins on its own copy (the in-kernel drop detector).
    // Created once — it accumulates across iters, so the spin target is cumulative.
    auto fabric_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh_device_.get(),
        CoreRangeSet(CoreRange{worker_cores.start_coord, worker_cores.start_coord}),
        /*initial_value=*/0,
        BufferType::L1);

    if (has_inbound) {
        inbound->release_fabric_links();
    }

    std::vector<std::byte> host_buf;
    if (d2h_service) {
        host_buf.resize(d2h_service->payload_size_bytes());
    }

    auto& cq = mesh_device_->mesh_command_queue();
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // (a) COMPUTE (all cores): fused gate (skipped iter 0 / no outbound) + wait inbound
        //     + mutate (+1) this worker's slice + ack inbound consumed. No data_ready yet.
        const uint32_t skip_gate = iter == 0 ? 1u : 0u;
        MeshWorkload compute;
        if (is_stage0) {
            compute = make_stress_compute_workload(
                h2d_service.get(), mesh_device_, dest_addr, gate_sender, skip_gate, kStressMetadataBytes);
        } else if (has_outbound) {
            compute = make_stress_compute_workload(
                inbound.get(), mesh_device_, dest_addr, gate_sender, skip_gate, kStressMetadataBytes);
        } else {
            compute = make_stress_d2h_compute_workload(
                inbound.get(), mesh_device_, d2h_service.get(), skip_gate, kStressMetadataBytes);
        }
        EnqueueMeshWorkload(cq, compute, /*blocking=*/false);
        Finish(cq);

        // (b) Wait BOTH services off their links (inbound receive done; prev forward done).
        if (has_inbound) {
            inbound->wait_for_fabric_links();
        }
        if (has_outbound) {
            outbound->wait_for_fabric_links();
        }

        // (c) Intra-mesh fabric stress op: opens all links to all neighbors (the D2D
        //     first-hop link is among them on interior chips -> exercises the lease) +
        //     an atomic-inc handshake (in-kernel drop detector). Runs on EVERY stage,
        //     bracketed by the wait (b) above and the release (d) below.
        {
            auto fabric = make_stress_fabric_workload(mesh_device_, worker_cores, fabric_sem, kStressIncsPerConn);
            EnqueueMeshWorkload(cq, fabric, /*blocking=*/false);
            Finish(cq);
        }

        // (d) Hand the links back: outbound = this forward, inbound = next receive (skip last).
        if (has_outbound) {
            outbound->release_fabric_links();
        }
        if (has_inbound && iter + 1 < num_iters) {
            inbound->release_fabric_links();
        }

        // (e) SIGNAL (all cores): inc outbound data_ready -> the sender forwards this iter's
        //     output. After the release, so the forward fires last.
        if (has_outbound) {
            auto signal = make_stress_signal_workload(outbound.get(), mesh_device_);
            EnqueueMeshWorkload(cq, signal, /*blocking=*/false);
            Finish(cq);
        }

        if (d2h_service) {
            // ForwardChainStress: also read + verify metadata
            std::vector<std::byte> metadata_out(kStressMetadataBytes);
            d2h_service->read_from_tensor(host_buf, metadata_out);
            d2h_service->barrier();
            const uint32_t delta = seeds[iter];
            auto expected = make_iota_u32(
                host_buf.size() / sizeof(uint32_t), kFillBase + iter + static_cast<uint32_t>(world_size) * delta);
            std::vector<uint32_t> actual(host_buf.size() / sizeof(uint32_t));
            std::memcpy(actual.data(), host_buf.data(), host_buf.size());
            EXPECT_EQ(actual, expected) << "D2H stress payload mismatch at iter " << iter;
            uint32_t meta_val;
            std::memcpy(&meta_val, metadata_out.data(), sizeof(uint32_t));
            EXPECT_EQ(meta_val, delta) << "D2H stress metadata mismatch at iter " << iter;
        }
    }

    if (h2d_feeder.joinable()) {
        h2d_feeder.join();
    }

    Synchronize(mesh_device_.get(), std::nullopt);
    ctx->barrier();  // [B2]
}

TEST_F(CrossProcessD2DFixture, ForwardChainStress) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    if (world_size < 2) {
        GTEST_SKIP() << "D2D forward chain needs >= 2 ranks; got " << world_size;
    }
    const int rank = *ctx->rank();

    int case_idx = 0;
    for (const auto& combo : kStressCombos) {
        // Per-case trace (printed on failure) + live progress line (so a hang's combo is
        // obvious from any rank under --tag-output). The combo order is identical on every
        // rank, so the per-combo rendezvous/barriers stay in lockstep.
        SCOPED_TRACE(::testing::Message() << "rank=" << rank << " case=" << case_idx << " " << combo.label);
        log_info(tt::LogMetal, "[xproc-d2d-stress] rank={} case={} {}", rank, case_idx, combo.label);
        ++case_idx;
        run_forward_chain_stress(mesh_device_, combo.global_shape, combo.worker_cores, combo.num_iters);
    }
}

}  // namespace
}  // namespace ttnn::distributed::test
