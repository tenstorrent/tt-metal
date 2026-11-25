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
// Fixture: CrossProcessD2DFixture — each rank opens its local mesh_device_, and world_size ==
// number of meshes in the mesh-graph descriptor (a split Galaxy, or a QuietBox with one 1x1 mesh
// per chip, chained over the ethernet links between them).
//
// Launch (no CI yet):
//   tt-run --rank-binding <rank_binding.yaml> --mpi-args "--allow-run-as-root --tag-output"
//       ./build/test/ttnn/multiprocess/unit_tests_cross_process_d2d

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "gtest/gtest.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal.hpp>
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
#include <internal/mesh_device.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"

#include <ttnn/api/ttnn/distributed/distributed_configs.hpp>
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/inbound_socket_service_sync/inbound_socket_service_sync.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/outbound_socket_service_sync/outbound_socket_service_sync.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/d2d_pipeline_descriptor.hpp"
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
using ::tt::tt_metal::distributed::MeshDeviceConfig;
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

// Each rank opens its whole local mesh with fabric on; world_size == number of meshes in the
// descriptor. Unlike MeshDeviceExaboxFixture this doesn't require a UBB Galaxy, and unlike
// MeshDeviceFixtureBase it allows a 1x1 mesh (its fabric links all go to other meshes), so it also
// runs on a QuietBox with one chip per rank.
class CrossProcessD2DFixture : public ::testing::Test {
protected:
    void SetUp() override {
        auto& context = tt::tt_metal::MetalContext::instance();
        if (static_cast<size_t>(*context.global_distributed_context().size()) !=
            context.get_control_plane().get_mesh_graph().get_mesh_ids().size()) {
            GTEST_SKIP() << "Needs one mesh per rank in the mesh graph descriptor.";
        }
        tt::tt_fabric::SetFabricConfig(
            tt::tt_fabric::FabricConfig::FABRIC_2D,
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(context.get_system_mesh().shape()));
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }
        mesh_device_->close();
        mesh_device_.reset();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};  // single worker core
// ForwardChainStress uses a multi-core grid so the num_workers machinery is actually
// exercised (consumed_sem mcast to N cores, data_ready reaching N acks, page split).
const CoreRange kStressWorkerCores{CoreCoord{0, 0}, CoreCoord{1, 1}};  // 2x2 = 4 worker cores
constexpr uint32_t kStressIncsPerConn = 4u;  // atomic-inc packets sent per fabric connection per iter
constexpr uint32_t kFillBase = 1u;           // stage-0 source iota base; end stage verifies kFillBase + iter
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

        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
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
    Upstream* inbound, const std::shared_ptr<MeshDevice>& mesh, D2HStreamService* d2h_service) {
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

        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
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

        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
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

        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
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

        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
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
        const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, sender_node, downstream_node);
        TT_FATAL(!links.empty(), "stub fabric op: no fabric link sender->downstream at coord {}", coord);

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_stub_fabric_op.cpp",
            stub_core_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> rt_args;
        tt::tt_fabric::append_fabric_connection_rt_args(
            *mesh, sender_node, downstream_node, links.front(), program, stub_core, rt_args);
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
        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
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
            for (auto link : tt::tt_fabric::get_forwarding_link_indices(*mesh, sender_node, neighbor_node)) {
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
            tt::tt_fabric::append_fabric_connection_rt_args(
                *mesh, sender_node, neighbor_node, link, program, fabric_core, rt);
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

    Synchronize(*mesh_device_, std::nullopt);
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

    Synchronize(*mesh_device_, std::nullopt);
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

// Rank 0 asks for relay mode and rank 1 does not. Only a relay side enters the relay
// exchange, so without the relay field in the startup settings check the sender would wait
// for an answer forever. Both sides must throw at that check instead.
TEST_F(CrossProcessD2DFixture, MismatchedRelayThrows) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();
    if (world_size < 2) {
        GTEST_SKIP() << "needs >= 2 ranks; got " << world_size;
    }
    if (tt::tt_fabric::is_intermesh_vc_disabled(*mesh_device_)) {
        GTEST_SKIP() << "TT_METAL_FABRIC_DISABLE_INTERMESH_VC=1 forces relay on for both sides";
    }

    const ttnn::Shape global_shape({1, 1, 32, 64});
    const D2DEndpointConfig endpoints{.sender_rank = Rank{0}, .receiver_rank = Rank{1}, .distributed_context = ctx};
    if (rank == 0) {
        auto cfg = make_cfg(mesh_device_, global_shape);
        cfg.relay = true;
        EXPECT_THROW(D2DStreamService::create_sender(mesh_device_, std::move(cfg), endpoints), std::exception);
    } else if (rank == 1) {
        EXPECT_THROW(
            D2DStreamService::create_receiver(mesh_device_, make_cfg(mesh_device_, global_shape), endpoints),
            std::exception);
    }
    ctx->barrier();
}

D2DStreamConfig make_relay_cfg(const std::shared_ptr<MeshDevice>& mesh, const ttnn::Shape& global_shape) {
    auto cfg = make_cfg(mesh, global_shape);
    cfg.relay = true;
    return cfg;
}

// Builds a relay-mode pair holder_sender -> the other rank of {0, 1} and keeps it, so its relays hold
// their fabric routers. Then builds a relay-mode pair rank 0 -> rank 1 whose relays need the same chips'
// routers, so `failing_rank`'s side finds none free. That side's claim fails before the relay exchange;
// it must tell its peer, and both ranks must throw instead of one waiting forever in the exchange.
void expect_relay_claim_failure_on_both_ranks(
    const std::shared_ptr<MeshDevice>& mesh,
    const std::shared_ptr<DistributedContext>& ctx,
    Rank holder_sender,
    Rank failing_rank) {
    const int rank = *ctx->rank();
    const ttnn::Shape global_shape({1, 1, 32, 64});

    const Rank holder_receiver{1 - *holder_sender};
    const D2DEndpointConfig holder_endpoints{
        .sender_rank = holder_sender, .receiver_rank = holder_receiver, .distributed_context = ctx};
    std::unique_ptr<D2DStreamServiceSender> holder_tx;
    std::unique_ptr<D2DStreamServiceReceiver> holder_rx;
    if (rank == *holder_sender) {
        holder_tx = D2DStreamService::create_sender(mesh, make_relay_cfg(mesh, global_shape), holder_endpoints);
    } else if (rank == *holder_receiver) {
        holder_rx = D2DStreamService::create_receiver(mesh, make_relay_cfg(mesh, global_shape), holder_endpoints);
    }

    const D2DEndpointConfig endpoints{.sender_rank = Rank{0}, .receiver_rank = Rank{1}, .distributed_context = ctx};
    std::string error;
    try {
        if (rank == 0) {
            (void)D2DStreamService::create_sender(mesh, make_relay_cfg(mesh, global_shape), endpoints);
        } else if (rank == 1) {
            (void)D2DStreamService::create_receiver(mesh, make_relay_cfg(mesh, global_shape), endpoints);
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (rank == *failing_rank) {
        EXPECT_NE(error.find("no free fabric router"), std::string::npos) << "rank " << rank << ": " << error;
    } else if (rank == 0 || rank == 1) {
        EXPECT_NE(error.find("failed to set up its relay side"), std::string::npos) << "rank " << rank << ": " << error;
    }
    ctx->barrier();
}

// The holder 0 -> 1 holds both cable links of each exit chip on rank 0, so the second sender's exit
// relays find no cable router. Its receiver claims fine and waits in the relay exchange: it must hear
// that the sender failed.
TEST_F(CrossProcessD2DFixture, RelaySenderClaimFailureThrowsOnBothRanks) {
    const auto& ctx = DistributedContext::get_current_world();
    if (*ctx->size() < 2) {
        GTEST_SKIP() << "needs >= 2 ranks; got " << *ctx->size();
    }
    expect_relay_claim_failure_on_both_ranks(mesh_device_, ctx, /*holder_sender=*/Rank{0}, /*failing_rank=*/Rank{0});
}

// The holder 1 -> 0 has its exit relays on rank 1's cable chips, which are the second pair's entry
// chips, and holds both cable links there. The second receiver's entry relays find no router back over
// the cable, after its sender has already sent its relay words: the sender must hear that the receiver failed.
TEST_F(CrossProcessD2DFixture, RelayReceiverClaimFailureThrowsOnBothRanks) {
    const auto& ctx = DistributedContext::get_current_world();
    if (*ctx->size() < 2) {
        GTEST_SKIP() << "needs >= 2 ranks; got " << *ctx->size();
    }
    expect_relay_claim_failure_on_both_ranks(mesh_device_, ctx, /*holder_sender=*/Rank{1}, /*failing_rank=*/Rank{1});
}

// Every rank exports its inbound boundary; rank 0 gets the whole chain. Each boundary has two
// channels, like the prefill runner's activations plus indexer indices: a gated one with
// metadata, then a plain one.
TEST_F(CrossProcessD2DFixture, ExportPipelineDescriptor) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();
    if (world_size < 2) {
        GTEST_SKIP() << "Pipeline export needs >= 2 ranks; got " << world_size;
    }

    constexpr uint32_t kMetadataBytes = 16;
    const ttnn::D2DStageGateConfig gate_cfg{.num_gates = 2, .slot_id_offset_bytes = 0, .gate_mode_offset_bytes = 4};
    const ttnn::Shape global_shape({1, 1, 32, 64});
    const D2DEndpointConfig from_prev{
        .sender_rank = Rank{rank - 1}, .receiver_rank = Rank{rank}, .distributed_context = ctx};
    const D2DEndpointConfig to_next{
        .sender_rank = Rank{rank}, .receiver_rank = Rank{rank + 1}, .distributed_context = ctx};
    // Channels between the same two ranks pair up in creation order, so every rank builds them
    // in the same order: inbound before outbound, gated before plain. Each direction's two
    // channels share the routers in one lease group, like the prefill runner.
    std::vector<std::unique_ptr<D2DStreamServiceReceiver>> inbound;
    std::vector<std::unique_ptr<D2DStreamServiceSender>> outbound;
    for (const bool gated : {true, false}) {
        const uint32_t metadata_bytes = gated ? kMetadataBytes : 0;
        if (rank > 0) {
            auto cfg = make_cfg(mesh_device_, global_shape, kWorkerCores, metadata_bytes);
            cfg.fabric_link_group = "export-inbound";
            if (gated) {
                cfg.stage_gate = gate_cfg;
            }
            inbound.push_back(D2DStreamService::create_receiver(mesh_device_, std::move(cfg), from_prev));
        }
        if (rank + 1 < world_size) {
            auto cfg = make_cfg(mesh_device_, global_shape, kWorkerCores, metadata_bytes);
            cfg.fabric_link_group = "export-outbound";
            outbound.push_back(D2DStreamService::create_sender(mesh_device_, std::move(cfg), to_next));
        }
    }

    std::vector<std::vector<const D2DStreamServiceReceiver*>> boundaries;
    if (!inbound.empty()) {
        boundaries.push_back({inbound[0].get(), inbound[1].get()});
    }
    // Only the root writes, so its pid keeps the path unique.
    auto file = D2DStreamService::export_pipeline_descriptor(
        "gtest_cross_process_export_" + std::to_string(getpid()), boundaries, Rank{0}, ctx);
    EXPECT_EQ(file != nullptr, rank == 0);
    if (rank == 0) {
        const auto desc = ttnn::D2DPipelineDescriptor::wait_and_read(file->path());
        ASSERT_EQ(desc.ranks.size(), static_cast<size_t>(world_size));
        // Ranks sharing a host each see their own chip as chip 0 (TT_VISIBLE_DEVICES), so the file
        // must carry the physical PCIe device number. Otherwise every rank on a host would name the
        // same device.
        std::map<std::pair<std::string, uint32_t>, int> owner_of_device;
        for (int r = 0; r < world_size; ++r) {
            const auto& rd = desc.ranks[r];
            EXPECT_EQ(rd.rank, r);
            EXPECT_FALSE(rd.host_name.empty());
            ASSERT_EQ(rd.boundaries.size(), r > 0 ? 1u : 0u);
            if (rd.boundaries.empty()) {
                continue;
            }
            const auto& receivers = rd.boundaries[0].receivers;
            ASSERT_EQ(receivers.size(), 2u);
            for (size_t i = 0; i < receivers.size(); ++i) {
                const auto& rx = receivers[i];
                const bool gated = i == 0;
                EXPECT_EQ(rx.sender_rank, r - 1);
                EXPECT_EQ(rx.metadata_size_bytes, gated ? kMetadataBytes : 0u);
                EXPECT_EQ(rx.stage_gate, gated ? gate_cfg : ttnn::D2DStageGateConfig{});
                EXPECT_EQ(rx.stage_gate_stride_bytes, gated ? tt::tt_metal::hal::get_l1_alignment() : 0u);
                EXPECT_EQ(rx.metadata_addr != 0u, gated);
                ASSERT_FALSE(rx.coords.empty());
                for (const auto& c : rx.coords) {
                    EXPECT_NE(c.backing_tensor_addr, 0u);
                    EXPECT_EQ(c.stage_gate_base_addr != 0u, gated);
                    EXPECT_EQ(c.metadata_core.has_value(), gated);
                    EXPECT_NE(c.sender_fabric_node_id, c.fabric_node_id);
                    const auto [it, inserted] =
                        owner_of_device.try_emplace({rd.host_name, c.service_core.pcie_device_id}, r);
                    EXPECT_TRUE(inserted || it->second == r)
                        << "ranks " << it->second << " and " << r << " both name PCIe device "
                        << c.service_core.pcie_device_id << " on host " << rd.host_name;
                }
            }
            // Two channels into one chip use different service cores.
            EXPECT_NE(receivers[0].coords.at(0).service_core, receivers[1].coords.at(0).service_core);
        }
    }
    ctx->barrier();
}

// A bad call on any rank makes every rank throw. Before, the failing rank threw on its own
// and the root waited for it forever. A regression here hangs instead of failing.
TEST(CrossProcessD2DExport, BadCallFailsOnEveryRank) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();
    if (world_size < 2) {
        GTEST_SKIP() << "Needs >= 2 ranks; got " << world_size;
    }
    const std::string service_id = "gtest_cross_process_export_fail_" + std::to_string(getpid());
    const std::vector<std::vector<const D2DStreamServiceReceiver*>> none;

    // Only the last rank fails, first on a null receiver, then on a boundary with no receivers.
    const std::vector<std::vector<const D2DStreamServiceReceiver*>> null_receiver{{nullptr}};
    EXPECT_ANY_THROW(D2DStreamService::export_pipeline_descriptor(
        service_id, rank == world_size - 1 ? null_receiver : none, Rank{0}, ctx));
    const std::vector<std::vector<const D2DStreamServiceReceiver*>> empty_boundary{{}};
    EXPECT_ANY_THROW(D2DStreamService::export_pipeline_descriptor(
        service_id, rank == world_size - 1 ? empty_boundary : none, Rank{0}, ctx));

    // The ranks disagree on the root.
    EXPECT_ANY_THROW(D2DStreamService::export_pipeline_descriptor(service_id, none, Rank{rank == 0 ? 1 : 0}, ctx));

    // Every rank passes the same out-of-range root.
    EXPECT_ANY_THROW(D2DStreamService::export_pipeline_descriptor(service_id, none, Rank{world_size}, ctx));

    // A good call still works afterwards.
    auto file = D2DStreamService::export_pipeline_descriptor(service_id, none, Rank{0}, ctx);
    EXPECT_EQ(file != nullptr, rank == 0);
    ctx->barrier();
}

// ===========================================================================
// Bidirectional Stage Sockets (Cable Mux + Bandwidth + Stage Compute Overlap)
// ===========================================================================

using ::tt::tt_metal::distributed::MeshBuffer;
using ::tt::tt_metal::distributed::ReadShard;
using ::tt::tt_metal::distributed::WriteShard;

std::shared_ptr<MeshBuffer> bidir_mesh_buffer_view(const ttnn::Tensor& tensor) {
    return std::shared_ptr<MeshBuffer>(const_cast<MeshBuffer*>(&tensor.mesh_buffer()), [](auto*) {});
}

class BidirectionalStageSocketsFixture : public ::testing::Test {
protected:
    void SetUp() override {
        auto& context = tt::tt_metal::MetalContext::instance();
        if (static_cast<size_t>(*context.global_distributed_context().size()) !=
            context.get_control_plane().get_mesh_graph().get_mesh_ids().size()) {
            GTEST_SKIP() << "Needs one mesh per rank in the mesh graph descriptor.";
        }
        auto fabric_cfg = tt::tt_fabric::FabricConfig::MGD_DEFAULT;
        if (const char* mode = std::getenv("BIDIR_FABRIC_MODE")) {
            const std::string m(mode);
            if (m == "2d") {
                fabric_cfg = tt::tt_fabric::FabricConfig::FABRIC_2D;
            } else if (m == "2d_torus_x") {
                fabric_cfg = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X;
            } else if (m == "2d_torus_y") {
                fabric_cfg = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y;
            } else if (m == "2d_torus_xy") {
                fabric_cfg = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
            }
        }
        tt::tt_fabric::FabricRouterConfig router_cfg{};
        if (const char* payload_env = std::getenv("BIDIR_MAX_PAYLOAD_BYTES")) {
            router_cfg.max_packet_payload_size_bytes = static_cast<size_t>(std::stoul(payload_env));
        }
        tt::tt_fabric::SetFabricConfig(
            fabric_cfg,
            tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE,
            std::nullopt,
            tt::tt_fabric::FabricTensixConfig::DISABLED,
            tt::tt_fabric::FabricUDMMode::DISABLED,
            tt::tt_fabric::FabricManagerMode::DEFAULT,
            router_cfg);
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(context.get_system_mesh().shape()));
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }
        mesh_device_->close();
        mesh_device_.reset();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

D2DStreamConfig make_bidir_relay_cfg(
    const std::shared_ptr<MeshDevice>& mesh,
    const ttnn::Shape& per_chip_shape,
    const CoreRange& worker_cores,
    const std::string& link_group,
    bool cable_mux,
    uint32_t cable_lanes,
    uint32_t in_mesh_lanes,
    bool reserve_links_for_later_service = true,
    uint32_t metadata_size_bytes = 0,
    bool instage_mux = false,
    uint32_t relay_slices = 16u) {
    const tt::tt_metal::TensorSpec global_spec(
        per_chip_shape,
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
        .share_fabric_links = true,
        .max_sender_lanes = in_mesh_lanes,
        .max_cable_lanes = cable_lanes,
        .cable_mux = cable_mux,
        .instage_mux = instage_mux,
        .relay_slices = relay_slices,
        .fabric_link_group = link_group,
        .relay = true,
        .reserve_links_for_later_service = reserve_links_for_later_service,
    };
}

// Zero-copy receiver handshake (placeholder_d2d_receiver_worker, num_iters = 1):
// waits data_ready_sem, resets it, and atomic-incs consumed_counter.
MeshWorkload make_recv_ack_workload(D2DStreamServiceReceiver* receiver, const std::shared_ptr<MeshDevice>& mesh) {
    const auto& coords = receiver->get_backing_tensor().tensor_topology().mesh_coords();
    const CoreRange worker_cores = receiver->get_worker_cores();
    const std::vector<uint32_t> ct_args = {
        static_cast<uint32_t>(receiver->get_data_ready_sem_addr()),
        /*num_iters=*/1u,
    };
    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/placeholder_d2d_receiver_worker.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});
        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
        const auto svc_phys = device->worker_core_from_logical_core(receiver->get_service_core(coord));
        const std::vector<uint32_t> rt_args = {
            static_cast<uint32_t>(receiver->get_consumed_counter_addr(coord)),
            static_cast<uint32_t>(svc_phys.x),
            static_cast<uint32_t>(svc_phys.y),
        };
        for (const auto& wc : worker_cores) {
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

void fill_backing_iota(const ttnn::Tensor& tensor, const std::shared_ptr<MeshDevice>& mesh, uint32_t base) {
    auto mb = bidir_mesh_buffer_view(tensor);
    const size_t num_u32 = tensor.buffer()->size() / sizeof(uint32_t);
    std::vector<uint32_t> vals = make_iota_u32(num_u32, base);
    for (const auto& coord : tensor.tensor_topology().mesh_coords()) {
        WriteShard(mesh->mesh_command_queue(), mb, vals, coord);
    }
    Finish(mesh->mesh_command_queue());
}

void verify_backing_iota(
    const ttnn::Tensor& tensor, const std::shared_ptr<MeshDevice>& mesh, uint32_t base, const char* label) {
    auto mb = bidir_mesh_buffer_view(tensor);
    const size_t num_u32 = tensor.buffer()->size() / sizeof(uint32_t);
    const std::vector<uint32_t> expected = make_iota_u32(num_u32, base);
    std::vector<uint32_t> actual;
    for (const auto& coord : tensor.tensor_topology().mesh_coords()) {
        actual.clear();
        ReadShard(mesh->mesh_command_queue(), actual, mb, coord);
        EXPECT_EQ(actual, expected) << label << " mismatch at coord " << coord << " (expected base " << base << ")";
    }
}

TEST_F(BidirectionalStageSocketsFixture, MismatchedCableMuxThrows) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();
    if (world_size < 2) {
        GTEST_SKIP() << "Needs >= 2 ranks; got " << world_size;
    }

    const ttnn::Shape shape({1, 1, 32, 64});
    const D2DEndpointConfig endpoints{.sender_rank = Rank{0}, .receiver_rank = Rank{1}, .distributed_context = ctx};
    if (rank == 0) {
        auto cfg = make_bidir_relay_cfg(mesh_device_, shape, kWorkerCores, "d2d-outbound", /*cable_mux=*/true, 2, 1);
        EXPECT_THROW(D2DStreamService::create_sender(mesh_device_, std::move(cfg), endpoints), std::exception);
    } else if (rank == 1) {
        auto cfg = make_bidir_relay_cfg(mesh_device_, shape, kWorkerCores, "d2d-inbound", /*cable_mux=*/false, 1, 1);
        EXPECT_THROW(D2DStreamService::create_receiver(mesh_device_, std::move(cfg), endpoints), std::exception);
    }
    ctx->barrier();
}

TEST_F(BidirectionalStageSocketsFixture, BidirectionalCableMuxCorrectnessAndTeardown) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();
    if (world_size < 2) {
        GTEST_SKIP() << "Needs >= 2 ranks; got " << world_size;
    }

    // 1. Check dispatch-column service core pool on every chip (>= 6 free cores before D2D claims).
    auto& svc = tt::tt_metal::internal::service_core_manager(*mesh_device_);
    size_t min_claimable = 1000;
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        auto* d = tt::tt_metal::internal::get_device(*mesh_device_, coord);
        const auto claimable = svc.get_claimable_cores(d);
        min_claimable = std::min(min_claimable, claimable.size());
        EXPECT_GE(claimable.size(), 6u) << "rank " << rank << " coord " << coord << " has too few service cores";
    }
    if (rank == 0) {
        log_info(
            tt::LogMetal,
            "[bidir-check] mesh shape {} ({} chips/rank), world_size={}, min claimable dispatch-column cores/chip={}",
            mesh_device_->shape(),
            mesh_device_->num_devices(),
            world_size,
            min_claimable);
    }

    const ttnn::Shape shape({1, 1, 256, 256});  // 256 KB/chip (multi-page, >16 packets/lane)
    auto fabric_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh_device_.get(),
        CoreRangeSet(CoreRange{kStressWorkerCores.start_coord, kStressWorkerCores.start_coord}),
        /*initial_value=*/0,
        BufferType::L1);

    auto build_all_streams = [&]() {
        std::unique_ptr<D2DStreamServiceReceiver> act_in;
        std::unique_ptr<D2DStreamServiceSender> act_out;
        std::unique_ptr<D2DStreamServiceReceiver> grad_in;
        std::unique_ptr<D2DStreamServiceSender> grad_out;

        // Section 2.2 creation order: boundaries b = 0 .. P-2, activation pair then gradient pair.
        // Uses BOTH cable_mux = true AND instage_mux = true with 2 cable lanes + 2 in-mesh lanes!
        for (int b = 0; b + 1 < world_size; ++b) {
            const D2DEndpointConfig act_ep{
                .sender_rank = Rank{b}, .receiver_rank = Rank{b + 1}, .distributed_context = ctx};
            if (rank == b) {
                act_out = D2DStreamService::create_sender(
                    mesh_device_,
                    make_bidir_relay_cfg(
                        mesh_device_,
                        shape,
                        kStressWorkerCores,
                        "d2d-outbound",
                        /*cable_mux=*/true,
                        2,
                        2,
                        true,
                        0,
                        /*instage_mux=*/true),
                    act_ep);
            } else if (rank == b + 1) {
                act_in = D2DStreamService::create_receiver(
                    mesh_device_,
                    make_bidir_relay_cfg(
                        mesh_device_,
                        shape,
                        kStressWorkerCores,
                        "d2d-inbound",
                        /*cable_mux=*/true,
                        2,
                        2,
                        /*reserve=*/true,
                        0,
                        /*instage_mux=*/true),
                    act_ep);
            }

            const D2DEndpointConfig grad_ep{
                .sender_rank = Rank{b + 1}, .receiver_rank = Rank{b}, .distributed_context = ctx};
            if (rank == b + 1) {
                grad_out = D2DStreamService::create_sender(
                    mesh_device_,
                    make_bidir_relay_cfg(
                        mesh_device_,
                        shape,
                        kStressWorkerCores,
                        "d2d-outbound",
                        /*cable_mux=*/true,
                        2,
                        2,
                        true,
                        0,
                        /*instage_mux=*/true),
                    grad_ep);
            } else if (rank == b) {
                const bool more_services_follow = (b + 1 < world_size - 1);
                grad_in = D2DStreamService::create_receiver(
                    mesh_device_,
                    make_bidir_relay_cfg(
                        mesh_device_,
                        shape,
                        kStressWorkerCores,
                        "d2d-inbound",
                        /*cable_mux=*/true,
                        2,
                        2,
                        more_services_follow,
                        0,
                        /*instage_mux=*/true),
                    grad_ep);
            }
        }
        return std::make_tuple(std::move(act_in), std::move(act_out), std::move(grad_in), std::move(grad_out));
    };

    // Round 1: simultaneous bidirectional transfers across all ranks + intra-mesh fabric contention,
    // followed by tearing down `act` first while `grad` continues streaming!
    {
        auto [act_in, act_out, grad_in, grad_out] = build_all_streams();
        auto& cq = mesh_device_->mesh_command_queue();

        const uint32_t act_base = 1000u + static_cast<uint32_t>(rank) * 100u;
        const uint32_t grad_base = 5000u + static_cast<uint32_t>(rank) * 100u;
        if (act_out) {
            fill_backing_iota(act_out->get_backing_tensor(), mesh_device_, act_base);
        }
        if (grad_out) {
            fill_backing_iota(grad_out->get_backing_tensor(), mesh_device_, grad_base);
        }
        ctx->barrier();

        MeshWorkload act_recv_wl = act_in ? make_recv_ack_workload(act_in.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload grad_recv_wl = grad_in ? make_recv_ack_workload(grad_in.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload act_sig_wl = act_out ? make_stress_signal_workload(act_out.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload grad_sig_wl =
            grad_out ? make_stress_signal_workload(grad_out.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload act_gate_wl = act_out ? make_gate_workload(act_out.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload grad_gate_wl = grad_out ? make_gate_workload(grad_out.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload fabric_wl = make_stress_fabric_workload(mesh_device_, kStressWorkerCores, fabric_sem, 4u);

        constexpr uint32_t kIters = 8;
        // Prime transfer 0 in BOTH directions simultaneously.
        if (act_out) {
            act_out->release_fabric_links();
            EnqueueMeshWorkload(cq, act_sig_wl, /*blocking=*/false);
        }
        if (grad_out) {
            grad_out->release_fabric_links();
            EnqueueMeshWorkload(cq, grad_sig_wl, /*blocking=*/false);
        }

        for (uint32_t iter = 0; iter < kIters; ++iter) {
            // Receive before wait (Section 2.1).
            if (act_in) {
                act_in->release_fabric_links();
                EnqueueMeshWorkload(cq, act_recv_wl, /*blocking=*/false);
            }
            if (grad_in) {
                grad_in->release_fabric_links();
                EnqueueMeshWorkload(cq, grad_recv_wl, /*blocking=*/false);
            }

            // Wait all open stream windows before intra-mesh collectives.
            if (act_in) {
                act_in->wait_for_fabric_links();
            }
            if (grad_in) {
                grad_in->wait_for_fabric_links();
            }
            if (act_out) {
                act_out->wait_for_fabric_links();
                EnqueueMeshWorkload(cq, act_gate_wl, /*blocking=*/false);
            }
            if (grad_out) {
                grad_out->wait_for_fabric_links();
                EnqueueMeshWorkload(cq, grad_gate_wl, /*blocking=*/false);
            }

            // Intra-mesh fabric collective stress on ALL links of ALL chips.
            EnqueueMeshWorkload(cq, fabric_wl, /*blocking=*/false);

            if (iter + 1 < kIters) {
                if (act_out) {
                    act_out->release_fabric_links();
                    EnqueueMeshWorkload(cq, act_sig_wl, /*blocking=*/false);
                }
                if (grad_out) {
                    grad_out->release_fabric_links();
                    EnqueueMeshWorkload(cq, grad_sig_wl, /*blocking=*/false);
                }
            }
        }
        Finish(cq);
        ctx->barrier();

        if (act_in) {
            verify_backing_iota(
                act_in->get_backing_tensor(),
                mesh_device_,
                1000u + static_cast<uint32_t>(rank - 1) * 100u,
                "act_in simultaneous bidir");
        }
        if (grad_in) {
            verify_backing_iota(
                grad_in->get_backing_tensor(),
                mesh_device_,
                5000u + static_cast<uint32_t>(rank + 1) * 100u,
                "grad_in simultaneous bidir");
        }

        // Teardown order A: destroy `act` first; prove `grad` keeps streaming through the shared cable mux!
        act_in.reset();
        act_out.reset();
        ctx->barrier();

        const uint32_t grad_after_act_teardown = 9000u + static_cast<uint32_t>(rank) * 100u;
        if (grad_out) {
            fill_backing_iota(grad_out->get_backing_tensor(), mesh_device_, grad_after_act_teardown);
        }
        ctx->barrier();
        for (uint32_t iter = 0; iter < 4u; ++iter) {
            if (grad_out) {
                grad_out->release_fabric_links();
                EnqueueMeshWorkload(cq, grad_sig_wl, /*blocking=*/false);
            }
            if (grad_in) {
                grad_in->release_fabric_links();
                EnqueueMeshWorkload(cq, grad_recv_wl, /*blocking=*/false);
                grad_in->wait_for_fabric_links();
            }
            if (grad_out) {
                grad_out->wait_for_fabric_links();
                EnqueueMeshWorkload(cq, grad_gate_wl, /*blocking=*/false);
            }
        }
        Finish(cq);
        ctx->barrier();
        if (grad_in) {
            verify_backing_iota(
                grad_in->get_backing_tensor(),
                mesh_device_,
                9000u + static_cast<uint32_t>(rank + 1) * 100u,
                "grad_in after act teardown");
        }

        grad_in.reset();
        grad_out.reset();
        ctx->barrier();
    }

    // Round 2: re-create all streams on the SAME MeshDevice (proving zero leaked service cores,
    // L1, or router claims!) and tear down in the OPPOSITE order (`grad` first, then `act`).
    {
        auto [act_in, act_out, grad_in, grad_out] = build_all_streams();
        auto& cq = mesh_device_->mesh_command_queue();

        // Destroy `grad` first, then run 4 transfers on `act`!
        grad_in.reset();
        grad_out.reset();
        ctx->barrier();

        const uint32_t act_after_grad_teardown = 12000u + static_cast<uint32_t>(rank) * 100u;
        if (act_out) {
            fill_backing_iota(act_out->get_backing_tensor(), mesh_device_, act_after_grad_teardown);
        }
        ctx->barrier();

        MeshWorkload act_recv_wl = act_in ? make_recv_ack_workload(act_in.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload act_sig_wl = act_out ? make_stress_signal_workload(act_out.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload act_gate_wl = act_out ? make_gate_workload(act_out.get(), mesh_device_) : MeshWorkload{};

        for (uint32_t iter = 0; iter < 4u; ++iter) {
            if (act_out) {
                act_out->release_fabric_links();
                EnqueueMeshWorkload(cq, act_sig_wl, /*blocking=*/false);
            }
            if (act_in) {
                act_in->release_fabric_links();
                EnqueueMeshWorkload(cq, act_recv_wl, /*blocking=*/false);
                act_in->wait_for_fabric_links();
            }
            if (act_out) {
                act_out->wait_for_fabric_links();
                EnqueueMeshWorkload(cq, act_gate_wl, /*blocking=*/false);
            }
        }
        Finish(cq);
        ctx->barrier();
        if (act_in) {
            verify_backing_iota(
                act_in->get_backing_tensor(),
                mesh_device_,
                12000u + static_cast<uint32_t>(rank - 1) * 100u,
                "act_in after grad teardown");
        }

        act_in.reset();
        act_out.reset();
        ctx->barrier();
    }
}

struct BenchResult {
    std::string name;
    size_t shard_bytes = 0;
    size_t stage_bytes_per_dir = 0;
    double ms_per_iter = 0.0;
    double gbps_per_dir = 0.0;
    double gbps_total_bidir = 0.0;
    double on_device_recv_wait_ms = 0.0;
    double on_device_lease_wait_ms = 0.0;
    double on_device_total_compute_wait_ms = 0.0;
};

TEST_F(BidirectionalStageSocketsFixture, BidirectionalBandwidthAndComputeBenchmark) {
    const auto& ctx = DistributedContext::get_current_world();
    const int world_size = *ctx->size();
    const int rank = *ctx->rank();
    if (world_size != 2) {
        GTEST_SKIP() << "2-stage benchmark requires world_size == 2; got " << world_size;
    }

    const size_t chips_per_stage = mesh_device_->num_devices();
    uint32_t default_mb = 8u;  // 8 MB/chip = Llama-70B TP4xCP4 boundary shard ([1,1,2048,2048] bf16)
    if (const char* env_mb = std::getenv("BIDIR_SHARD_MB")) {
        default_mb = static_cast<uint32_t>(std::stoul(env_mb));
    }

    constexpr uint32_t kWarmupIters = 5;
    constexpr uint32_t kTimedIters = 25;
    constexpr uint32_t kMaxTsSamples = 3u * (kTimedIters + 2u);
    const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t raw_ts_bytes = (2u + 2u * kMaxTsSamples) * static_cast<uint32_t>(sizeof(uint32_t));
    const uint32_t ts_buf_bytes = ((raw_ts_bytes + l1_align - 1u) / l1_align) * l1_align;

    // Allocate persistent L1 timestamp buffer on worker core (0,0) across the mesh.
    const CoreCoord ts_core{0, 0};
    const CoreRange ts_core_range{ts_core, ts_core};
    tt::tt_metal::distributed::DeviceLocalBufferConfig ts_local = {
        .page_size = ts_buf_bytes,
        .buffer_type = BufferType::L1,
        .sharding_args = tt::tt_metal::BufferShardingArgs(
            tt::tt_metal::ShardSpecBuffer(
                CoreRangeSet(ts_core_range), {1, 1}, tt::tt_metal::ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    tt::tt_metal::distributed::MeshBufferConfig ts_mesh_cfg =
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = ts_buf_bytes};
    auto ts_buffer = MeshBuffer::create(ts_mesh_cfg, ts_local, mesh_device_.get());
    const uint32_t ts_l1_addr = static_cast<uint32_t>(ts_buffer->address());

    MeshWorkload ts_wl;
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        auto program = CreateProgram();
        CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_timestamp.cpp",
            ts_core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {ts_l1_addr, kMaxTsSamples}});
        ts_wl.add_program(MeshCoordinateRange(coord), std::move(program));
    }

    auto reset_ts_buffers = [&]() {
        std::vector<uint32_t> zeros(ts_buf_bytes / sizeof(uint32_t), 0u);
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            tt::tt_metal::detail::WriteToDeviceL1(
                tt::tt_metal::internal::get_device(*mesh_device_, coord), ts_core, ts_l1_addr, zeros);
        }
    };

    const auto compute_grid = mesh_device_->compute_with_storage_grid_size();
    const CoreRange full_worker_grid{CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1}};

    auto run_socket_cell = [&](const std::string& label,
                               uint32_t shard_mb,
                               bool bidirectional,
                               bool cable_mux,
                               bool instage_mux,
                               uint32_t cable_lanes,
                               uint32_t in_mesh_lanes,
                               bool use_ttnn_runtime_ops,
                               int num_matmuls,
                               const ttnn::Tensor* mm_a,
                               const ttnn::Tensor* mm_b,
                               ttnn::Tensor* mm_out,
                               MeshWorkload* intra_fabric_wl,
                               uint32_t relay_slices = 16u) -> BenchResult {
        // Page = 2048 * 4B = 8192B; rows = shard_mb * 128 -> exact shard_mb MiB/chip.
        const ttnn::Shape shape({1, 1, shard_mb * 128u, 2048u});
        const size_t shard_bytes = static_cast<size_t>(shard_mb) * 1024ull * 1024ull;
        const size_t stage_bytes = shard_bytes * chips_per_stage;
        const CoreRange workers = use_ttnn_runtime_ops ? full_worker_grid : kStressWorkerCores;

        std::unique_ptr<D2DStreamServiceSender> act_out;
        std::unique_ptr<D2DStreamServiceReceiver> act_in;
        std::unique_ptr<D2DStreamServiceSender> grad_out;
        std::unique_ptr<D2DStreamServiceReceiver> grad_in;

        const D2DEndpointConfig act_ep{.sender_rank = Rank{0}, .receiver_rank = Rank{1}, .distributed_context = ctx};
        if (rank == 0) {
            act_out = D2DStreamService::create_sender(
                mesh_device_,
                make_bidir_relay_cfg(
                    mesh_device_,
                    shape,
                    workers,
                    "d2d-outbound",
                    cable_mux,
                    cable_lanes,
                    in_mesh_lanes,
                    true,
                    0,
                    instage_mux,
                    relay_slices),
                act_ep);
        } else {
            act_in = D2DStreamService::create_receiver(
                mesh_device_,
                make_bidir_relay_cfg(
                    mesh_device_,
                    shape,
                    workers,
                    "d2d-inbound",
                    cable_mux,
                    cable_lanes,
                    in_mesh_lanes,
                    /*reserve=*/bidirectional,
                    0,
                    instage_mux,
                    relay_slices),
                act_ep);
        }

        if (bidirectional) {
            const D2DEndpointConfig grad_ep{
                .sender_rank = Rank{1}, .receiver_rank = Rank{0}, .distributed_context = ctx};
            if (rank == 1) {
                grad_out = D2DStreamService::create_sender(
                    mesh_device_,
                    make_bidir_relay_cfg(
                        mesh_device_,
                        shape,
                        workers,
                        "d2d-outbound",
                        cable_mux,
                        cable_lanes,
                        in_mesh_lanes,
                        true,
                        0,
                        instage_mux,
                        relay_slices),
                    grad_ep);
            } else {
                grad_in = D2DStreamService::create_receiver(
                    mesh_device_,
                    make_bidir_relay_cfg(
                        mesh_device_,
                        shape,
                        workers,
                        "d2d-inbound",
                        cable_mux,
                        cable_lanes,
                        in_mesh_lanes,
                        /*reserve=*/false,
                        0,
                        instage_mux,
                        relay_slices),
                    grad_ep);
            }
        }

        std::optional<ttnn::Tensor> act_send_src, grad_send_src, act_recv_dst, grad_recv_dst;
        if (use_ttnn_runtime_ops) {
            if (act_out) {
                act_send_src = ttnn::create_device_tensor(
                    act_out->get_backing_tensor().tensor_spec(),
                    mesh_device_.get(),
                    act_out->get_backing_tensor().tensor_topology());
                fill_backing_iota(*act_send_src, mesh_device_, 0x100u + shard_mb);
            }
            if (grad_out) {
                grad_send_src = ttnn::create_device_tensor(
                    grad_out->get_backing_tensor().tensor_spec(),
                    mesh_device_.get(),
                    grad_out->get_backing_tensor().tensor_topology());
                fill_backing_iota(*grad_send_src, mesh_device_, 0x900u + shard_mb);
            }
            if (act_in) {
                act_recv_dst = ttnn::create_device_tensor(
                    act_in->get_backing_tensor().tensor_spec(),
                    mesh_device_.get(),
                    act_in->get_backing_tensor().tensor_topology());
            }
            if (grad_in) {
                grad_recv_dst = ttnn::create_device_tensor(
                    grad_in->get_backing_tensor().tensor_spec(),
                    mesh_device_.get(),
                    grad_in->get_backing_tensor().tensor_topology());
            }
        } else {
            if (act_out) {
                fill_backing_iota(act_out->get_backing_tensor(), mesh_device_, 0x100u + shard_mb);
            }
            if (grad_out) {
                fill_backing_iota(grad_out->get_backing_tensor(), mesh_device_, 0x900u + shard_mb);
            }
        }

        auto& cq = mesh_device_->mesh_command_queue();
        MeshWorkload act_recv_wl =
            (!use_ttnn_runtime_ops && act_in) ? make_recv_ack_workload(act_in.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload grad_recv_wl =
            (!use_ttnn_runtime_ops && grad_in) ? make_recv_ack_workload(grad_in.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload act_sig_wl = (!use_ttnn_runtime_ops && act_out)
                                      ? make_stress_signal_workload(act_out.get(), mesh_device_)
                                      : MeshWorkload{};
        MeshWorkload grad_sig_wl = (!use_ttnn_runtime_ops && grad_out)
                                       ? make_stress_signal_workload(grad_out.get(), mesh_device_)
                                       : MeshWorkload{};
        MeshWorkload act_gate_wl =
            (!use_ttnn_runtime_ops && act_out) ? make_gate_workload(act_out.get(), mesh_device_) : MeshWorkload{};
        MeshWorkload grad_gate_wl =
            (!use_ttnn_runtime_ops && grad_out) ? make_gate_workload(grad_out.get(), mesh_device_) : MeshWorkload{};

        auto do_send = [&](D2DStreamServiceSender* s, const std::optional<ttnn::Tensor>& src, MeshWorkload& sig_wl) {
            if (!s) {
                return;
            }
            if (use_ttnn_runtime_ops) {
                (void)ttnn::experimental::outbound_socket_service_sync(*s, *src);
                s->release_fabric_links();
            } else {
                s->release_fabric_links();
                EnqueueMeshWorkload(cq, sig_wl, /*blocking=*/false);
            }
        };

        auto do_recv = [&](D2DStreamServiceReceiver* r, const std::optional<ttnn::Tensor>& dst, MeshWorkload& recv_wl) {
            if (!r) {
                return;
            }
            r->release_fabric_links();
            if (use_ttnn_runtime_ops) {
                (void)ttnn::experimental::inbound_socket_service_sync(*r, 0u, 0u, dst);
            } else {
                EnqueueMeshWorkload(cq, recv_wl, /*blocking=*/false);
            }
        };

        auto run_loop = [&](uint32_t iters, bool record_timestamps) {
            do_send(act_out.get(), act_send_src, act_sig_wl);
            do_send(grad_out.get(), grad_send_src, grad_sig_wl);

            for (uint32_t i = 0; i < iters; ++i) {
                if (record_timestamps) {
                    EnqueueMeshWorkload(cq, ts_wl, /*blocking=*/false);  // t0: before Recv
                }

                do_recv(act_in.get(), act_recv_dst, act_recv_wl);
                do_recv(grad_in.get(), grad_recv_dst, grad_recv_wl);

                if (record_timestamps) {
                    EnqueueMeshWorkload(cq, ts_wl, /*blocking=*/false);  // t1: after Recv, before lease WAIT
                }

                if (act_in) {
                    act_in->wait_for_fabric_links();
                }
                if (grad_in) {
                    grad_in->wait_for_fabric_links();
                }
                if (act_out) {
                    act_out->wait_for_fabric_links();
                    if (!use_ttnn_runtime_ops) {
                        EnqueueMeshWorkload(cq, act_gate_wl, /*blocking=*/false);
                    }
                }
                if (grad_out) {
                    grad_out->wait_for_fabric_links();
                    if (!use_ttnn_runtime_ops) {
                        EnqueueMeshWorkload(cq, grad_gate_wl, /*blocking=*/false);
                    }
                }

                if (record_timestamps) {
                    EnqueueMeshWorkload(cq, ts_wl, /*blocking=*/false);  // t2: after WAIT (exact start of Compute!)
                }

                if (num_matmuls > 0) {
                    for (int m = 0; m < num_matmuls; ++m) {
                        (void)ttnn::operations::matmul::matmul(
                            *mm_a,
                            *mm_b,
                            /*transpose_a=*/false,
                            /*transpose_b=*/false,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            *mm_out);
                    }
                    if (intra_fabric_wl != nullptr) {
                        EnqueueMeshWorkload(cq, *intra_fabric_wl, /*blocking=*/false);
                    }
                }

                if (i + 1 < iters) {
                    do_send(act_out.get(), act_send_src, act_sig_wl);
                    do_send(grad_out.get(), grad_send_src, grad_sig_wl);
                }
            }
            Finish(cq);
        };

        ctx->barrier();
        run_loop(kWarmupIters, /*record_timestamps=*/false);
        reset_ts_buffers();
        Synchronize(*mesh_device_, std::nullopt);
        ctx->barrier();

        const auto t0 = std::chrono::steady_clock::now();
        run_loop(kTimedIters, /*record_timestamps=*/true);
        Synchronize(*mesh_device_, std::nullopt);
        ctx->barrier();
        const auto t1 = std::chrono::steady_clock::now();

        if (act_in) {
            const auto& check_t = use_ttnn_runtime_ops ? *act_recv_dst : act_in->get_backing_tensor();
            verify_backing_iota(check_t, mesh_device_, 0x100u + shard_mb, label.c_str());
        }
        if (grad_in) {
            const auto& check_t = use_ttnn_runtime_ops ? *grad_recv_dst : grad_in->get_backing_tensor();
            verify_backing_iota(check_t, mesh_device_, 0x900u + shard_mb, label.c_str());
        }

        // Read back on-device Tensix cycle timestamps across all chips of this stage and
        // average the worst-case chip wait per iteration (since step collectives synchronize chips).
        const double total_sec = std::chrono::duration<double>(t1 - t0).count();
        double sum_recv_ms = 0.0;
        double sum_lease_ms = 0.0;
        // Drop iter 0 (priming send preceded t0) and average steady-state iters 1 .. kTimedIters - 1.
        std::vector<std::vector<uint64_t>> chip_ts;
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> raw;
            tt::tt_metal::detail::ReadFromDeviceL1(
                tt::tt_metal::internal::get_device(*mesh_device_, coord), ts_core, ts_l1_addr, ts_buf_bytes, raw);
            const uint32_t count = raw.at(0);
            EXPECT_EQ(count, 3u * kTimedIters);
            std::vector<uint64_t> ts(count);
            for (uint32_t s = 0; s < count; ++s) {
                ts[s] = (static_cast<uint64_t>(raw[2u + 2u * s + 1u]) << 32) | raw[2u + 2u * s];
            }
            chip_ts.push_back(std::move(ts));
        }

        constexpr double kCyclesPerMs = 1343000.0;
        const uint32_t steady_iters = kTimedIters > 1 ? (kTimedIters - 1u) : 1u;
        for (uint32_t i = 1; i < kTimedIters; ++i) {
            double max_chip_recv_ms = 0.0;
            double max_chip_lease_ms = 0.0;
            for (const auto& ts : chip_ts) {
                const double r_ms = static_cast<double>(ts[3u * i + 1u] - ts[3u * i + 0u]) / kCyclesPerMs;
                const double l_ms = static_cast<double>(ts[3u * i + 2u] - ts[3u * i + 1u]) / kCyclesPerMs;
                if (r_ms + l_ms > max_chip_recv_ms + max_chip_lease_ms) {
                    max_chip_recv_ms = r_ms;
                    max_chip_lease_ms = l_ms;
                }
            }
            sum_recv_ms += max_chip_recv_ms;
            sum_lease_ms += max_chip_lease_ms;
        }
        double avg_recv_wait_ms = sum_recv_ms / static_cast<double>(steady_iters);
        double avg_lease_wait_ms = sum_lease_ms / static_cast<double>(steady_iters);
        // For OneWay cells, rank 0 only sends while rank 1 receives; report the receiver rank's wait on rank 0.
        if (!bidirectional) {
            std::array<double, 2> waits{avg_recv_wait_ms, avg_lease_wait_ms};
            auto span = ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(waits.data()), sizeof(waits));
            if (rank == 1) {
                ctx->send(span, Rank{0}, tt::tt_metal::distributed::multihost::Tag{0x4257});
            } else if (rank == 0) {
                ctx->recv(span, Rank{1}, tt::tt_metal::distributed::multihost::Tag{0x4257});
                avg_recv_wait_ms = waits[0];
                avg_lease_wait_ms = waits[1];
            }
        }

        act_in.reset();
        act_out.reset();
        grad_in.reset();
        grad_out.reset();
        ctx->barrier();

        const double ms_per_iter = (total_sec * 1000.0) / static_cast<double>(kTimedIters);
        const double gbps_per_dir =
            (static_cast<double>(stage_bytes) * static_cast<double>(kTimedIters)) / (total_sec * 1e9);
        return BenchResult{
            .name = label,
            .shard_bytes = shard_bytes,
            .stage_bytes_per_dir = stage_bytes,
            .ms_per_iter = ms_per_iter,
            .gbps_per_dir = gbps_per_dir,
            .gbps_total_bidir = bidirectional ? (2.0 * gbps_per_dir) : gbps_per_dir,
            .on_device_recv_wait_ms = avg_recv_wait_ms,
            .on_device_lease_wait_ms = avg_lease_wait_ms,
            .on_device_total_compute_wait_ms = avg_recv_wait_ms + avg_lease_wait_ms,
        };
    };

    std::vector<BenchResult> results;
    {
        const uint32_t mb = default_mb;
        const std::string tag = std::to_string(mb) + "MB/chip";
        results.push_back(run_socket_cell(
            "OneWay_2Lane_Direct_S1_" + tag,
            mb,
            false,
            false,
            false,
            2,
            2,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            1u));
        results.push_back(run_socket_cell(
            "OneWay_2Lane_Direct_S16_" + tag,
            mb,
            false,
            false,
            false,
            2,
            2,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            16u));
        results.push_back(run_socket_cell(
            "OneWay_2Lane_AllMux_S1_" + tag,
            mb,
            false,
            true,
            true,
            2,
            2,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            1u));
        results.push_back(run_socket_cell(
            "OneWay_2Lane_AllMux_S16_" + tag,
            mb,
            false,
            true,
            true,
            2,
            2,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            16u));
        results.push_back(run_socket_cell(
            "Bidir_1Lane_Plain_S1_" + tag,
            mb,
            true,
            false,
            false,
            1,
            1,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            1u));
        results.push_back(run_socket_cell(
            "Bidir_1Lane_Plain_S16_" + tag,
            mb,
            true,
            false,
            false,
            1,
            1,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            16u));
        results.push_back(run_socket_cell(
            "Bidir_2Lane_CableMux_S16_" + tag,
            mb,
            true,
            true,
            false,
            2,
            1,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            16u));
        results.push_back(run_socket_cell(
            "Bidir_2Lane_AllMux_S1_" + tag,
            mb,
            true,
            true,
            true,
            2,
            2,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            1u));
        results.push_back(run_socket_cell(
            "Bidir_2Lane_AllMux_S16_" + tag,
            mb,
            true,
            true,
            true,
            2,
            2,
            false,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            16u));
        results.push_back(run_socket_cell(
            "Bidir_2Lane_AllMux_TTNNOps_S16_" + tag,
            mb,
            true,
            true,
            true,
            2,
            2,
            true,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            16u));
    }

    // Stage compute + intra-mesh fabric CCL contention benchmark (at default_mb = 8 MB/chip):
    const auto mm_mapper =
        create_mesh_mapper(*mesh_device_, MeshMapperConfig{.placements = replicate_all(*mesh_device_)});
    const TensorLayout bf16_tile_dram(
        DataType::BFLOAT16,
        PageConfig(Layout::TILE),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const tt::tt_metal::TensorSpec spec_a(ttnn::Shape({1, 1, 2048, 4096}), bf16_tile_dram);
    const tt::tt_metal::TensorSpec spec_b(ttnn::Shape({1, 1, 4096, 2048}), bf16_tile_dram);
    const tt::tt_metal::TensorSpec spec_out(ttnn::Shape({1, 1, 2048, 2048}), bf16_tile_dram);
    const auto dummy_a = (*mm_mapper)(
        ttnn::Tensor::from_vector<bfloat16>(std::vector<bfloat16>(2048ull * 4096ull, bfloat16(0.01f)), spec_a));
    const auto dummy_b = (*mm_mapper)(
        ttnn::Tensor::from_vector<bfloat16>(std::vector<bfloat16>(4096ull * 2048ull, bfloat16(0.01f)), spec_b));
    ttnn::Tensor mm_a = dummy_a.to_device(mesh_device_.get());
    ttnn::Tensor mm_b = dummy_b.to_device(mesh_device_.get());
    const auto dummy_out = (*mm_mapper)(
        ttnn::Tensor::from_vector<bfloat16>(std::vector<bfloat16>(2048ull * 2048ull, bfloat16(0.0f)), spec_out));
    ttnn::Tensor mm_out =
        ttnn::create_device_tensor(dummy_out.tensor_spec(), mesh_device_.get(), dummy_out.tensor_topology());

    auto fabric_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh_device_.get(),
        CoreRangeSet(CoreRange{kStressWorkerCores.start_coord, kStressWorkerCores.start_coord}),
        /*initial_value=*/0,
        BufferType::L1);
    MeshWorkload intra_fabric_wl =
        make_stress_fabric_workload(mesh_device_, kStressWorkerCores, fabric_sem, /*incs_per_conn=*/8u);

    auto run_compute_only = [&](int num_matmuls) -> double {
        auto& cq = mesh_device_->mesh_command_queue();
        auto body = [&](uint32_t iters) {
            for (uint32_t i = 0; i < iters; ++i) {
                for (int m = 0; m < num_matmuls; ++m) {
                    (void)ttnn::operations::matmul::matmul(
                        mm_a,
                        mm_b,
                        false,
                        false,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        mm_out);
                }
                EnqueueMeshWorkload(cq, intra_fabric_wl, /*blocking=*/false);
            }
            Finish(cq);
        };
        ctx->barrier();
        body(kWarmupIters);
        Synchronize(*mesh_device_, std::nullopt);
        ctx->barrier();
        const auto t0 = std::chrono::steady_clock::now();
        body(kTimedIters);
        Synchronize(*mesh_device_, std::nullopt);
        ctx->barrier();
        const auto t1 = std::chrono::steady_clock::now();
        return (std::chrono::duration<double>(t1 - t0).count() * 1000.0) / static_cast<double>(kTimedIters);
    };

    for (int nm : {4, 16}) {
        const double compute_only_ms = run_compute_only(nm);
        const auto with_1lane_s1 = run_socket_cell(
            "Compute" + std::to_string(nm) + "MM_Bidir_1Lane_S1",
            default_mb,
            true,
            false,
            false,
            1,
            1,
            false,
            nm,
            &mm_a,
            &mm_b,
            &mm_out,
            &intra_fabric_wl,
            1u);
        const auto with_all_mux_s1 = run_socket_cell(
            "Compute" + std::to_string(nm) + "MM_Bidir_AllMux_S1",
            default_mb,
            true,
            true,
            true,
            2,
            2,
            false,
            nm,
            &mm_a,
            &mm_b,
            &mm_out,
            &intra_fabric_wl,
            1u);
        const auto with_all_mux_s16 = run_socket_cell(
            "Compute" + std::to_string(nm) + "MM_Bidir_AllMux_S16",
            default_mb,
            true,
            true,
            true,
            2,
            2,
            false,
            nm,
            &mm_a,
            &mm_b,
            &mm_out,
            &intra_fabric_wl,
            16u);
        const auto with_all_mux_ttnn_s16 = run_socket_cell(
            "Compute" + std::to_string(nm) + "MM_Bidir_AllMux_TTNNOps_S16",
            default_mb,
            true,
            true,
            true,
            2,
            2,
            true,
            nm,
            &mm_a,
            &mm_b,
            &mm_out,
            &intra_fabric_wl,
            16u);

        if (rank == 0) {
            auto fmt_cell = [&](const BenchResult& b) {
                return fmt::format(
                    "step={:.3f}ms (+{:.3f}ms) [on-dev wait={:.3f}ms: recv={:.3f}ms, lease={:.3f}ms]",
                    b.ms_per_iter,
                    b.ms_per_iter - compute_only_ms,
                    b.on_device_total_compute_wait_ms,
                    b.on_device_recv_wait_ms,
                    b.on_device_lease_wait_ms);
            };
            log_info(
                tt::LogMetal,
                "[bidir-compute-wait] mesh={} ({}x matmul + intra-mesh fabric): compute_only={:.3f}ms\n"
                "  1-lane plain (S=1)   : {}\n"
                "  2-lane AllMux (S=1)  : {}\n"
                "  2-lane AllMux (S=16) : {}\n"
                "  2-lane AllMux+TTNN16 : {}",
                mesh_device_->shape(),
                nm,
                compute_only_ms,
                fmt_cell(with_1lane_s1),
                fmt_cell(with_all_mux_s1),
                fmt_cell(with_all_mux_s16),
                fmt_cell(with_all_mux_ttnn_s16));
        }
    }

    if (rank == 0) {
        log_info(
            tt::LogMetal,
            "==================== BIDIRECTIONAL SOCKET BENCHMARK ({}) ====================",
            mesh_device_->shape());
        for (const auto& r : results) {
            log_info(
                tt::LogMetal,
                "[bidir-bench] {:<36s} | {:6.1f} MiB/dir | {:7.3f} ms/iter | {:6.2f} GB/s/dir | {:6.2f} GB/s total | "
                "wait={:.3f}ms (recv={:.3f}, lease={:.3f})",
                r.name,
                static_cast<double>(r.stage_bytes_per_dir) / (1024.0 * 1024.0),
                r.ms_per_iter,
                r.gbps_per_dir,
                r.gbps_total_bidir,
                r.on_device_total_compute_wait_ms,
                r.on_device_recv_wait_ms,
                r.on_device_lease_wait_ms);
        }
        log_info(tt::LogMetal, "====================================================================================");
    }
}

}  // namespace
}  // namespace ttnn::distributed::test
