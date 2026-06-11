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
// Why a thread and not a separate "host" process/rank: in this unified fabric world
// every rank owns a mesh, and a device-owning process can't drive cross-process H2D
// into another process's device (a mesh-less feeder rank would also fight the fixture's
// world_size==num_meshes requirement). forward_to_tensor's push is a direct PCIe/sysmem
// write, independent of the mesh command queue, so a separate thread of stage 0's own
// (device-owning) process is the way to feed concurrently.
//
// UNBUNDLED op/sync model — the point of this test. A real model-graph op (matmul
// / CCL) OWNS its handshake SIGNALS (it incs data_ready_counter after it produces,
// and inbound consumed_counter after it reads) but CANNOT self-gate: it must not
// overwrite the outbound D2D backing tensor until the sender service has forwarded
// the previous iter. A black-box op can't spin a semaphore before it runs, so that
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
//   tt-run --rank-binding <rank_binding.yaml> --mpi-args "--allow-run-as-root --tag-output" \
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
#include "ttnn/tensor/d2d_stream_service.hpp"
#include "ttnn/tensor/socket_services.hpp"
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
using ::tt::tt_metal::create_device_tensor;
using ::tt::tt_metal::CreateCircularBuffer;
using ::tt::tt_metal::CreateKernel;
using ::tt::tt_metal::CreateProgram;
using ::tt::tt_metal::D2DEndpointConfig;
using ::tt::tt_metal::D2DStreamConfig;
using ::tt::tt_metal::D2DStreamService;
using ::tt::tt_metal::D2DStreamServiceReceiver;
using ::tt::tt_metal::D2DStreamServiceSender;
using ::tt::tt_metal::DataMovementConfig;
using ::tt::tt_metal::DataMovementProcessor;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::H2DStreamService;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::NOC;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::SetRuntimeArgs;
using ::tt::tt_metal::Tensor;
using ::tt::tt_metal::TensorAccessorArgs;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::EnqueueMeshWorkload;
using ::tt::tt_metal::distributed::Finish;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshDevice;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ::tt::tt_metal::distributed::MeshWorkload;
using ::tt::tt_metal::distributed::ReadShard;
using ::tt::tt_metal::distributed::SocketMemoryConfig;
using ::tt::tt_metal::distributed::Synchronize;
using ::tt::tt_metal::distributed::multihost::DistributedContext;
using ::tt::tt_metal::distributed::multihost::Rank;

// Each rank owns its local mesh; world_size == number of meshes in the descriptor.
using CrossProcessD2DFixture = tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture;

const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};  // single worker core
constexpr uint32_t kFillBase = 1u;  // stage-0 source iota base; end stage verifies kFillBase + iter

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

// Fresh config per call (the mapper is moved into each create_* call; a middle rank
// builds two services so it needs two mappers). UINT32 ROW_MAJOR DRAM, replicated on
// the local mesh, L1 socket FIFO, LEASE mode (the host drives the per-transfer grant).
D2DStreamConfig make_cfg(const std::shared_ptr<MeshDevice>& mesh, const ttnn::Shape& global_shape) {
    const TensorSpec global_spec(
        global_shape,
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}));
    return D2DStreamConfig{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh, MeshMapperConfig{.placements = replicate_all(*mesh)}),
        .socket_mem_config = SocketMemoryConfig{BufferType::L1, /*fifo_size=*/4096u},
        .sender_worker_cores = kWorkerCores,
        .receiver_worker_cores = kWorkerCores,
        .share_fabric_links = true,  // LEASE mode (host-driven per-transfer grants)
    };
}

// --- stage-0 host feed (in-process H2DStreamService) ---

// H2D socket FIFO / scratch sizing: one tensor page rounded up to a 4 KB floor
// (mirrors fifo_bytes_for in test_d2d_stream_service.cpp; H2D streams page-by-page,
// so a page-sized FIFO is correct and keeps L1 use bounded).
uint32_t fifo_bytes_for(const TensorSpec& spec) {
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
    const std::shared_ptr<MeshDevice>& mesh, const TensorSpec& global_spec) {
    const uint32_t fifo_bytes = fifo_bytes_for(global_spec);
    H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh, MeshMapperConfig{.placements = replicate_all(*mesh)}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = fifo_bytes,
        .scratch_cb_size_bytes = fifo_bytes,
        .worker_cores = kWorkerCores,
        .metadata_size_bytes = 0,
    };
    return std::make_unique<H2DStreamService>(mesh, std::move(cfg));
}

// Push one iter's iota token (global element i = base + i) into the H2D service. The
// mapper replicates it to every shard; the stage-0 relay then copies it verbatim into
// the outbound D2D backing. forward_to_tensor returns once the bytes are in the socket
// FIFOs; the caller's barrier() blocks until the device kernel drained them.
void h2d_push_token(H2DStreamService& h2d_service, uint32_t num_u32, uint32_t base) {
    const std::vector<uint32_t> token = make_iota_u32(num_u32, base);
    const auto bytes =
        ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(token.data()), token.size() * sizeof(uint32_t));
    h2d_service.forward_to_tensor(bytes);
}

// Body of stage 0's feeder THREAD: streams num_iters iota tokens into stage 0's own
// (in-process) H2D service, concurrently with the main thread's per-iter device loop.
// This is what decouples the host feed from the device loop — the push runs ahead,
// flow-controlled by the page-sized FIFO + the service<->relay handshake (it blocks when
// the FIFO fills; the relay drains one token per iter), so stage 0's device loop is
// uniform with every other stage (no in-loop push).
//
// In-process ON PURPOSE: a dedicated feeder RANK can't do this — in the unified fabric
// world every rank owns a mesh, and a device-owning process can't drive cross-process
// H2D into another process's device (and a mesh-less feeder rank fights the fixture's
// world_size==num_meshes requirement). forward_to_tensor's push is a direct PCIe/sysmem
// write, independent of the mesh command queue the main thread drives — so a separate
// thread of the SAME (device-owning) process is the way to feed concurrently.
void run_h2d_feed_loop(H2DStreamService& h2d_service, const ttnn::Shape& global_shape, uint32_t num_iters) {
    const uint32_t num_u32 = static_cast<uint32_t>(global_shape.volume());
    for (uint32_t i = 0; i < num_iters; ++i) {
        h2d_push_token(h2d_service, num_u32, kFillBase + i);
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
MeshWorkload make_gate_workload(D2DStreamServiceSender* sender, const std::shared_ptr<MeshDevice>& mesh) {
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
// arbitrates real contention on that EDM channel. The downstream FabricNodeId is
// re-derived here from the receiver rank (the socket resolved the same one
// internally) — kept in the test because it's purely workload-specific, not a
// D2DStreamService API concern. Runs on the sender's worker grid (off the service
// core), one program per coord.
MeshWorkload make_stub_fabric_workload(
    D2DStreamServiceSender* sender, const std::shared_ptr<MeshDevice>& mesh, Rank downstream_rank) {
    const auto& coords = sender->get_backing_tensor().tensor_topology().mesh_coords();
    const CoreRange worker_cores = sender->get_worker_cores();

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
            worker_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        for (const auto& wc : worker_cores) {
            std::vector<uint32_t> rt_args;
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_node, downstream_node, links.front(), program, wc, rt_args);
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// Assert `output` holds the iota (base + i) on every coord.
void expect_output_tensor_iota(const Tensor& output, const std::shared_ptr<MeshDevice>& mesh, uint32_t base) {
    auto mesh_buffer = output.device_storage().get_mesh_buffer_leak_ownership();
    const size_t num_u32 = output.buffer()->size() / sizeof(uint32_t);
    const std::vector<uint32_t> expected = make_iota_u32(num_u32, base);
    std::vector<uint32_t> readback;
    for (const auto& coord : output.tensor_topology().mesh_coords()) {
        readback.clear();
        ReadShard(mesh->mesh_command_queue(), readback, mesh_buffer, coord);
        EXPECT_EQ(readback, expected) << "output mismatch at " << coord << " (iota base " << base << ")";
    }
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

    // [B1] All services resident across all ranks before any data flows.
    ctx->barrier();

    // Stage 0: spawn the feeder thread now (its H2D service is up). It streams tokens
    // concurrently with the device loop below — decoupling the push from the loop — and
    // is joined after the loop. Default-constructed (non-joinable) on every other stage.
    std::thread h2d_feeder;
    if (is_stage0) {
        h2d_feeder = std::thread([&] { run_h2d_feed_loop(*h2d_service, global_shape, kNumIters); });
    }

    // End stage's output tensor (same per-shard spec/topology as its inbound backing).
    Tensor output;
    if (has_inbound && !has_outbound) {
        output = create_device_tensor(
            inbound->get_per_shard_spec(), mesh_device_.get(), inbound->get_backing_tensor().tensor_topology());
    }

    // Op parameters are loop-invariant (backing/output addresses are fixed at allocation):
    // every stage produces downstream except the last; the dest is the outbound backing
    // while producing, else the end-stage output tensor.
    const bool produce = has_outbound;
    const uint32_t dest_addr = produce ? static_cast<uint32_t>(outbound->get_backing_tensor().buffer()->address())
                                       : static_cast<uint32_t>(output.buffer()->address());
    D2DStreamServiceSender* const downstream = produce ? outbound.get() : nullptr;

    // FILL the pipeline: prime the inbound receiver so iter 0's input is received before the
    // first op reads it. Each iteration's release then grants the NEXT iter's receive
    // (steady-state pipelining), so this priming is the only "extra" release.
    if (has_inbound) {
        inbound->release_fabric_links();
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
        MeshWorkload op =
            is_stage0 ? make_relay_like_workload(h2d_service.get(), mesh_device_, dest_addr, produce, downstream)
                      : make_relay_like_workload(inbound.get(), mesh_device_, dest_addr, produce, downstream);
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
    }

    // Stage 0: the feeder has pushed all tokens (its barrier() returns once the device
    // drained them, which the loop above did). Join before leaving the scope its lambda
    // captured by reference.
    if (h2d_feeder.joinable()) {
        h2d_feeder.join();
    }

    // End stage: the final iter's iota (kFillBase + kNumIters - 1) that the host feed
    // produced must have survived every fabric hop, copied verbatim at each relay.
    if (has_inbound && !has_outbound) {
        expect_output_tensor_iota(output, mesh_device_, kFillBase + kNumIters - 1);
    }

    Synchronize(mesh_device_.get(), std::nullopt);
    ctx->barrier();  // [B2]
}

}  // namespace
}  // namespace ttnn::distributed::test
