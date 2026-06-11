// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-process (multi-host) D2DStreamService forward-pipeline test.
//
// A forward pipeline of `world_size` stages, one process per stage, derived
// entirely from the DistributedContext: rank == stage, with no hardcoded rank
// count and no assumptions about mesh shape/size. The launch (tt-run
// --rank-binding + mesh-graph descriptor) decides how many meshes/processes exist
// and how they are fabric-chained; the test just adapts (GTEST_SKIP if < 2 ranks).
//
//     rank 0           rank i (0<i<N-1)          rank N-1
//   ┌──────────┐     ┌────────────────┐        ┌──────────┐
//   │ source   │ D2D │ relay (recv +  │  D2D   │ consumer │
//   │  (iota)  │ ──► │  forward)      │ ─ … ─► │ + verify │
//   └──────────┘     └────────────────┘        └──────────┘
//
// UNBUNDLED op/sync model — the point of this test. A real model-graph op (matmul
// / CCL) OWNS its handshake SIGNALS (it incs data_ready_counter after it produces,
// and inbound consumed_counter after it reads) but CANNOT self-gate: it must not
// overwrite the outbound D2D backing tensor until the sender service has forwarded
// the previous iter. A black-box op can't spin a semaphore before it runs, so that
// overwrite-gate is a SEPARATE op — `d2d_sync` — that waits the sender's
// consumed_sem. So each stage's per-iter sequence is the production shape:
//
//     [d2d_sync GATE: wait outbound consumed_sem]   // separate op; skip iter 0
//     op  (produces; owns data_ready / consumed_counter incs)
//     wait_for_fabric_links → (ccl) → release_fabric_links   // one release/iter, after the op
//
// The lease (wait/release_fabric_links) is the service↔graph fabric-link
// arbitration; it is NOT the data handshake and there is exactly one release per
// iter, after the op. (A CCL is just another op; it syncs its own fabric use.)
//
// Kernels: rank 0 uses the gate-free `d2d_chain_source` (write iota + inc
// data_ready); middle/last reuse `pipeline_relay_worker` (it bundles only the
// natural inbound input-wait + both signals, no outbound gate); the outbound
// overwrite-gate is the standalone `d2d_sync`. All by repo-relative path.
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
#include <tt_stl/small_vector.hpp>

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"

#include <ttnn/api/ttnn/distributed/distributed_configs.hpp>
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"
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
constexpr uint32_t kFillBase = 1u;  // rank-0 source iota base; end stage verifies kFillBase + iter

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

// --- per-stage workloads (one program per coord; reuse kernels by repo path) ---

// rank 0 op: gate-free source. Writes iota (kFillBase + iter base) into the outbound
// sender backing + incs the sender's data_ready_counter (owns the signal). NO gate.
MeshWorkload make_source_workload(
    D2DStreamServiceSender* sender, const std::shared_ptr<MeshDevice>& mesh, uint32_t fill_base) {
    const auto& coords = sender->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* buf = sender->get_backing_tensor().buffer();
    const uint32_t page_size = buf->aligned_page_size();
    const uint32_t num_pages = buf->num_pages();
    const CoreRange worker_cores = sender->get_worker_cores();
    const uint32_t num_workers = core_range_volume(worker_cores);
    constexpr auto kScratchCb = CBIndex::c_0;

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        const auto* dbuf = sender->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(buf->address()),
            page_size,
            fill_base,
            static_cast<uint32_t>(kScratchCb),
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_chain_source.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh->get_device(coord);
        const auto svc_phys = device->worker_core_from_logical_core(sender->get_service_core(coord));
        for (const auto& wc : worker_cores) {
            const auto [start_page, end_page] =
                worker_page_range(worker_index(wc, worker_cores), num_workers, num_pages);
            const std::vector<uint32_t> rt_args = {
                static_cast<uint32_t>(sender->get_data_ready_counter_addr(coord)),
                static_cast<uint32_t>(svc_phys.x),
                static_cast<uint32_t>(svc_phys.y),
                start_page,
                end_page,
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// middle/last op: pipeline_relay_worker copies the inbound receiver backing into a
// downstream dest, spinning the inbound data_ready_sem (natural input-wait, bundled)
// and incing the inbound consumed_counter + (if producing) the outbound
// data_ready_counter — the op OWNS those signals. NO outbound gate (that's d2d_sync).
// produce=true → dest is the outbound sender backing; produce=false → dest is `output`.
MeshWorkload make_relay_like_workload(
    D2DStreamServiceReceiver* inbound,
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
// LEASE mode with one release per iter after the op. Hangs loudly (a gate spin, a
// MeshSocket handshake, or a Finish never returns) on any desync; the end stage's
// per-coord readback catches stale/dropped data.
TEST_F(CrossProcessD2DFixture, ForwardChain) {
    const auto& ctx = DistributedContext::get_current_world();
    // int (not uint32_t): multihost::Rank wraps int, so Rank{stage ± 1} must not narrow.
    const int world_size = *ctx->size();
    const int stage = *ctx->rank();
    if (world_size < 2) {
        GTEST_SKIP() << "D2D forward chain needs >= 2 ranks; got " << world_size;
    }

    constexpr uint32_t kNumIters = 10;
    const ttnn::Shape global_shape({1, 1, 32, 64});  // shape is irrelevant (mapper replicates)

    const bool has_inbound = stage > 0;                // receiver of pair (stage-1, stage)
    const bool has_outbound = stage + 1 < world_size;  // sender   of pair (stage, stage+1)

    // Construction ordering is load-bearing: inbound receiver BEFORE outbound sender,
    // so the per-endpoint handshakes cascade 0↔1, 1↔2, … without deadlock.
    std::unique_ptr<D2DStreamServiceReceiver> inbound;
    std::unique_ptr<D2DStreamServiceSender> outbound;
    if (has_inbound) {
        inbound = D2DStreamService::create_receiver(
            mesh_device_,
            make_cfg(mesh_device_, global_shape),
            D2DEndpointConfig{
                .sender_rank = Rank{stage - 1}, .receiver_rank = Rank{stage}, .distributed_context = ctx});
    }
    if (has_outbound) {
        outbound = D2DStreamService::create_sender(
            mesh_device_,
            make_cfg(mesh_device_, global_shape),
            D2DEndpointConfig{
                .sender_rank = Rank{stage}, .receiver_rank = Rank{stage + 1}, .distributed_context = ctx});
    }

    // All services across all ranks must be resident before any data flows.
    ctx->barrier();

    // End stage's output tensor (same per-shard spec/topology as its inbound backing).
    Tensor output;
    if (has_inbound && !has_outbound) {
        output = create_device_tensor(
            inbound->get_per_shard_spec(), mesh_device_.get(), inbound->get_backing_tensor().tensor_topology());
    }

    auto& cq = mesh_device_->mesh_command_queue();
    for (uint32_t iter = 0; iter < kNumIters; ++iter) {
        // (a) Outbound overwrite-gate — the SEPARATE d2d_sync op, before the producing
        //     op. Skipped on iter 0 (no prior forward to wait on).
        if (has_outbound && iter > 0) {
            auto gate = make_gate_workload(outbound.get(), mesh_device_);
            EnqueueMeshWorkload(cq, gate, /*blocking=*/false);
            Finish(cq);
        }

        // (b) Grant BOTH endpoints together — a boundary's sender (rank i) and receiver
        //     (rank i+1) are one socket, so its two halves must be granted in the same
        //     step; releasing per-rank at the same loop position lines them up. The
        //     receiver then drains this iter's data (so the op's input-wait fires) and
        //     the sender, granted, waits for the op's data_ready before forwarding.
        if (has_inbound) {
            inbound->release_fabric_links();
        }
        if (has_outbound) {
            outbound->release_fabric_links();
        }

        // (c) The op (one iter): waits its upstream's data (cascade — stages unblock in
        //     order within the iter), produces, owns its handshake signals.
        MeshWorkload op;
        if (!has_inbound) {
            op = make_source_workload(outbound.get(), mesh_device_, kFillBase + iter);
        } else if (has_outbound) {
            op = make_relay_like_workload(
                inbound.get(),
                mesh_device_,
                static_cast<uint32_t>(outbound->get_backing_tensor().buffer()->address()),
                /*produce=*/true,
                outbound.get());
        } else {
            op = make_relay_like_workload(
                inbound.get(),
                mesh_device_,
                static_cast<uint32_t>(output.buffer()->address()),
                /*produce=*/false,
                nullptr);
        }
        EnqueueMeshWorkload(cq, op, /*blocking=*/false);
        Finish(cq);

        // (d) Wait BOTH endpoints back off the link before the next iter — the
        //     lease ping-pong (one grant per iter; don't re-grant until the prior
        //     transfer drained and the service released the link).
        if (has_inbound) {
            inbound->wait_for_fabric_links();
        }
        if (has_outbound) {
            outbound->wait_for_fabric_links();
        }
    }

    // End stage: the final iter's iota (kFillBase + kNumIters - 1) that rank 0 produced
    // must have survived all N-1 fabric hops, copied verbatim at each relay.
    if (has_inbound && !has_outbound) {
        expect_output_tensor_iota(output, mesh_device_, kFillBase + kNumIters - 1);
    }

    Synchronize(mesh_device_.get(), std::nullopt);
    ctx->barrier();
}

}  // namespace
}  // namespace ttnn::distributed::test
