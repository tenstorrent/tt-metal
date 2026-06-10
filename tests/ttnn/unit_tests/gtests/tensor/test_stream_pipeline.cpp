// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-host streaming-pipeline integration test: chains H2DStreamService into
// one or more D2DStreamService stages and harvests the result with a plain host
// read, proving the streaming services compose on one host.
//
//   Host ─H2D.forward_to_tensor─► h2d_backing (stage 0)
//        ─ relay worker ────────► d2d_sender_backing (stage 0)
//        ─ D2D fabric ──────────► d2d_receiver_backing (stage 1)
//        ─ relay worker ────────► [d2d_sender_backing (stage 1) ─► … for >2 stages]
//        ─ … last stage relay ──► output_tensor (last stage)
//        ─ host ReadShard(output_tensor) ⇒ verify == source
//
// A single generic relay worker (pipeline_relay_worker.cpp) serves every stage:
// the consumer handshake is identical whether the upstream is the H2D service or a
// D2D receiver, and the producer handshake is always the D2D sender's. Stage 0's
// upstream is H2D; every other stage's upstream is the previous boundary's D2D
// receiver; the last stage writes the output tensor instead of producing downstream.
//
// V0 single-host only (D2D create_pair owns both submeshes); multi-host is a
// separate effort. D2D runs in LEASE mode and the host drives the fabric-link
// handshake (release → transfer → wait) once per iteration, mirroring the
// production path where the model graph grants the links around each transfer.
// No CCLs compete here, so every grant succeeds — the point is to exercise the
// real per-iter control flow, not contention (that is the FabricLease* tests).
//
// Hardware requirements (skips cleanly otherwise): FABRIC_2D (D2D builds a
// MeshSocket pair), service cores (Blackhole / UBB Galaxy under Fast Dispatch), and
// >= num_stages devices to carve one 1x1 submesh per stage.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <ttnn/api/ttnn/distributed/distributed_configs.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>
#include <umd/device/types/arch.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

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
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshDevice;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ::tt::tt_metal::distributed::MeshShape;
using ::tt::tt_metal::distributed::MeshWorkload;
using ::tt::tt_metal::distributed::ReadShard;
using ::tt::tt_metal::distributed::SocketMemoryConfig;

// FABRIC_2D over the system mesh (D2D needs fabric; H2D is PCIe and unaffected).
using StreamPipelineTest = ::tt::tt_metal::GenericMeshDeviceFabric2DFixture;

// Mirror the ServiceCoreManager precondition so we skip cleanly elsewhere.
bool service_cores_supported() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    return cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE;
}

// Single worker core per stage (num_workers == 1).
const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};

// Fully-replicated placement sized to the submesh dimensionality (identity on 1x1).
ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(const MeshDevice& mesh) {
    return ttsl::SmallVector<MeshMapperConfig::Placement>(mesh.shape().dims(), MeshMapperConfig::Replicate{});
}

TensorSpec make_spec(const ttnn::Shape& global_shape) {
    return TensorSpec(
        global_shape,
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}));
}

// Carve `n` distinct 1x1 submeshes (one per pipeline stage) from the parent mesh,
// walking coords in row-major order.
std::vector<std::shared_ptr<MeshDevice>> carve_stages(MeshDevice& parent, uint32_t n) {
    const auto shape = parent.shape();
    std::vector<std::shared_ptr<MeshDevice>> stages;
    stages.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        stages.push_back(parent.create_submesh(MeshShape(1, 1), MeshCoordinate(i / shape[1], i % shape[1])));
    }
    return stages;
}

// The consumer (upstream) side a relay worker reads from + acks. Built from either
// an H2DStreamService or a D2DStreamServiceReceiver — both expose these getters, so
// one template covers both.
struct UpstreamHandle {
    const Tensor* backing = nullptr;
    DeviceAddr data_ready_sem_addr = 0;
    std::function<DeviceAddr(const MeshCoordinate&)> consumed_counter_addr;
    std::function<CoreCoord(const MeshCoordinate&)> service_core;
};

template <typename Svc>
UpstreamHandle make_upstream(Svc& svc) {
    return UpstreamHandle{
        .backing = &svc.get_backing_tensor(),
        .data_ready_sem_addr = svc.get_data_ready_sem_addr(),
        .consumed_counter_addr = [&svc](const MeshCoordinate& c) { return svc.get_consumed_counter_addr(c); },
        .service_core = [&svc](const MeshCoordinate& c) { return svc.get_service_core(c); },
    };
}

// The producer (downstream) side a relay worker writes to. Either a D2D sender
// (produce == true: the inverted sender handshake) or the terminal output tensor
// (produce == false: write only, no downstream handshake).
struct DownstreamHandle {
    const Tensor* dest = nullptr;
    bool produce = false;
    std::function<DeviceAddr(const MeshCoordinate&)> data_ready_counter_addr;
    std::function<CoreCoord(const MeshCoordinate&)> service_core;
};

// The relay only SIGNALS the D2D sender (data_ready_counter) and returns — it does
// not wait on the sender's consumed_sem (the host drives release/wait), so the
// handle doesn't carry consumed_sem.
DownstreamHandle make_downstream_producer(D2DStreamServiceSender& s) {
    return DownstreamHandle{
        .dest = &s.get_backing_tensor(),
        .produce = true,
        .data_ready_counter_addr = [&s](const MeshCoordinate& c) { return s.get_data_ready_counter_addr(c); },
        .service_core = [&s](const MeshCoordinate& c) { return s.get_service_core(c); },
    };
}

DownstreamHandle make_downstream_terminal(const Tensor& output) {
    return DownstreamHandle{.dest = &output, .produce = false};
}

// Build one stage's relay MeshWorkload (one program per coord). Uniform CT args;
// per-coord RT args for the two service cores' counters + physical NoC coords.
MeshWorkload build_relay_workload(
    MeshDevice& stage,
    const CoreRange& workers,
    uint32_t num_iters,
    const UpstreamHandle& up,
    const DownstreamHandle& down) {
    const auto* up_buf = up.backing->buffer();
    const uint32_t page_size = up_buf->aligned_page_size();
    const uint32_t num_pages = up_buf->num_pages();
    constexpr auto kScratchCb = CBIndex::c_0;

    MeshWorkload workload;
    for (const auto& coord : up.backing->tensor_topology().mesh_coords()) {
        auto program = CreateProgram();

        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, workers, cb_cfg);

        // Upstream backing and downstream dest share the per-shard spec, so one
        // TensorAccessorArgs set serves both (built from the upstream device buffer).
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
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/pipeline_relay_worker.cpp",
            workers,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = stage.get_device(coord);
        const auto up_svc_phys = device->worker_core_from_logical_core(up.service_core(coord));
        CoreCoord down_svc_phys{0, 0};
        uint32_t down_counter_addr = 0;
        if (down.produce) {
            down_svc_phys = device->worker_core_from_logical_core(down.service_core(coord));
            down_counter_addr = static_cast<uint32_t>(down.data_ready_counter_addr(coord));
        }

        for (const auto& wc : workers) {
            // Single-core worker grid → this worker owns the whole page range.
            // (Partition start/end_page here to scale to multi-core grids.)
            const std::vector<uint32_t> rt_args = {
                0u,
                num_pages,
                static_cast<uint32_t>(up.consumed_counter_addr(coord)),
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

// Build an N-stage pipeline once, then run num_iters single-tensor passes through
// it, validating each iter's data at the last stage before the next (see the
// per-iteration loop below for the production-mirroring cadence). Hangs loudly
// (Finish / wait_for_fabric_links never returns) if any stage's handshake or the
// fabric lease desyncs; mismatches surface per-iter so stale/dropped data is caught.
void run_pipeline(MeshDevice& parent, uint32_t num_stages, const ttnn::Shape& global_shape, uint32_t num_iters) {
    ASSERT_GE(num_stages, 2u);
    auto stages = carve_stages(parent, num_stages);
    const TensorSpec global_spec = make_spec(global_shape);

    // Stage 0 front end: H2D service with a worker grid (so the relay can handshake).
    H2DStreamService h2d(
        stages[0],
        H2DStreamService::Config{
            .global_spec = global_spec,
            .mapper = create_mesh_mapper(*stages[0], MeshMapperConfig{.placements = replicate_all(*stages[0])}),
            .socket_buffer_type = BufferType::L1,
            .fifo_size_bytes = 4096,
            .scratch_cb_size_bytes = 4096,
            .worker_cores = kWorkerCores,
        });

    // One D2D pair per stage boundary: d2d[i] forwards stage i → stage i+1.
    std::vector<std::pair<std::unique_ptr<D2DStreamServiceSender>, std::unique_ptr<D2DStreamServiceReceiver>>> d2d;
    for (uint32_t i = 0; i + 1 < num_stages; ++i) {
        d2d.push_back(D2DStreamService::create_pair(
            stages[i],
            stages[i + 1],
            D2DStreamConfig{
                .global_spec = global_spec,
                .mapper = create_mesh_mapper(*stages[i], MeshMapperConfig{.placements = replicate_all(*stages[i])}),
                .socket_mem_config = SocketMemoryConfig{BufferType::L1, /*fifo_size=*/4096},
                .sender_worker_cores = kWorkerCores,
                .receiver_worker_cores = kWorkerCores,
                // LEASE mode: drive the fabric-link handshake per iteration, mirroring
                // the production path where the model graph grants the links around
                // each transfer. No CCLs compete here, so every grant succeeds, but the
                // control flow (release → transfer → wait) is the real one.
                .share_fabric_links = true,
            }));
    }

    // Output tensor on the last stage, same per-shard spec/topology as the last D2D
    // receiver backing (so the terminal relay's accessor matches 1:1).
    D2DStreamServiceReceiver& last_recv = *d2d.back().second;
    Tensor output_tensor = create_device_tensor(
        last_recv.get_per_shard_spec(), stages[num_stages - 1].get(), last_recv.get_backing_tensor().tensor_topology());

    auto out_buf = output_tensor.device_storage().get_mesh_buffer_leak_ownership();
    const size_t num_u32 = global_spec.compute_packed_buffer_size_bytes() / sizeof(uint32_t);
    ASSERT_GT(num_u32, 0u);

    // Per-iteration loop. Push one tensor, then walk the stages IN ORDER. The body of
    // the stage loop is EXACTLY what one host does per iter in the distributed
    // deployment — see the comment inside. The `for` loop only exists because this one
    // process is standing in for all N hosts; in the real multi-host pipeline the loop
    // disappears and each host runs the body once, concurrently, touching only its own
    // local D2D endpoints (the four phases below become four single calls per host).
    //
    // On a single host the stages run sequentially, so each stage's outbound release
    // feeds the next stage's inbound within the same iter — carrying src all the way to
    // the output, so each iter validates its own data with no pipeline-fill latency.
    std::vector<uint32_t> readback;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        std::vector<uint32_t> src(num_u32);
        std::iota(src.begin(), src.end(), 1u + iter * 0x1000u);  // distinct per iter

        h2d.forward_to_tensor(
            ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(src.data()), src.size() * sizeof(uint32_t)));

        for (uint32_t i = 0; i < num_stages; ++i) {
            // ---- host i's per-iter work (collapses to single calls in multi-host) ----
            // Host i owns only its LOCAL endpoints: its inbound = the receiver of the
            // upstream boundary (d2d[i-1].second, physically on stage i), and its
            // outbound = the sender of the downstream boundary (d2d[i].first, on stage i).
            const bool has_inbound = (i > 0);                // stage 0's inbound is H2D, not a leased link
            const bool has_outbound = (i + 1 < num_stages);  // last stage harvests to output, no outbound

            // (1) wait: my endpoints are free — their previous-iter transfer drained
            //     (grant == 0). No-op on iter 0.
            if (has_inbound) {
                d2d[i - 1].second->wait_for_fabric_links();
            }
            if (has_outbound) {
                d2d[i].first->wait_for_fabric_links();
            }

            // (2) grant my INBOUND receiver — BEFORE my op — so it drains this iter's
            //     incoming data (the upstream host granted its sender at the end of its
            //     own turn, so the boundary now transfers into my inbound backing).
            if (has_inbound) {
                d2d[i - 1].second->release_fabric_links();
            }

            // (3) my op: consume inbound, produce outbound, return (decoupled from the D2D).
            UpstreamHandle up = has_inbound ? make_upstream(*d2d[i - 1].second) : make_upstream(h2d);
            DownstreamHandle down =
                has_outbound ? make_downstream_producer(*d2d[i].first) : make_downstream_terminal(output_tensor);
            MeshWorkload relay = build_relay_workload(*stages[i], kWorkerCores, /*num_iters=*/1, up, down);
            EnqueueMeshWorkload(stages[i]->mesh_command_queue(), relay, /*blocking=*/false);
            Finish(stages[i]->mesh_command_queue());

            // (4) grant my OUTBOUND sender — AFTER my op — so it forwards my output (the
            //     downstream host grants its receiver at the start of ITS turn, completing
            //     the two-host boundary handshake).
            if (has_outbound) {
                d2d[i].first->release_fabric_links();
            }
        }

        // src has reached the output tensor (last stage's relay Finished above).
        for (const auto& coord : output_tensor.tensor_topology().mesh_coords()) {
            readback.clear();
            ReadShard(stages[num_stages - 1]->mesh_command_queue(), readback, out_buf, coord);
            EXPECT_EQ(readback, src) << "pipeline output mismatch at iter " << iter << " coord " << coord;
        }
    }
}

bool enough_devices(MeshDevice& mesh, uint32_t num_stages) {
    return mesh.shape().dims() == 2 && mesh.num_devices() >= num_stages;
}

// 2 stages, one transfer: proves Host→H2D→D2D→output composes.
TEST_F(StreamPipelineTest, TwoStageSingleTransfer) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "Pipeline service cores require Blackhole or UBB Galaxy.";
    }
    if (!enough_devices(*this->mesh_device_, /*num_stages=*/2)) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices; got " << this->mesh_device_->shape();
    }
    run_pipeline(*this->mesh_device_, /*num_stages=*/2, ttnn::Shape({1, 1, 32, 64}), /*num_iters=*/1);
}

// 2 stages, several transfers: proves the persistent pipeline streams repeatedly.
TEST_F(StreamPipelineTest, TwoStageReuse) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "Pipeline service cores require Blackhole or UBB Galaxy.";
    }
    if (!enough_devices(*this->mesh_device_, /*num_stages=*/2)) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices; got " << this->mesh_device_->shape();
    }
    run_pipeline(*this->mesh_device_, /*num_stages=*/2, ttnn::Shape({1, 1, 32, 64}), /*num_iters=*/4);
}

// 3 stages: exercises a middle stage whose relay is D2D-consumer + D2D-producer (a
// D2D pair on both ends of one device), the multi-stage generalization.
TEST_F(StreamPipelineTest, ThreeStage) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "Pipeline service cores require Blackhole or UBB Galaxy.";
    }
    if (!enough_devices(*this->mesh_device_, /*num_stages=*/3)) {
        GTEST_SKIP() << "Need a 2D mesh with >= 3 devices; got " << this->mesh_device_->shape();
    }
    run_pipeline(*this->mesh_device_, /*num_stages=*/3, ttnn::Shape({1, 1, 32, 64}), /*num_iters=*/4);
}

}  // namespace
}  // namespace ttnn::distributed::test
