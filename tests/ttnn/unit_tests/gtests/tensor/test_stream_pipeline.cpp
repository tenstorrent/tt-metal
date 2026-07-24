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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/bfloat16.hpp>
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
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

#include "stream_service_test_utils.hpp"  // replicate_all, service_cores_supported, make_spec, worker_page_range, ...

namespace ttnn::distributed::test {
namespace {

using ::tt::CBIndex;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CircularBufferConfig;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CreateCircularBuffer;
using ::tt::tt_metal::CreateKernel;
using ::tt::tt_metal::CreateProgram;
using ::tt::tt_metal::DataMovementConfig;
using ::tt::tt_metal::DataMovementProcessor;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::DeviceAddr;
using ::tt::tt_metal::H2DStreamService;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::NOC;
using ::tt::tt_metal::SetRuntimeArgs;
using ::tt::tt_metal::TensorAccessorArgs;
using ::tt::tt_metal::distributed::EnqueueMeshWorkload;
using ::tt::tt_metal::distributed::Finish;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshDevice;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ::tt::tt_metal::distributed::MeshShape;
using ::tt::tt_metal::distributed::MeshWorkload;
using ::tt::tt_metal::distributed::SocketMemoryConfig;
using ttnn::D2DStreamConfig;
using ttnn::D2DStreamService;
using ttnn::D2DStreamServiceReceiver;
using ttnn::D2DStreamServiceSender;
using ttnn::Tensor;

// FABRIC_2D over the system mesh (D2D needs fabric; H2D is PCIe and unaffected).
using StreamPipelineTest = ::tt::tt_metal::GenericMeshDeviceFabric2DFixture;

// service_cores_supported / h2d_host_pinning_supported / replicate_all / make_spec /
// all_cores_for / worker_index / worker_page_range / fifo_bytes_for live in
// stream_service_test_utils.hpp (shared with the D2D stream-service gtest).

// Single worker core per stage (num_workers == 1) for the basic smoke tests; the
// shape/dtype/longer-pipeline tests pass the full compute grid (all_cores_for).
const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};

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
    // L1 address (uniform mesh-wide) where this service multicasts the metadata blob
    // into every worker core. Only invoked when metadata is enabled.
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

// The producer (downstream) side a relay worker writes to. Either a D2D sender
// (produce == true: the inverted sender handshake) or the terminal output tensor
// (produce == false: write only, no downstream handshake).
struct DownstreamHandle {
    const Tensor* dest = nullptr;
    bool produce = false;
    std::function<DeviceAddr(const MeshCoordinate&)> data_ready_counter_addr;
    std::function<CoreCoord(const MeshCoordinate&)> service_core;
    // Per-coord L1 address of the D2D sender service core's metadata buffer (the
    // designated worker forwards into it). Producer-only; only invoked with metadata.
    std::function<DeviceAddr(const MeshCoordinate&)> metadata_addr;
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
        .metadata_addr = [&s](const MeshCoordinate& c) { return s.get_metadata_addr(c); },
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
    const DownstreamHandle& down,
    uint32_t metadata_size_bytes = 0) {
    const auto* up_buf = up.backing->buffer();
    const uint32_t page_size = up_buf->aligned_page_size();
    const uint32_t num_pages = up_buf->num_pages();
    const bool metadata_enabled = metadata_size_bytes > 0;
    // The designated metadata writer (only when this stage produces downstream) is the
    // highest-id worker, matching the receiver service's multicast-to-all fan-out.
    const bool forwards_metadata = metadata_enabled && down.produce;
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
            metadata_enabled ? 1u : 0u,
            metadata_size_bytes,
            metadata_enabled ? static_cast<uint32_t>(up.metadata_addr()) : 0u,
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
            // Each worker owns its row-major page slice (remainder spread over the
            // first workers); empty ranges still handshake, so the service's
            // num_workers ack count is always satisfied.
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

// Build an N-stage pipeline once, then run num_iters single-tensor passes through
// it, validating each iter's data at the last stage before the next (see the
// per-iteration loop below for the production-mirroring cadence). Hangs loudly
// (Finish / wait_for_fabric_links never returns) if any stage's handshake or the
// fabric lease desyncs; mismatches surface per-iter so stale/dropped data is caught.
//
// Templated on the element type T so one harness covers every dtype: source data is
// built with from_vector<T> and verified with to_vector<T>, byte-faithful because
// every hop is a raw page copy. `modulus` bounds low-precision dtypes (src is a
// rotating modular iota (iter+i) % modulus, distinct per element AND rotated per iter
// so a stale page reads back wrong); pass modulus == 0 for a full-range uint32 iota.
// `use_all_cores` selects the full compute grid (exercises per-core page partitioning)
// vs a single worker core. When `metadata_size_bytes > 0`, each transfer also carries an
// inline metadata blob (made by `make_metadata`, default {-1,0,1+iter}); it is forwarded
// stage-by-stage and the last iter's blob is verified on every terminal worker core.
template <typename T>
void run_pipeline(
    MeshDevice& parent,
    uint32_t num_stages,
    const ttnn::Shape& global_shape,
    uint32_t num_iters,
    DataType dtype = DataType::UINT32,
    Layout layout = Layout::ROW_MAJOR,
    uint32_t modulus = 0,
    bool use_all_cores = false,
    uint32_t metadata_size_bytes = 0,
    std::function<std::vector<uint32_t>(uint32_t iter)> make_metadata = {}) {
    ASSERT_GE(num_stages, 2u);
    auto stages = carve_stages(parent, num_stages);
    const tt::tt_metal::TensorSpec global_spec = make_spec(global_shape, dtype, layout);
    const CoreRange workers = use_all_cores ? all_cores_for(*stages[0]) : kWorkerCores;
    const uint32_t fifo_bytes = fifo_bytes_for(global_spec);
    const bool metadata_enabled = metadata_size_bytes > 0;
    if (metadata_enabled && !make_metadata) {
        make_metadata = [](uint32_t iter) { return std::vector<uint32_t>{static_cast<uint32_t>(-1), 0u, 1u + iter}; };
    }

    // Stage 0 front end: H2D service with a worker grid (so the relay can handshake).
    H2DStreamService h2d(
        stages[0],
        H2DStreamService::Config{
            .global_spec = global_spec,
            .mapper = create_mesh_mapper(*stages[0], MeshMapperConfig{.placements = replicate_all(*stages[0])}),
            .socket_buffer_type = BufferType::L1,
            .fifo_size_bytes = fifo_bytes,
            .max_socket_page_size_bytes = fifo_bytes,
            .worker_cores = workers,
            .metadata_size_bytes = metadata_size_bytes,
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
                .socket_mem_config = SocketMemoryConfig{BufferType::L1, fifo_bytes},
                .sender_worker_cores = workers,
                .receiver_worker_cores = workers,
                .metadata_size_bytes = metadata_size_bytes,
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
    Tensor output_tensor = ttnn::create_device_tensor(
        last_recv.get_per_shard_spec(), stages[num_stages - 1].get(), last_recv.get_backing_tensor().tensor_topology());

    const uint32_t num_elems = static_cast<uint32_t>(global_shape.volume());
    ASSERT_GT(num_elems, 0u);
    auto mapper = create_mesh_mapper(*stages[0], MeshMapperConfig{.placements = replicate_all(*stages[0])});
    auto make_src = [&](uint32_t iter) {
        std::vector<T> v(num_elems);
        for (uint32_t i = 0; i < num_elems; ++i) {
            v[i] = modulus ? static_cast<T>(static_cast<float>((iter + i) % modulus))
                           : static_cast<T>(1u + iter * 0x1000u + i);  // full-range uint32 iota
        }
        return v;
    };

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
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const Tensor host_src = Tensor::from_vector<T>(make_src(iter), global_spec);
        std::vector<uint32_t> md;
        ttsl::Span<const std::byte> md_span;
        if (metadata_enabled) {
            md = make_metadata(iter);
            ASSERT_EQ(md.size() * sizeof(uint32_t), metadata_size_bytes);
            md_span = ttsl::Span<const std::byte>(
                reinterpret_cast<const std::byte*>(md.data()), md.size() * sizeof(uint32_t));
        }
        h2d.forward_to_tensor(ttnn::distributed::distribute_tensor(host_src, *mapper), md_span);

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
            MeshWorkload relay =
                build_relay_workload(*stages[i], workers, /*num_iters=*/1, up, down, metadata_size_bytes);
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
        // Single-chip (1x1) stages, so to_vector returns the one replica unambiguously.
        const std::vector<T> expected = host_src.to_vector<T>();
        const std::vector<T> got = output_tensor.to_vector<T>();
        EXPECT_EQ(got, expected) << "pipeline output mismatch at iter " << iter << " (dtype " << static_cast<int>(dtype)
                                 << ")";
    }

    // Metadata is forwarded stage-by-stage and the last D2D receiver multicasts it into
    // every terminal worker core's L1 (at the uniform get_metadata_addr()). The blob is
    // overwritten each iter, so assert the final iter's blob persisted everywhere.
    if (metadata_enabled) {
        const std::vector<uint32_t> expected_md = make_metadata(num_iters - 1);
        const uint32_t recv_md_addr = static_cast<uint32_t>(last_recv.get_metadata_addr());
        const CoreRange recv_workers = last_recv.get_worker_cores();
        std::vector<uint32_t> rb;
        for (const auto& coord : last_recv.get_backing_tensor().tensor_topology().mesh_coords()) {
            auto* device = stages[num_stages - 1]->get_device(coord);
            for (const auto& wc : recv_workers) {
                rb.clear();
                tt::tt_metal::detail::ReadFromDeviceL1(device, wc, recv_md_addr, metadata_size_bytes, rb);
                EXPECT_EQ(rb, expected_md)
                    << "pipeline metadata mismatch at coord " << coord << " core (" << wc.x << "," << wc.y << ")";
            }
        }
    }
}

bool enough_devices(MeshDevice& mesh, uint32_t num_stages) {
    return mesh.shape().dims() == 2 && mesh.num_devices() >= num_stages;
}

// Shared skip guard: service cores (Blackhole / UBB Galaxy), H2D host-DMA pinning
// (the front end pins a DMA buffer; iommu=pt hosts can't), and >= n_stages devices
// to carve one 1x1 submesh per stage. Lease mode leases the link per transfer, so
// the stages need not be colinear (unlike the own-mode chains).
#define PIPELINE_GUARD(n_stages)                                                                                    \
    if (!service_cores_supported()) {                                                                               \
        GTEST_SKIP() << "Pipeline service cores require Blackhole or UBB Galaxy.";                                  \
    }                                                                                                               \
    if (!h2d_host_pinning_supported()) {                                                                            \
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not "                     \
                        "iommu=pt); see notes/d2d_galaxy_h2d_pinning_failure.md.";                                  \
    }                                                                                                               \
    if (!enough_devices(*this->mesh_device_, (n_stages))) {                                                         \
        GTEST_SKIP() << "Need a 2D mesh with >= " << (n_stages) << " devices; got " << this->mesh_device_->shape(); \
    }

// ===========================================================================
// Basic 2/3-stage smoke (uint32, single worker core): the minimal composition
// proofs. Larger grids, shapes, dtypes, and longer pipelines are below.
// ===========================================================================

// 2 stages, several transfers: proves Host→H2D→D2D→output composes AND the persistent
// pipeline streams repeatedly (iter 0 alone is the single-transfer case).
TEST_F(StreamPipelineTest, TwoStageReuse) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(*this->mesh_device_, /*num_stages=*/2, ttnn::Shape({1, 1, 32, 64}), /*num_iters=*/4);
}

// 3 stages: exercises a middle stage whose relay is D2D-consumer + D2D-producer (a
// D2D pair on both ends of one device), the multi-stage generalization.
TEST_F(StreamPipelineTest, ThreeStage) {
    PIPELINE_GUARD(3);
    run_pipeline<uint32_t>(*this->mesh_device_, /*num_stages=*/3, ttnn::Shape({1, 1, 32, 64}), /*num_iters=*/4);
}

// ===========================================================================
// Shape matrix (uint32, full grid, 2 stages): exercises the page partition +
// socket chunk plan across edge shapes. The unevenness that stresses the
// partition is the PAGE COUNT (row count); the last dim only sets the page size.
//   - large         big page count, wide page > FIFO (one tensor page per chunk)
//   - uneven        711 pages mod num_workers -> remainder partition; wide page
//   - long-uneven   thin remainder (just over num_workers); wide page
//   - narrow-uneven 256 B page -> multi-page chunks AND 711 not divisible by the
//                   per-chunk count, so derive_chunk_plan's reduction loop fires
// ===========================================================================

// ~536 MB / tensor; 2 iters (still catches stale/ignored transfers via the
// per-iteration value bump) to bound runtime + fabric traffic.
TEST_F(StreamPipelineTest, ShapeLarge) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 8, 4096, 4096}),
        /*num_iters=*/2,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, ShapeUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, ShapeLongUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 155, 3712}),
        /*num_iters=*/3,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, ShapeNarrowUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 711, 64}),
        /*num_iters=*/3,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

// ===========================================================================
// Multi-dtype matrix (full grid, 2 stages, no metadata). BFLOAT16 / FLOAT32 /
// UINT8 are ROW_MAJOR; BFLOAT8_B / BFLOAT4_B are block-float -> TILE. Values are
// a rotating modular iota (bounded so low-precision dtypes don't saturate);
// verification is exact because every hop is a byte copy. Fixed-per-core shape
// (32 pages/core) for the dense even split, plus a [1,1,711,5120] uneven variant.
// ===========================================================================

// fixed_per_core_shape == 32 pages/core, 512 u32/page (dense, perfectly even).
ttnn::Shape fixed_per_core_shape(const CoreRange& workers) {
    return ttnn::Shape({1, 1, 32u * core_range_volume(workers), 512u});
}

TEST_F(StreamPipelineTest, DtypeBfloat16) {
    PIPELINE_GUARD(2);
    const auto shape = fixed_per_core_shape(all_cores_for(*this->mesh_device_));
    run_pipeline<bfloat16>(
        *this->mesh_device_,
        /*num_stages=*/2,
        shape,
        /*num_iters=*/2,
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        /*modulus=*/256,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeFloat32) {
    PIPELINE_GUARD(2);
    const auto shape = fixed_per_core_shape(all_cores_for(*this->mesh_device_));
    run_pipeline<float>(
        *this->mesh_device_,
        /*num_stages=*/2,
        shape,
        /*num_iters=*/2,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        /*modulus=*/4096,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeUint8) {
    PIPELINE_GUARD(2);
    const auto shape = fixed_per_core_shape(all_cores_for(*this->mesh_device_));
    run_pipeline<uint8_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        shape,
        /*num_iters=*/2,
        DataType::UINT8,
        Layout::ROW_MAJOR,
        /*modulus=*/128,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeBfloat8B) {
    PIPELINE_GUARD(2);
    const auto shape = fixed_per_core_shape(all_cores_for(*this->mesh_device_));
    run_pipeline<float>(
        *this->mesh_device_,
        /*num_stages=*/2,
        shape,
        /*num_iters=*/2,
        DataType::BFLOAT8_B,
        Layout::TILE,
        /*modulus=*/256,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeBfloat4B) {
    PIPELINE_GUARD(2);
    const auto shape = fixed_per_core_shape(all_cores_for(*this->mesh_device_));
    run_pipeline<float>(
        *this->mesh_device_,
        /*num_stages=*/2,
        shape,
        /*num_iters=*/2,
        DataType::BFLOAT4_B,
        Layout::TILE,
        /*modulus=*/64,
        /*use_all_cores=*/true);
}

// Same dtype matrix on a fixed uneven shape [1,1,711,5120] (711 pages mod workers).
TEST_F(StreamPipelineTest, DtypeBfloat16ShapeUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<bfloat16>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        /*modulus=*/256,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeFloat32ShapeUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<float>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        /*modulus=*/4096,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeUint8ShapeUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<uint8_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::UINT8,
        Layout::ROW_MAJOR,
        /*modulus=*/128,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeBfloat8BShapeUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<float>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::BFLOAT8_B,
        Layout::TILE,
        /*modulus=*/256,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, DtypeBfloat4BShapeUneven) {
    PIPELINE_GUARD(2);
    run_pipeline<float>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::BFLOAT4_B,
        Layout::TILE,
        /*modulus=*/64,
        /*use_all_cores=*/true);
}

// ===========================================================================
// Longer pipelines (full grid): 4-stage (~Quietbox) and 8-stage (~Loudbox). In
// lease mode each boundary leases the link per transfer, so the stages need not
// be colinear. num_iters > 1 covers reuse (the chain is rebuilt-free; each iter
// re-enqueues the per-stage relays against the persistent services).
// ===========================================================================

TEST_F(StreamPipelineTest, FourStageAllCores) {
    PIPELINE_GUARD(4);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/4,
        ttnn::Shape({1, 1, 32, 64}),
        /*num_iters=*/4,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, FourStageLargeAllCores) {
    PIPELINE_GUARD(4);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/4,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, EightStageAllCores) {
    PIPELINE_GUARD(8);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/8,
        ttnn::Shape({1, 1, 32, 64}),
        /*num_iters=*/4,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

TEST_F(StreamPipelineTest, EightStageLargeAllCores) {
    PIPELINE_GUARD(8);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/8,
        ttnn::Shape({1, 1, 711, 5120}),
        /*num_iters=*/3,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true);
}

// ===========================================================================
// Metadata (uint32, full grid): an inline blob ships with every transfer and is
// forwarded stage-by-stage; the last iter's blob is verified on every terminal
// worker core. 1-word (smallest), 3-word triple {-1,0,base}, and a full
// socket-page iota (4096 B) cover the metadata size range. The 3-stage variant
// proves a middle relay forwards the blob (consume + produce) like the stage-0
// bridge does.
// ===========================================================================

constexpr uint32_t kMetadataWord = static_cast<uint32_t>(sizeof(uint32_t));
constexpr uint32_t kMetadataTriple = 3u * static_cast<uint32_t>(sizeof(uint32_t));
constexpr uint32_t kMetadataFullPageWords = 1024u;  // 4096 B == one socket page

// 2 stages, 3-word triple {-1,0,1+iter}: basic end-to-end metadata.
TEST_F(StreamPipelineTest, MetadataTriple) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 32, 512}),
        /*num_iters=*/4,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true,
        kMetadataTriple);
}

// Single metadata word (4 B): smallest payload, distinct from the 3-word triple.
TEST_F(StreamPipelineTest, MetadataOneWord) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 32, 512}),
        /*num_iters=*/4,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true,
        kMetadataWord,
        [](uint32_t iter) { return std::vector<uint32_t>{1u + iter}; });
}

// Full socket-page metadata: 1024 uint32 words (4096 B) iota [0..1023].
TEST_F(StreamPipelineTest, MetadataFullPage) {
    PIPELINE_GUARD(2);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/2,
        ttnn::Shape({1, 1, 32, 512}),
        /*num_iters=*/2,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true,
        kMetadataFullPageWords * kMetadataWord,
        [](uint32_t /*iter*/) {
            std::vector<uint32_t> words(kMetadataFullPageWords);
            std::iota(words.begin(), words.end(), 0u);
            return words;
        });
}

// 3 stages with metadata: the middle relay forwards the blob (consume + produce) like
// the stage-0 bridge does, so the final {-1,0,base} must reach the terminal workers.
TEST_F(StreamPipelineTest, MetadataThreeStage) {
    PIPELINE_GUARD(3);
    run_pipeline<uint32_t>(
        *this->mesh_device_,
        /*num_stages=*/3,
        ttnn::Shape({1, 1, 32, 512}),
        /*num_iters=*/4,
        DataType::UINT32,
        Layout::ROW_MAJOR,
        /*modulus=*/0,
        /*use_all_cores=*/true,
        kMetadataTriple);
}

}  // namespace
}  // namespace ttnn::distributed::test
