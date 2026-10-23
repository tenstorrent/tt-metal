// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2DStreamService unit tests, ordered from cheapest to most thorough:
//   * creatability        — create_pair runs end-to-end and the getters return
//                            sane state (no kernels exercised),
//   * sync resources      — every worker-sync address is allocated / non-zero,
//   * transfer            — a single host-driven transfer lands correctly,
//   * handshake           — real worker workloads drive transfers through the
//                            full sender/receiver handshake,
//   * reuse / recreate    — the persistent service survives many transfers, and
//                            a torn-down pair fully releases its resources.
//
// These are OWN-mode (the service holds the fabric link) basic/unit tests plus one
// own-mode 3-stage chain. The exhaustive lease-mode coverage — every shape/dtype, a
// metadata size matrix, and longer (4/8-stage) pipelines — lives in the StreamPipeline
// gtest (test_stream_pipeline.cpp), since lease mode is the production path.
//
// Hardware requirements (tests skip cleanly otherwise):
//   * Fabric: create_pair builds a MeshSocket pair, which handshakes over the
//     control plane. We use a FABRIC_2D fixture.
//   * Service cores: ServiceCoreManager only claims cores on Blackhole / UBB
//     Galaxy clusters under Fast Dispatch (mirrors the precondition inside
//     create_pair), so we skip on other clusters.
//   * >= 2x2 mesh: a sender/receiver pair needs two distinct submeshes
//     (1x1<->1x1 for single-chip cases; 1x2<->1x2 for multi-socket cases).

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
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
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/small_vector.hpp>
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

#include "stream_service_test_utils.hpp"  // replicate_all

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
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::NOC;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::SetRuntimeArgs;
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
using ::tt::tt_metal::distributed::WriteShard;
using ttnn::D2DStreamConfig;
using ttnn::D2DStreamService;
using ttnn::D2DStreamServiceReceiver;
using ttnn::D2DStreamServiceSender;

// service_cores_supported / h2d_host_pinning_supported live in
// stream_service_test_utils.hpp (shared with the stream-pipeline gtest).

// FABRIC_2D over the system mesh (the full Galaxy on a UBB system).
using D2DStreamServiceTest = tt::tt_metal::GenericMeshDeviceFabric2DFixture;

// Single worker core per side (num_workers == 1).
const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};

// Multi-core worker grids for the metadata tests (exercise the "last core writes"
// disambiguation on the sender and the broadcast-to-all-workers fan-out on the
// receiver). 2 cores, 4 cores; "all cores" is derived per-mesh in the test.
const CoreRange kWorkerCores2{CoreCoord{0, 0}, CoreCoord{0, 1}};
const CoreRange kWorkerCores4{CoreCoord{0, 0}, CoreCoord{1, 1}};

// core_range_volume / fifo_bytes_for live in stream_service_test_utils.hpp.

// Standard config: UINT32 ROW_MAJOR DRAM-interleaved, replicated on every
// device, L1 socket FIFO sized to the tensor page (see fifo_bytes_for). The
// mapper is built fresh per call (create_pair moves it out). share_fabric_links
// defaults to false (OWN mode) so the functional tests stream without needing
// per-transfer grants; the lease tests pass true and drive the grant/release
// handshake explicitly.
D2DStreamConfig make_config(
    const std::shared_ptr<MeshDevice>& sender_mesh, const ttnn::Shape& global_shape, bool share_fabric_links = false) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const tt::tt_metal::TensorSpec global_spec(global_shape, tensor_layout);
    return D2DStreamConfig{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*sender_mesh, MeshMapperConfig{.placements = replicate_all(*sender_mesh)}),
        .socket_mem_config = SocketMemoryConfig{BufferType::L1, fifo_bytes_for(global_spec)},
        .sender_worker_cores = kWorkerCores,
        .receiver_worker_cores = kWorkerCores,
        .share_fabric_links = share_fabric_links,
    };
}

// make_spec(shape, dtype, layout) lives in stream_service_test_utils.hpp.

// "Fixed work per core" shape: every worker core gets exactly 32 ROW_MAJOR pages
// of 512 u32 each, so num_pages == 32 * num_workers — a dense, perfectly even
// split with real per-core work (unlike the tiny [1,1,32,64], which leaves most
// of a full grid idle). 512 u32 = 2048 B page keeps a 2-page socket chunk.
ttnn::Shape fixed_per_core_shape(const CoreRange& worker_cores) {
    return ttnn::Shape({1, 1, 32u * core_range_volume(worker_cores), 512u});
}

// Run create_pair and assert the construction-time getters return sane state.
void verify_creatable(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape) {
    ASSERT_EQ(sender_mesh->shape(), receiver_mesh->shape());

    auto [sender, receiver] =
        D2DStreamService::create_pair(sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape));

    ASSERT_NE(sender, nullptr);
    ASSERT_NE(receiver, nullptr);

    // Backing tensors are allocated on both sides with the same per-shard spec.
    EXPECT_TRUE(sender->get_backing_tensor().is_allocated());
    EXPECT_TRUE(receiver->get_backing_tensor().is_allocated());
    EXPECT_EQ(sender->get_per_shard_spec(), receiver->get_per_shard_spec());
    EXPECT_EQ(sender->get_backing_tensor().tensor_spec(), sender->get_per_shard_spec());

    // Worker cores are echoed back verbatim.
    EXPECT_EQ(sender->get_worker_cores(), kWorkerCores);
    EXPECT_EQ(receiver->get_worker_cores(), kWorkerCores);

    // A service core is claimed and queryable for every participating coord on
    // both sides (the connection list is built from these).
    const auto& sender_coords = sender->get_backing_tensor().tensor_topology().mesh_coords();
    for (const auto& coord : sender_coords) {
        EXPECT_NO_THROW((void)sender->get_service_core(coord));
    }
    const auto& receiver_coords = receiver->get_backing_tensor().tensor_topology().mesh_coords();
    for (const auto& coord : receiver_coords) {
        EXPECT_NO_THROW((void)receiver->get_service_core(coord));
    }
}

// Every worker-sync resource address (per-coord service-core L1 slots +
// mesh-wide GlobalSemaphores) is allocated and non-zero. The GlobalSemaphore
// addresses are uniform by construction (a single getter, no coord).
void verify_sync_resources(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape) {
    auto [sender, receiver] =
        D2DStreamService::create_pair(sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape));

    // Mesh-wide worker-grid GlobalSemaphores: one address each, non-zero.
    EXPECT_NE(sender->get_consumed_sem_addr(), 0u);
    EXPECT_NE(receiver->get_data_ready_sem_addr(), 0u);

    // Per-coord service-core L1 slots: allocated and non-zero on every
    // participating coord.
    const auto& sender_coords = sender->get_backing_tensor().tensor_topology().mesh_coords();
    for (const auto& coord : sender_coords) {
        EXPECT_NE(sender->get_data_ready_counter_addr(coord), 0u) << "sender data_ready_counter at " << coord;
    }
    const auto& receiver_coords = receiver->get_backing_tensor().tensor_topology().mesh_coords();
    for (const auto& coord : receiver_coords) {
        EXPECT_NE(receiver->get_consumed_counter_addr(coord), 0u) << "receiver consumed_counter at " << coord;
    }
}

// A single end-to-end transfer without real worker ops. Host-loads the sender
// backing tensor with iota, simulates the sender worker grid (bumps
// data_ready_counter by num_workers on each sender service core so the
// persistent sender does exactly one transfer), then polls the receiver backing
// tensor until it matches.
void verify_transfer(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape) {
    auto [sender, receiver] =
        D2DStreamService::create_pair(sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape));

    const auto& coords = sender->get_backing_tensor().tensor_topology().mesh_coords();

    // Size the payload to the per-device backing buffer (raw bytes, including any
    // page padding) so the sender/receiver page transfer is compared 1:1.
    const auto* sender_buffer = sender->get_backing_tensor().buffer();
    ASSERT_NE(sender_buffer, nullptr);
    const size_t num_u32 = sender_buffer->size() / sizeof(uint32_t);
    ASSERT_GT(num_u32, 0u);

    // Start at 1 so freshly-allocated (zero) receiver L1/DRAM never matches before
    // the transfer actually lands.
    std::vector<uint32_t> iota(num_u32);
    std::iota(iota.begin(), iota.end(), 1u);

    // WriteShard/ReadShard need a shared_ptr<MeshBuffer>; the Tensor only exposes a
    // `const MeshBuffer&`, so wrap it in a non-owning view (see mesh_buffer_view).
    auto sender_mesh_buffer = mesh_buffer_view(sender->get_backing_tensor());
    auto receiver_mesh_buffer = mesh_buffer_view(receiver->get_backing_tensor());

    // Load the sender backing tensor on every participating coord and make sure
    // the writes land before we trigger the sender.
    for (const auto& coord : coords) {
        WriteShard(sender_mesh->mesh_command_queue(), sender_mesh_buffer, iota, coord);
    }
    Finish(sender_mesh->mesh_command_queue());

    // Simulate the sender worker grid: write num_workers into each coord's
    // data_ready_counter so (cur - 0) == num_workers triggers exactly one
    // transfer (no real producer op in this test).
    const uint32_t num_workers = core_range_volume(sender->get_worker_cores());
    std::vector<uint32_t> trigger{num_workers};
    for (const auto& coord : coords) {
        tt::tt_metal::detail::WriteToDeviceL1(
            sender_mesh->get_device(coord),
            sender->get_service_core(coord),
            static_cast<uint32_t>(sender->get_data_ready_counter_addr(coord)),
            trigger);
    }

    // No host-visible completion signal without a consumer op, so poll the
    // receiver backing tensor until the transfer lands.
    constexpr auto kTimeout = std::chrono::milliseconds(1500);
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    bool all_match = false;
    std::vector<uint32_t> readback;
    while (std::chrono::steady_clock::now() < deadline) {
        all_match = true;
        for (const auto& coord : coords) {
            readback.clear();
            ReadShard(receiver_mesh->mesh_command_queue(), readback, receiver_mesh_buffer, coord);
            if (readback != iota) {
                all_match = false;
                break;
            }
        }
        if (all_match) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_TRUE(all_match) << "receiver backing tensor did not match sender iota within timeout";
}

// Build a sender-side placeholder worker workload: one program per coord
// (uniform CT args; per-coord RT args for the service-core counter + NoC coords).
// Each worker produces value (fill_base + iter) for num_iters then exits. Runs on
// the service's full worker grid. When metadata_size_bytes > 0 the designated
// (highest-id) worker writes a {-1, 0, fill_base+iter} blob into the sender
// service core's metadata L1 before acking.
MeshWorkload make_sender_worker_workload(
    D2DStreamServiceSender* sender,
    const std::shared_ptr<MeshDevice>& mesh,
    uint32_t num_iters,
    uint32_t fill_base,
    uint32_t metadata_size_bytes = 0) {
    const auto& coords = sender->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* sender_buffer = sender->get_backing_tensor().buffer();
    const uint32_t tensor_page_size = sender_buffer->aligned_page_size();
    const uint32_t num_pages = sender_buffer->num_pages();
    const CoreRange worker_cores = sender->get_worker_cores();
    const bool metadata_enabled = metadata_size_bytes > 0;
    constexpr auto kScratchCb = CBIndex::c_0;

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();

        auto cb_cfg = CircularBufferConfig(tensor_page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, tensor_page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        const auto* dbuf = sender->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(sender->get_consumed_sem_addr()),
            static_cast<uint32_t>(sender_buffer->address()),
            num_pages,
            tensor_page_size,
            num_iters,
            static_cast<uint32_t>(kScratchCb),
            fill_base,
            metadata_enabled ? 1u : 0u,
            metadata_size_bytes,
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/placeholder_d2d_sender_worker.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh->get_device(coord);
        const auto service_phys = device->worker_core_from_logical_core(sender->get_service_core(coord));
        const uint32_t md_addr = metadata_enabled ? static_cast<uint32_t>(sender->get_metadata_addr(coord)) : 0u;
        for (const auto& wc : worker_cores) {
            // Designate the highest-id worker core as the sole metadata writer.
            const uint32_t is_metadata_writer = (metadata_enabled && wc == worker_cores.end_coord) ? 1u : 0u;
            const std::vector<uint32_t> rt_args = {
                static_cast<uint32_t>(sender->get_data_ready_counter_addr(coord)),
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

// Build a receiver-side placeholder worker workload: handshake only (no data
// read needed). Acks num_iters times then exits.
MeshWorkload make_receiver_worker_workload(
    D2DStreamServiceReceiver* receiver, const std::shared_ptr<MeshDevice>& mesh, uint32_t num_iters) {
    const auto& coords = receiver->get_backing_tensor().tensor_topology().mesh_coords();
    const CoreRange worker_cores = receiver->get_worker_cores();

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(receiver->get_data_ready_sem_addr()),
            num_iters,
        };
        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/placeholder_d2d_receiver_worker.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh->get_device(coord);
        const auto service_phys = device->worker_core_from_logical_core(receiver->get_service_core(coord));
        const std::vector<uint32_t> rt_args = {
            static_cast<uint32_t>(receiver->get_consumed_counter_addr(coord)),
            static_cast<uint32_t>(service_phys.x),
            static_cast<uint32_t>(service_phys.y),
        };
        for (const auto& wc : worker_cores) {
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// STEP 2 driver: launch real sender workers with metadata enabled and verify the
// designated worker wrote the metadata blob into the sender SERVICE core's L1.
// The sender service does NOT yet ship it over fabric. fill_base=3 => {-1,0,3}.
void verify_metadata_sender_write(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores) {
    constexpr uint32_t kMdValue = 3;
    const std::vector<uint32_t> expected_md = {static_cast<uint32_t>(-1), 0u, kMdValue};
    const uint32_t md_bytes = static_cast<uint32_t>(expected_md.size() * sizeof(uint32_t));

    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = md_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto sender_workload = make_sender_worker_workload(
        sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/kMdValue, /*metadata_size_bytes=*/md_bytes);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);
    Finish(sender_mesh->mesh_command_queue());

    std::vector<uint32_t> readback;
    for (const auto& coord : sender->get_backing_tensor().tensor_topology().mesh_coords()) {
        readback.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(
            sender_mesh->get_device(coord),
            sender->get_service_core(coord),
            static_cast<uint32_t>(sender->get_metadata_addr(coord)),
            md_bytes,
            readback);
        EXPECT_EQ(readback, expected_md) << "sender service-core metadata mismatch at " << coord;
    }
}

// Row-major iota of `n` uint32 starting at `base`: v[i] = base + i. The sender
// worker writes this pattern (base = fill_base + iter), so the readback catches
// transposed pages (per-element distinct) and stale/ignored transfers (the base
// shifts by 1 each iteration).
std::vector<uint32_t> make_iota_u32(size_t n, uint32_t base) {
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = base + static_cast<uint32_t>(i);
    }
    return v;
}

// Assert the receiver backing tensor holds the iota (base + i) on every coord.
void expect_receiver_backing_iota(
    D2DStreamServiceReceiver* receiver, const std::shared_ptr<MeshDevice>& mesh, uint32_t base) {
    auto mesh_buffer = mesh_buffer_view(receiver->get_backing_tensor());
    const size_t num_u32 = receiver->get_backing_tensor().buffer()->size() / sizeof(uint32_t);
    const std::vector<uint32_t> expected = make_iota_u32(num_u32, base);
    std::vector<uint32_t> readback;
    for (const auto& coord : receiver->get_backing_tensor().tensor_topology().mesh_coords()) {
        readback.clear();
        ReadShard(mesh->mesh_command_queue(), readback, mesh_buffer, coord);
        EXPECT_EQ(readback, expected) << "receiver backing mismatch at " << coord << " (iota base " << base << ")";
    }
}

// Full worker handshake. Launches placeholder sender + receiver worker
// workloads that drive `num_iters` end-to-end transfers through the real
// handshakes (sender: data_ready_counter / consumed_sem; receiver:
// data_ready_sem / consumed_counter). With fill_base=1 the sender writes values
// 1..num_iters, so the receiver backing tensor must hold `num_iters` after the
// loop — and the test only completes if nothing deadlocked.
void verify_handshake(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    uint32_t num_iters) {
    auto [sender, receiver] =
        D2DStreamService::create_pair(sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape));

    auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, num_iters);
    auto sender_workload = make_sender_worker_workload(sender.get(), sender_mesh, num_iters, /*fill_base=*/1);

    // Launch receivers first so they're ready to ack, then the senders produce.
    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);

    // Wait for all num_iters on both sides. Finishing the receiver workers
    // guarantees the final transfer is durable in the backing tensor (the
    // service barriers its DRAM writes before mcasting data_ready_sem).
    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    expect_receiver_backing_iota(receiver.get(), receiver_mesh, num_iters);
}

// Reuse check. Builds the pair ONCE (persistent service kernels launched once),
// then drives `num_rounds` independent single-transfer rounds against that same
// service, each with a distinct seed, host-verifying the readback every round.
// If a persistent kernel had exited after round 0, later rounds would either
// deadlock (Finish never returns) or read back round 0's seed — both fail loudly.
void verify_reuse(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    uint32_t num_rounds) {
    auto [sender, receiver] =
        D2DStreamService::create_pair(sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape));

    for (uint32_t round = 0; round < num_rounds; ++round) {
        // Distinct, non-trivial seed per round (avoids 0 / small values that
        // could collide with uninitialised state).
        const uint32_t seed = 0x1000u + round * 0x111u;

        auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, /*num_iters=*/1);
        auto sender_workload =
            make_sender_worker_workload(sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/seed);

        EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
        EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);

        Finish(sender_mesh->mesh_command_queue());
        Finish(receiver_mesh->mesh_command_queue());

        expect_receiver_backing_iota(receiver.get(), receiver_mesh, seed);
    }
}

// Fabric-link lease round-trip (share_fabric_links == true). With lease mode the service
// holds no fabric connection and does NOTHING until granted a turn. The lease is
// CQ-ordered: each round enqueues, on each mesh's queue, [release (grant) -> worker ->
// wait (fence)]. Granting BEFORE the worker means the worker's handshake completes
// against an already-granted service (granting after the worker would deadlock — the
// grant would sit behind a worker that is itself waiting for the grant). The wait kernel
// fences the next round. Two rounds with distinct seeds prove the service re-acquires +
// releases the link per transfer and keeps streaming. A service that never re-acquired
// would hang the Finish; one that never released would hang the wait kernel.
void verify_fabric_lease(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape) {
    auto [sender, receiver] = D2DStreamService::create_pair(
        sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape, /*share_fabric_links=*/true));

    auto run_round = [&](uint32_t seed) {
        // Grant each service its turn FIRST (CQ-ordered; receiver before sender so it is
        // ready to drain), so the grant is in place when the worker workloads run and
        // each service completes its handshake in the same window.
        receiver->release_fabric_links();
        sender->release_fabric_links();

        auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, /*num_iters=*/1);
        auto sender_workload =
            make_sender_worker_workload(sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/seed);
        EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
        EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);

        // Fence: each service must hand its link back (link_grant == 0) before the next
        // round grants again — in production this is the point a fabric op would launch.
        // CQ-ordered after the workers, so no host poll.
        sender->wait_for_fabric_links();
        receiver->wait_for_fabric_links();

        // Drain both queues before the host reads the landed tensor back.
        Finish(sender_mesh->mesh_command_queue());
        Finish(receiver_mesh->mesh_command_queue());

        expect_receiver_backing_iota(receiver.get(), receiver_mesh, seed);
    };

    run_round(0x1234u);
    run_round(0x5678u);
}

// Fabric-link lease stress. Build the pair once, then drive num_iters grant -> transfer
// -> release cycles. Each iteration enqueues, on each mesh's queue, [release (grant) ->
// single-iteration worker -> wait (fence)] and Finishes before the next iteration. The
// grant always lands before its worker, so the worker's handshake completes against an
// already-granted service; the per-iteration Finish drains the queue so the loop-local
// worker workloads stay valid until executed (a MeshWorkload owns the on-device program
// binaries the queue runs, so it must outlive its non-blocking enqueue). fill_base =
// i + 1, so the sender writes 1..num_iters and the receiver backing tensor must hold
// num_iters at the end. Exercises the open/close + link_grant ping-pong many times;
// catches state drift a 2-round test would miss: a stuck grant word, a failed connection
// re-open, or monotonic-counter desync would hang a wait kernel / a Finish mid-stream.
void verify_fabric_lease_stress(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    uint32_t num_iters) {
    auto [sender, receiver] = D2DStreamService::create_pair(
        sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape, /*share_fabric_links=*/true));

    for (uint32_t i = 0; i < num_iters; ++i) {
        // Grant before the worker (receiver first so it is ready to drain).
        receiver->release_fabric_links();
        sender->release_fabric_links();

        auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, /*num_iters=*/1);
        auto sender_workload =
            make_sender_worker_workload(sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/i + 1);
        EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
        EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);

        // Fence this transfer, then drain both queues before the worker workloads (loop
        // locals) go out of scope.
        sender->wait_for_fabric_links();
        receiver->wait_for_fabric_links();
        Finish(sender_mesh->mesh_command_queue());
        Finish(receiver_mesh->mesh_command_queue());
    }

    expect_receiver_backing_iota(receiver.get(), receiver_mesh, num_iters);
}

// Assert the metadata blob equals `expected` on every receiver worker core's L1
// (where the receiver service multicast it). Factored out so the multi-hop chain
// driver can validate the terminal receiver without a paired sender on the same mesh.
void expect_metadata_on_receiver_workers(
    D2DStreamServiceReceiver* receiver,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const std::vector<uint32_t>& expected) {
    const uint32_t md_bytes = static_cast<uint32_t>(expected.size() * sizeof(uint32_t));
    const uint32_t recv_md_addr = static_cast<uint32_t>(receiver->get_metadata_addr());
    const CoreRange recv_workers = receiver->get_worker_cores();
    std::vector<uint32_t> rb;
    for (const auto& coord : receiver->get_backing_tensor().tensor_topology().mesh_coords()) {
        auto* device = receiver_mesh->get_device(coord);
        for (const auto& wc : recv_workers) {
            rb.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(device, wc, recv_md_addr, md_bytes, rb);
            EXPECT_EQ(rb, expected) << "receiver metadata mismatch at coord " << coord << " core (" << wc.x << ","
                                    << wc.y << ")";
        }
    }
}

// Assert the metadata blob equals `expected` on BOTH the sender service core L1
// (where the designated worker wrote it) and every receiver worker core's L1
// (where the receiver service multicast it).
void expect_metadata_everywhere(
    D2DStreamServiceSender* sender,
    D2DStreamServiceReceiver* receiver,
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const std::vector<uint32_t>& expected) {
    const uint32_t md_bytes = static_cast<uint32_t>(expected.size() * sizeof(uint32_t));
    std::vector<uint32_t> rb;

    for (const auto& coord : sender->get_backing_tensor().tensor_topology().mesh_coords()) {
        rb.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(
            sender_mesh->get_device(coord),
            sender->get_service_core(coord),
            static_cast<uint32_t>(sender->get_metadata_addr(coord)),
            md_bytes,
            rb);
        EXPECT_EQ(rb, expected) << "sender service-core metadata mismatch at " << coord;
    }

    expect_metadata_on_receiver_workers(receiver, receiver_mesh, expected);
}

std::vector<uint32_t> make_metadata_triple(uint32_t base) { return {static_cast<uint32_t>(-1), 0u, base}; }

uint32_t metadata_bytes_for_words(uint32_t num_words) { return num_words * static_cast<uint32_t>(sizeof(uint32_t)); }

ttsl::Span<const std::byte> as_metadata_bytes(const std::vector<uint32_t>& metadata_words) {
    return ttsl::Span<const std::byte>(
        reinterpret_cast<const std::byte*>(metadata_words.data()), metadata_bytes_for_words(metadata_words.size()));
}

// STEP 3 driver: full end-to-end metadata. Real sender + receiver workers drive
// one transfer; the designated sender worker's {-1,0,fill_base} blob must land on
// the sender service core AND every receiver worker core.
void verify_metadata_end_to_end(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores) {
    constexpr uint32_t kFillBase = 0x140u;  // metadata {-1, 0, 0x140}
    const std::vector<uint32_t> expected_md = {static_cast<uint32_t>(-1), 0u, kFillBase};
    const uint32_t md_bytes = static_cast<uint32_t>(expected_md.size() * sizeof(uint32_t));

    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = md_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, /*num_iters=*/1);
    auto sender_workload =
        make_sender_worker_workload(sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/kFillBase, md_bytes);
    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);
    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    expect_metadata_everywhere(sender.get(), receiver.get(), sender_mesh, receiver_mesh, expected_md);
}

// STEP 3 reuse: build once, drive several rounds with DISTINCT metadata per round.
// Catches early kernel exit / stale-buffer reuse on the metadata path.
void verify_metadata_reuse(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_rounds) {
    const uint32_t md_bytes = 3u * sizeof(uint32_t);

    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = md_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t fill_base = 0x200u + round * 0x11u;
        auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, /*num_iters=*/1);
        auto sender_workload =
            make_sender_worker_workload(sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/fill_base, md_bytes);
        EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
        EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);
        Finish(sender_mesh->mesh_command_queue());
        Finish(receiver_mesh->mesh_command_queue());

        expect_metadata_everywhere(
            sender.get(), receiver.get(), sender_mesh, receiver_mesh, {static_cast<uint32_t>(-1), 0u, fill_base});
    }
}

// ===========================================================================
// Receiver consumer workload builder. A consumer worker reads its page slice of
// the receiver backing tensor into a SEPARATE output tensor. Used by the H2D
// bridge tests (Part B) below.
// ===========================================================================

// worker_index / worker_page_range live in stream_service_test_utils.hpp.

// Build a receiver-side CONSUMER worker workload: one program per coord. Each
// worker copies its page slice of the receiver backing tensor into `output_tensor`
// (same spec) and runs the receiver handshake num_iters times. Grid-agnostic.
MeshWorkload make_receiver_consumer_workload(
    D2DStreamServiceReceiver* receiver,
    const std::shared_ptr<MeshDevice>& mesh,
    const ttnn::Tensor& output_tensor,
    uint32_t num_iters) {
    const auto& coords = receiver->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* backing_buffer = receiver->get_backing_tensor().buffer();
    const uint32_t page_size = backing_buffer->aligned_page_size();
    const uint32_t num_pages = backing_buffer->num_pages();
    const CoreRange worker_cores = receiver->get_worker_cores();
    const uint32_t num_workers = core_range_volume(worker_cores);
    constexpr auto kScratchCb = CBIndex::c_0;

    const uint32_t input_addr = static_cast<uint32_t>(backing_buffer->address());
    const uint32_t output_addr = static_cast<uint32_t>(output_tensor.buffer()->address());

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();

        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        const auto* dbuf = receiver->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(receiver->get_data_ready_sem_addr()),
            input_addr,
            output_addr,
            page_size,
            num_iters,
            static_cast<uint32_t>(kScratchCb),
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/d2d_receiver_consumer_worker.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh->get_device(coord);
        const auto service_phys = device->worker_core_from_logical_core(receiver->get_service_core(coord));
        for (const auto& wc : worker_cores) {
            const auto [start_page, end_page] =
                worker_page_range(worker_index(wc, worker_cores), num_workers, num_pages);
            const std::vector<uint32_t> rt_args = {
                start_page,
                end_page,
                static_cast<uint32_t>(receiver->get_consumed_counter_addr(coord)),
                static_cast<uint32_t>(service_phys.x),
                static_cast<uint32_t>(service_phys.y),
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// Assert `output_tensor` holds the iota (base + i) on every coord.
void expect_output_tensor_iota(
    const ttnn::Tensor& output_tensor, const std::shared_ptr<MeshDevice>& mesh, uint32_t base) {
    auto mesh_buffer = mesh_buffer_view(output_tensor);
    const size_t num_u32 = output_tensor.buffer()->size() / sizeof(uint32_t);
    const std::vector<uint32_t> expected = make_iota_u32(num_u32, base);
    std::vector<uint32_t> readback;
    for (const auto& coord : output_tensor.tensor_topology().mesh_coords()) {
        readback.clear();
        ReadShard(mesh->mesh_command_queue(), readback, mesh_buffer, coord);
        EXPECT_EQ(readback, expected) << "output tensor mismatch at " << coord << " (iota base " << base << ")";
    }
}

// ===========================================================================
// Part B: H2D front-end (Host -> H2D -> bridge -> D2D -> Host). A bridge worker on
// the sender mesh drains the H2D backing tensor into the D2D sender backing tensor
// and drives both handshakes; the receiver consumer lands the result in an output
// tensor the host validates.
// ===========================================================================

// Build the bridge worker workload on the sender mesh: one program per coord. Each
// worker copies its page slice of the UPSTREAM backing tensor into the D2D sender
// backing tensor and runs both handshakes num_iters times. The designated
// (highest-id) worker forwards the metadata from its L1 to the D2D sender service.
//
// Templated on the upstream type so the same builder serves both the stage-0 bridge
// (upstream == H2DStreamService) and a middle-stage bridge in a multi-hop chain
// (upstream == D2DStreamServiceReceiver). Both expose the identical upstream-handshake
// getters (get_backing_tensor / get_per_shard_spec / get_data_ready_sem_addr /
// get_consumed_counter_addr / get_service_core / get_metadata_addr), and the bridge
// kernel is upstream-agnostic, so the only difference is the deduced type.
template <typename Upstream>
MeshWorkload make_bridge_workload(
    Upstream& upstream,
    D2DStreamServiceSender* d2d_sender,
    const std::shared_ptr<MeshDevice>& mesh,
    const CoreRange& worker_cores,
    uint32_t num_iters,
    uint32_t metadata_size_bytes) {
    // The kernel reuses a single TensorAccessorArgs (built from the D2D sender backing)
    // for both the upstream read and the downstream write, so the two specs must match.
    TT_FATAL(
        upstream.get_per_shard_spec() == d2d_sender->get_per_shard_spec(),
        "make_bridge_workload: upstream and D2D sender per-shard specs must match");

    const auto& coords = d2d_sender->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* backing_buffer = d2d_sender->get_backing_tensor().buffer();
    const uint32_t page_size = backing_buffer->aligned_page_size();
    const uint32_t num_pages = backing_buffer->num_pages();
    const uint32_t num_workers = core_range_volume(worker_cores);
    const bool metadata_enabled = metadata_size_bytes > 0;
    constexpr auto kScratchCb = CBIndex::c_0;

    const uint32_t h2d_input_addr = static_cast<uint32_t>(upstream.get_backing_tensor().buffer()->address());
    const uint32_t d2d_backing_addr = static_cast<uint32_t>(backing_buffer->address());
    const uint32_t h2d_metadata_l1_addr = metadata_enabled ? static_cast<uint32_t>(upstream.get_metadata_addr()) : 0u;

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();

        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        const auto* dbuf = d2d_sender->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(upstream.get_data_ready_sem_addr()),
            h2d_input_addr,
            d2d_backing_addr,
            page_size,
            num_iters,
            static_cast<uint32_t>(kScratchCb),
            static_cast<uint32_t>(d2d_sender->get_consumed_sem_addr()),
            metadata_enabled ? 1u : 0u,
            metadata_size_bytes,
            h2d_metadata_l1_addr,
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/h2d_d2d_bridge_worker.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = mesh->get_device(coord);
        const auto h2d_service_phys = device->worker_core_from_logical_core(upstream.get_service_core(coord));
        const auto d2d_service_phys = device->worker_core_from_logical_core(d2d_sender->get_service_core(coord));
        const uint32_t d2d_md_addr =
            metadata_enabled ? static_cast<uint32_t>(d2d_sender->get_metadata_addr(coord)) : 0u;
        for (const auto& wc : worker_cores) {
            const auto [start_page, end_page] =
                worker_page_range(worker_index(wc, worker_cores), num_workers, num_pages);
            const uint32_t is_metadata_writer = (metadata_enabled && wc == worker_cores.end_coord) ? 1u : 0u;
            const std::vector<uint32_t> rt_args = {
                start_page,
                end_page,
                static_cast<uint32_t>(upstream.get_consumed_counter_addr(coord)),
                static_cast<uint32_t>(h2d_service_phys.x),
                static_cast<uint32_t>(h2d_service_phys.y),
                static_cast<uint32_t>(d2d_sender->get_data_ready_counter_addr(coord)),
                static_cast<uint32_t>(d2d_service_phys.x),
                static_cast<uint32_t>(d2d_service_phys.y),
                is_metadata_writer,
                d2d_md_addr,
            };
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// Build an H2D service on the sender mesh. FIFO / scratch are sized to one
// tensor page (see fifo_bytes_for), NOT the whole tensor: H2D streams page-by-
// page and blocks the host when the FIFO fills, and the receiver kernel signals
// data_ready only after draining the full token, so a page-sized FIFO is correct
// and keeps L1 use bounded for large shapes (a whole-tensor FIFO is impossible in
// L1 past a few MB). fifo_bytes_for is dtype/layout-aware (block-float TILE specs
// that pack exponents alongside data are covered).
std::unique_ptr<H2DStreamService> make_h2d_service(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const tt::tt_metal::TensorSpec& global_spec,
    const CoreRange& worker_cores,
    uint32_t metadata_size_bytes) {
    const uint32_t fifo_bytes = fifo_bytes_for(global_spec);
    H2DStreamService::Config h2d_cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*sender_mesh, MeshMapperConfig{.placements = replicate_all(*sender_mesh)}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = fifo_bytes,
        .max_socket_page_size_bytes = fifo_bytes,
        .worker_cores = worker_cores,
        .metadata_size_bytes = metadata_size_bytes,
    };
    return std::make_unique<H2DStreamService>(sender_mesh, std::move(h2d_cfg));
}

// Push one token + uint32 metadata words through the H2D service.
void h2d_push_token(
    H2DStreamService& h2d_service,
    ttsl::Span<const std::byte> token_bytes,
    const std::vector<uint32_t>& metadata_words) {
    h2d_service.forward_to_tensor(token_bytes, as_metadata_bytes(metadata_words));
}

// Uniform u32 iota token; optional {-1,0,base} metadata when metadata_size_bytes > 0.
void h2d_push_token_u32(H2DStreamService& h2d_service, uint32_t num_u32, uint32_t base, uint32_t metadata_size_bytes) {
    const std::vector<uint32_t> token = make_iota_u32(num_u32, base);
    const auto token_bytes =
        ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(token.data()), token.size() * sizeof(uint32_t));
    if (metadata_size_bytes == 0) {
        h2d_service.forward_to_tensor(token_bytes, {});
        return;
    }
    const auto metadata_words = make_metadata_triple(base);
    ASSERT_EQ(metadata_bytes_for_words(metadata_words.size()), metadata_size_bytes);
    h2d_push_token(h2d_service, token_bytes, metadata_words);
}

// Part B driver: full Host -> H2D -> bridge -> D2D -> consumer -> Host chain. The
// host streams num_iters tokens (value fill_base+i); the final value must appear
// in the receiver output tensor. When metadata is enabled, make_metadata(iter)
// supplies the per-iteration uint32 word blob (defaults to {-1,0,fill_base+iter}).
void verify_h2d_d2d_bridge(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_iters,
    uint32_t metadata_size_bytes = 0,
    std::function<std::vector<uint32_t>(uint32_t iter)> make_metadata = {}) {
    constexpr uint32_t kFillBase = 1u;
    const bool metadata_enabled = metadata_size_bytes > 0;
    if (metadata_enabled && !make_metadata) {
        make_metadata = [](uint32_t iter) { return make_metadata_triple(kFillBase + iter); };
    }

    const auto global_spec = make_config(sender_mesh, global_shape).global_spec;
    auto h2d_service = make_h2d_service(sender_mesh, global_spec, worker_cores, metadata_size_bytes);

    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = metadata_size_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto output_tensor = ttnn::create_device_tensor(
        receiver->get_backing_tensor().tensor_spec(),
        receiver_mesh.get(),
        receiver->get_backing_tensor().tensor_topology());

    auto consumer_workload = make_receiver_consumer_workload(receiver.get(), receiver_mesh, output_tensor, num_iters);
    auto bridge_workload =
        make_bridge_workload(*h2d_service, sender.get(), sender_mesh, worker_cores, num_iters, metadata_size_bytes);

    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), consumer_workload, /*blocking=*/false);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), bridge_workload, /*blocking=*/false);

    const uint32_t num_u32 = static_cast<uint32_t>(global_shape.volume());
    for (uint32_t i = 0; i < num_iters; ++i) {
        const std::vector<uint32_t> token = make_iota_u32(num_u32, kFillBase + i);
        const auto token_bytes = ttsl::Span<const std::byte>(
            reinterpret_cast<const std::byte*>(token.data()), token.size() * sizeof(uint32_t));
        if (metadata_enabled) {
            const auto metadata_words = make_metadata(i);
            ASSERT_EQ(metadata_bytes_for_words(metadata_words.size()), metadata_size_bytes);
            h2d_push_token(*h2d_service, token_bytes, metadata_words);
        } else {
            h2d_service->forward_to_tensor(token_bytes, {});
        }
    }
    h2d_service->barrier();
    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    const uint32_t final_value = kFillBase + num_iters - 1;
    expect_output_tensor_iota(output_tensor, receiver_mesh, final_value);
    if (metadata_enabled) {
        expect_metadata_everywhere(
            sender.get(), receiver.get(), sender_mesh, receiver_mesh, make_metadata(num_iters - 1));
    }
}

// 1->1->1 chain driver: full Host -> H2D -> bridge -> D2D0 -> MIDDLE bridge -> D2D1 ->
// consumer -> Host across three colinear devices, all D2D pairs in OWN mode. stage1 is
// the MIDDLE device: it hosts BOTH the D2D0 receiver (inbound) and the D2D1 sender
// (outbound) at once. Their fabric connections route in opposite directions
// (stage1->stage0 credit-return vs stage1->stage2 data), so on colinear devices they
// land on distinct EDM channels and never collide -- which is why OWN mode is safe here
// without the per-transfer lease handshake. The MIDDLE worker reuses the bridge kernel
// with its upstream = the D2D0 receiver (instead of an H2D service); see
// make_bridge_workload. The host streams num_iters tokens (value kFillBase+i); the final
// value must reach the stage2 output tensor, with metadata {-1,0,final} multicast to the
// stage2 receiver worker cores when enabled.
void verify_three_stage_chain(
    const std::shared_ptr<MeshDevice>& stage0,
    const std::shared_ptr<MeshDevice>& stage1,
    const std::shared_ptr<MeshDevice>& stage2,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_iters,
    uint32_t metadata_size_bytes = 0) {
    constexpr uint32_t kFillBase = 1u;
    const bool metadata_enabled = metadata_size_bytes > 0;

    const auto global_spec = make_config(stage0, global_shape).global_spec;
    auto h2d_service = make_h2d_service(stage0, global_spec, worker_cores, metadata_size_bytes);

    // make_config defaults share_fabric_links to false (OWN mode).
    auto cfg0 = make_config(stage0, global_shape);
    cfg0.sender_worker_cores = worker_cores;
    cfg0.receiver_worker_cores = worker_cores;
    cfg0.metadata_size_bytes = metadata_size_bytes;
    auto [sender0, receiver0] = D2DStreamService::create_pair(stage0, stage1, std::move(cfg0));

    auto cfg1 = make_config(stage1, global_shape);
    cfg1.sender_worker_cores = worker_cores;
    cfg1.receiver_worker_cores = worker_cores;
    cfg1.metadata_size_bytes = metadata_size_bytes;
    auto [sender1, receiver1] = D2DStreamService::create_pair(stage1, stage2, std::move(cfg1));

    auto output_tensor = ttnn::create_device_tensor(
        receiver1->get_backing_tensor().tensor_spec(), stage2.get(), receiver1->get_backing_tensor().tensor_topology());

    auto consumer_workload = make_receiver_consumer_workload(receiver1.get(), stage2, output_tensor, num_iters);
    // MIDDLE bridge: upstream = D2D0 receiver (on stage1), downstream = D2D1 sender (on stage1).
    auto middle_workload =
        make_bridge_workload(*receiver0, sender1.get(), stage1, worker_cores, num_iters, metadata_size_bytes);
    // Stage-0 bridge: upstream = H2D service, downstream = D2D0 sender (both on stage0).
    auto stage0_workload =
        make_bridge_workload(*h2d_service, sender0.get(), stage0, worker_cores, num_iters, metadata_size_bytes);

    EnqueueMeshWorkload(stage2->mesh_command_queue(), consumer_workload, /*blocking=*/false);
    EnqueueMeshWorkload(stage1->mesh_command_queue(), middle_workload, /*blocking=*/false);
    EnqueueMeshWorkload(stage0->mesh_command_queue(), stage0_workload, /*blocking=*/false);

    const uint32_t num_u32 = static_cast<uint32_t>(global_shape.volume());
    for (uint32_t i = 0; i < num_iters; ++i) {
        h2d_push_token_u32(*h2d_service, num_u32, kFillBase + i, metadata_size_bytes);
    }
    h2d_service->barrier();
    Finish(stage0->mesh_command_queue());
    Finish(stage1->mesh_command_queue());
    Finish(stage2->mesh_command_queue());

    const uint32_t final_value = kFillBase + num_iters - 1;
    expect_output_tensor_iota(output_tensor, stage2, final_value);
    if (metadata_enabled) {
        // The D2D1 receiver service multicasts the blob into every stage2 receiver worker
        // core's L1; assert the final token's {-1,0,final} landed there.
        expect_metadata_on_receiver_workers(receiver1.get(), stage2, {static_cast<uint32_t>(-1), 0u, final_value});
    }
}

// Part B reuse: build the H2D service + D2D pair once, drive num_rounds rounds with
// distinct seeds (re-enqueue the bridge + consumer per round), verifying each.
void verify_h2d_d2d_bridge_reuse(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_rounds,
    uint32_t metadata_size_bytes = 0) {
    const bool metadata_enabled = metadata_size_bytes > 0;

    const auto global_spec = make_config(sender_mesh, global_shape).global_spec;
    auto h2d_service = make_h2d_service(sender_mesh, global_spec, worker_cores, metadata_size_bytes);

    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = metadata_size_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto output_tensor = ttnn::create_device_tensor(
        receiver->get_backing_tensor().tensor_spec(),
        receiver_mesh.get(),
        receiver->get_backing_tensor().tensor_topology());

    const uint32_t num_u32 = static_cast<uint32_t>(global_shape.volume());
    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t seed = 0x1000u + round * 0x111u;
        auto consumer_workload =
            make_receiver_consumer_workload(receiver.get(), receiver_mesh, output_tensor, /*num_iters=*/1);
        auto bridge_workload = make_bridge_workload(
            *h2d_service, sender.get(), sender_mesh, worker_cores, /*num_iters=*/1, metadata_size_bytes);
        EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), consumer_workload, /*blocking=*/false);
        EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), bridge_workload, /*blocking=*/false);

        h2d_push_token_u32(*h2d_service, num_u32, seed, metadata_size_bytes);
        h2d_service->barrier();
        Finish(sender_mesh->mesh_command_queue());
        Finish(receiver_mesh->mesh_command_queue());

        expect_output_tensor_iota(output_tensor, receiver_mesh, seed);
        if (metadata_enabled) {
            expect_metadata_everywhere(
                sender.get(), receiver.get(), sender_mesh, receiver_mesh, {static_cast<uint32_t>(-1), 0u, seed});
        }
    }
}

// 1x1 <-> 1x1 (single chip each).
TEST_F(D2DStreamServiceTest, CreatableSingleChipPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_creatable(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}));
}

// 1x2 <-> 1x2 (exercises the per-coord SocketConnection list and the 1:1 coord
// mapping with > 1 socket). Row 0 is the sender, row 1 is the receiver.
TEST_F(D2DStreamServiceTest, CreatableRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 2) {
        GTEST_SKIP() << "Need a >= 2x2 mesh to carve 1x2 <-> 1x2 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(1, 0));

    verify_creatable(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}));
}

// Worker-sync resources (data_ready_counter / consumed_counter L1 slots +
// consumed_sem / data_ready_sem GlobalSemaphores) are allocated with non-zero
// addresses.
TEST_F(D2DStreamServiceTest, SyncResourceAddresses) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_sync_resources(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}));
}

// Single end-to-end device-to-device transfer (single chip pair). Host-readback
// of the receiver backing tensor must equal the sender iota.
TEST_F(D2DStreamServiceTest, TransferSingleChipPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_transfer(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}));
}

// Same single transfer across a 1x2 row pair (exercises > 1 socket / coord).
TEST_F(D2DStreamServiceTest, TransferRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 2) {
        GTEST_SKIP() << "Need a >= 2x2 mesh to carve 1x2 <-> 1x2 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(1, 0));

    verify_transfer(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}));
}

// Worker handshake end-to-end on a single chip pair, driven by real placeholder
// worker workloads over several iterations.
TEST_F(D2DStreamServiceTest, HandshakeSingleChipPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_handshake(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}), /*num_iters=*/4);
}

// Worker handshake across a 1x2 row pair (multiple sockets / coords).
TEST_F(D2DStreamServiceTest, HandshakeRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 2) {
        GTEST_SKIP() << "Need a >= 2x2 mesh to carve 1x2 <-> 1x2 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(1, 0));

    verify_handshake(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}), /*num_iters=*/4);
}

// Reuse the persistent service across several rounds with distinct seeds,
// verifying each round. Fails loudly if a persistent kernel exited early.
TEST_F(D2DStreamServiceTest, ReuseSingleChipPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_reuse(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}), /*num_rounds=*/4);
}

// Reuse across a 1x2 row pair.
TEST_F(D2DStreamServiceTest, ReuseRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 2) {
        GTEST_SKIP() << "Need a >= 2x2 mesh to carve 1x2 <-> 1x2 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(1, 0));

    verify_reuse(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}), /*num_rounds=*/4);
}

// Fabric-link lease round-trip on a single chip pair: grant a transfer, run it,
// confirm the links are released, repeat.
TEST_F(D2DStreamServiceTest, FabricLeaseSingleChipPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_fabric_lease(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}));
}

// Fabric-link lease across a 1x2 row pair (the lease covers every coord on both
// meshes).
TEST_F(D2DStreamServiceTest, FabricLeaseRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 2) {
        GTEST_SKIP() << "Need a >= 2x2 mesh to carve 1x2 <-> 1x2 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(1, 0));

    verify_fabric_lease(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}));
}

// Fabric-link lease stress: 100 interleaved grant/release cycles against a single
// pair of long-running dummy worker programs.
TEST_F(D2DStreamServiceTest, FabricLeaseStressSingleChipPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_fabric_lease_stress(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 512}), /*num_iters=*/100);
}

// Per-handle teardown must release service cores / L1 / sockets so a fresh pair
// can be built on the same submeshes. Each verify_handshake creates and destroys
// a pair; the second call only succeeds if the first dtor released everything
// (otherwise create_pair's service-core claim TT_FATALs).
TEST_F(D2DStreamServiceTest, RecreateAfterTeardown) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }

    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    const auto shape_arg = ttnn::Shape({1, 1, 32, 512});
    verify_handshake(sender_mesh, receiver_mesh, shape_arg, /*num_iters=*/2);
    verify_handshake(sender_mesh, receiver_mesh, shape_arg, /*num_iters=*/2);
}

// STEP 2: the designated (highest-id) sender worker writes metadata into the
// sender service core's L1. Single worker core (degenerate: the only core writes).
TEST_F(D2DStreamServiceTest, MetadataSenderWriteSingleChipPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_metadata_sender_write(sender_mesh, receiver_mesh, fixed_per_core_shape(kWorkerCores), kWorkerCores);
}

// STEP 2 with 4 worker cores: only the highest-id core writes metadata; the other
// three must NOT (a stuck/duplicate writer would corrupt the service-core value).
TEST_F(D2DStreamServiceTest, MetadataSenderWrite4Cores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);

    verify_metadata_sender_write(sender_mesh, receiver_mesh, fixed_per_core_shape(kWorkerCores4), kWorkerCores4);
}

// STEP 3: full end-to-end metadata across the worker-grid matrix (1/2/4/all). The
// designated sender worker's blob must land on the sender service core AND every
// receiver worker core.
TEST_F(D2DStreamServiceTest, MetadataEndToEndSingleCore) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    verify_metadata_end_to_end(sender_mesh, receiver_mesh, fixed_per_core_shape(kWorkerCores), kWorkerCores);
}

TEST_F(D2DStreamServiceTest, MetadataEndToEnd2Cores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    verify_metadata_end_to_end(sender_mesh, receiver_mesh, fixed_per_core_shape(kWorkerCores2), kWorkerCores2);
}

TEST_F(D2DStreamServiceTest, MetadataEndToEnd4Cores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    verify_metadata_end_to_end(sender_mesh, receiver_mesh, fixed_per_core_shape(kWorkerCores4), kWorkerCores4);
}

TEST_F(D2DStreamServiceTest, MetadataEndToEndAllCores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    const auto grid = receiver_mesh->compute_with_storage_grid_size();
    const CoreRange all_cores{CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}};
    verify_metadata_end_to_end(sender_mesh, receiver_mesh, fixed_per_core_shape(all_cores), all_cores);
}

// STEP 3 reuse: the persistent service handles several metadata transfers with
// distinct per-round values (4 cores so the last-core-writer path is exercised).
TEST_F(D2DStreamServiceTest, MetadataReuse4Cores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    verify_metadata_reuse(
        sender_mesh, receiver_mesh, fixed_per_core_shape(kWorkerCores4), kWorkerCores4, /*num_rounds=*/4);
}

// ===========================================================================
// Part B tests: full Host -> H2D -> bridge -> D2D -> consumer -> Host chain.
// ===========================================================================

// Single worker core: one bridge core drains the whole tensor H2D -> D2D.
TEST_F(D2DStreamServiceTest, H2DtoD2DBridge1Core) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    if (!h2d_host_pinning_supported()) {
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not iommu=pt); "
                        "see notes/d2d_galaxy_h2d_pinning_failure.md.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}), kWorkerCores, /*num_iters=*/4);
}

// 4 cores with a page count NOT divisible by 4 (30 pages) — partition remainder.
TEST_F(D2DStreamServiceTest, H2DtoD2DBridge4Cores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    if (!h2d_host_pinning_supported()) {
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not iommu=pt); "
                        "see notes/d2d_galaxy_h2d_pinning_failure.md.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 30, 64}), kWorkerCores4, /*num_iters=*/3);
}

// Full worker grid (the realistic case).
TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeAllCores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    if (!h2d_host_pinning_supported()) {
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not iommu=pt); "
                        "see notes/d2d_galaxy_h2d_pinning_failure.md.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    const auto grid = receiver_mesh->compute_with_storage_grid_size();
    const CoreRange all_cores{CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}};
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, fixed_per_core_shape(all_cores), all_cores, /*num_iters=*/4);
}

// Full grid + metadata end-to-end through both services.
TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeMetadataAllCores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    if (!h2d_host_pinning_supported()) {
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not iommu=pt); "
                        "see notes/d2d_galaxy_h2d_pinning_failure.md.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    const auto grid = receiver_mesh->compute_with_storage_grid_size();
    const CoreRange all_cores{CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}};
    verify_h2d_d2d_bridge(
        sender_mesh,
        receiver_mesh,
        fixed_per_core_shape(all_cores),
        all_cores,
        /*num_iters=*/4,
        /*metadata_size_bytes=*/3u * sizeof(uint32_t));
}

// Full grid + reuse: build the chain once, drive several rounds.
TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeReuseAllCores) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    if (!h2d_host_pinning_supported()) {
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not iommu=pt); "
                        "see notes/d2d_galaxy_h2d_pinning_failure.md.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape;
    }
    const auto coord0 = MeshCoordinate(0, 0);
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);
    const auto grid = receiver_mesh->compute_with_storage_grid_size();
    const CoreRange all_cores{CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}};
    verify_h2d_d2d_bridge_reuse(
        sender_mesh, receiver_mesh, fixed_per_core_shape(all_cores), all_cores, /*num_rounds=*/4);
}

// Multi-device variant: mirrors the existing RowPair setup (1x2 <-> 1x2).
TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    if (!h2d_host_pinning_supported()) {
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not iommu=pt); "
                        "see notes/d2d_galaxy_h2d_pinning_failure.md.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 2) {
        GTEST_SKIP() << "Need a >= 2x2 mesh to carve 1x2 <-> 1x2 submeshes; got " << shape;
    }
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(1, 0));
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}), kWorkerCores, /*num_iters=*/4);
}

// ===========================================================================
// 1->1->1 chain test: Host -> H2D -> bridge -> D2D0 -> MIDDLE bridge -> D2D1 ->
// consumer -> Host across three colinear devices, all D2D pairs in OWN mode. The
// middle device (stage1) runs an inbound D2D receiver and an outbound D2D sender
// simultaneously -- consecutive device-to-device data movement without a host
// round trip. Requires three COLINEAR devices so the middle's inbound (->stage0)
// and outbound (->stage2) fabric connections route in opposite directions (distinct
// EDM channels), which is what lets them coexist in OWN mode. This is the only own-
// mode multi-stage chain kept; exhaustive shapes/dtypes and longer (4/8-stage)
// pipelines moved to the lease-mode StreamPipeline gtest (test_stream_pipeline.cpp).
// ===========================================================================
#define D2D_THREE_STAGE_GUARD()                                                                                       \
    if (!service_cores_supported()) {                                                                                 \
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";                            \
    }                                                                                                                 \
    if (!h2d_host_pinning_supported()) {                                                                              \
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not "                       \
                        "iommu=pt); see notes/d2d_galaxy_h2d_pinning_failure.md.";                                    \
    }                                                                                                                 \
    const auto shape = this->mesh_device_->shape();                                                                   \
    if (shape.dims() != 2 || (shape[1] < 3 && shape[0] < 3)) {                                                        \
        GTEST_SKIP() << "Need a 2D mesh with >= 3 colinear devices for a 1->1->1 chain; got " << shape;               \
    }                                                                                                                 \
    const bool kRowChain = shape[1] >= 3;                                                                             \
    auto stage0 = this->mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));                          \
    auto stage1 =                                                                                                     \
        this->mesh_device_->create_submesh(MeshShape(1, 1), kRowChain ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0)); \
    auto stage2 =                                                                                                     \
        this->mesh_device_->create_submesh(MeshShape(1, 1), kRowChain ? MeshCoordinate(0, 2) : MeshCoordinate(2, 0))

// Single core, several iterations in one launch: proves the persistent middle device
// keeps consuming inbound and producing outbound across iterations without desync.
TEST_F(D2DStreamServiceTest, ThreeStageChainOwnIteration) {
    D2D_THREE_STAGE_GUARD();
    verify_three_stage_chain(stage0, stage1, stage2, ttnn::Shape({1, 1, 32, 64}), kWorkerCores, /*num_iters=*/4);
}

}  // namespace
}  // namespace ttnn::distributed::test
