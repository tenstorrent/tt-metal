// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2DStreamService creatability checks (milestone M1).
//
// These verify that `D2DStreamService::create_pair` runs end-to-end through the
// validated -> mapper -> allocate -> claim-service-cores -> MeshSocket-pair
// build and returns two queryable handles. No kernels run yet (steps 5/6), so
// there is no data-path assertion here — only that construction succeeds and
// the M1-live getters return sane state.
//
// Hardware requirements (tests skip cleanly otherwise):
//   * Fabric: create_pair builds a MeshSocket pair, which handshakes over the
//     control plane. We use a FABRIC_2D fixture.
//   * Service cores: ServiceCoreManager only claims cores on Blackhole / UBB
//     Galaxy clusters under Fast Dispatch (mirrors the precondition inside
//     create_pair), so we skip on other clusters.
//   * >= 2 devices in a 2D mesh: a sender/receiver pair needs two distinct
//     submeshes (escalation ladder steps 1-2: 1x1<->1x1, 1x4<->1x4).

#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

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
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

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
using ::tt::tt_metal::D2DStreamConfig;
using ::tt::tt_metal::D2DStreamService;
using ::tt::tt_metal::DataMovementConfig;
using ::tt::tt_metal::DataMovementProcessor;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::DeviceAddr;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::NOC;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::Program;
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

// Mirror the ServiceCoreManager precondition (Blackhole / UBB Galaxy under Fast
// Dispatch) so we skip cleanly elsewhere instead of fataling inside create_pair.
bool service_cores_supported() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    return cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE;
}

// FABRIC_2D over the system mesh (the full Galaxy on a UBB system).
using D2DStreamServiceTest = tt::tt_metal::GenericMeshDeviceFabric2DFixture;

// Fully-replicated placements sized to a submesh's dimensionality (identity on
// a 1x1 submesh; full tensor on every device otherwise).
ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(const MeshDevice& mesh) {
    return ttsl::SmallVector<MeshMapperConfig::Placement>(mesh.shape().dims(), MeshMapperConfig::Replicate{});
}

// Single worker core per side for the bring-up tests (num_workers == 1).
const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};

uint32_t core_range_volume(const CoreRange& cr) {
    return (cr.end_coord.x - cr.start_coord.x + 1) * (cr.end_coord.y - cr.start_coord.y + 1);
}

// Standard V0 config: UINT32 ROW_MAJOR DRAM-interleaved, replicated on every
// device, L1 socket FIFO. The mapper is built fresh per call (create_pair
// moves it out).
D2DStreamConfig make_config(const std::shared_ptr<MeshDevice>& sender_mesh, const ttnn::Shape& global_shape) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    return D2DStreamConfig{
        .global_spec = TensorSpec(global_shape, tensor_layout),
        .mapper = create_mesh_mapper(*sender_mesh, MeshMapperConfig{.placements = replicate_all(*sender_mesh)}),
        .socket_mem_config = SocketMemoryConfig{BufferType::L1, /*fifo_size=*/4096},
        .sender_worker_cores = kWorkerCores,
        .receiver_worker_cores = kWorkerCores,
    };
}

// Build a D2DStreamConfig, run create_pair, and assert the M1-live getters.
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

// M2: every worker-sync resource address (per-coord service-core L1 slots +
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

// M3: first end-to-end transfer. Host-loads the sender backing tensor with iota,
// simulates the sender worker grid (bumps data_ready_counter by num_workers on
// each sender service core so the persistent sender does exactly one transfer),
// then polls the receiver backing tensor until it matches.
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

    // WriteShard/ReadShard need a shared_ptr<MeshBuffer>; the Tensor only exposes
    // it through the device storage.
    auto sender_mesh_buffer = sender->get_backing_tensor().device_storage().get_mesh_buffer_leak_ownership();
    auto receiver_mesh_buffer = receiver->get_backing_tensor().device_storage().get_mesh_buffer_leak_ownership();

    // Load the sender backing tensor on every participating coord and make sure
    // the writes land before we trigger the sender.
    for (const auto& coord : coords) {
        WriteShard(sender_mesh->mesh_command_queue(), sender_mesh_buffer, iota, coord);
    }
    Finish(sender_mesh->mesh_command_queue());

    // Simulate the (not-yet-wired, step 8) sender worker grid: write num_workers
    // into each coord's data_ready_counter so (cur - 0) == num_workers triggers
    // exactly one transfer.
    const uint32_t num_workers = core_range_volume(sender->get_worker_cores());
    std::vector<uint32_t> trigger{num_workers};
    for (const auto& coord : coords) {
        tt::tt_metal::detail::WriteToDeviceL1(
            sender_mesh->get_device(coord),
            sender->get_service_core(coord),
            static_cast<uint32_t>(sender->get_data_ready_counter_addr(coord)),
            trigger);
    }

    // No host-visible completion signal in M3 (worker handshake lands in steps
    // 7/8), so poll the receiver backing tensor until the transfer lands.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
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

// M4: full worker handshake (steps 7-8). Launches placeholder sender + receiver
// worker workloads that drive `num_iters` end-to-end transfers through the real
// handshakes (sender: data_ready_counter / consumed_sem; receiver:
// data_ready_sem / consumed_counter). The sender worker writes value (iter+1)
// uniformly each iteration, so the receiver backing tensor must hold `num_iters`
// after the loop — and the test only completes if nothing deadlocked.
void verify_handshake(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    uint32_t num_iters) {
    auto [sender, receiver] =
        D2DStreamService::create_pair(sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape));

    const auto& coords = sender->get_backing_tensor().tensor_topology().mesh_coords();

    const auto* sender_buffer = sender->get_backing_tensor().buffer();
    ASSERT_NE(sender_buffer, nullptr);
    const uint32_t tensor_page_size = sender_buffer->aligned_page_size();
    const uint32_t num_pages = sender_buffer->num_pages();

    constexpr auto kScratchCb = CBIndex::c_0;

    // ---- sender worker workload: one program per coord (uniform CT args, per-
    //      coord RT args for the service-core counter + NoC coords) ----
    MeshWorkload sender_workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();

        auto cb_cfg = CircularBufferConfig(tensor_page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, tensor_page_size);
        CreateCircularBuffer(program, kWorkerCores, cb_cfg);

        const auto* dbuf = sender->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(sender->get_consumed_sem_addr()),
            static_cast<uint32_t>(sender_buffer->address()),
            num_pages,
            tensor_page_size,
            num_iters,
            static_cast<uint32_t>(kScratchCb),
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/placeholder_d2d_sender_worker.cpp",
            kWorkerCores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = sender_mesh->get_device(coord);
        const auto service_phys = device->worker_core_from_logical_core(sender->get_service_core(coord));
        const std::vector<uint32_t> rt_args = {
            static_cast<uint32_t>(sender->get_data_ready_counter_addr(coord)),
            static_cast<uint32_t>(service_phys.x),
            static_cast<uint32_t>(service_phys.y),
        };
        for (const auto& wc : kWorkerCores) {
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        sender_workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }

    // ---- receiver worker workload: handshake only (no data read needed) ----
    MeshWorkload receiver_workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(receiver->get_data_ready_sem_addr()),
            num_iters,
        };
        auto kernel = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/placeholder_d2d_receiver_worker.cpp",
            kWorkerCores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

        auto* device = receiver_mesh->get_device(coord);
        const auto service_phys = device->worker_core_from_logical_core(receiver->get_service_core(coord));
        const std::vector<uint32_t> rt_args = {
            static_cast<uint32_t>(receiver->get_consumed_counter_addr(coord)),
            static_cast<uint32_t>(service_phys.x),
            static_cast<uint32_t>(service_phys.y),
        };
        for (const auto& wc : kWorkerCores) {
            SetRuntimeArgs(program, kernel, wc, rt_args);
        }
        receiver_workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }

    // Launch receivers first so they're ready to ack, then the senders produce.
    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);

    // Wait for all num_iters on both sides. Finishing the receiver workers
    // guarantees the final transfer is durable in the backing tensor (the
    // service barriers its DRAM writes before mcasting data_ready_sem).
    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    auto receiver_mesh_buffer = receiver->get_backing_tensor().device_storage().get_mesh_buffer_leak_ownership();
    const size_t num_u32 = sender_buffer->size() / sizeof(uint32_t);
    const std::vector<uint32_t> expected(num_u32, num_iters);
    std::vector<uint32_t> readback;
    for (const auto& coord : coords) {
        readback.clear();
        ReadShard(receiver_mesh->mesh_command_queue(), readback, receiver_mesh_buffer, coord);
        EXPECT_EQ(readback, expected) << "receiver backing mismatch at " << coord;
    }
}

// Escalation-ladder step 1: 1x1 <-> 1x1 (single chip each).
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

    verify_creatable(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}));
}

// Escalation-ladder step 2: 1x4 <-> 1x4 (exercises the per-coord SocketConnection
// list and the 1:1 coord mapping with > 1 socket).
TEST_F(D2DStreamServiceTest, CreatableRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 4) {
        GTEST_SKIP() << "Need a >= 2x4 mesh to carve 1x4 <-> 1x4 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(1, 0));

    verify_creatable(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}));
}

// M2: worker-sync resources (data_ready_counter / consumed_counter L1 slots +
// consumed_sem / data_ready_sem GlobalSemaphores) are allocated with non-zero
// addresses.
TEST_F(D2DStreamServiceTest, Milestone2SyncResources) {
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

    verify_sync_resources(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}));
}

// M3: first end-to-end device-to-device transfer (single chip pair). Host-readback
// of the receiver backing tensor must equal the sender iota.
TEST_F(D2DStreamServiceTest, Milestone3TransferSingleChipPair) {
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

    verify_transfer(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}));
}

// M3: same first transfer across a 1x4 row pair (exercises > 1 socket / coord).
TEST_F(D2DStreamServiceTest, Milestone3TransferRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 4) {
        GTEST_SKIP() << "Need a >= 2x4 mesh to carve 1x4 <-> 1x4 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(1, 0));

    verify_transfer(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}));
}

// M4: worker handshake end-to-end on a single chip pair, driven by real
// placeholder worker workloads over several iterations.
TEST_F(D2DStreamServiceTest, Milestone4HandshakeSingleChipPair) {
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

    verify_handshake(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}), /*num_iters=*/4);
}

// M4: worker handshake across a 1x4 row pair (multiple sockets / coords).
TEST_F(D2DStreamServiceTest, Milestone4HandshakeRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 4) {
        GTEST_SKIP() << "Need a >= 2x4 mesh to carve 1x4 <-> 1x4 submeshes; got " << shape;
    }

    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 4), MeshCoordinate(1, 0));

    verify_handshake(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}), /*num_iters=*/4);
}

}  // namespace
}  // namespace ttnn::distributed::test
