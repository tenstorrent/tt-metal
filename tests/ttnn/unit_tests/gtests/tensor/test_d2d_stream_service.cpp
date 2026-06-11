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
#include <cstdint>
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
#include "ttnn/tensor/socket_services.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

#include "stream_service_test_utils.hpp"

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

// The H2D front-end (make_h2d_service) pins a host DMA buffer for its FIFO. On a
// host booted with the IOMMU in passthrough/identity mode (e.g. iommu=pt, as on
// the Blackhole Galaxy bh-glx-*), UMD must pin physically-contiguous pages
// (TENSTORRENT_PIN_PAGES_CONTIGUOUS) and rejects the multi-page, non-contiguous
// H2D buffer with EINVAL ("Failed to pin pages for DMA buffer ..."). This is a
// host-config / H2D-infra limitation, NOT a D2D bug: the pure-D2D tests below run
// fine. Tests that build an H2DStreamService skip cleanly when the IOMMU isn't in
// DMA-translation mode. See notes/d2d_galaxy_h2d_pinning_failure.md.
bool h2d_host_pinning_supported() { return tt::tt_metal::MetalContext::instance().get_cluster().is_iommu_enabled(); }

// FABRIC_2D over the system mesh (the full Galaxy on a UBB system).
using D2DStreamServiceTest = tt::tt_metal::GenericMeshDeviceFabric2DFixture;

// replicate_all() lives in stream_service_test_utils.hpp (shared with the H2D
// stream-service tests).

// Single worker core per side (num_workers == 1).
const CoreRange kWorkerCores{CoreCoord{0, 0}, CoreCoord{0, 0}};

// Multi-core worker grids for the metadata tests (exercise the "last core writes"
// disambiguation on the sender and the broadcast-to-all-workers fan-out on the
// receiver). 2 cores, 4 cores; "all cores" is derived per-mesh in the test.
const CoreRange kWorkerCores2{CoreCoord{0, 0}, CoreCoord{0, 1}};
const CoreRange kWorkerCores4{CoreCoord{0, 0}, CoreCoord{1, 1}};

uint32_t core_range_volume(const CoreRange& cr) {
    return (cr.end_coord.x - cr.start_coord.x + 1) * (cr.end_coord.y - cr.start_coord.y + 1);
}

// Socket / scratch FIFO size for a given spec. derive_chunk_plan FATALs unless
// the FIFO holds at least one tensor page, so wide last dims (page > 4096 B)
// need a bigger FIFO than the historical 4096. Round the page up to a 4096
// multiple: narrow pages keep a 4096 FIFO (so the FIFO still holds *several*
// pages, exercising chunk packing and the pages_per_chunk reduction loop),
// while a wide page gets a FIFO sized to exactly one page. 4096 is a multiple
// of the L1 alignment, so this is always >= the buffer's aligned_page_size.
uint32_t fifo_bytes_for(const TensorSpec& spec) {
    constexpr uint32_t kMinFifo = 4096u;
    const uint32_t page = static_cast<uint32_t>(spec.compute_page_size_bytes());
    return std::max(kMinFifo, ((page + kMinFifo - 1u) / kMinFifo) * kMinFifo);
}

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
    const TensorSpec global_spec(global_shape, tensor_layout);
    return D2DStreamConfig{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*sender_mesh, MeshMapperConfig{.placements = replicate_all(*sender_mesh)}),
        .socket_mem_config = SocketMemoryConfig{BufferType::L1, fifo_bytes_for(global_spec)},
        .sender_worker_cores = kWorkerCores,
        .receiver_worker_cores = kWorkerCores,
        .share_fabric_links = share_fabric_links,
    };
}

// DRAM-interleaved spec with an explicit dtype + layout (block-float formats
// require TILE). Used by the multi-dtype chain tests.
TensorSpec make_spec(const ttnn::Shape& global_shape, DataType dtype, Layout layout) {
    return TensorSpec(
        global_shape,
        TensorLayout(
            dtype, PageConfig(layout), MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}));
}

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
    auto mesh_buffer = receiver->get_backing_tensor().device_storage().get_mesh_buffer_leak_ownership();
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

// Fabric-link lease round-trip (share_fabric_links == true). With lease mode the
// service holds no fabric connection and does NOTHING until granted a turn, so each
// round must drive the full ping-pong: enqueue the worker workloads (which block on
// the service), grant both sides one transfer via release_fabric_links(), Finish,
// then wait_for_fabric_links() to confirm both services released the links again.
// Two rounds with distinct seeds prove the service re-acquires + releases the link
// per transfer and keeps streaming. A service that never re-acquired would hang the
// Finish; one that never released would hang wait_for_fabric_links.
void verify_fabric_lease(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape) {
    auto [sender, receiver] = D2DStreamService::create_pair(
        sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape, /*share_fabric_links=*/true));

    auto run_round = [&](uint32_t seed) {
        auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, /*num_iters=*/1);
        auto sender_workload =
            make_sender_worker_workload(sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/seed);
        EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
        EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);

        // Nothing moves until granted: lease the links to each service for one
        // transfer (a device hosting both an inbound receiver and an outbound sender
        // must grant both). Grant before Finish, else the workers block forever.
        receiver->release_fabric_links();
        sender->release_fabric_links();

        Finish(sender_mesh->mesh_command_queue());
        Finish(receiver_mesh->mesh_command_queue());

        // Both services must hand the links back (link_grant == 0) before the next
        // round — in production this is the point a fabric op would launch.
        sender->wait_for_fabric_links();
        receiver->wait_for_fabric_links();

        expect_receiver_backing_iota(receiver.get(), receiver_mesh, seed);
    };

    run_round(0x1234u);
    run_round(0x5678u);
}

// Fabric-link lease stress. Build the pair once, launch sender + receiver dummy
// worker programs that each loop num_iters INTERNALLY, then drive num_iters
// grant/release cycles interleaved with those worker iterations. The workers only
// advance an iteration when their service is granted the link, so the host loop
// paces every transfer — exercising the open/close + link_grant ping-pong many
// times against a single long-running workload. Catches state drift a 2-round test
// would miss: a stuck grant word, a failed connection re-open, or monotonic-counter
// desync would hang a wait_for_fabric_links() / Finish() mid-loop. The sender writes
// values 1..num_iters, so the receiver backing tensor must hold num_iters at the end.
void verify_fabric_lease_stress(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    uint32_t num_iters) {
    auto [sender, receiver] = D2DStreamService::create_pair(
        sender_mesh, receiver_mesh, make_config(sender_mesh, global_shape, /*share_fabric_links=*/true));

    auto receiver_workload = make_receiver_worker_workload(receiver.get(), receiver_mesh, num_iters);
    auto sender_workload = make_sender_worker_workload(sender.get(), sender_mesh, num_iters, /*fill_base=*/1);
    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), receiver_workload, /*blocking=*/false);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);

    // One grant per transfer. Grant both services (receiver first so it's ready to
    // drain), then wait for both to hand the links back before the next grant.
    for (uint32_t i = 0; i < num_iters; ++i) {
        receiver->release_fabric_links();
        sender->release_fabric_links();
        sender->wait_for_fabric_links();
        receiver->wait_for_fabric_links();
    }

    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    expect_receiver_backing_iota(receiver.get(), receiver_mesh, num_iters);
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

    const uint32_t recv_md_addr = static_cast<uint32_t>(receiver->get_metadata_addr());
    const CoreRange recv_workers = receiver->get_worker_cores();
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
// Part A: real receiver consumer (D2D -> Host). A consumer worker reads its page
// slice of the receiver backing tensor into a SEPARATE output tensor; the host
// validates the output tensor (not the backing tensor). See
// notes/d2d_test_flow_vs_realistic_workload.md.
// ===========================================================================

// Row-major worker index within a CoreRange (robust to iterator order).
uint32_t worker_index(const CoreCoord& wc, const CoreRange& worker_cores) {
    const uint32_t width = worker_cores.end_coord.x - worker_cores.start_coord.x + 1;
    return (wc.y - worker_cores.start_coord.y) * width + (wc.x - worker_cores.start_coord.x);
}

// Per-worker [start_page, end_page) over num_pages, distributing the remainder to
// the first `rem` workers. Empty ranges (num_pages < num_workers) are valid: that
// worker copies nothing but still handshakes, so the service's num_workers ack
// count is always satisfied.
std::pair<uint32_t, uint32_t> worker_page_range(uint32_t worker_idx, uint32_t num_workers, uint32_t num_pages) {
    const uint32_t base = num_pages / num_workers;
    const uint32_t rem = num_pages % num_workers;
    const uint32_t start = worker_idx * base + std::min(worker_idx, rem);
    const uint32_t end = start + base + (worker_idx < rem ? 1u : 0u);
    return {start, end};
}

// Build a receiver-side CONSUMER worker workload: one program per coord. Each
// worker copies its page slice of the receiver backing tensor into `output_tensor`
// (same spec) and runs the receiver handshake num_iters times. Grid-agnostic.
MeshWorkload make_receiver_consumer_workload(
    D2DStreamServiceReceiver* receiver,
    const std::shared_ptr<MeshDevice>& mesh,
    const tt::tt_metal::Tensor& output_tensor,
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
    const tt::tt_metal::Tensor& output_tensor, const std::shared_ptr<MeshDevice>& mesh, uint32_t base) {
    auto mesh_buffer = output_tensor.device_storage().get_mesh_buffer_leak_ownership();
    const size_t num_u32 = output_tensor.buffer()->size() / sizeof(uint32_t);
    const std::vector<uint32_t> expected = make_iota_u32(num_u32, base);
    std::vector<uint32_t> readback;
    for (const auto& coord : output_tensor.tensor_topology().mesh_coords()) {
        readback.clear();
        ReadShard(mesh->mesh_command_queue(), readback, mesh_buffer, coord);
        EXPECT_EQ(readback, expected) << "output tensor mismatch at " << coord << " (iota base " << base << ")";
    }
}

// Part A driver: real consumer end-to-end (no H2D). The existing placeholder
// sender worker produces value (fill_base + iter) into the sender backing tensor;
// the receiver consumer copies the landed data into a separate output tensor. The
// final value (fill_base + num_iters - 1) must appear in the output tensor on
// every coord. With metadata enabled, the {-1,0,final} blob must also land on the
// sender service core and every receiver worker core.
void verify_receiver_consume(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_iters,
    uint32_t metadata_size_bytes = 0) {
    constexpr uint32_t kFillBase = 1u;
    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = metadata_size_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto output_tensor = tt::tt_metal::create_device_tensor(
        receiver->get_backing_tensor().tensor_spec(),
        receiver_mesh.get(),
        receiver->get_backing_tensor().tensor_topology());

    auto consumer_workload = make_receiver_consumer_workload(receiver.get(), receiver_mesh, output_tensor, num_iters);
    auto sender_workload =
        make_sender_worker_workload(sender.get(), sender_mesh, num_iters, kFillBase, metadata_size_bytes);

    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), consumer_workload, /*blocking=*/false);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);
    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    const uint32_t final_value = kFillBase + num_iters - 1;
    expect_output_tensor_iota(output_tensor, receiver_mesh, final_value);
    if (metadata_size_bytes > 0) {
        expect_metadata_everywhere(
            sender.get(), receiver.get(), sender_mesh, receiver_mesh, {static_cast<uint32_t>(-1), 0u, final_value});
    }
}

// Part A reuse: build the pair once, drive num_rounds single-transfer rounds with
// distinct seeds against the same persistent service + a reused output tensor,
// verifying each round. Catches early kernel exit / stale-buffer reuse.
void verify_receiver_consume_reuse(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_rounds,
    uint32_t metadata_size_bytes = 0) {
    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = metadata_size_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto output_tensor = tt::tt_metal::create_device_tensor(
        receiver->get_backing_tensor().tensor_spec(),
        receiver_mesh.get(),
        receiver->get_backing_tensor().tensor_topology());

    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t seed = 0x1000u + round * 0x111u;
        auto consumer_workload =
            make_receiver_consumer_workload(receiver.get(), receiver_mesh, output_tensor, /*num_iters=*/1);
        auto sender_workload = make_sender_worker_workload(
            sender.get(), sender_mesh, /*num_iters=*/1, /*fill_base=*/seed, metadata_size_bytes);
        EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), consumer_workload, /*blocking=*/false);
        EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), sender_workload, /*blocking=*/false);
        Finish(sender_mesh->mesh_command_queue());
        Finish(receiver_mesh->mesh_command_queue());

        expect_output_tensor_iota(output_tensor, receiver_mesh, seed);
        if (metadata_size_bytes > 0) {
            expect_metadata_everywhere(
                sender.get(), receiver.get(), sender_mesh, receiver_mesh, {static_cast<uint32_t>(-1), 0u, seed});
        }
    }
}

// ===========================================================================
// Part B: H2D front-end (Host -> H2D -> bridge -> D2D -> Host). A bridge worker on
// the sender mesh drains the H2D backing tensor into the D2D sender backing tensor
// and drives both handshakes; the receiver consumer (Part A) lands the result in
// an output tensor the host validates.
// ===========================================================================

// Build the bridge worker workload on the sender mesh: one program per coord. Each
// worker copies its page slice of the H2D backing tensor into the D2D sender
// backing tensor and runs both handshakes num_iters times. The designated
// (highest-id) worker forwards the metadata from its L1 to the D2D sender service.
MeshWorkload make_bridge_workload(
    H2DStreamService& h2d_service,
    D2DStreamServiceSender* d2d_sender,
    const std::shared_ptr<MeshDevice>& mesh,
    const CoreRange& worker_cores,
    uint32_t num_iters,
    uint32_t metadata_size_bytes) {
    const auto& coords = d2d_sender->get_backing_tensor().tensor_topology().mesh_coords();
    const auto* backing_buffer = d2d_sender->get_backing_tensor().buffer();
    const uint32_t page_size = backing_buffer->aligned_page_size();
    const uint32_t num_pages = backing_buffer->num_pages();
    const uint32_t num_workers = core_range_volume(worker_cores);
    const bool metadata_enabled = metadata_size_bytes > 0;
    constexpr auto kScratchCb = CBIndex::c_0;

    const uint32_t h2d_input_addr = static_cast<uint32_t>(h2d_service.get_backing_tensor().buffer()->address());
    const uint32_t d2d_backing_addr = static_cast<uint32_t>(backing_buffer->address());
    const uint32_t h2d_metadata_l1_addr =
        metadata_enabled ? static_cast<uint32_t>(h2d_service.get_metadata_addr()) : 0u;

    MeshWorkload workload;
    for (const auto& coord : coords) {
        auto program = CreateProgram();

        auto cb_cfg = CircularBufferConfig(page_size, {{kScratchCb, tt::DataFormat::UInt32}})
                          .set_page_size(kScratchCb, page_size);
        CreateCircularBuffer(program, worker_cores, cb_cfg);

        const auto* dbuf = d2d_sender->get_backing_tensor().mesh_buffer().get_device_buffer(coord);
        auto accessor_ct = TensorAccessorArgs(*dbuf).get_compile_time_args();
        std::vector<uint32_t> ct_args = {
            static_cast<uint32_t>(h2d_service.get_data_ready_sem_addr()),
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
        const auto h2d_service_phys = device->worker_core_from_logical_core(h2d_service.get_service_core(coord));
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
                static_cast<uint32_t>(h2d_service.get_consumed_counter_addr(coord)),
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
    const TensorSpec& global_spec,
    const CoreRange& worker_cores,
    uint32_t metadata_size_bytes) {
    const uint32_t fifo_bytes = fifo_bytes_for(global_spec);
    H2DStreamService::Config h2d_cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*sender_mesh, MeshMapperConfig{.placements = replicate_all(*sender_mesh)}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = fifo_bytes,
        .scratch_cb_size_bytes = fifo_bytes,
        .worker_cores = worker_cores,
        .metadata_size_bytes = metadata_size_bytes,
    };
    return std::make_unique<H2DStreamService>(sender_mesh, std::move(h2d_cfg));
}

// Push one token (uniform `value`) + its {-1,0,value} metadata through the H2D
// service. A zero-length metadata span when disabled.
void h2d_push_token(H2DStreamService& h2d_service, uint32_t num_u32, uint32_t base, uint32_t metadata_size_bytes) {
    const std::vector<uint32_t> token = make_iota_u32(num_u32, base);
    const std::vector<uint32_t> meta_words = {static_cast<uint32_t>(-1), 0u, base};
    const auto bytes =
        ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(token.data()), token.size() * sizeof(uint32_t));
    const auto meta =
        ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(meta_words.data()), metadata_size_bytes);
    h2d_service.forward_to_tensor(bytes, meta);
}

// Part B driver: full Host -> H2D -> bridge -> D2D -> consumer -> Host chain. The
// host streams num_iters tokens (value fill_base+i); the final value must appear
// in the receiver output tensor, with metadata {-1,0,final} on the sender service
// core and every receiver worker core when enabled.
void verify_h2d_d2d_bridge(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    const CoreRange& worker_cores,
    uint32_t num_iters,
    uint32_t metadata_size_bytes = 0) {
    constexpr uint32_t kFillBase = 1u;
    const bool metadata_enabled = metadata_size_bytes > 0;

    const auto global_spec = make_config(sender_mesh, global_shape).global_spec;
    auto h2d_service = make_h2d_service(sender_mesh, global_spec, worker_cores, metadata_size_bytes);

    auto cfg = make_config(sender_mesh, global_shape);
    cfg.sender_worker_cores = worker_cores;
    cfg.receiver_worker_cores = worker_cores;
    cfg.metadata_size_bytes = metadata_size_bytes;
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto output_tensor = tt::tt_metal::create_device_tensor(
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
        h2d_push_token(*h2d_service, num_u32, kFillBase + i, metadata_size_bytes);
    }
    h2d_service->barrier();
    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    const uint32_t final_value = kFillBase + num_iters - 1;
    expect_output_tensor_iota(output_tensor, receiver_mesh, final_value);
    if (metadata_enabled) {
        expect_metadata_everywhere(
            sender.get(), receiver.get(), sender_mesh, receiver_mesh, {static_cast<uint32_t>(-1), 0u, final_value});
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

    auto output_tensor = tt::tt_metal::create_device_tensor(
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

        h2d_push_token(*h2d_service, num_u32, seed, metadata_size_bytes);
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

// Multi-dtype chain driver (no metadata). Streams `num_iters` tokens of element
// type T through Host -> H2D -> bridge -> D2D -> consumer -> Host and asserts the
// final token survives byte-faithfully. Values are a rotating modular iota
// src[i] = (iter + i) % modulus: bounded so low-precision dtypes don't saturate,
// distinct per element, and rotated by 1 each iteration so a stale/unwritten page
// reads back the wrong rotation. Verification compares the output tensor's
// dequantized values against the (identically quantized) input — exact, because
// every hop in the pipeline is a byte copy. Single-chip only so to_vector returns
// one replica unambiguously.
template <typename T>
void verify_dtype_chain(
    const std::shared_ptr<MeshDevice>& sender_mesh,
    const std::shared_ptr<MeshDevice>& receiver_mesh,
    const ttnn::Shape& global_shape,
    DataType dtype,
    Layout layout,
    const CoreRange& worker_cores,
    uint32_t num_iters,
    uint32_t modulus) {
    const auto global_spec = make_spec(global_shape, dtype, layout);
    const uint32_t num_elems = static_cast<uint32_t>(global_shape.volume());

    auto make_src = [&](uint32_t base) {
        std::vector<T> v(num_elems);
        for (uint32_t i = 0; i < num_elems; ++i) {
            v[i] = static_cast<T>(static_cast<float>((base + i) % modulus));
        }
        return v;
    };

    auto h2d_service = make_h2d_service(sender_mesh, global_spec, worker_cores, /*metadata_size_bytes=*/0);

    D2DStreamConfig cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*sender_mesh, MeshMapperConfig{.placements = replicate_all(*sender_mesh)}),
        .socket_mem_config = SocketMemoryConfig{BufferType::L1, fifo_bytes_for(global_spec)},
        .sender_worker_cores = worker_cores,
        .receiver_worker_cores = worker_cores,
        .share_fabric_links = false,  // OWN mode: stream without per-transfer grants
    };
    auto [sender, receiver] = D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));

    auto output_tensor = tt::tt_metal::create_device_tensor(
        receiver->get_backing_tensor().tensor_spec(),
        receiver_mesh.get(),
        receiver->get_backing_tensor().tensor_topology());

    auto consumer_workload = make_receiver_consumer_workload(receiver.get(), receiver_mesh, output_tensor, num_iters);
    auto bridge_workload = make_bridge_workload(
        *h2d_service, sender.get(), sender_mesh, worker_cores, num_iters, /*metadata_size_bytes=*/0);
    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), consumer_workload, /*blocking=*/false);
    EnqueueMeshWorkload(sender_mesh->mesh_command_queue(), bridge_workload, /*blocking=*/false);

    auto mapper = create_mesh_mapper(*sender_mesh, MeshMapperConfig{.placements = replicate_all(*sender_mesh)});
    tt::tt_metal::Tensor last_input;
    for (uint32_t i = 0; i < num_iters; ++i) {
        last_input = tt::tt_metal::Tensor::from_vector<T>(make_src(/*base=*/i), global_spec);
        auto distributed = ttnn::distributed::distribute_tensor(last_input, *mapper);
        h2d_service->forward_to_tensor(distributed);
    }
    h2d_service->barrier();
    Finish(sender_mesh->mesh_command_queue());
    Finish(receiver_mesh->mesh_command_queue());

    const std::vector<T> expected = last_input.to_vector<T>();
    const std::vector<T> got = output_tensor.to_vector<T>();
    EXPECT_EQ(got, expected) << "dtype chain mismatch (dtype " << static_cast<int>(dtype) << ")";
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
// Part A tests: real receiver consumer (D2D -> Host).
// ===========================================================================

// Single worker core: the consumer copies the whole landed tensor into the output
// tensor over several iterations.
TEST_F(D2DStreamServiceTest, ReceiverConsume1Core) {
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
    verify_receiver_consume(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}), kWorkerCores, /*num_iters=*/4);
}

// 4 cores with a page count NOT divisible by 4 (30 pages) to exercise the
// partition remainder branch (first 2 workers carry one extra page).
TEST_F(D2DStreamServiceTest, ReceiverConsume4Cores) {
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
    verify_receiver_consume(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 30, 64}), kWorkerCores4, /*num_iters=*/3);
}

// Full worker grid (the realistic case). num_pages < num_workers exercises the
// empty-range workers (they still handshake).
TEST_F(D2DStreamServiceTest, ReceiverConsumeAllCores) {
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
    verify_receiver_consume(sender_mesh, receiver_mesh, fixed_per_core_shape(all_cores), all_cores, /*num_iters=*/4);
}

// Full grid + metadata: the {-1,0,final} blob must land on the sender service core
// and every receiver worker core, alongside the consumed output tensor.
TEST_F(D2DStreamServiceTest, ReceiverConsumeMetadataAllCores) {
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
    verify_receiver_consume(
        sender_mesh,
        receiver_mesh,
        fixed_per_core_shape(all_cores),
        all_cores,
        /*num_iters=*/4,
        /*metadata_size_bytes=*/3u * sizeof(uint32_t));
}

// Full grid + reuse: build once, drive several rounds with distinct seeds.
TEST_F(D2DStreamServiceTest, ReceiverConsumeReuseAllCores) {
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
    verify_receiver_consume_reuse(
        sender_mesh, receiver_mesh, fixed_per_core_shape(all_cores), all_cores, /*num_rounds=*/4);
}

// Multi-device variant: mirrors the existing RowPair setup (1x2 <-> 1x2).
TEST_F(D2DStreamServiceTest, ReceiverConsumeRowPair) {
    if (!service_cores_supported()) {
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";
    }
    const auto shape = this->mesh_device_->shape();
    if (shape.dims() != 2 || shape[0] < 2 || shape[1] < 2) {
        GTEST_SKIP() << "Need a >= 2x2 mesh to carve 1x2 <-> 1x2 submeshes; got " << shape;
    }
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(0, 0));
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 2), MeshCoordinate(1, 0));
    verify_receiver_consume(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 32, 64}), kWorkerCores, /*num_iters=*/4);
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
// Single-chip + full-grid setup shared by the shape-matrix and multi-dtype
// tests below: carve two 1x1 submeshes (sender at (0,0), receiver adjacent) and
// an all-cores worker grid, skipping cleanly when the host can't provide them.
// ===========================================================================
#define D2D_SINGLECHIP_ALLCORES_GUARD()                                                                \
    if (!service_cores_supported()) {                                                                  \
        GTEST_SKIP() << "D2DStreamService service cores require Blackhole or UBB Galaxy.";             \
    }                                                                                                  \
    if (!h2d_host_pinning_supported()) {                                                               \
        GTEST_SKIP() << "H2D front-end host-DMA pinning requires a DMA-translation IOMMU (not "        \
                        "iommu=pt); see notes/d2d_galaxy_h2d_pinning_failure.md.";                     \
    }                                                                                                  \
    const auto shape = this->mesh_device_->shape();                                                    \
    if (shape.dims() != 2 || this->mesh_device_->num_devices() < 2) {                                  \
        GTEST_SKIP() << "Need a 2D mesh with >= 2 devices to carve distinct submeshes; got " << shape; \
    }                                                                                                  \
    const auto coord0 = MeshCoordinate(0, 0);                                                          \
    const auto coord1 = (shape[1] >= 2) ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);                 \
    auto sender_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord0);                    \
    auto receiver_mesh = this->mesh_device_->create_submesh(MeshShape(1, 1), coord1);                  \
    const auto grid = receiver_mesh->compute_with_storage_grid_size();                                 \
    const CoreRange all_cores { CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1} }

// ===========================================================================
// Part B shape matrix (uint32, full grid): exercises the page partition + socket
// chunk plan across the shapes that stress different edges. The baseline (dense,
// even per-core split) and tiny (empty-range workers) cases are already covered
// by H2DtoD2DBridgeAllCores / H2DtoD2DBridge1Core. These add:
//   - large       big page count, wide page > FIFO (one tensor page per chunk)
//   - uneven      711 pages mod num_workers -> remainder partition; wide page
//   - long-uneven thin remainder (just over num_workers); wide page
//   - narrow-uneven 256 B page -> multi-page chunks AND 711 not divisible by the
//                 per-chunk count, so derive_chunk_plan's reduction loop fires
// The unevenness that exercises the partition is the PAGE COUNT (711 / 155 rows);
// the last dim only sets the page size. The H2D socket requires the page size
// (last_dim * 4 B) to be PCIE-aligned, so the wide last dims are chosen as
// multiples of 64 u32 (256 B) while keeping the odd row counts.
// ===========================================================================

// ~536 MB / tensor; capped at 2 iters (still catches stale/ignored transfers via
// the per-iteration value bump) to bound runtime + fabric traffic.
TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeShapeLarge) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, ttnn::Shape({1, 8, 4096, 4096}), all_cores, /*num_iters=*/2);
}

TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeShapeUneven) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 711, 5120}), all_cores, /*num_iters=*/3);
}

TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeShapeLongUneven) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 155, 3712}), all_cores, /*num_iters=*/3);
}

TEST_F(D2DStreamServiceTest, H2DtoD2DBridgeShapeNarrowUneven) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_h2d_d2d_bridge(sender_mesh, receiver_mesh, ttnn::Shape({1, 1, 711, 64}), all_cores, /*num_iters=*/3);
}

// ===========================================================================
// Multi-dtype tests: the full chain transferring several element types (no
// metadata). Single-chip pair, full worker grid, 2 iterations (rotating data).
// BFLOAT16 / FLOAT32 / UINT8 are ROW_MAJOR; BFLOAT8_B / BFLOAT4_B are block-float
// formats that require TILE layout.
// ===========================================================================

TEST_F(D2DStreamServiceTest, DtypeBfloat16) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_dtype_chain<bfloat16>(
        sender_mesh,
        receiver_mesh,
        fixed_per_core_shape(all_cores),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        all_cores,
        /*num_iters=*/2,
        /*modulus=*/256);
}

TEST_F(D2DStreamServiceTest, DtypeFloat32) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_dtype_chain<float>(
        sender_mesh,
        receiver_mesh,
        fixed_per_core_shape(all_cores),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        all_cores,
        /*num_iters=*/2,
        /*modulus=*/4096);
}

TEST_F(D2DStreamServiceTest, DtypeUint8) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_dtype_chain<uint8_t>(
        sender_mesh,
        receiver_mesh,
        fixed_per_core_shape(all_cores),
        DataType::UINT8,
        Layout::ROW_MAJOR,
        all_cores,
        /*num_iters=*/2,
        /*modulus=*/128);
}

TEST_F(D2DStreamServiceTest, DtypeBfloat8B) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_dtype_chain<float>(
        sender_mesh,
        receiver_mesh,
        fixed_per_core_shape(all_cores),
        DataType::BFLOAT8_B,
        Layout::TILE,
        all_cores,
        /*num_iters=*/2,
        /*modulus=*/256);
}

TEST_F(D2DStreamServiceTest, DtypeBfloat4B) {
    D2D_SINGLECHIP_ALLCORES_GUARD();
    verify_dtype_chain<float>(
        sender_mesh,
        receiver_mesh,
        fixed_per_core_shape(all_cores),
        DataType::BFLOAT4_B,
        Layout::TILE,
        all_cores,
        /*num_iters=*/2,
        /*modulus=*/64);
}

}  // namespace
}  // namespace ttnn::distributed::test
