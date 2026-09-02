// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/host_api.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t num_elems = 1024;
constexpr uint32_t buf_size = num_elems * sizeof(uint32_t);
constexpr uint32_t page_size = buf_size / 8;

std::shared_ptr<distributed::MeshBuffer> make_interleaved_mesh_buffer(
    BufferType buffer_type, distributed::MeshDevice* mesh_device) {
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = page_size,
        .buffer_type = buffer_type,
    };
    distributed::ReplicatedBufferConfig global_cfg{.size = buf_size};
    return distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device);
}

std::shared_ptr<distributed::MeshBuffer> make_sharded_l1_mesh_buffer(
    distributed::MeshDevice* mesh_device, TensorMemoryLayout layout, const CoreCoord& worker_grid) {
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t sharded_page_size = tile_height * tile_width * sizeof(uint32_t);
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    // One page per worker node. Height-sharded stacks pages along H; width-sharded along W.
    const std::array<uint32_t, 2> tensor_shape_in_pages = layout == TensorMemoryLayout::HEIGHT_SHARDED
                                                              ? std::array<uint32_t, 2>{num_nodes, 1}
                                                              : std::array<uint32_t, 2>{1, num_nodes};
    ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1})),
        {tile_height, tile_width},
        ShardOrientation::ROW_MAJOR,
        {tile_height, tile_width},
        tensor_shape_in_pages);
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = sharded_page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(std::move(shard_spec), layout),
        .bottom_up = false};
    distributed::ReplicatedBufferConfig global_cfg{.size = num_nodes * sharded_page_size};
    return distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device);
}

}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferWriteReadDRAM) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_interleaved_mesh_buffer(BufferType::DRAM, devices_[0].get());

    std::vector<uint32_t> src(num_elems);
    std::iota(src.begin(), src.end(), 0xabcd0000u);

    distributed::EnqueueWriteMeshBuffer(cq, buf, src);

    std::vector<uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst, src);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferMultipleWriteReadRoundsDRAM) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_interleaved_mesh_buffer(BufferType::DRAM, devices_[0].get());

    for (uint32_t round = 0; round < 10; round++) {
        std::vector<uint32_t> src(num_elems);
        std::iota(src.begin(), src.end(), round * 1000u);

        distributed::EnqueueWriteMeshBuffer(cq, buf, src);

        std::vector<uint32_t> dst;
        distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);

        ASSERT_EQ(dst, src);
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferWriteReadL1) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_interleaved_mesh_buffer(BufferType::L1, devices_[0].get());

    std::vector<uint32_t> src(num_elems);
    std::iota(src.begin(), src.end(), 0xabcd0000u);

    distributed::EnqueueWriteMeshBuffer(cq, buf, src);

    std::vector<uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst, src);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferMultipleWriteReadRoundsL1) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_interleaved_mesh_buffer(BufferType::L1, devices_[0].get());

    for (uint32_t round = 0; round < 10; round++) {
        std::vector<uint32_t> src(num_elems);
        std::iota(src.begin(), src.end(), round * 1000u);

        distributed::EnqueueWriteMeshBuffer(cq, buf, src);

        std::vector<uint32_t> dst;
        distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);

        ASSERT_EQ(dst, src);
    }
}

TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, MeshBufferCrossCQWriteReadRoundsDRAM) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq0 = devices_[0]->mesh_command_queue(0);
    distributed::MeshCommandQueue& cq1 = devices_[0]->mesh_command_queue(1);
    std::shared_ptr<distributed::MeshBuffer> buf = make_interleaved_mesh_buffer(BufferType::DRAM, devices_[0].get());

    for (uint32_t round = 0; round < 10; round++) {
        std::vector<uint32_t> src(num_elems);
        std::iota(src.begin(), src.end(), round * 1000u);

        distributed::EnqueueWriteMeshBuffer(cq0, buf, src);
        distributed::MeshEvent write_event = cq0.enqueue_record_event();
        cq1.enqueue_wait_for_event(write_event);

        std::vector<uint32_t> dst;
        distributed::EnqueueReadMeshBuffer(cq1, dst, buf, /*blocking=*/true);

        ASSERT_EQ(dst, src);
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferHeightShardedL1) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    auto buffer = make_sharded_l1_mesh_buffer(mesh_device.get(), TensorMemoryLayout::HEIGHT_SHARDED, worker_grid);
    std::vector<uint32_t> src(num_nodes * 32 * 32);
    std::iota(src.begin(), src.end(), 0x5a5a0000u);

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, buffer, src);
    std::vector<uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, buffer, /*blocking=*/true);

    ASSERT_EQ(dst, src);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferWidthShardedL1) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    auto buffer = make_sharded_l1_mesh_buffer(mesh_device.get(), TensorMemoryLayout::WIDTH_SHARDED, worker_grid);
    std::vector<uint32_t> src(num_nodes * 32 * 32);
    std::iota(src.begin(), src.end(), 0x5a5a0000u);

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, buffer, src);
    std::vector<uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, buffer, /*blocking=*/true);

    ASSERT_EQ(dst, src);
}
