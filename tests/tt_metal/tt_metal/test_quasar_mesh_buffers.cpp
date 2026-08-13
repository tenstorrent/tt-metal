// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/host_api.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
#include "tt_metal/distributed/mesh_command_queue_base.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t num_elems = 1024;
constexpr uint32_t buf_size = num_elems * sizeof(uint32_t);
constexpr uint32_t page_size = buf_size / 8;

std::shared_ptr<distributed::MeshBuffer> make_mesh_buffer(
    BufferType buffer_type, distributed::MeshDevice* mesh_device) {
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = page_size,
        .buffer_type = buffer_type,
    };
    distributed::ReplicatedBufferConfig global_cfg{.size = buf_size};
    return distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device);
}

}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferWriteReadDRAM) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_mesh_buffer(BufferType::DRAM, devices_[0].get());

    std::vector<uint32_t> src(num_elems);
    std::iota(src.begin(), src.end(), 0xabcd0000u);

    distributed::as_mesh_command_queue_base(cq).enqueue_write_mesh_buffer(buf, src.data(), false);

    std::vector<uint32_t> dst;
    if ((buf)->global_layout() == tt::tt_metal::distributed::MeshBufferLayout::SHARDED) {
        (dst).resize((buf)->global_shard_spec().global_size / sizeof(typename std::decay_t<decltype(dst)>::value_type));
    } else {
        (dst).resize((buf)->size() / sizeof(typename std::decay_t<decltype(dst)>::value_type));
    }
    distributed::as_mesh_command_queue_base(cq).enqueue_read_mesh_buffer((dst).data(), buf, /*blocking=*/true);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst, src);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferMultipleWriteReadRoundsDRAM) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_mesh_buffer(BufferType::DRAM, devices_[0].get());

    for (uint32_t round = 0; round < 10; round++) {
        std::vector<uint32_t> src(num_elems);
        std::iota(src.begin(), src.end(), round * 1000u);

        distributed::as_mesh_command_queue_base(cq).enqueue_write_mesh_buffer(buf, src.data(), false);

        std::vector<uint32_t> dst;
        if ((buf)->global_layout() == tt::tt_metal::distributed::MeshBufferLayout::SHARDED) {
            (dst).resize(
                (buf)->global_shard_spec().global_size / sizeof(typename std::decay_t<decltype(dst)>::value_type));
        } else {
            (dst).resize((buf)->size() / sizeof(typename std::decay_t<decltype(dst)>::value_type));
        }
        distributed::as_mesh_command_queue_base(cq).enqueue_read_mesh_buffer((dst).data(), buf, /*blocking=*/true);

        ASSERT_EQ(dst, src);
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferWriteReadL1) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_mesh_buffer(BufferType::L1, devices_[0].get());

    std::vector<uint32_t> src(num_elems);
    std::iota(src.begin(), src.end(), 0xabcd0000u);

    distributed::as_mesh_command_queue_base(cq).enqueue_write_mesh_buffer(buf, src.data(), false);

    std::vector<uint32_t> dst;
    if ((buf)->global_layout() == tt::tt_metal::distributed::MeshBufferLayout::SHARDED) {
        (dst).resize((buf)->global_shard_spec().global_size / sizeof(typename std::decay_t<decltype(dst)>::value_type));
    } else {
        (dst).resize((buf)->size() / sizeof(typename std::decay_t<decltype(dst)>::value_type));
    }
    distributed::as_mesh_command_queue_base(cq).enqueue_read_mesh_buffer((dst).data(), buf, /*blocking=*/true);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst, src);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, MeshBufferMultipleWriteReadRoundsL1) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    distributed::MeshCommandQueue& cq = devices_[0]->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> buf = make_mesh_buffer(BufferType::L1, devices_[0].get());

    for (uint32_t round = 0; round < 10; round++) {
        std::vector<uint32_t> src(num_elems);
        std::iota(src.begin(), src.end(), round * 1000u);

        distributed::as_mesh_command_queue_base(cq).enqueue_write_mesh_buffer(buf, src.data(), false);

        std::vector<uint32_t> dst;
        if ((buf)->global_layout() == tt::tt_metal::distributed::MeshBufferLayout::SHARDED) {
            (dst).resize(
                (buf)->global_shard_spec().global_size / sizeof(typename std::decay_t<decltype(dst)>::value_type));
        } else {
            (dst).resize((buf)->size() / sizeof(typename std::decay_t<decltype(dst)>::value_type));
        }
        distributed::as_mesh_command_queue_base(cq).enqueue_read_mesh_buffer((dst).data(), buf, /*blocking=*/true);

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
    std::shared_ptr<distributed::MeshBuffer> buf = make_mesh_buffer(BufferType::DRAM, devices_[0].get());

    for (uint32_t round = 0; round < 10; round++) {
        std::vector<uint32_t> src(num_elems);
        std::iota(src.begin(), src.end(), round * 1000u);

        distributed::as_mesh_command_queue_base(cq0).enqueue_write_mesh_buffer(buf, src.data(), false);
        distributed::MeshEvent write_event = cq0.enqueue_record_event();
        cq1.enqueue_wait_for_event(write_event);

        std::vector<uint32_t> dst;
        if ((buf)->global_layout() == tt::tt_metal::distributed::MeshBufferLayout::SHARDED) {
            (dst).resize(
                (buf)->global_shard_spec().global_size / sizeof(typename std::decay_t<decltype(dst)>::value_type));
        } else {
            (dst).resize((buf)->size() / sizeof(typename std::decay_t<decltype(dst)>::value_type));
        }
        distributed::as_mesh_command_queue_base(cq1).enqueue_read_mesh_buffer((dst).data(), buf, /*blocking=*/true);

        ASSERT_EQ(dst, src);
    }
}
