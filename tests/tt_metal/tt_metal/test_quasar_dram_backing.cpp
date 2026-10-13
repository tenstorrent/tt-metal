// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"
#include "llrt/hal/tt-2xx/quasar/qa_emulator_models.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <numeric>
#include <stdexcept>

using namespace tt;
using namespace tt::tt_metal;

// The 1x3, 2x3 and 2x3_DISPATCH emulator models back 64 MiB of DRAM per channel although their soc
// descriptor declares 1 GiB.
TEST_F(QuasarMeshDeviceSingleCardFixture, DramStaysInsideTheEmulatorBacking) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }
    const quasar_emu::ModelInfo* model =
        quasar_emu::find_model(MetalContext::instance().rtoptions().get_simulator_name());
    if (model == nullptr) {
        GTEST_SKIP() << "Not one of the emulator models with a known DRAM backing (1x3, 2x3, 2x3_DISPATCH).";
    }

    auto mesh_device = devices_[0];
    EXPECT_EQ(mesh_device->dram_size_per_channel(), model->dram_size_per_channel);

    constexpr uint32_t page_size = 4096;
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
        .bottom_up = false,
    };
    distributed::ReplicatedBufferConfig global_cfg{.size = page_size};
    auto buf = distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());
    EXPECT_LE(buf->address() + page_size, model->dram_size_per_channel);

    std::vector<uint32_t> src(page_size / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 0x1000u);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, buf, src);
    std::vector<uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);
    EXPECT_EQ(dst, src);

    // Above the backing storage: one page larger than the backing cannot fit in any DRAM bank.
    const DeviceAddr too_big = model->dram_size_per_channel + page_size;
    distributed::DeviceLocalBufferConfig too_big_cfg{.page_size = too_big, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig too_big_global{.size = too_big};
    EXPECT_THROW(distributed::MeshBuffer::create(too_big_global, too_big_cfg, mesh_device.get()), std::runtime_error);
}
