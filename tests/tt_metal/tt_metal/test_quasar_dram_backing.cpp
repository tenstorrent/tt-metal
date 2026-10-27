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
// descriptor declares 1 GiB. With TT_METAL_SIMULATOR_DRAM_UNCAPPED=1 the clamp is off, the channel
// reports the descriptor's size, a buffer larger than the backing allocates, and a page above the
// backing round-trips through the command queue.
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
    const DeviceAddr backing = model->dram_size_per_channel;
    constexpr uint32_t page_size = 4096;

    // Write one page to the buffer through the command queue and read it back.
    auto round_trip = [&](std::shared_ptr<distributed::MeshBuffer>& buf) {
        std::vector<uint32_t> src(page_size / sizeof(uint32_t));
        std::iota(src.begin(), src.end(), 0x1000u);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, buf, src);
        std::vector<uint32_t> dst;
        distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);
        EXPECT_EQ(dst, src);
    };
    // One page at the top of the DRAM bank.
    distributed::DeviceLocalBufferConfig top_page_cfg{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
        .bottom_up = false,
    };
    distributed::ReplicatedBufferConfig page_global{.size = page_size};
    // One page larger than the backing.
    const DeviceAddr too_big = backing + page_size;
    distributed::DeviceLocalBufferConfig too_big_cfg{.page_size = too_big, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig too_big_global{.size = too_big};

    if (MetalContext::instance().rtoptions().get_simulator_dram_uncapped()) {
        EXPECT_GT(mesh_device->dram_size_per_channel(), backing);
        // An ATT map whose DRAM window is smaller than the descriptor (horizon_2x3) still caps the bank.
        if (mesh_device->allocator()->get_bank_size(BufferType::DRAM) <= backing) {
            return;
        }
        auto big = distributed::MeshBuffer::create(too_big_global, too_big_cfg, mesh_device.get());
        EXPECT_GT(big->address() + too_big, backing);
        // The top page lies above the backing. Its write wraps onto the low DRAM and the read follows it there.
        auto top = distributed::MeshBuffer::create(page_global, top_page_cfg, mesh_device.get());
        EXPECT_GE(top->address(), backing);
        round_trip(top);
        return;
    }

    EXPECT_EQ(mesh_device->dram_size_per_channel(), backing);
    auto top = distributed::MeshBuffer::create(page_global, top_page_cfg, mesh_device.get());
    EXPECT_LE(top->address() + page_size, backing);
    round_trip(top);
    // Above the backing storage: one page larger than the backing cannot fit in any DRAM bank.
    EXPECT_THROW(distributed::MeshBuffer::create(too_big_global, too_big_cfg, mesh_device.get()), std::runtime_error);
}
