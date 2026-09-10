// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Buffer holds a non-owning allocator pointer, while the allocator tracks live buffers.
// These lifetime invariants are architecture-independent, so the tests run under mock.

#include <gtest/gtest.h>

#include <memory>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include <umd/device/types/arch.hpp>

#include "impl/device/mock_device_util.hpp"

namespace tt::tt_metal {
namespace {

constexpr DeviceAddr kBufferSize = 2048;

class BufferAllocatorLifetimeTest : public ::testing::Test {
protected:
    std::unique_ptr<MetalEnv> env_;
    std::shared_ptr<distributed::MeshDevice> mesh_device_;

    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(
            MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)));
        mesh_device_ = env_->create_mesh_device(distributed::MeshDeviceConfig(env_->get_system_mesh().shape()));
        ASSERT_GT(mesh_device_->num_devices(), 0u);
    }

    void TearDown() override {
        mesh_device_.reset();
        env_.reset();
    }

    IDevice* device() { return mesh_device_->get_devices()[0]; }
};

// Models a graph capture that begins after allocation and suppresses deallocation.
class DeallocateBlockingHooks : public IGraphHooks {
public:
    bool hook_allocate(const Buffer*) override { return false; }
    bool hook_deallocate(Buffer*) override { return true; }
    bool hook_program(Program*) override { return false; }
    bool hook_write_to_device(const Buffer*) override { return false; }
    bool hook_write_to_device(const distributed::MeshBuffer*) override { return false; }
    bool hook_read_from_device(Buffer*) override { return false; }
    bool hook_read_from_device(const distributed::MeshBuffer*) override { return false; }
};

}  // namespace

// Buffers that outlive their allocator must be marked deallocated before its pointer dangles.
TEST_F(BufferAllocatorLifetimeTest, BufferOutlivingItsAllocatorIsMarkedDeallocated) {
    auto buffer = Buffer::create(device(), kBufferSize, kBufferSize, BufferType::DRAM);
    ASSERT_TRUE(buffer->is_allocated());

    mesh_device_.reset();

    EXPECT_FALSE(buffer->is_allocated())
        << "buffer still claims to be allocated after its allocator was destroyed; destroying it "
           "will dereference the dangling allocator_";

    // Destroyed here: without the fix this is the dangling call.
    buffer.reset();
}

// Hook-suppressed deallocation must still remove a real buffer from allocator tracking.
TEST_F(BufferAllocatorLifetimeTest, HookSuppressedDeallocateUntracksBuffer) {
    const auto& allocator = device()->allocator();
    const size_t baseline = allocator->get_allocated_buffers().size();

    auto buffer = Buffer::create(device(), kBufferSize, kBufferSize, BufferType::DRAM);
    ASSERT_EQ(allocator->get_allocated_buffers().size(), baseline + 1)
        << "buffer was not registered with the allocator; the rest of this test proves nothing";

    auto hooks = std::make_shared<DeallocateBlockingHooks>();
    ASSERT_TRUE(GraphTracker::instance().add_hook(hooks));
    buffer.reset();
    GraphTracker::instance().clear_hook();

    EXPECT_EQ(allocator->get_allocated_buffers().size(), baseline)
        << "hook-suppressed deallocation left a dangling Buffer* in the allocator's tracking set";
}

}  // namespace tt::tt_metal
