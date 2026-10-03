// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the TTNN-owned thread-local "current command queue id" stack.
// The CommandQueueIdStack tests need no device: the stack lives in TTNN and never touches Metal / MetalContext.
// The CommandQueueSelectionFixture test needs a fast-dispatch device with 2 CQs and checks the Metal/TTNN boundary.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/async_runtime.hpp"
#include "ttnn/core.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace {

TEST(CommandQueueIdStack, DefaultsToZero) { EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0)); }

TEST(CommandQueueIdStack, PushPopNest) {
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0));
    push_current_command_queue_id_for_thread(QueueId(1));
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(1));
    push_current_command_queue_id_for_thread(QueueId(0));
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0));
    EXPECT_EQ(pop_current_command_queue_id_for_thread(), QueueId(0));
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(1));
    EXPECT_EQ(pop_current_command_queue_id_for_thread(), QueueId(1));
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0));
}

TEST(CommandQueueIdStack, ScopeGuardRestores) {
    {
        auto guard = with_command_queue_id(QueueId(1));
        EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(1));
        with_command_queue_id(QueueId(0), [&]() { EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0)); });
        EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(1));
    }
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0));
}

TEST(CommandQueueIdStack, IsPerThread) {
    auto guard = with_command_queue_id(QueueId(1));
    QueueId seen_in_thread(0);
    QueueId seen_in_thread_inside(0);
    std::thread worker([&]() {
        seen_in_thread = get_current_command_queue_id_for_thread();
        auto inner = with_command_queue_id(QueueId(1));
        seen_in_thread_inside = get_current_command_queue_id_for_thread();
    });
    worker.join();
    EXPECT_EQ(seen_in_thread, QueueId(0));
    EXPECT_EQ(seen_in_thread_inside, QueueId(1));
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(1));
}

TEST(CommandQueueIdStack, PopOnEmptyStackThrows) {
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0));
    EXPECT_ANY_THROW(pop_current_command_queue_id_for_thread());
    EXPECT_EQ(get_current_command_queue_id_for_thread(), QueueId(0));
}

// Runs without hardware on a 2-chip mock cluster: the explicit-queue reset overload must reject a queue that
// belongs to a different mesh than the semaphore.
class GlobalSemaphoreMockMeshTest : public ::testing::Test {
protected:
    void TearDown() override { tt::tt_metal::experimental::disable_mock_mode(); }
};

TEST_F(GlobalSemaphoreMockMeshTest, ExplicitQueueResetRejectsQueueFromAnotherMesh) {
    tt::tt_metal::experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 2);
    auto meshes = tt::tt_metal::distributed::MeshDevice::create_unit_meshes({0, 1});
    ASSERT_EQ(meshes.size(), 2u);
    auto& mesh_a = meshes.at(0);
    auto& mesh_b = meshes.at(1);
    {
        const CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
        auto semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_a.get(), cores, /*initial_value=*/0);
        EXPECT_NO_THROW(semaphore.reset_semaphore_value(1, mesh_a->mesh_command_queue()));
        EXPECT_ANY_THROW(semaphore.reset_semaphore_value(1, mesh_b->mesh_command_queue()));
    }
    for (auto& [_, mesh] : meshes) {
        mesh->close();
    }
}

using CommandQueueSelectionFixture = ttnn::MultiCommandQueueSingleDeviceFixture;

// The central boundary of the design: Metal has no implicit queue state (no-arg mesh_command_queue() is always
// cq 0), while TTNN's resolver follows the thread's selection.
TEST_F(CommandQueueSelectionFixture, MetalNoArgQueueIsZeroWhileTtnnResolverFollowsThread) {
    ASSERT_GE(device_->num_hw_cqs(), 2);

    EXPECT_EQ(device_->mesh_command_queue().id(), 0u);
    EXPECT_EQ(current_mesh_command_queue(*device_).id(), 0u);
    {
        auto guard = with_command_queue_id(QueueId(1));
        // Metal: still cq 0, regardless of what TTNN selected for this thread.
        EXPECT_EQ(device_->mesh_command_queue().id(), 0u);
        // TTNN: the thread's current queue.
        EXPECT_EQ(current_mesh_command_queue(*device_).id(), 1u);
        EXPECT_EQ(&current_mesh_command_queue(*device_), &device_->mesh_command_queue(1));
        // An explicit id wins over the thread's selection.
        EXPECT_EQ(current_mesh_command_queue(*device_, QueueId(0)).id(), 0u);
    }
    EXPECT_EQ(current_mesh_command_queue(*device_).id(), 0u);
}

// ttnn::global_semaphore::reset_global_semaphore_value issues the (blocking) reset on the thread's current queue, so
// it stays ordered behind work dispatched under with_command_queue_id; the explicit-queue Metal overload targets the
// queue it is given.
TEST_F(CommandQueueSelectionFixture, GlobalSemaphoreResetFollowsSelectedQueue) {
    ASSERT_GE(device_->num_hw_cqs(), 2);
    auto* device = device_;

    const CoreCoord core{0, 0};
    const CoreRangeSet cores(CoreRange(core, core));
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(device, cores, /*initial_value=*/0);

    // The semaphore lives in L1 of `core` on the single device of this unit mesh; read it back over the slow
    // (non-CQ) path once the queue that wrote it has been drained.
    auto read_semaphore = [&]() {
        std::vector<uint32_t> value;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device->get_device(tt::tt_metal::distributed::MeshCoordinate(0, 0)),
            core,
            static_cast<uint32_t>(semaphore.address()),
            sizeof(uint32_t),
            value);
        return value.at(0);
    };

    // Upload an input on cq 0 and make sure it has landed before cq 1 consumes it.
    constexpr uint32_t num_elements = 32 * 32;
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elements]);
    for (uint32_t i = 0; i < num_elements; i++) {
        host_data[i] = bfloat16(2.0f);
    }
    const tt::tt_metal::TensorLayout layout(
        DataType::BFLOAT16,
        tt::tt_metal::PageConfig(Layout::TILE),
        MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    auto input = ttnn::create_device_tensor(TensorSpec(Shape({1, 1, 32, 32}), layout), device);
    ttnn::write_buffer(QueueId(0), input, {host_data});
    ttnn::queue_synchronize(device->mesh_command_queue(0));

    Tensor output;
    {
        auto guard = with_command_queue_id(QueueId(1));
        // cq-1 work, then a reset that resolves to cq 1 as well (ordered behind the op in the queue).
        output = ttnn::neg(input);
        ttnn::global_semaphore::reset_global_semaphore_value(semaphore, 7);
        ttnn::queue_synchronize(device->mesh_command_queue(1));
        EXPECT_EQ(read_semaphore(), 7u);
    }
    auto readback = std::shared_ptr<bfloat16[]>(new bfloat16[num_elements]);
    ttnn::read_buffer(QueueId(1), output, {readback});
    ttnn::queue_synchronize(device->mesh_command_queue(1));
    for (uint32_t i = 0; i < num_elements; i++) {
        ASSERT_EQ(static_cast<float>(readback[i]), -2.0f) << "at index " << i;
    }

    // Explicit queue: the Metal overload writes through the queue it is given.
    semaphore.reset_semaphore_value(3, device->mesh_command_queue(0));
    ttnn::queue_synchronize(device->mesh_command_queue(0));
    EXPECT_EQ(read_semaphore(), 3u);

    // Outside any TTNN scope the TTNN wrapper resolves to cq 0.
    ttnn::global_semaphore::reset_global_semaphore_value(semaphore, 11);
    ttnn::queue_synchronize(device->mesh_command_queue(0));
    EXPECT_EQ(read_semaphore(), 11u);
}

}  // namespace
}  // namespace ttnn
