// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the TTNN-owned thread-local "current command queue id" stack.
// The CommandQueueIdStack tests need no device: the stack lives in TTNN and never touches Metal / MetalContext.
// The CommandQueueSelectionFixture test needs a fast-dispatch device with 2 CQs and checks the Metal/TTNN boundary.
// The SingleCommandQueueFixture tests select a queue the device does not have, in paths that never look a queue up.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/async_runtime.hpp"
#include "ttnn/core.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "impl/context/metal_context.hpp"

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

Tensor make_tile_tensor(MeshDevice* device) {
    const tt::tt_metal::TensorLayout layout(
        DataType::BFLOAT16,
        tt::tt_metal::PageConfig(Layout::TILE),
        MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    return ttnn::create_device_tensor(TensorSpec(Shape({1, 1, 32, 32}), layout), device);
}

using CommandQueueSelectionFixture = ttnn::MultiCommandQueueSingleDeviceFixture;

// Metal ignores the TTNN selection; only TTNN's resolver follows it.
TEST_F(CommandQueueSelectionFixture, MetalNoArgQueueIsZeroWhileTtnnResolverFollowsThread) {
    ASSERT_GE(device_->num_hw_cqs(), 2);

    EXPECT_EQ(device_->mesh_command_queue().id(), 0u);
    EXPECT_EQ(current_mesh_command_queue(*device_).id(), 0u);
    {
        auto guard = with_command_queue_id(QueueId(1));
        EXPECT_EQ(device_->mesh_command_queue().id(), 0u);
        EXPECT_EQ(current_mesh_command_queue(*device_).id(), 1u);
        EXPECT_EQ(&current_mesh_command_queue(*device_), &device_->mesh_command_queue(1));
        EXPECT_EQ(current_mesh_command_queue(*device_, QueueId(0)).id(), 0u);
    }
    EXPECT_EQ(current_mesh_command_queue(*device_).id(), 0u);
}

// ttnn::global_semaphore::reset_global_semaphore_value issues the (blocking) reset on the thread's current queue, so
// it stays ordered behind work dispatched under with_command_queue_id; an explicit cq_id passed to Metal targets that
// queue.
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
    auto input = make_tile_tensor(device);
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

    // Explicit queue id: Metal writes through the queue it is given.
    semaphore.reset_semaphore_value(3, /*cq_id=*/0);
    ttnn::queue_synchronize(device->mesh_command_queue(0));
    EXPECT_EQ(read_semaphore(), 3u);

    // Outside any TTNN scope the TTNN wrapper resolves to cq 0.
    ttnn::global_semaphore::reset_global_semaphore_value(semaphore, 11);
    ttnn::queue_synchronize(device->mesh_command_queue(0));
    EXPECT_EQ(read_semaphore(), 11u);
}

using SingleCommandQueueFixture = ttnn::TTNNFixtureWithDevice;

// Graph capture in NO_DISPATCH mode returns before the workload is enqueued, so the thread's queue is never looked up
// and an op can be captured even under a queue id the device does not have.
TEST_F(SingleCommandQueueFixture, NoDispatchGraphCaptureDoesNotResolveCommandQueue) {
    // Inspector (when enabled) reads the current trace id from the dispatch queue; keep it out of the way so only the
    // dispatch path is exercised.
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const bool inspector_enabled = rtoptions.get_inspector_enabled();
    rtoptions.set_inspector_enabled(false);

    auto input = make_tile_tensor(device_);
    const QueueId missing_queue(static_cast<uint8_t>(device_->num_hw_cqs()));
    EXPECT_NO_THROW({
        auto capture = ttnn::graph::ScopedGraphCapture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
        auto guard = with_command_queue_id(missing_queue);
        auto output = ttnn::neg(input);
        capture.end_graph_capture();
    });
    {
        // Sanity check: with dispatch, the same selection does reach the queue lookup.
        auto guard = with_command_queue_id(missing_queue);
        EXPECT_ANY_THROW(ttnn::neg(input));
    }

    rtoptions.set_inspector_enabled(inspector_enabled);
}

// With slow dispatch the global semaphore is written directly, without a command queue, so the thread's queue
// selection is ignored.
TEST_F(SingleCommandQueueFixture, SlowDispatchGlobalSemaphoreIgnoresSelectedQueue) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        GTEST_SKIP() << "Slow dispatch only";
    }
    const CoreCoord core{0, 0};
    const CoreRangeSet cores(CoreRange(core, core));
    const QueueId missing_queue(static_cast<uint8_t>(device_->num_hw_cqs()));

    auto guard = with_command_queue_id(missing_queue);
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(device_, cores, /*initial_value=*/5);
    auto read_semaphore = [&]() {
        std::vector<uint32_t> value;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device_->get_device(tt::tt_metal::distributed::MeshCoordinate(0, 0)),
            core,
            static_cast<uint32_t>(semaphore.address()),
            sizeof(uint32_t),
            value);
        return value.at(0);
    };
    EXPECT_EQ(read_semaphore(), 5u);
    ttnn::global_semaphore::reset_global_semaphore_value(semaphore, 9);
    EXPECT_EQ(read_semaphore(), 9u);
}

}  // namespace
}  // namespace ttnn
