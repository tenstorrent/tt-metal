// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the TTNN-owned thread-local "current command queue id" stack.
// The CommandQueueIdStack tests need no device: the stack lives in TTNN and never touches Metal / MetalContext.
// The CommandQueueSelectionFixture test needs a fast-dispatch device with 2 CQs and checks the Metal/TTNN boundary.

#include <gtest/gtest.h>

#include <thread>

#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/core.hpp"
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

}  // namespace
}  // namespace ttnn
