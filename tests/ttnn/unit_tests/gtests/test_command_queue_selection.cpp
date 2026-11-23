// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNN owns the per-thread "current command queue id". These tests check that ttnn::core::current_mesh_command_queue
// resolves an explicit id first, then the thread's current id, then cq 0.

#include <gtest/gtest.h>

#include <thread>

#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace {

using CommandQueueSelectionFixture = MultiCommandQueueSingleDeviceFixture;

TEST_F(CommandQueueSelectionFixture, ResolverDefaultsToQueueZero) {
    EXPECT_EQ(ttnn::core::get_current_command_queue_id_for_thread(), QueueId(0));
    EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_).id(), 0);
}

TEST_F(CommandQueueSelectionFixture, ResolverFollowsThreadScopeAndRestores) {
    {
        auto outer = ttnn::with_command_queue_id(QueueId(1));
        EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_).id(), 1);
        {
            auto inner = ttnn::with_command_queue_id(QueueId(0));
            EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_).id(), 0);
        }
        EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_).id(), 1);
    }
    EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_).id(), 0);
}

TEST_F(CommandQueueSelectionFixture, ExplicitQueueWinsOverThreadScope) {
    auto guard = ttnn::with_command_queue_id(QueueId(1));
    EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_, QueueId(0)).id(), 0);
    EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_, QueueId(1)).id(), 1);
}

// TTNN's selection is not visible to Metal: a Metal lookup without a queue id stays cq 0 inside a TTNN scope, which is
// why every TTNN call site resolves its queue through current_mesh_command_queue.
TEST_F(CommandQueueSelectionFixture, MetalNoArgQueueIsZeroInsideTtnnScope) {
    auto guard = ttnn::with_command_queue_id(QueueId(1));
    EXPECT_EQ(device_->mesh_command_queue().id(), 0);
    EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_).id(), 1);
}

TEST_F(CommandQueueSelectionFixture, ThreadScopeIsPerThread) {
    auto guard = ttnn::with_command_queue_id(QueueId(1));
    uint8_t other_thread_cq = 99;
    std::thread t([&] { other_thread_cq = ttnn::core::current_mesh_command_queue(*device_).id(); });
    t.join();
    EXPECT_EQ(other_thread_cq, 0);
    EXPECT_EQ(ttnn::core::current_mesh_command_queue(*device_).id(), 1);
}

TEST_F(CommandQueueSelectionFixture, WithCommandQueueIdRunsFunctionInScope) {
    uint8_t seen = 99;
    ttnn::with_command_queue_id(QueueId(1), [&] { seen = ttnn::core::current_mesh_command_queue(*device_).id(); });
    EXPECT_EQ(seen, 1);
    EXPECT_EQ(ttnn::core::get_current_command_queue_id_for_thread(), QueueId(0));
}

}  // namespace
}  // namespace ttnn
