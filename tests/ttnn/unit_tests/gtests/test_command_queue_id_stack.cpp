// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The thread-local "current command queue id" stack is owned by TTNN and does not touch Metal, so these tests run
// without a device.

#include <gtest/gtest.h>

#include <thread>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"

namespace ttnn {
namespace {

TEST(CommandQueueIdStack, DefaultsToQueueZero) {
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(0));
}

TEST(CommandQueueIdStack, PushPopNest) {
    ttnn::push_current_command_queue_id_for_thread(QueueId(1));
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(1));
    ttnn::push_current_command_queue_id_for_thread(QueueId(3));
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(3));
    EXPECT_EQ(ttnn::pop_current_command_queue_id_for_thread(), QueueId(3));
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(1));
    EXPECT_EQ(ttnn::pop_current_command_queue_id_for_thread(), QueueId(1));
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(0));
}

TEST(CommandQueueIdStack, PopOnEmptyStackThrows) { EXPECT_ANY_THROW(ttnn::pop_current_command_queue_id_for_thread()); }

TEST(CommandQueueIdStack, ScopeGuardRestoresOnExit) {
    {
        auto guard = ttnn::with_command_queue_id(QueueId(2));
        EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(2));
    }
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(0));
}

TEST(CommandQueueIdStack, FunctionOverloadRestoresAfterThrow) {
    EXPECT_ANY_THROW(ttnn::with_command_queue_id(QueueId(1), [] {
        EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(1));
        throw std::runtime_error("boom");
    }));
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(0));
}

TEST(CommandQueueIdStack, StackIsPerThread) {
    auto guard = ttnn::with_command_queue_id(QueueId(1));
    QueueId other_thread_value(99);
    std::thread t([&] {
        other_thread_value = ttnn::get_current_command_queue_id_for_thread();
        auto inner = ttnn::with_command_queue_id(QueueId(2));
    });
    t.join();
    EXPECT_EQ(other_thread_value, QueueId(0));
    EXPECT_EQ(ttnn::get_current_command_queue_id_for_thread(), QueueId(1));
}

}  // namespace
}  // namespace ttnn
