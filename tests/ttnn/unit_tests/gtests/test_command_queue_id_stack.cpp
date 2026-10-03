// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the TTNN-owned thread-local "current command queue id" stack.
// These need no device: the stack lives in TTNN and never touches Metal / MetalContext.

#include <gtest/gtest.h>

#include <thread>

#include "ttnn/core.hpp"

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

}  // namespace
}  // namespace ttnn
