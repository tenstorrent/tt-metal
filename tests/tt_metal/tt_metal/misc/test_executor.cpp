// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#if defined(__linux__)
#include <sched.h>
#endif
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "common/executor.hpp"

namespace tt::tt_metal {

TEST(ExecutorTest, ThreadCountRespectsCpuAffinity) {
#if defined(__linux__)
    cpu_set_t original_affinity;
    ASSERT_EQ(sched_getaffinity(0, sizeof(original_affinity), &original_affinity), 0);

    int selected_cpu = -1;
    for (int cpu_index = 0; cpu_index < CPU_SETSIZE; ++cpu_index) {
        if (CPU_ISSET(cpu_index, &original_affinity)) {
            selected_cpu = cpu_index;
            break;
        }
    }
    ASSERT_NE(selected_cpu, -1);

    cpu_set_t restricted_affinity;
    CPU_ZERO(&restricted_affinity);
    CPU_SET(selected_cpu, &restricted_affinity);
    ASSERT_EQ(sched_setaffinity(0, sizeof(restricted_affinity), &restricted_affinity), 0);

    const size_t detected_thread_count = detail::get_executor_thread_count();
    const int restore_result = sched_setaffinity(0, sizeof(original_affinity), &original_affinity);

    EXPECT_EQ(detected_thread_count, 1);
    ASSERT_EQ(restore_result, 0);
#else
    GTEST_SKIP() << "CPU affinity detection is Linux-specific";
#endif
}

TEST(ExecutorTest, AsyncRunsInline) {
    std::atomic<int> value{0};
    auto fut = detail::async([&value] { value.store(42); });
    fut.get();
    EXPECT_EQ(value.load(), 42);
}

TEST(ExecutorTest, ForkSafety) {
    std::atomic<int> pre{0};
    detail::async([&pre] { pre.store(1); }).get();
    ASSERT_EQ(pre.load(), 1);

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        // Child: the atfork handler should have replaced the dead executor.
        std::atomic<int> child_val{0};
        auto fut = detail::async([&child_val] { child_val.store(99); });
        fut.get();
        _exit(child_val.load() == 99 ? 0 : 1);
    }

    // Parent: verify the executor still works here too.
    std::atomic<int> parent_val{0};
    detail::async([&parent_val] { parent_val.store(77); }).get();
    EXPECT_EQ(parent_val.load(), 77);

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0) << "Child failed: detail::async did not work after fork";
}

}  // namespace tt::tt_metal
