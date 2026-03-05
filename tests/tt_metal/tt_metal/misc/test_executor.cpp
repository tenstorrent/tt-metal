// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "common/executor.hpp"

namespace tt::tt_metal {

TEST(ExecutorTest, AsyncRunsInline) {
    std::atomic<int> value{0};
    auto fut = detail::async([&value] { value.store(42); });
    fut.get();
    EXPECT_EQ(value.load(), 42);
}

TEST(ExecutorDeathTest, ForkWithInflightWorkAborts) {
    ASSERT_DEATH(
        {
            detail::GetExecutor().silent_async([] { std::this_thread::sleep_for(std::chrono::seconds(5)); });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            fork();
        },
        "fork.*in-flight work");
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
