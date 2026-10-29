// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

#include "common/executor.hpp"

namespace tt::tt_metal {

TEST(ExecutorTest, AsyncRunsInline) {
    std::atomic<int> value{0};
    auto fut = detail::async([&value] { value.store(42); });
    fut.get();
    EXPECT_EQ(value.load(), 42);
}

TEST(ExecutorTest, RunAllLetsWorkersThatFreeUpTakeQueuedJobs) {
    const size_t workers = detail::GetExecutor().num_workers();
    constexpr size_t kJobs = 3;
    if (workers < kJobs) {
        GTEST_SKIP() << "needs " << kJobs << " executor workers, has " << workers;
    }
    // Hold every worker but one, and call run_all from that one, as a kernel's build step does. Executor::async always
    // queues: detail::async could run a holder on this thread, which would then wait for the gate forever.
    std::promise<void> gate;
    std::shared_future<void> released = gate.get_future().share();
    std::atomic<size_t> held{0};
    std::vector<std::future<void>> holders;
    for (size_t i = 0; i + 1 < workers; ++i) {
        holders.push_back(detail::GetExecutor().async([&] {
            held.fetch_add(1);
            released.wait();
        }));
    }
    while (held.load() < workers - 1) {
        std::this_thread::yield();
    }
    // The first job frees the held workers, then each job waits until all kJobs jobs are running at once.
    constexpr auto kTimeout = std::chrono::seconds(10);
    std::atomic<size_t> started{0};
    std::atomic<bool> timed_out{false};
    auto job = [&] {
        if (started.fetch_add(1) == 0) {
            gate.set_value();
        }
        const auto deadline = std::chrono::steady_clock::now() + kTimeout;
        while (started.load() < kJobs) {
            if (std::chrono::steady_clock::now() > deadline) {
                timed_out.store(true);
                return;
            }
            std::this_thread::yield();
        }
    };
    detail::GetExecutor().async([&] { detail::run_all(std::vector<std::function<void()>>(kJobs, job)); }).get();
    for (auto& holder : holders) {
        holder.get();
    }
    EXPECT_FALSE(timed_out.load()) << "the queued jobs never ran at the same time";
}

TEST(ExecutorTest, RunAllFromEveryWorkerDoesNotDeadlock) {
    const size_t workers = detail::GetExecutor().num_workers();
    std::atomic<size_t> started{0};
    std::atomic<size_t> inner_done{0};
    std::vector<std::future<void>> outer;
    for (size_t i = 0; i < workers; ++i) {
        outer.push_back(detail::GetExecutor().async([&] {
            started.fetch_add(1);
            while (started.load() < workers) {
                std::this_thread::yield();
            }
            // Every worker is here, so no other worker can run these jobs.
            detail::run_all({[&] { inner_done.fetch_add(1); }, [&] { inner_done.fetch_add(1); }});
        }));
    }
    for (auto& f : outer) {
        f.get();
    }
    EXPECT_EQ(inner_done.load(), 2 * workers);
}

TEST(ExecutorTest, RunAllRethrowsAfterEveryJobFinishes) {
    std::atomic<int> finished{0};
    auto slow = [&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        finished.fetch_add(1);
    };
    EXPECT_THROW(detail::run_all({[] { throw std::runtime_error("build failed"); }, slow, slow}), std::runtime_error);
    EXPECT_EQ(finished.load(), 2);
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
