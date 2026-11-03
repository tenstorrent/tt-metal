// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "impl/context/simulator_termination.hpp"

namespace simulator_termination_test {

bool sigint_is_default() {
    struct sigaction current{};
    if (sigaction(SIGINT, nullptr, &current) != 0) {
        return false;
    }
    return (current.sa_flags & SA_SIGINFO) == 0 && current.sa_handler == SIG_DFL;
}

// Reap a child that should exit on its own. Kill it if teardown stalls, so a hung atexit handler
// fails the test instead of holding the runner.
bool wait_for_child(pid_t pid, int* status) {
    for (int attempt = 0; attempt < 50; ++attempt) {
        const pid_t result = waitpid(pid, status, WNOHANG);
        if (result == pid) {
            return true;
        }
        if (result < 0) {
            return false;
        }
        usleep(100 * 1000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, status, 0);
    return false;
}

int child_write_fd = -1;

void write_marker() {
    if (child_write_fd < 0) {
        return;
    }
    const char marker = 'x';
    (void)write(child_write_fd, &marker, 1);
}

void custom_sigint_handler(int) { _exit(42); }

std::atomic<bool> request_segfault{false};

void raise_sigsegv_when_requested() {
    while (!request_segfault.load()) {
        usleep(1000);
    }
    raise(SIGSEGV);
}

// Runs first among the test's atexit handlers. Wakes the other thread so it faults while this thread
// is still inside std::exit, then records that teardown got past that fault.
void fault_other_thread_then_mark() {
    request_segfault.store(true);
    usleep(200 * 1000);
    write_marker();
}

// gtest is already multithreaded. A child that only fork()s inherits locks held by threads that were
// not copied, and std::exit then deadlocks. exec a fresh image so the scenario is single-threaded.
[[noreturn]] void exec_child(const char* mode, int write_fd) {
    if (write_fd >= 0) {
        (void)fcntl(write_fd, F_SETFD, 0);
        char fd_text[16];
        (void)snprintf(fd_text, sizeof(fd_text), "%d", write_fd);
        setenv("TT_METAL_SIM_TERM_FD", fd_text, 1);
    }
    setenv("TT_METAL_SIM_TERM_CHILD", mode, 1);
    struct sigaction default_action{};
    default_action.sa_handler = SIG_DFL;
    (void)sigaction(SIGINT, &default_action, nullptr);
    (void)sigaction(SIGTERM, &default_action, nullptr);
    execl("/proc/self/exe", "simterm-child", nullptr);
    _exit(127);
}

void run_child_mode() {
    const char* mode = std::getenv("TT_METAL_SIM_TERM_CHILD");
    if (mode == nullptr) {
        return;
    }
    if (const char* fd_text = std::getenv("TT_METAL_SIM_TERM_FD")) {
        child_write_fd = std::atoi(fd_text);
    }
    if (std::strcmp(mode, "noop") == 0) {
        unsetenv("TT_METAL_SIMULATOR");
        _exit(tt::tt_metal::release_simulator_session(1, "dispatch_core_manager") ? 1 : 0);
    }

    setenv("TT_METAL_SIMULATOR", "libttsim.so", 1);
    tt::tt_metal::install_simulator_termination_handler();
    if (std::strcmp(mode, "release") == 0) {
        std::atexit(write_marker);
        tt::tt_metal::release_simulator_session(7, "Trying to get dispatch_core_manager before initializing it.");
    } else if (std::strcmp(mode, "segv") == 0) {
        std::thread(raise_sigsegv_when_requested).detach();
        std::atexit(fault_other_thread_then_mark);
        tt::tt_metal::release_simulator_session(7, "Trying to get dispatch_core_manager before initializing it.");
    }
    _exit(99);
}

struct RunChildBeforeMain {
    RunChildBeforeMain() { run_child_mode(); }
};

RunChildBeforeMain run_child_before_main;

}  // namespace simulator_termination_test

TEST(SimulatorTermination, SigintRunsAtexitWhenSimulatorIsSet) {
    if (!simulator_termination_test::sigint_is_default()) {
        GTEST_SKIP() << "SIGINT already has a handler in this process";
    }

    int pipe_fds[2];
    ASSERT_EQ(pipe(pipe_fds), 0);
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        close(pipe_fds[0]);
        setenv("TT_METAL_SIMULATOR", "libttsim.so", 1);
        simulator_termination_test::child_write_fd = pipe_fds[1];
        tt::tt_metal::install_simulator_termination_handler();
        // Registered after the handler's own atexit, so this runs first and the parent can observe it
        // before the rest of process teardown.
        std::atexit(simulator_termination_test::write_marker);
        raise(SIGINT);
        for (;;) {
            pause();
        }
    }

    close(pipe_fds[1]);
    pollfd ready{.fd = pipe_fds[0], .events = POLLIN, .revents = 0};
    ASSERT_EQ(poll(&ready, 1, 5000), 1);
    char marker = 0;
    const ssize_t nread = read(pipe_fds[0], &marker, 1);
    close(pipe_fds[0]);

    int status = 0;
    const bool finished = simulator_termination_test::wait_for_child(pid, &status);
    EXPECT_EQ(nread, 1);
    EXPECT_EQ(marker, 'x');
    ASSERT_TRUE(finished);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 128 + SIGINT);
}

TEST(SimulatorTermination, SigintKillsWhenSimulatorIsUnset) {
    if (!simulator_termination_test::sigint_is_default()) {
        GTEST_SKIP() << "SIGINT already has a handler in this process";
    }

    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        unsetenv("TT_METAL_SIMULATOR");
        tt::tt_metal::install_simulator_termination_handler();
        raise(SIGINT);
        _exit(0);
    }

    int status = 0;
    ASSERT_TRUE(simulator_termination_test::wait_for_child(pid, &status));
    EXPECT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGINT);
}

TEST(SimulatorTermination, LeavesAnExistingHandlerInPlace) {
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        setenv("TT_METAL_SIMULATOR", "libttsim.so", 1);
        std::signal(SIGINT, simulator_termination_test::custom_sigint_handler);
        tt::tt_metal::install_simulator_termination_handler();
        raise(SIGINT);
        _exit(0);
    }

    int status = 0;
    ASSERT_TRUE(simulator_termination_test::wait_for_child(pid, &status));
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 42);
}

TEST(SimulatorTermination, ReleaseSessionNoopsWhenSimulatorIsUnset) {
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        simulator_termination_test::exec_child("noop", -1);
    }

    int status = 0;
    ASSERT_TRUE(simulator_termination_test::wait_for_child(pid, &status));
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(SimulatorTermination, ReleaseSessionRunsAtexitWhenSimulatorIsSet) {
    int pipe_fds[2];
    ASSERT_EQ(pipe(pipe_fds), 0);
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        close(pipe_fds[0]);
        simulator_termination_test::exec_child("release", pipe_fds[1]);
    }

    close(pipe_fds[1]);
    pollfd ready{.fd = pipe_fds[0], .events = POLLIN, .revents = 0};
    ASSERT_EQ(poll(&ready, 1, 5000), 1);
    char marker = 0;
    const ssize_t nread = read(pipe_fds[0], &marker, 1);
    close(pipe_fds[0]);

    int status = 0;
    const bool finished = simulator_termination_test::wait_for_child(pid, &status);
    EXPECT_EQ(nread, 1);
    EXPECT_EQ(marker, 'x');
    ASSERT_TRUE(finished);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 7);
}

TEST(SimulatorTermination, SegfaultDuringTeardownStillRunsAtexit) {
    int pipe_fds[2];
    ASSERT_EQ(pipe(pipe_fds), 0);
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        close(pipe_fds[0]);
        simulator_termination_test::exec_child("segv", pipe_fds[1]);
    }

    close(pipe_fds[1]);
    pollfd ready{.fd = pipe_fds[0], .events = POLLIN, .revents = 0};
    ASSERT_EQ(poll(&ready, 1, 5000), 1);
    char marker = 0;
    const ssize_t nread = read(pipe_fds[0], &marker, 1);
    close(pipe_fds[0]);

    int status = 0;
    const bool finished = simulator_termination_test::wait_for_child(pid, &status);
    EXPECT_EQ(nread, 1);
    EXPECT_EQ(marker, 'x');
    ASSERT_TRUE(finished);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 7);
}
