// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "impl/context/simulator_termination.hpp"

namespace simulator_termination_test {

// Exit code the release-path children ask for, to tell a real release from an incidental exit.
constexpr int kReleaseExitCode = 7;
// Exit code of the child that exits normally while a signal arrives mid-teardown.
constexpr int kNormalExitCode = 5;
constexpr int kMarkerTimeoutMs = 5000;
constexpr int kChildTimeoutMs = 5000;

int child_write_fd = -1;

void write_marker() {
    if (child_write_fd < 0) {
        return;
    }
    const char marker = 'x';
    (void)write(child_write_fd, &marker, 1);
}

std::atomic<bool> fault_requested{false};
std::atomic<bool> fault_raised{false};

void raise_sigsegv_when_requested() {
    while (!fault_requested.load()) {
        usleep(1000);
    }
    fault_raised.store(true);
    raise(SIGSEGV);
}

// Runs while the exiting thread is still inside exit(). Faults another thread and waits for the
// fault to actually be raised, so the marker cannot be written on a run where it never happened.
void fault_other_thread_then_mark() {
    fault_requested.store(true);
    for (int waited_ms = 0; waited_ms < 2000 && !fault_raised.load(); ++waited_ms) {
        usleep(1000);
    }
    if (!fault_raised.load()) {
        return;
    }
    usleep(200 * 1000);
    write_marker();
}

// Stands in for MetalContext's teardown atexit: slow enough for a signal to land inside it.
void slow_teardown_then_mark() {
    raise(SIGINT);
    usleep(500 * 1000);
    write_marker();
}

void slow_teardown() { usleep(3000 * 1000); }

void custom_sigint_handler(int) { _exit(42); }

// Signals this process and expects to die from it, so the parent can tell a default disposition
// from a handler that ran teardown.
int die_from_signal_in_fork_child() {
    const pid_t pid = fork();
    if (pid == -1) {
        return 1;
    }
    if (pid == 0) {
        raise(SIGTERM);
        // Reached only if the inherited handler swallowed the signal.
        usleep(2000 * 1000);
        _exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        return 1;
    }
    return (WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM) ? 0 : 1;
}

// gtest is already multithreaded. A child that only fork()s inherits locks held by threads that
// were not copied, and std::exit then deadlocks, so every scenario runs in a fresh image.
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

    if (std::strcmp(mode, "release-without-simulator") == 0) {
        unsetenv("TT_METAL_SIMULATOR");
        _exit(tt::tt_metal::release_simulator_session(1, "dispatch_core_manager") ? 1 : 0);
    }
    if (std::strcmp(mode, "sigint-without-simulator") == 0) {
        unsetenv("TT_METAL_SIMULATOR");
        tt::tt_metal::install_simulator_termination_handler();
        raise(SIGINT);
        _exit(0);
    }

    setenv("TT_METAL_SIMULATOR", "libttsim.so", 1);
    if (std::strcmp(mode, "existing-handler") == 0) {
        std::signal(SIGINT, custom_sigint_handler);
        tt::tt_metal::install_simulator_termination_handler();
        raise(SIGINT);
        _exit(0);
    }

    tt::tt_metal::install_simulator_termination_handler();
    if (std::strcmp(mode, "sigint") == 0) {
        std::atexit(write_marker);
        tt::tt_metal::install_simulator_exit_guard();
        raise(SIGINT);
        for (;;) {
            pause();
        }
    } else if (std::strcmp(mode, "release") == 0) {
        std::atexit(write_marker);
        tt::tt_metal::install_simulator_exit_guard();
        tt::tt_metal::release_simulator_session(kReleaseExitCode, "dispatch_core_manager");
    } else if (std::strcmp(mode, "fault-during-teardown") == 0) {
        std::thread(raise_sigsegv_when_requested).detach();
        std::atexit(fault_other_thread_then_mark);
        tt::tt_metal::install_simulator_exit_guard();
        tt::tt_metal::release_simulator_session(kReleaseExitCode, "dispatch_core_manager");
    } else if (std::strcmp(mode, "signal-during-exit") == 0) {
        std::atexit(slow_teardown_then_mark);
        tt::tt_metal::install_simulator_exit_guard();
        std::exit(kNormalExitCode);
    } else if (std::strcmp(mode, "second-signal") == 0) {
        std::atexit(slow_teardown);
        tt::tt_metal::install_simulator_exit_guard();
        raise(SIGINT);
        usleep(500 * 1000);
        raise(SIGTERM);
        for (;;) {
            pause();
        }
    } else if (std::strcmp(mode, "fork-child") == 0) {
        _exit(die_from_signal_in_fork_child());
    }
    _exit(99);
}

struct RunChildMode {
    RunChildMode() { run_child_mode(); }
};

// The exec'd image re-enters here before gtest starts, and never returns to main.
RunChildMode run_child_mode_before_main;

// Reap a child that should exit on its own. Kill it if teardown stalls, so a hung atexit handler
// fails the test instead of holding the runner.
bool wait_for_child(pid_t pid, int* status) {
    for (int waited_ms = 0; waited_ms < kChildTimeoutMs; waited_ms += 100) {
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

void kill_and_reap(pid_t pid) {
    kill(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);
}

// Runs a scenario in a fresh image and checks it wrote the marker from atexit and then exited with
// the expected status. Returns false (with the failure already recorded) if anything went wrong.
void expect_marker_and_exit_status(const char* mode, int expected_status) {
    int pipe_fds[2];
    ASSERT_EQ(pipe(pipe_fds), 0);
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        close(pipe_fds[0]);
        exec_child(mode, pipe_fds[1]);
    }
    close(pipe_fds[1]);

    pollfd ready{.fd = pipe_fds[0], .events = POLLIN, .revents = 0};
    const int poll_result = poll(&ready, 1, kMarkerTimeoutMs);
    char marker = 0;
    const ssize_t nread = poll_result == 1 ? read(pipe_fds[0], &marker, 1) : 0;
    close(pipe_fds[0]);
    if (poll_result != 1) {
        kill_and_reap(pid);
        FAIL() << "child never reached its atexit handler";
    }

    int status = 0;
    const bool finished = wait_for_child(pid, &status);
    EXPECT_EQ(nread, 1);
    EXPECT_EQ(marker, 'x');
    ASSERT_TRUE(finished) << "child did not exit on its own";
    ASSERT_TRUE(WIFEXITED(status)) << "child died from signal " << WTERMSIG(status);
    EXPECT_EQ(WEXITSTATUS(status), expected_status);
}

}  // namespace simulator_termination_test

TEST(SimulatorTermination, SigintRunsAtexitWhenSimulatorIsSet) {
    simulator_termination_test::expect_marker_and_exit_status("sigint", 128 + SIGINT);
}

TEST(SimulatorTermination, SigintKillsWhenSimulatorIsUnset) {
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        simulator_termination_test::exec_child("sigint-without-simulator", -1);
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
        simulator_termination_test::exec_child("existing-handler", -1);
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
        simulator_termination_test::exec_child("release-without-simulator", -1);
    }

    int status = 0;
    ASSERT_TRUE(simulator_termination_test::wait_for_child(pid, &status));
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(SimulatorTermination, ReleaseSessionRunsAtexitWhenSimulatorIsSet) {
    simulator_termination_test::expect_marker_and_exit_status("release", simulator_termination_test::kReleaseExitCode);
}

TEST(SimulatorTermination, FaultOnAnotherThreadDuringTeardownStillRunsAtexit) {
    simulator_termination_test::expect_marker_and_exit_status(
        "fault-during-teardown", simulator_termination_test::kReleaseExitCode);
}

// A signal landing while exit() is already running teardown must not start a second exit: that is
// undefined, and it races the handler that releases the simulator.
TEST(SimulatorTermination, SignalDuringExitDoesNotStartASecondExit) {
    simulator_termination_test::expect_marker_and_exit_status(
        "signal-during-exit", simulator_termination_test::kNormalExitCode);
}

// The first signal hands over to teardown; the next one, of either kind, kills immediately.
TEST(SimulatorTermination, SecondTerminationSignalKillsImmediately) {
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        simulator_termination_test::exec_child("second-signal", -1);
    }

    int status = 0;
    ASSERT_TRUE(simulator_termination_test::wait_for_child(pid, &status));
    ASSERT_TRUE(WIFSIGNALED(status)) << "child exited " << WEXITSTATUS(status) << " instead of dying";
    EXPECT_EQ(WTERMSIG(status), SIGTERM);
}

// A forked child does not own the simulator session, so it must not run the parent's teardown.
TEST(SimulatorTermination, ForkChildKeepsDefaultSignalBehaviour) {
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";
    if (pid == 0) {
        simulator_termination_test::exec_child("fork-child", -1);
    }

    int status = 0;
    ASSERT_TRUE(simulator_termination_test::wait_for_child(pid, &status));
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0) << "forked child did not die from SIGTERM";
}
