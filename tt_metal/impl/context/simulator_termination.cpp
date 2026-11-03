// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simulator_termination.hpp"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <pthread.h>
#include <semaphore.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace tt::tt_metal {
namespace {

// Default bound on MetalContext teardown after Ctrl-C. A second SIGINT/SIGTERM kills immediately.
// Override with TT_METAL_SIMULATOR_TERMINATION_GRACE_SECONDS.
constexpr unsigned int kDefaultGraceSeconds = 30;

unsigned int grace_seconds = kDefaultGraceSeconds;
volatile std::sig_atomic_t termination_signal = 0;
volatile std::sig_atomic_t shutdown_exit_code = 0;
volatile std::sig_atomic_t shutdown_tid = 0;
sem_t shutdown_semaphore;
bool handlers_installed = false;

pid_t current_tid() { return static_cast<pid_t>(::syscall(SYS_gettid)); }

void write_stderr(const char* data, size_t size) {
    const int saved_errno = errno;
    while (size > 0) {
        const ssize_t written = ::write(STDERR_FILENO, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        data += written;
        size -= static_cast<size_t>(written);
    }
    errno = saved_errno;
}

void write_stderr(const char* text) { write_stderr(text, std::strlen(text)); }

void write_stderr_number(const char* prefix, long number, const char* suffix) {
    char digits[24];
    char* end = digits + sizeof(digits);
    char* cursor = end;
    const bool negative = number < 0;
    unsigned long magnitude = negative ? 0UL - static_cast<unsigned long>(number) : static_cast<unsigned long>(number);
    do {
        *--cursor = static_cast<char>('0' + (magnitude % 10));
        magnitude /= 10;
    } while (magnitude != 0);
    if (negative) {
        *--cursor = '-';
    }
    write_stderr(prefix);
    write_stderr(cursor, static_cast<size_t>(end - cursor));
    write_stderr(suffix);
}

extern "C" void handle_simulator_teardown_alarm(int signal_number);

void arm_teardown_alarm() {
    struct sigaction alarm_action{};
    alarm_action.sa_handler = handle_simulator_teardown_alarm;
    ::sigemptyset(&alarm_action.sa_mask);
    (void)::sigaction(SIGALRM, &alarm_action, nullptr);
    ::alarm(grace_seconds);
}

extern "C" void handle_simulator_teardown_alarm(int /*signal_number*/) {
    write_stderr_number(
        "[tt-metal] simulator teardown did not finish within ",
        static_cast<long>(grace_seconds),
        "s; the emulator may still be held.\n");
    _exit(shutdown_exit_code != 0 ? shutdown_exit_code : 128 + termination_signal);
}

// Teardown has to take SimulationChip::device_lock. The thread that received the signal is often
// inside a device read, already holding that non-recursive mutex. Exiting on that thread deadlocks
// and libttsim_exit never runs. Posting here lets that call return and drop the lock; the shutdown
// thread then exits between device calls.
// A fault on the test thread must not kill the process. Teardown runs on the shutdown thread via
// std::exit, and the test thread is still inside the test. Once the cluster is gone that thread
// faults in get_cluster(), and the default SIGSEGV action kills teardown before DEVICE_COMMAND_EXIT
// is sent. Park every thread except the one actually exiting. Same idea as the hang-app handler,
// which parks a thread whose device call fails while shutdown is in progress.
extern "C" void handle_fault_during_shutdown(int signal_number) {
    if (termination_signal != 0 && shutdown_tid != 0 && current_tid() != shutdown_tid) {
        for (;;) {
            ::pause();
        }
    }
    std::signal(signal_number, SIG_DFL);
    ::raise(signal_number);
}

extern "C" void handle_simulator_termination(int signal_number) {
    termination_signal = signal_number;
    shutdown_exit_code = 128 + signal_number;
    write_stderr_number(
        "[tt-metal] signal ", signal_number, ": releasing the simulator (interrupt again to kill immediately).\n");

    // Return to the interrupted call so it can drop SimulationChip::device_lock. The second signal
    // restores the default action and kills the process. Faults on any other thread park instead.
    std::signal(signal_number, SIG_DFL);
    std::signal(SIGSEGV, handle_fault_during_shutdown);
    std::signal(SIGABRT, handle_fault_during_shutdown);
    arm_teardown_alarm();
    (void)::sem_post(&shutdown_semaphore);
}

[[noreturn]] void park_until_process_exits() {
    write_stderr("[tt-metal] a device call failed while the simulator was shutting down; waiting for teardown.\n");
    for (;;) {
        ::pause();
    }
}

std::terminate_handler previous_terminate = nullptr;

void handle_terminate_during_shutdown() {
    if (termination_signal != 0) {
        park_until_process_exits();
    }
    if (previous_terminate != nullptr) {
        previous_terminate();
    }
    std::abort();
}

void report_simulator_released() {
    if (termination_signal != 0) {
        write_stderr("[tt-metal] simulator released.\n");
    }
}

void run_shutdown_thread() {
    shutdown_tid = current_tid();
    while (::sem_wait(&shutdown_semaphore) != 0) {
        if (errno != EINTR) {
            _exit(128 + SIGINT);
        }
    }
    // std::exit runs MetalContext's atexit handler, which destroys the cluster and calls libttsim_exit
    // (or sends DEVICE_COMMAND_EXIT). Returning from main would not: the signal interrupted it.
    std::exit(shutdown_exit_code != 0 ? shutdown_exit_code : 128 + termination_signal);
}

// fork() does not duplicate the shutdown thread. Restart it in the child or Ctrl-C there is swallowed.
void restart_shutdown_thread_after_fork() {
    if (!handlers_installed) {
        return;
    }
    termination_signal = 0;
    shutdown_exit_code = 0;
    shutdown_tid = 0;
    // The parent's waiter was not duplicated, so this copy has no waiters. Re-init it for a new thread.
    ::sem_destroy(&shutdown_semaphore);
    if (::sem_init(&shutdown_semaphore, 0, 0) != 0) {
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
        handlers_installed = false;
        return;
    }
    try {
        std::thread(run_shutdown_thread).detach();
    } catch (...) {
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
        handlers_installed = false;
    }
}

bool disposition_is_default(int signal_number) {
    struct sigaction current{};
    if (::sigaction(signal_number, nullptr, &current) != 0) {
        return false;
    }
    if ((current.sa_flags & SA_SIGINFO) != 0) {
        return false;
    }
    return current.sa_handler == SIG_DFL;
}

bool install_handler(int signal_number) {
    struct sigaction action{};
    action.sa_handler = handle_simulator_termination;
    ::sigemptyset(&action.sa_mask);
    // The interrupted device call should resume and drop its lock; the shutdown thread exits it.
    action.sa_flags = SA_RESTART;
    return ::sigaction(signal_number, &action, nullptr) == 0;
}

}  // namespace

void install_simulator_termination_handler() {
    if (handlers_installed) {
        return;
    }
    const char* simulator = std::getenv("TT_METAL_SIMULATOR");
    if (simulator == nullptr || simulator[0] == '\0') {
        return;
    }

    const bool sigint_default = disposition_is_default(SIGINT);
    const bool sigterm_default = disposition_is_default(SIGTERM);
    // Python (and the hang apps) already turn these into a normal shutdown. Replacing that handler
    // would skip their cleanup.
    if (!sigint_default && !sigterm_default) {
        return;
    }

    if (const char* grace = std::getenv("TT_METAL_SIMULATOR_TERMINATION_GRACE_SECONDS")) {
        const long parsed = std::strtol(grace, nullptr, 10);
        if (parsed > 0) {
            grace_seconds = static_cast<unsigned int>(parsed);
        }
    }

    if (::sem_init(&shutdown_semaphore, 0, 0) != 0) {
        return;
    }
    try {
        std::thread(run_shutdown_thread).detach();
    } catch (...) {
        return;
    }

    bool installed_one = false;
    if (sigint_default) {
        installed_one = install_handler(SIGINT) || installed_one;
    }
    if (sigterm_default) {
        installed_one = install_handler(SIGTERM) || installed_one;
    }
    if (!installed_one) {
        return;
    }

    handlers_installed = true;
    // Registered before MetalContext::destroy_all_instances, so it runs after that teardown.
    std::atexit(report_simulator_released);
    previous_terminate = std::set_terminate(handle_terminate_during_shutdown);
    (void)::pthread_atfork(nullptr, nullptr, restart_shutdown_thread_after_fork);
}

bool release_simulator_session(int exit_code, const char* reason) {
    const char* simulator = std::getenv("TT_METAL_SIMULATOR");
    if (simulator == nullptr || simulator[0] == '\0') {
        return false;
    }

    write_stderr("[tt-metal] ");
    write_stderr(reason);
    write_stderr(" Releasing the simulator.\n");

    // The shutdown thread is what makes Ctrl-C safe. A fatal error is on a normal call stack, so exiting
    // here still runs atexit. Prefer that thread when it exists, so a second failure during teardown parks
    // instead of aborting past libttsim_exit.
    if (!handlers_installed) {
        std::exit(exit_code);
    }
    if (termination_signal == 0) {
        termination_signal = 1;
    }
    shutdown_exit_code = exit_code;
    std::signal(SIGSEGV, handle_fault_during_shutdown);
    std::signal(SIGABRT, handle_fault_during_shutdown);
    arm_teardown_alarm();
    (void)::sem_post(&shutdown_semaphore);
    for (;;) {
        ::pause();
    }
}

}  // namespace tt::tt_metal
