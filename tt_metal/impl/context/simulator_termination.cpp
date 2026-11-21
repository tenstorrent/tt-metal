// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simulator_termination.hpp"

#include <atomic>
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

// Bound on teardown once shutdown starts. A second termination signal kills immediately.
// Override with TT_METAL_SIMULATOR_TERMINATION_GRACE_SECONDS.
constexpr unsigned int kDefaultGraceSeconds = 30;

// Read from signal handlers and from ordinary code on other threads, so plain sig_atomic_t would be
// a data race. These types are lock-free, which is what makes them usable from a handler.
using AtomicFlag = std::atomic<int>;
static_assert(AtomicFlag::is_always_lock_free, "signal handlers need a lock-free flag");

unsigned int grace_seconds = kDefaultGraceSeconds;
AtomicFlag shutdown_requested{0};
AtomicFlag shutdown_exit_code{0};
// Set once exit() has started running handlers, with the thread that is running them.
AtomicFlag exit_in_progress{0};
AtomicFlag teardown_tid{0};
sem_t shutdown_semaphore;
bool handlers_installed = false;
bool exit_guard_installed = false;
bool installed_sigint = false;
bool installed_sigterm = false;

int current_tid() { return static_cast<int>(::syscall(SYS_gettid)); }

bool simulator_configured() {
    const char* simulator = std::getenv("TT_METAL_SIMULATOR");
    return simulator != nullptr && simulator[0] != '\0';
}

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

extern "C" void handle_simulator_teardown_alarm(int /*signal_number*/) {
    write_stderr_number(
        "[tt-metal] simulator teardown did not finish within ",
        static_cast<long>(grace_seconds),
        "s; the emulator may still be held.\n");
    _exit(shutdown_exit_code.load());
}

void arm_teardown_alarm() {
    struct sigaction alarm_action{};
    alarm_action.sa_handler = handle_simulator_teardown_alarm;
    ::sigemptyset(&alarm_action.sa_mask);
    (void)::sigaction(SIGALRM, &alarm_action, nullptr);
    ::alarm(grace_seconds);
}

// Only the thread running exit() can make progress once teardown starts. Any other thread that
// faults would kill the process before the simulator is told to exit, so it waits here instead.
[[noreturn]] void park_until_process_exits() {
    for (;;) {
        ::pause();
    }
}

bool is_teardown_thread() { return current_tid() == teardown_tid.load(); }

bool shutdown_started() { return shutdown_requested.load() != 0 || exit_in_progress.load() != 0; }

extern "C" void handle_fault_during_shutdown(int signal_number) {
    if (shutdown_started() && !is_teardown_thread()) {
        write_stderr_number("[tt-metal] signal ", signal_number, " during teardown; parking this thread.\n");
        park_until_process_exits();
    }
    std::signal(signal_number, SIG_DFL);
    ::raise(signal_number);
}

void install_fault_handlers() {
    std::signal(SIGSEGV, handle_fault_during_shutdown);
    std::signal(SIGABRT, handle_fault_during_shutdown);
}

// Async-signal-safe. Arm before writing any diagnostics: stderr can block on a full pipe, and the
// grace alarm is what releases the simulator if it does.
void arm_shutdown(int exit_code) {
    shutdown_exit_code.store(exit_code);
    install_fault_handlers();
    arm_teardown_alarm();
}

void wake_shutdown_thread() { (void)::sem_post(&shutdown_semaphore); }

extern "C" void handle_simulator_termination(int signal_number) {
    // Whoever is signalling has stopped waiting for a clean release.
    if (shutdown_requested.exchange(1) != 0) {
        std::signal(signal_number, SIG_DFL);
        ::raise(signal_number);
        return;
    }
    std::signal(signal_number, SIG_DFL);

    arm_shutdown(128 + signal_number);

    // exit() is already running the teardown. Starting a second one concurrently is undefined and
    // races the handler that releases the simulator, so only bound the one in flight.
    if (exit_in_progress.load() != 0) {
        write_stderr_number(
            "[tt-metal] signal ", signal_number, ": already releasing the simulator (signal again to kill).\n");
        return;
    }

    write_stderr_number(
        "[tt-metal] signal ", signal_number, ": releasing the simulator (signal again to kill immediately).\n");
    // Wake last: the interrupted call has to return and drop SimulationChip::device_lock before
    // teardown can take it.
    wake_shutdown_thread();
}

std::terminate_handler previous_terminate = nullptr;

void handle_terminate_during_shutdown() {
    // Parking the thread that is running teardown would stall the release it is performing.
    if (shutdown_started() && !is_teardown_thread()) {
        write_stderr("[tt-metal] a call failed while the simulator was shutting down; waiting for teardown.\n");
        park_until_process_exits();
    }
    if (previous_terminate != nullptr) {
        previous_terminate();
    }
    std::abort();
}

void report_simulator_released() {
    if (shutdown_requested.load() != 0) {
        write_stderr("[tt-metal] simulator released.\n");
    }
}

void note_exit_started() {
    teardown_tid.store(current_tid());
    // A plain exit() releases the simulator too, so a fault on another thread must not kill it.
    install_fault_handlers();
    exit_in_progress.store(1);
}

void run_shutdown_thread() {
    while (::sem_wait(&shutdown_semaphore) != 0) {
        if (errno != EINTR) {
            _exit(shutdown_exit_code.load());
        }
    }
    // exit() became visible only after the handler decided to wake this thread. Running the exit
    // handlers a second time, concurrently, is undefined, so let the one in flight finish; the
    // handler already armed the grace alarm in case it wedges.
    if (exit_in_progress.load() != 0) {
        park_until_process_exits();
    }
    // std::exit runs MetalContext's atexit handler, which destroys the cluster and tells the
    // simulator to exit. Returning from main would not: the signal interrupted it.
    std::exit(shutdown_exit_code.load());
}

void restore_default_disposition(int signal_number) {
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    ::sigemptyset(&action.sa_mask);
    (void)::sigaction(signal_number, &action, nullptr);
}

// A forked child does not own the parent's session, so teardown there would release it. Restore
// what the child had before this handler existed, using only async-signal-safe calls.
void restore_defaults_in_fork_child() {
    if (!handlers_installed) {
        return;
    }
    if (installed_sigint) {
        restore_default_disposition(SIGINT);
    }
    if (installed_sigterm) {
        restore_default_disposition(SIGTERM);
    }
    handlers_installed = false;
    shutdown_requested.store(0);
    shutdown_exit_code.store(0);
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
    if (handlers_installed || !simulator_configured()) {
        return;
    }

    const bool sigint_default = disposition_is_default(SIGINT);
    const bool sigterm_default = disposition_is_default(SIGTERM);
    // Python (and the hang apps) already turn these into a normal shutdown; replacing their handler
    // would skip that cleanup.
    if (!sigint_default && !sigterm_default) {
        return;
    }

    if (const char* grace = std::getenv("TT_METAL_SIMULATOR_TERMINATION_GRACE_SECONDS")) {
        const long parsed = std::strtol(grace, nullptr, 10);
        if (parsed > 0) {
            grace_seconds = static_cast<unsigned int>(parsed);
        }
    }

    if (::sem_init(&shutdown_semaphore, 0 /*pshared*/, 0 /*value*/) != 0) {
        return;
    }
    try {
        std::thread(run_shutdown_thread).detach();
    } catch (...) {
        return;
    }

    installed_sigint = sigint_default && install_handler(SIGINT);
    installed_sigterm = sigterm_default && install_handler(SIGTERM);
    if (!installed_sigint && !installed_sigterm) {
        return;
    }

    handlers_installed = true;
    // Registered before MetalContext's teardown, so it runs after it.
    std::atexit(report_simulator_released);
    previous_terminate = std::set_terminate(handle_terminate_during_shutdown);
    (void)::pthread_atfork(nullptr, nullptr, restore_defaults_in_fork_child);
}

void install_simulator_exit_guard() {
    if (exit_guard_installed || !simulator_configured()) {
        return;
    }
    exit_guard_installed = true;
    // Registered after MetalContext's teardown, so it runs before it.
    std::atexit(note_exit_started);
}

bool release_simulator_session(int exit_code, const char* reason) {
    if (!simulator_configured()) {
        return false;
    }
    // exit() is already releasing the simulator; exiting again from inside it is undefined.
    if (exit_in_progress.load() != 0) {
        return false;
    }

    // Arm before the diagnostics, for the same reason the signal handler does.
    const bool owns_shutdown = handlers_installed && shutdown_requested.exchange(1) == 0;
    if (owns_shutdown) {
        arm_shutdown(exit_code);
    }

    write_stderr("[tt-metal] ");
    write_stderr(reason);
    write_stderr(" Releasing the simulator.\n");

    // Without the shutdown thread there is nothing to hand over to, and this is an ordinary call
    // stack rather than a signal handler, so exiting here still runs atexit.
    if (!handlers_installed) {
        std::exit(exit_code);
    }
    if (owns_shutdown) {
        wake_shutdown_thread();
    }
    // The context is unusable, so this thread has nothing left to do but let teardown finish.
    park_until_process_exits();
}

}  // namespace tt::tt_metal
