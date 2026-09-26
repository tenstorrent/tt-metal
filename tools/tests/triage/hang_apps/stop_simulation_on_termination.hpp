// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <execinfo.h>
#include <semaphore.h>
#include <thread>
#include <unistd.h>

namespace tt::tt_metal::triage_hang_apps {

namespace detail {

// Everything below runs from a signal handler, so it uses write(2) rather than printf and keeps to
// async-signal-safe calls.
inline void trace(const char* message) {
    const char* prefix = "[hang-app] ";
    (void)::write(STDERR_FILENO, prefix, std::strlen(prefix));
    (void)::write(STDERR_FILENO, message, std::strlen(message));
    (void)::write(STDERR_FILENO, "\n", 1);
}

inline void trace_with_number(const char* message, long number) {
    // No snprintf here: format the number by hand so this stays usable from a handler.
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

    const char* prefix = "[hang-app] ";
    (void)::write(STDERR_FILENO, prefix, std::strlen(prefix));
    (void)::write(STDERR_FILENO, message, std::strlen(message));
    (void)::write(STDERR_FILENO, cursor, static_cast<size_t>(end - cursor));
    (void)::write(STDERR_FILENO, "\n", 1);
}

inline void print_backtrace() {
    void* frames[64];
    const int count = ::backtrace(frames, 64);
    // backtrace_symbols_fd does not allocate, unlike backtrace_symbols.
    ::backtrace_symbols_fd(frames, count, STDERR_FILENO);
}

// Set once, at install time, so the handlers below only read them.
inline unsigned int grace_seconds = 15;
inline bool hold_for_debugger = false;
// Written by the handler, read from ordinary code: sig_atomic_t is the only type that is safe both ways.
inline volatile std::sig_atomic_t termination_signal = 0;
// Posted by the handler, waited on by the shutdown thread. sem_post is async-signal-safe; this is
// the only way the handler is allowed to wake another thread.
inline sem_t shutdown_semaphore;

}  // namespace detail

extern "C" inline void handle_teardown_alarm(int /*signal_number*/) {
    detail::trace_with_number("teardown did not finish within seconds: ", detail::grace_seconds);
    detail::trace("the simulator job is still running and will be orphaned.");
    detail::trace("stack of the thread that took the alarm (may not be the thread that is stuck):");
    detail::print_backtrace();

    if (detail::hold_for_debugger) {
        detail::trace_with_number("holding for a debugger, pid ", ::getpid());
        detail::trace("attach with: gdb -p <pid> -ex 'thread apply all bt' -batch");
        for (;;) {
            ::pause();
        }
    }
    ::_exit(128 + detail::termination_signal);
}

// Teardown runs here rather than in the handler, and the reason is the whole point of this file.
//
// The handler runs on whichever thread took the signal, which is normally the main thread -- and the
// main thread is inside a UMD read, holding SimulationChip::device_lock (simulation_chip.cpp:79) and
// waiting on the simulator's reply. Exiting from there runs tt-metal's teardown on that same thread,
// and the first thing it does is assert the cores into reset, which takes that very same
// non-recursive mutex (simulation_chip.cpp:90). The thread deadlocks against the lock it is already
// holding, teardown never reaches UMD, and the simulation is orphaned anyway.
//
// So the handler only posts, and returns. The interrupted thread resumes its read, releases
// device_lock, and carries on polling; this thread meanwhile exits the process, and teardown can take
// the lock between two of those polls.
inline void run_shutdown_thread() {
    while (::sem_wait(&detail::shutdown_semaphore) != 0) {
        // Interrupted by a signal; keep waiting.
    }
    detail::trace("running teardown off the interrupted thread.");
    std::exit(128 + detail::termination_signal);
}

extern "C" inline void handle_termination_signal(int signal_number) {
    detail::termination_signal = signal_number;
    detail::trace_with_number("received signal ", signal_number);
    detail::trace("handing over to the shutdown thread so the simulator is released.");

    // A second signal kills us outright: whoever is killing us has stopped waiting.
    std::signal(signal_number, SIG_DFL);
    // Teardown still has to take locks the rest of the process holds, so bound it: a wedged teardown
    // costs a delay rather than a hang.
    std::signal(SIGALRM, handle_teardown_alarm);
    alarm(detail::grace_seconds);

    // Wake the shutdown thread and RETURN, so this thread can release whatever UMD lock it holds.
    (void)::sem_post(&detail::shutdown_semaphore);
}

// Whether a termination signal has been seen and teardown is on its way.
inline bool termination_requested() { return detail::termination_signal != 0; }

// Wait here until the shutdown thread exits the process.
//
// Teardown closes the link to the simulator, so every device call still in flight fails -- the
// polling thread gets "Failed to receive response from device" the moment the notification thread
// stops (rtl_sim_communicator.cpp:170). Letting that failure propagate ends the process through
// std::terminate and abort, which kills teardown before it sends DEVICE_COMMAND_EXIT. So a thread
// that fails while terminating parks here instead, and teardown finishes.
[[noreturn]] inline void park_until_process_exits() {
    detail::trace("a device call failed while shutting down, as expected; waiting for teardown.");
    for (;;) {
        ::pause();
    }
}

// Backstop for a failure that nothing catches: without this, terminate calls abort and teardown dies
// with it.
inline void handle_terminate_during_shutdown() {
    if (termination_requested()) {
        park_until_process_exits();
    }
    std::abort();
}

// Runs last of all the exit handlers, because it is registered before any of tt-metal's: atexit
// handlers and static destructors share one LIFO queue. Seeing this line means teardown got all the
// way through; not seeing it means it stopped somewhere in tt-metal or UMD.
inline void report_teardown_finished() {
    if (detail::termination_signal != 0) {
        detail::trace("teardown finished; the simulator has been released.");
    }
}

// Make SIGINT and SIGTERM stop the simulation instead of orphaning it.
//
// An app running against an RTL simulator or emulator is what started it: UMD spawns it as a
// detached child and then forgets it, logging the pid and throwing it away
// (umd/device/simulation/rtl_sim_communicator.cpp:94-122). The one thing that stops that job is the
// DEVICE_COMMAND_EXIT UMD sends from ~RtlSimulationTTDevice, so an app that *dies* from a signal
// runs no destructors and leaves the simulation running until its own timeout -- for the Quasar
// emulator, holding a shared Zebu job for minutes after the app is gone. These apps exist to be
// killed while hung, so they are exactly the ones that have to exit rather than die.
//
// Only a real exit helps: the teardown that reaches UMD is the atexit handler MetalContext registers
// (metal_context.cpp:500-507).
//
// Environment:
//   TT_TRIAGE_HANG_APP_TERMINATION_GRACE_SECONDS  how long teardown gets before the alarm (default 15)
//   TT_TRIAGE_HANG_APP_HOLD_FOR_DEBUGGER=1        on alarm, stay alive so gdb can attach, instead of exiting
inline void stop_simulation_on_termination() {
    // Only when a simulator is in play. On silicon the device outlives the process, so there is
    // nothing to release and dying promptly is the better behaviour.
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        return;
    }

    if (const char* grace = std::getenv("TT_TRIAGE_HANG_APP_TERMINATION_GRACE_SECONDS")) {
        const long parsed = std::strtol(grace, nullptr, 10);
        if (parsed > 0) {
            detail::grace_seconds = static_cast<unsigned int>(parsed);
        }
    }
    const char* hold = std::getenv("TT_TRIAGE_HANG_APP_HOLD_FOR_DEBUGGER");
    detail::hold_for_debugger = hold != nullptr && std::strcmp(hold, "1") == 0;

    // backtrace() can allocate the first time it runs, which a signal handler must not do. Calling
    // it once here loads libgcc's unwinder up front so the handler's call does not have to.
    void* warmup[4];
    (void)::backtrace(warmup, 4);

    // Registered before tt-metal registers any of its own, so it runs after all of them.
    std::atexit(report_teardown_finished);
    std::set_terminate(handle_terminate_during_shutdown);

    (void)::sem_init(&detail::shutdown_semaphore, 0, 0);
    std::thread(run_shutdown_thread).detach();

    std::signal(SIGINT, handle_termination_signal);
    std::signal(SIGTERM, handle_termination_signal);
}

}  // namespace tt::tt_metal::triage_hang_apps
