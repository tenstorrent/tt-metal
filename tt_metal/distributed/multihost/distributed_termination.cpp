// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// How a launched rank leaves a job: its own clean-up and nothing more. A rank never stops the job and never talks to
// the launcher; that is tt-run-rank's, the wrapper every rank runs in (tools/scaleout/src/tt_run_rank.cpp). The rank
// sees two signals and two pipes:
//   - SIGTERM: the job is stopping. The handler sets a flag; every context wait polls it and throws. A Python rank's
//     handler goes to teardown(), which for now says so and exits 128 + SIGTERM without MPI_Finalize (leave_job).
//   - "failed <code> <reason>" on TT_RUN_RANK_FD: what a rank that fails writes before it closes (fail_rank). The
//     wrapper reports it at once and stops the job.
//   - SIGUSR1 with "exit <rank> <code> <signal> <reason>" lines on TT_RUN_RANK_EVENTS_FD: another rank exited and the
//     wrapper runs in --on-peer-exit notify mode, so the job goes on. The handler sets a flag; every context wait
//     throws PeerExitError, and the application decides (peer_exits() lists them).
// Without a wrapper (a bare mpirun line) a failing rank falls back to the launcher's own abort.

#include "api/tt-metalium/distributed_context.hpp"
#include "job_stop_internal.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace tt::tt_metal::distributed::multihost {

namespace {
std::atomic<bool> g_terminating{false};
std::atomic<bool> g_stop_signal{false};  // set by the SIGTERM handler only
std::atomic<bool> g_failed{false};
std::atomic<int> g_exit_code{0};  // 0: not set; this rank's own code once it failed
std::string g_reason;             // this rank's own failure, the first one; read by teardown
std::atomic<bool> g_peer_exited{false};
struct sigaction g_previous_sigterm{};
struct sigaction g_previous_sigusr1{};
std::mutex g_events_mu;
std::vector<DistributedContext::PeerExit> g_events;
std::string g_events_rest;

int env_fd(const char* name) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : -1;
}

// One line: a TT_FATAL's message spans several and ends in a backtrace, so the reason is cut at the backtrace, its
// line breaks become spaces, and it is capped. The wrapper reads the pipe one line at a time.
std::string one_line(const std::string& reason) {
    std::string out = reason.substr(0, reason.find("backtrace:"));
    std::string squeezed;
    for (char c : out) {
        if (c == '\n' || c == '\r' || c == '\t') {
            c = ' ';
        }
        if (!(c == ' ' && (squeezed.empty() || squeezed.back() == ' '))) {
            squeezed += c;
        }
    }
    while (!squeezed.empty() && squeezed.back() == ' ') {
        squeezed.pop_back();
    }
    return squeezed.size() > 400 ? squeezed.substr(0, 400) + "..." : squeezed;
}

// Async-signal-safe: one atomic, then the handler that was installed before ours (Python's).
void chain(const struct sigaction& previous, int signal_number, siginfo_t* info, void* context) {
    if ((previous.sa_flags & SA_SIGINFO) != 0) {
        if (previous.sa_sigaction != nullptr) {
            previous.sa_sigaction(signal_number, info, context);
        }
    } else if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN) {
        previous.sa_handler(signal_number);
    }
    // The default disposition is not re-raised: the flag makes the rank's waits throw, and it leaves after closing.
}

extern "C" void on_sigterm(int signal_number, siginfo_t* info, void* context) {
    g_terminating.store(true, std::memory_order_relaxed);
    g_stop_signal.store(true, std::memory_order_relaxed);
    chain(g_previous_sigterm, signal_number, info, context);
}

extern "C" void on_sigusr1(int signal_number, siginfo_t* info, void* context) {
    g_peer_exited.store(true, std::memory_order_relaxed);
    chain(g_previous_sigusr1, signal_number, info, context);
}

void install(int signal_number, void (*handler)(int, siginfo_t*, void*), struct sigaction& previous) {
    struct sigaction current{};
    if (::sigaction(signal_number, nullptr, &current) == 0 && (current.sa_flags & SA_SIGINFO) != 0 &&
        current.sa_sigaction == handler) {
        return;  // ours is the one installed
    }
    struct sigaction action{};
    action.sa_sigaction = handler;
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    ::sigaction(signal_number, &action, &previous);
}
}  // namespace

namespace job_stop {
bool in_rank_wrapper() { return env_fd("TT_RUN_RANK_FD") >= 0; }
bool stop_signal_seen() { return g_stop_signal.load(std::memory_order_relaxed); }
}  // namespace job_stop

void DistributedContext::mark_terminating() { g_terminating.store(true, std::memory_order_relaxed); }
bool DistributedContext::is_terminating() { return g_terminating.load(std::memory_order_relaxed); }
bool DistributedContext::peer_exited() { return g_peer_exited.load(std::memory_order_relaxed); }

void DistributedContext::install_termination_handler() {
    if (std::getenv("TT_METAL_SIMULATOR") != nullptr) {
        return;  // under a simulator, simulator_termination.cpp owns SIGTERM and exits through atexit
    }
    install(SIGTERM, on_sigterm, g_previous_sigterm);
    if (env_fd("TT_RUN_RANK_EVENTS_FD") >= 0) {
        install(SIGUSR1, on_sigusr1, g_previous_sigusr1);
    }
}

bool DistributedContext::fail_rank(int exit_code, const std::string& reason) {
    if (g_failed.exchange(true)) {
        return false;  // the first reason wins
    }
    const int code = exit_code != 0 ? exit_code : 1;
    int unset = 0;
    g_exit_code.compare_exchange_strong(unset, code);
    g_reason = one_line(reason);
    g_terminating.store(true);
    if (const int fd = env_fd("TT_RUN_RANK_FD"); fd >= 0) {
        const std::string line = "failed " + std::to_string(code) + " " + one_line(reason) + "\n";
        (void)!::write(fd, line.data(), line.size());
    }
    return true;
}

void DistributedContext::teardown(const std::string& reason) {
    g_terminating.store(true);
    std::string who = "rank";
    if (is_initialized()) {
        who += " " + std::to_string(*get_current_world()->rank());
    }
    const int code = g_exit_code.load() != 0 ? g_exit_code.load() : 128 + SIGTERM;
    const std::string& why = g_reason.empty() ? reason : g_reason;  // a rank that failed keeps its first reason
    std::fprintf(stderr, "[%s] teardown initiated: %s; exiting with code %d\n", who.c_str(), why.c_str(), code);
    leave_job();
}

void DistributedContext::leave_job() {
    std::fflush(nullptr);
    const int code = g_exit_code.load();
    ::_exit(code != 0 ? code : 128 + SIGTERM);
}

std::vector<DistributedContext::PeerExit> DistributedContext::peer_exits() {
    std::lock_guard<std::mutex> lock(g_events_mu);
    const int fd = env_fd("TT_RUN_RANK_EVENTS_FD");
    if (fd >= 0) {
        ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);
        char chunk[512];
        for (ssize_t n = 0; (n = ::read(fd, chunk, sizeof chunk)) > 0;) {
            g_events_rest.append(chunk, static_cast<size_t>(n));
        }
        for (size_t nl = 0; (nl = g_events_rest.find('\n')) != std::string::npos; g_events_rest.erase(0, nl + 1)) {
            std::istringstream line(g_events_rest.substr(0, nl));
            std::string word;
            PeerExit e{};
            if (line >> word >> e.rank >> e.exit_code >> e.signal && word == "exit") {
                std::getline(line >> std::ws, e.reason);
                g_events.push_back(e);
            }
        }
    }
    return g_events;
}

std::string DistributedContext::PeerExit::describe() const {
    std::ostringstream s;
    s << "rank " << rank << " exited with code " << exit_code;
    if (signal != 0) {
        s << " (signal " << signal << ")";
    }
    if (!reason.empty()) {
        s << ": " << reason;
    }
    return s.str();
}

}  // namespace tt::tt_metal::distributed::multihost
