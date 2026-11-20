// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// How a launched rank leaves a job that is stopping. Its own translation unit, independent of which context
// implementation (MPI or single-host) is in use: the flag is process-wide, and every context's waits watch it.
//
// A job stops in one of three ways, and every rank leaves the same way in all of them (docs: tt-run's README):
//   - tt-run is stopped (Ctrl-C, SIGTERM): tt-run asks the launcher for SIGTERM on every rank (PMIx job control);
//   - one rank fails (an exception, a non-zero exit, abort()): that rank asks the launcher for the same SIGTERM,
//     request_job_stop below, before it unwinds;
//   - one rank receives SIGTERM from outside the launcher (kill, scancel): the watcher thread sends it on to every
//     rank.
// On the SIGTERM a rank's waits throw, the rank unwinds and closes its devices, then leave_stopped_job waits for the
// other ranks (a PMIx fence) and exits, so no rank's exit makes the launcher kill a peer that is still closing. A rank
// that dies outright (SIGKILL, a segfault) is covered by tt-run's per-rank wrapper (tools/scaleout tt-run-rank),
// which stops the job in its place.

#include "api/tt-metalium/distributed_context.hpp"
#include "job_stop_internal.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <semaphore.h>
#include <unistd.h>

#ifdef TT_METAL_HAS_PMIX
#include <pmix.h>
#endif

namespace tt::tt_metal::distributed::multihost {

namespace {
std::atomic<bool> g_terminating{false};
std::atomic<bool> g_stop_requested{false};
std::atomic<int> g_exit_code{0};         // 0: not set; the stop's code once a rank chose one
std::atomic<pid_t> g_sigterm_sender{0};  // who sent the first SIGTERM (si_pid), read by the watcher thread
std::atomic<bool> g_watchdog_armed{false};
struct sigaction g_previous_sigterm{};
sem_t g_wake;
std::atomic<bool> g_watcher_running{false};
void (*g_force_abort)(int) = nullptr;

constexpr double kDefaultStopTimeoutS = 30.0;
constexpr double kWatchdogMarginS = 10.0;

// TT_METAL_STOP_TIMEOUT_S: the bound on a stopping rank's wait for the other ranks at its exit; the watchdog ends the
// job (MPI_Abort) if the rank is still running kWatchdogMarginS after that. 0 turns both bounds off.
double stop_timeout_s() {
    const char* v = std::getenv("TT_METAL_STOP_TIMEOUT_S");
    if (v == nullptr || *v == '\0') {
        return kDefaultStopTimeoutS;
    }
    char* end = nullptr;
    const double s = std::strtod(v, &end);
    return (end != v && s >= 0.0) ? s : kDefaultStopTimeoutS;
}

std::string rank_label() {
    const char* r = std::getenv("OMPI_COMM_WORLD_RANK");
    if (r == nullptr) {
        r = std::getenv("PMIX_RANK");
    }
    char host[256] = "?";
    ::gethostname(host, sizeof host - 1);
    std::ostringstream s;
    s << "rank " << (r != nullptr ? r : "?") << " on " << host << " (pid " << ::getpid() << ")";
    return s.str();
}

pid_t parent_of(pid_t pid) {
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!std::getline(stat, line)) {
        return 0;
    }
    const auto close_paren = line.rfind(')');  // the comm field may hold spaces; ppid is the 2nd field after it
    if (close_paren == std::string::npos) {
        return 0;
    }
    std::istringstream rest(line.substr(close_paren + 1));
    char state = 0;
    long ppid = 0;
    rest >> state >> ppid;
    return static_cast<pid_t>(ppid);
}

// The launcher's SIGTERM comes from this rank's daemon (its parent) or, under tt-run, through the per-rank wrapper
// (the parent, with the daemon above it). Anything else is a signal for this rank alone that the job must hear about.
bool from_launcher(pid_t sender) {
    pid_t p = ::getppid();
    for (int depth = 0; depth < 3 && p > 1; ++depth) {
        if (p == sender) {
            return true;
        }
        p = parent_of(p);
    }
    return false;
}

#ifdef TT_METAL_HAS_PMIX
std::once_flag g_pmix_once;
bool g_pmix_ok = false;
pmix_proc_t g_me{};

bool pmix_ready() {
    std::call_once(g_pmix_once, [] {
        if (std::getenv("PMIX_RANK") == nullptr && std::getenv("PMIX_NAMESPACE") == nullptr) {
            return;  // not started by a PMIx launcher
        }
        // MPI_Init already made this process a PMIx client; this only takes a reference and reads its name.
        g_pmix_ok = PMIx_Init(&g_me, nullptr, 0) == PMIX_SUCCESS;
    });
    return g_pmix_ok;
}
#endif

bool send_job_sigterm(std::string& why) {
#ifdef TT_METAL_HAS_PMIX
    if (!pmix_ready()) {
        why = "not started by a PMIx launcher";
        return false;
    }
    pmix_proc_t all;
    PMIX_LOAD_PROCID(&all, g_me.nspace, PMIX_RANK_WILDCARD);
    int32_t signum = SIGTERM;
    pmix_info_t directive;
    PMIX_INFO_LOAD(&directive, PMIX_JOB_CTRL_SIGNAL, &signum, PMIX_INT32);
    const pmix_status_t rc = PMIx_Job_control(&all, 1, &directive, 1, nullptr, nullptr);
    PMIX_INFO_DESTRUCT(&directive);
    if (rc != PMIX_SUCCESS) {
        why = std::string("PMIx_Job_control: ") + PMIx_Error_string(rc);
        return false;
    }
    return true;
#else
    why = "tt-metal was built without PMIx (the ULFM Open MPI install)";
    return false;
#endif
}

void arm_watchdog() {
    if (g_watchdog_armed.exchange(true)) {
        return;
    }
    const double timeout = stop_timeout_s();
    if (timeout <= 0.0) {
        return;
    }
    // Later than the exit fence's own bound (the same timeout), so a rank that closed and waited out the fence for a
    // peer that never came still leaves through the exit, not through the watchdog.
    std::thread([timeout] {
        std::this_thread::sleep_for(std::chrono::duration<double>(timeout + kWatchdogMarginS));
        const int code = g_exit_code.load() != 0 ? g_exit_code.load() : 128 + SIGTERM;
        std::fprintf(
            stderr,
            "[tt-stop] %s: still running %.0f s after the stop (TT_METAL_STOP_TIMEOUT_S); ending the job without a "
            "graceful close (exit %d)\n",
            rank_label().c_str(),
            timeout,
            code);
        std::fflush(stderr);
        if (g_force_abort != nullptr) {
            g_force_abort(code);
        }
        ::_exit(code);
    }).detach();
}

// The signal handler only records; everything that may block (PMIx, a log line) happens on this thread.
void watcher() {
    for (;;) {
        while (::sem_wait(&g_wake) != 0) {
        }
        arm_watchdog();
        if (g_stop_requested.load()) {
            continue;  // this rank already stopped the job (the SIGTERM is its own broadcast coming back)
        }
        const pid_t sender = g_sigterm_sender.load();
        if (sender > 0 && sender != ::getpid() && !from_launcher(sender)) {
            DistributedContext::request_job_stop(
                128 + SIGTERM, "received SIGTERM from pid " + std::to_string(sender) + " (not from the launcher)");
        }
    }
}

// Async-signal-safe: atomics, sem_post, then the handler that was installed before ours.
extern "C" void on_sigterm(int signal_number, siginfo_t* info, void* context) {
    g_terminating.store(true, std::memory_order_relaxed);
    pid_t none = 0;
    g_sigterm_sender.compare_exchange_strong(none, info != nullptr ? info->si_pid : -1);
    if (g_watcher_running.load(std::memory_order_relaxed)) {
        ::sem_post(&g_wake);
    }
    if ((g_previous_sigterm.sa_flags & SA_SIGINFO) != 0) {
        if (g_previous_sigterm.sa_sigaction != nullptr && g_previous_sigterm.sa_sigaction != on_sigterm) {
            g_previous_sigterm.sa_sigaction(signal_number, info, context);
        }
        return;
    }
    // The default disposition is not re-raised: the flag makes the rank's waits throw, and it leaves after closing.
    if (g_previous_sigterm.sa_handler != SIG_DFL && g_previous_sigterm.sa_handler != SIG_IGN) {
        g_previous_sigterm.sa_handler(signal_number);
    }
}
}  // namespace

namespace job_stop {
void set_force_abort(void (*force_abort)(int)) { g_force_abort = force_abort; }

bool launcher_reachable() {
#ifdef TT_METAL_HAS_PMIX
    return pmix_ready();
#else
    return false;
#endif
}
}  // namespace job_stop

void DistributedContext::mark_terminating() { g_terminating.store(true, std::memory_order_relaxed); }
bool DistributedContext::is_terminating() { return g_terminating.load(std::memory_order_relaxed); }

void DistributedContext::install_termination_handler() {
    if (std::getenv("TT_METAL_SIMULATOR") != nullptr) {
        return;  // under a simulator, simulator_termination.cpp owns SIGTERM and exits through atexit
    }
    struct sigaction current{};
    if (::sigaction(SIGTERM, nullptr, &current) == 0 && (current.sa_flags & SA_SIGINFO) != 0 &&
        current.sa_sigaction == on_sigterm) {
        return;  // ours is the one installed
    }
    if (!g_watcher_running.exchange(true)) {
        ::sem_init(&g_wake, 0, 0);
        std::thread(watcher).detach();
    }
    struct sigaction action{};
    action.sa_sigaction = on_sigterm;
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    ::sigaction(SIGTERM, &action, &g_previous_sigterm);  // chains to whatever was there (Python's handler)
}

bool DistributedContext::request_job_stop(int exit_code, const std::string& reason) {
    if (g_stop_requested.exchange(true)) {
        return true;  // the first reason wins; the job is already stopping
    }
    int unset = 0;
    g_exit_code.compare_exchange_strong(unset, exit_code);
    g_terminating.store(true);
    std::fprintf(
        stderr,
        "[tt-stop] %s: %s; stopping the job (exit %d)\n",
        rank_label().c_str(),
        reason.c_str(),
        g_exit_code.load());
    std::string why;
    const bool sent = send_job_sigterm(why);
    if (!sent) {
        std::fprintf(
            stderr, "[tt-stop] %s: could not send SIGTERM to the other ranks: %s\n", rank_label().c_str(), why.c_str());
    }
    std::fflush(stderr);
    arm_watchdog();
    return sent;
}

void DistributedContext::leave_stopped_job() {
    int code = g_exit_code.load();
    if (code == 0) {
        code = 128 + SIGTERM;  // this rank was told to stop
    }
    std::fflush(nullptr);
#ifdef TT_METAL_HAS_PMIX
    if (pmix_ready()) {
        pmix_proc_t all;
        PMIX_LOAD_PROCID(&all, g_me.nspace, PMIX_RANK_WILDCARD);
        const double timeout = stop_timeout_s();
        pmix_info_t bound;
        int seconds = static_cast<int>(timeout);
        PMIX_INFO_LOAD(&bound, PMIX_TIMEOUT, &seconds, PMIX_INT);
        const pmix_status_t rc = PMIx_Fence(&all, 1, timeout > 0.0 ? &bound : nullptr, timeout > 0.0 ? 1 : 0);
        PMIX_INFO_DESTRUCT(&bound);
        if (rc != PMIX_SUCCESS) {
            std::fprintf(
                stderr,
                "[tt-stop] %s: not every rank reached the exit (%s); leaving anyway\n",
                rank_label().c_str(),
                PMIx_Error_string(rc));
        }
    }
#endif
    // Under tt-run the wrapper that started this rank (tt-run-rank) is told the stop was handled here, so it does not
    // stop the job again in this rank's place.
    if (const char* fd = std::getenv("TT_RUN_RANK_FD"); fd != nullptr && *fd != '\0') {
        static const char kStopped[] = "stopped\n";
        (void)!::write(std::atoi(fd), kStopped, sizeof kStopped - 1);
    }
    std::fflush(nullptr);
    ::_exit(code);
}

}  // namespace tt::tt_metal::distributed::multihost
