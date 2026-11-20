// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// tt-run-rank: the wrapper tt-run starts each rank in, so a rank that dies still stops the job gracefully.
//
//   tt-run-rank -- <program> [args...]
//
// It runs the program as its child and passes it every signal the launcher sends. When the program ends:
//   - with 0, or after it stopped the job itself (tt-metal's context writes "stopped" to TT_RUN_RANK_FD), or after a
//     stop signal from the launcher it did not crash on: exit with the program's code;
//   - otherwise (a segfault, a SIGKILL, an exit with an error from a program that did not stop the job): print one
//     "[tt-stop] rank R on HOST (pid P): <how it ended>; stopping the job (exit CODE)" line, which tt-run reads to
//     name the rank and return CODE, then send SIGTERM to every rank through the launcher (PMIx job control). When
//     the program was a tt-metal rank (its context wrote "context"), stay up in its place until the launcher ends the
//     job once the other ranks have closed and exited (at most TT_METAL_STOP_TIMEOUT_S), so this exit does not make
//     the launcher kill them mid-close.
// A program killed by a signal is reported with exit 128 + the signal.

#include <pmix.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int kForwarded[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2, SIGCONT};
volatile pid_t g_child = -1;
volatile sig_atomic_t g_stop_signal = 0;  // a stop signal (SIGTERM, SIGINT, SIGHUP) the launcher sent this rank

void forward(int signal_number) {
    if (signal_number == SIGTERM || signal_number == SIGINT || signal_number == SIGHUP) {
        g_stop_signal = signal_number;
    }
    if (g_child > 0) {
        ::kill(g_child, signal_number);
    }
}

// The same bound as the ranks' exit wait (DistributedContext): TT_METAL_STOP_TIMEOUT_S, else tt-run's stop grace
// (TT_RUN_STOP_GRACE_S, 120 s by default) plus 10 s.
double stop_timeout_s() {
    const char* v = std::getenv("TT_METAL_STOP_TIMEOUT_S");
    if (v != nullptr && *v != '\0') {
        return std::atof(v);
    }
    const char* grace = std::getenv("TT_RUN_STOP_GRACE_S");
    return ((grace != nullptr && *grace != '\0') ? std::atof(grace) : 120.0) + 10.0;
}

// Whether the process is writing a core dump right now (/proc/<pid>/status "CoreDumping: 1").
bool core_dumping(pid_t pid) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/status", static_cast<int>(pid));
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) {
        return false;
    }
    char line[256];
    bool dumping = false;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        if (std::strncmp(line, "CoreDumping:", 12) == 0) {
            dumping = std::atoi(line + 12) == 1;
            break;
        }
    }
    std::fclose(f);
    return dumping;
}

}  // namespace

int main(int argc, char** argv) {
    int first = 1;
    if (first < argc && std::strcmp(argv[first], "--") == 0) {
        ++first;
    }
    if (first >= argc) {
        std::fprintf(stderr, "usage: %s -- <program> [args...]\n", argv[0]);
        return 2;
    }

    int status_pipe[2] = {-1, -1};
    if (::pipe2(status_pipe, O_CLOEXEC) != 0) {
        status_pipe[0] = status_pipe[1] = -1;
    }
    struct sigaction action{};
    action.sa_handler = forward;
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    for (const int s : kForwarded) {
        ::sigaction(s, &action, nullptr);
    }

    const pid_t wrapper = ::getpid();
    const pid_t child = ::fork();
    if (child < 0) {
        std::perror("tt-run-rank: fork");
        return 127;
    }
    if (child == 0) {
        for (const int s : kForwarded) {
            ::signal(s, SIG_DFL);
        }
        // The launcher kills the process it started, this wrapper, not its child: without this a program the
        // launcher's SIGKILL reached only through the wrapper would outlive the job.
        ::prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (::getppid() != wrapper) {
            ::_exit(128 + SIGKILL);  // the wrapper is already gone
        }
        if (status_pipe[1] >= 0) {
            const int fd = ::dup(status_pipe[1]);  // without O_CLOEXEC, so the program inherits it
            ::setenv("TT_RUN_RANK_FD", std::to_string(fd).c_str(), 1);
        }
        ::execvp(argv[first], argv + first);
        std::fprintf(stderr, "tt-run-rank: cannot run %s: %s\n", argv[first], std::strerror(errno));
        ::_exit(127);
    }
    g_child = child;
    if (status_pipe[1] >= 0) {
        ::close(status_pipe[1]);
    }

    // Wait for the program. If the job is stopping while the program is still writing a core dump (a crash that is
    // not over yet), end it: a core dump never holds the job (teardown plan P8); the core is cut short.
    int status = 0;
    bool dump_ended = false;
    for (;;) {
        const pid_t done = ::waitpid(child, &status, WNOHANG);
        if (done == child || (done < 0 && errno != EINTR)) {
            break;
        }
        if (g_stop_signal != 0 && !dump_ended && core_dumping(child)) {
            std::fprintf(
                stderr,
                "[tt-run-rank] rank %s was still writing a core dump when the job stopped; ending it (core cut short)\n",
                std::getenv("OMPI_COMM_WORLD_RANK") != nullptr ? std::getenv("OMPI_COMM_WORLD_RANK") : "?");
            ::kill(child, SIGKILL);
            dump_ended = true;
        }
        ::usleep(100000);
    }
    const bool signalled = WIFSIGNALED(status);
    const int code = signalled ? 128 + WTERMSIG(status) : WEXITSTATUS(status);

    char note[256] = {};
    if (status_pipe[0] >= 0) {
        ::fcntl(status_pipe[0], F_SETFL, O_NONBLOCK);  // a helper the program forked may still hold the write end
        (void)!::read(status_pipe[0], note, sizeof note - 1);
    }
    const bool stopped_by_program = std::strstr(note, "stopped") != nullptr;
    const bool context_rank = std::strstr(note, "context") != nullptr;  // tt-metal's context was up in the program
    // A program killed by a signal other than a stop signal died on its own (a SIGKILL, a segfault), even when a
    // stop reached this wrapper first: across hosts a peer's MPI can see the death and stop the job before this
    // wrapper does, and the job must still name this rank.
    const bool crashed = signalled && WTERMSIG(status) != SIGTERM && WTERMSIG(status) != SIGINT &&
                         WTERMSIG(status) != SIGHUP;
    if (code == 0 || stopped_by_program || (g_stop_signal != 0 && !crashed)) {
        return code;
    }

    // The program died, or failed without stopping the job: stop it in its place.
    ::signal(SIGTERM, SIG_IGN);  // our own broadcast comes back to this rank
    const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
    char host[256] = "?";
    ::gethostname(host, sizeof host - 1);
    std::string how;
    if (signalled) {
        const char* name = sigabbrev_np(WTERMSIG(status));
        how = "was killed by signal " + std::to_string(WTERMSIG(status)) + " (SIG" + (name ? name : "?") + ")";
    } else {
        how = "exited with code " + std::to_string(code) + " without stopping the job";
    }
    std::fprintf(
        stderr,
        "[tt-stop] rank %s on %s (pid %d): %s; stopping the job (exit %d)\n",
        rank != nullptr ? rank : "?",
        host,
        static_cast<int>(child),
        how.c_str(),
        code);
    std::fflush(stderr);

    pmix_proc_t me;
    pmix_status_t rc = PMIx_Init(&me, nullptr, 0);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(
            stderr, "[tt-stop] tt-run-rank: PMIx_Init: %s; the launcher ends the job\n", PMIx_Error_string(rc));
        return code;
    }
    pmix_proc_t all;
    PMIX_LOAD_PROCID(&all, me.nspace, PMIX_RANK_WILDCARD);
    int32_t signum = SIGTERM;
    pmix_info_t directive;
    PMIX_INFO_LOAD(&directive, PMIX_JOB_CTRL_SIGNAL, &signum, PMIX_INT32);
    rc = PMIx_Job_control(&all, 1, &directive, 1, nullptr, nullptr);
    PMIX_INFO_DESTRUCT(&directive);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(stderr, "[tt-stop] tt-run-rank: PMIx_Job_control: %s\n", PMIx_Error_string(rc));
        return code;
    }
    if (!context_rank) {
        return code;  // the other ranks are not tt-metal ranks waiting at an exit fence: the launcher ends the job
    }
    // Stand in for the dead program at the other ranks' exit fence (DistributedContext::leave_stopped_job), so they
    // leave as soon as they have closed, then stay up: this rank's exit would make the launcher kill a rank that is
    // still closing. When they have exited (non-zero, so the launcher ends the job) the launcher ends this wrapper
    // too; otherwise it leaves after TT_METAL_STOP_TIMEOUT_S. The fence alone cannot hold it: with this rank's own
    // client gone it may return at once.
    const double timeout = stop_timeout_s();
    int seconds = static_cast<int>(timeout);
    pmix_info_t bound;
    PMIX_INFO_LOAD(&bound, PMIX_TIMEOUT, &seconds, PMIX_INT);
    (void)PMIx_Fence(&all, 1, timeout > 0.0 ? &bound : nullptr, timeout > 0.0 ? 1 : 0);
    PMIX_INFO_DESTRUCT(&bound);
    for (double waited = 0.0; timeout <= 0.0 || waited < timeout; waited += 0.1) {
        ::usleep(100000);
    }
    return code;
}
