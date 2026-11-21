// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mpi_probe: a fault-injection rank program that mirrors how tt-metal uses MPI
// (MPI_Init_thread(MULTIPLE), atexit(MPI_Finalize), MPI_Comm_dup(WORLD) with
// MPI_ERRORS_RETURN, blocking collectives on the dup) so the behaviour of
// OpenMPI/PRRTE on each kind of rank failure can be measured without tt-metal.
//
// Every rank prints timestamped markers to stdout so the runner can tell which
// teardown step each rank reached: barrier entered, barrier rc, finalize
// entered, finalize done.
//
// Env:
//   PROBE_SCENARIO        clean | victim_abort | victim_exit1 | victim_exit0_nofinalize
//                         | victim_exit1_nofinalize | victim_sigkill | victim_sigterm
//                         | victim_hang | victim_mpi_abort | victim_late | all_hang
//   PROBE_VICTIM          rank that misbehaves (default 1)
//   PROBE_DELAY_MS        delay before the victim misbehaves (default 300)
//   PROBE_SURVIVOR_ACTION finalize (default; what tt-metal does today: unwind to exit)
//                         | revoke_shrink | abort | exit_nofinalize
//   PROBE_SECOND_BARRIER  1 -> survivors try a second collective after an error
//   PROBE_PIDDIR          directory to write rank_<r>.pid into (external per-rank signals)
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <unistd.h>  // getpid / getppid / _exit: POSIX, no C++ header equivalent

static int g_rank = -1;
static double g_t0;

// Set from a SIGINT/SIGTERM handler that is installed ONLY for the all_hang scenario (other scenarios must still
// die on the default disposition). It lets a spinning rank show it RECEIVED the forwarded stop — the launcher
// relayed it to every rank, the ones on remote hosts included — and exit gracefully, instead of being killed.
static volatile sig_atomic_t g_caught_signal = 0;
static void on_terminate_signal(int signum) { g_caught_signal = signum; }

// all_hang only: the signals tt-run passes on to the ranks unchanged (ttrun.py PASSED_SIGNALS, SIGTSTP aside, which
// keeps its default and pauses the rank). Each one is recorded and reported, and the rank goes on.
static constexpr int kPassedSignals[] = {SIGQUIT, SIGUSR1, SIGUSR2, SIGALRM, SIGCONT, SIGWINCH};
static volatile sig_atomic_t g_passed_signal[65] = {};
static void on_passed_signal(int signum) { g_passed_signal[signum] = 1; }

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A printf-style marker line. A variadic template (not a C-style variadic function) keeps the type-safe call
// while satisfying clang-tidy; the no-argument case copies the literal so a non-literal format with no args
// never reaches snprintf (-Wformat-security).
template <typename... Args>
static void mark(const char* fmt, Args... args) {
    char buf[512];
    if constexpr (sizeof...(Args) == 0) {
        std::snprintf(buf, sizeof buf, "%s", fmt);
    } else {
        std::snprintf(buf, sizeof buf, fmt, args...);
    }
    std::printf("[probe r=%d pid=%d t=%.3f] %s\n", g_rank, static_cast<int>(getpid()), now_s() - g_t0, buf);
    std::fflush(stdout);
}

static const char* errstr(int rc) {
    static char s[MPI_MAX_ERROR_STRING + 1];
    int len = 0;
    if (rc == MPI_SUCCESS) {
        return "MPI_SUCCESS";
    }
    if (MPI_Error_string(rc, s, &len) != MPI_SUCCESS) {
        snprintf(s, sizeof s, "rc=%d", rc);
    }
    return s;
}

// Same as tt-metal's MPIContext::init_env: finalize from atexit, unconditionally.
static void finalize_at_exit(void) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        return;
    }
    double t = now_s();
    mark("finalize_enter");
    int rc = MPI_Finalize();
    mark("finalize_done rc=%s dt=%.3f", errstr(rc), now_s() - t);
}

static void sleep_ms(long ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

static const char* env_or(const char* k, const char* d) {
    const char* v = getenv(k);
    return (v && *v) ? v : d;
}

int main(int argc, char** argv) {
    g_t0 = now_s();
    const char* scenario = env_or("PROBE_SCENARIO", "clean");
    int victim = atoi(env_or("PROBE_VICTIM", "1"));
    long delay_ms = atol(env_or("PROBE_DELAY_MS", "300"));
    const char* survivor_action = env_or("PROBE_SURVIVOR_ACTION", "finalize");
    int second_barrier = atoi(env_or("PROBE_SECOND_BARRIER", "0"));

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    atexit(finalize_at_exit);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

#if defined(MPIX_ERR_PROC_FAILED)
    const int have_ulfm = 1;
#else
    const int have_ulfm = 0;
#endif
    char host[64] = "?";
    gethostname(host, sizeof host - 1);
    mark(
        "start scenario=%s size=%d victim=%d ulfm_compiled=%d ppid=%d pgid=%d sid=%d host=%s",
        scenario,
        size,
        victim,
        have_ulfm,
        (int)getppid(),
        (int)getpgrp(),
        (int)getsid(0),
        host);
    // PROBE_PIDDIR: write rank_<r>.pid so a runner can send a signal to ONE rank from outside (rank_signal cases)
    const char* piddir = getenv("PROBE_PIDDIR");
    if (piddir && *piddir) {
        char path[512];
        snprintf(path, sizeof path, "%s/rank_%d.pid", piddir, g_rank);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "%d\n", (int)getpid());
            fclose(f);
        }
        // readiness marker: this rank is past MPI_Init and about to hang/barrier, so the parent harness can gate a
        // launcher signal on the ranks actually being up, instead of guessing a wall-clock delay (review #9).
        snprintf(path, sizeof path, "%s/rank_%d.ready", piddir, g_rank);
        FILE* rf = fopen(path, "w");
        if (rf) {
            fclose(rf);
        }
    }

    const int is_victim = (g_rank == victim);
    if (!strcmp(scenario, "all_hang")) {
        // Catch the stop ONLY in this scenario, so a rank can prove it RECEIVED the signal (and which one) rather
        // than being killed outright. Every other scenario keeps the default disposition (SIGTERM ends the rank).
        // A stop at tt-run reaches every rank as SIGTERM through PRRTE job control; SIGINT is caught too for a
        // signal delivered straight to the rank.
        std::signal(SIGINT, on_terminate_signal);
        std::signal(SIGTERM, on_terminate_signal);
        for (const int s : kPassedSignals) {
            std::signal(s, on_passed_signal);
        }
        // Poll on a short granularity: sleep_for does not return early on a signal, so a long sleep would let the
        // launcher escalate to SIGKILL before this loop noticed the flag. 100 ms keeps the catch well inside the
        // launcher's SIGTERM->SIGKILL grace. Heartbeat stays ~5 s (every 50 ticks).
        for (int i = 0;; i++) {
            if (g_caught_signal) {
                mark("caught signal %d; exiting gracefully", (int)g_caught_signal);
                mark("main_exit");
                return 0;  // atexit -> MPI_Finalize: a graceful exit, not a kill
            }
            for (const int s : kPassedSignals) {
                if (g_passed_signal[s]) {
                    g_passed_signal[s] = 0;
                    mark("got signal %d; going on", s);
                }
            }
            if (i % 50 == 0) {
                mark("all_hang heartbeat");
            }
            sleep_ms(100);
        }
    }
    if (is_victim && strcmp(scenario, "clean") != 0) {
        sleep_ms(delay_ms);
        mark("victim acting: %s", scenario);
        if (!strcmp(scenario, "victim_abort")) {
            abort();
        }
        if (!strcmp(scenario, "victim_exit1")) {
            exit(1);
        }
        if (!strcmp(scenario, "victim_exit0_nofinalize")) {
            _exit(0);
        }
        if (!strcmp(scenario, "victim_exit1_nofinalize")) {
            _exit(1);
        }
        if (!strcmp(scenario, "victim_sigkill")) {
            raise(SIGKILL);
        }
        if (!strcmp(scenario, "victim_sigterm")) {
            raise(SIGTERM);
        }
        if (!strcmp(scenario, "victim_mpi_abort")) {
            MPI_Abort(MPI_COMM_WORLD, 7);
        }
        if (!strcmp(scenario, "victim_hang")) {
            for (int i = 0;; i++) {
                if (i % 5 == 0) {
                    mark("victim_hang heartbeat");
                }
                sleep_ms(1000);
            }
        }
        if (!strcmp(scenario, "victim_late")) {
            sleep_ms(2000);
        }
        // unknown scenario or victim_late: fall through to the barrier.
    }

    double t = now_s();
    mark("barrier_enter");
    int rc = MPI_Barrier(comm);
    mark("barrier_rc=%s dt=%.3f", errstr(rc), now_s() - t);

    if (rc != MPI_SUCCESS) {
        if (!strcmp(survivor_action, "abort")) {
            // KNOWN LAUNCHER BUG: when a peer was killed and the survivors reach here together and call
            // MPI_Abort within a millisecond of each other, the PRRTE this image ships can INTERMITTENTLY fail to
            // tear the job down — prterun stays up with all its daemons after every rank is gone, because a rank
            // re-activated by the launcher's own kill path is never counted as terminated (a race in PRRTE's
            // CALLED_ABORT accounting), so the job never completes. The matching test row is a non-strict xfail.
            //   bug:   https://github.com/openpmix/prrte/issues/2815
            //   fixed: https://github.com/open-mpi/ompi/pull/14590 (main, 2026-10-23) and
            //          https://github.com/open-mpi/ompi/pull/14713 (v6.0.x, 2026-10-26); NOT in the 5.0.x series.
            //          This image is PRRTE 3.0.8 inside Open MPI 5.0.7rc2, before the fix.
            // (A second shape — prterun exits 0 though ranks aborted — is ULFM-only; tt-run launches the plain
            //  mpirun to avoid it: https://github.com/tenstorrent/tt-metal/pull/60468.)
            mark("survivor MPI_Abort");
            MPI_Abort(MPI_COMM_WORLD, 9);
        } else if (!strcmp(survivor_action, "exit_nofinalize")) {
            mark("survivor _exit(2)");
            _exit(2);
        } else if (!strcmp(survivor_action, "revoke_shrink")) {
#if defined(MPIX_ERR_PROC_FAILED)
            t = now_s();
            int rrc = MPIX_Comm_revoke(comm);
            mark("revoke rc=%s dt=%.3f", errstr(rrc), now_s() - t);
            MPI_Comm shrunk;
            t = now_s();
            int src = MPIX_Comm_shrink(comm, &shrunk);
            int nsize = -1;
            if (src == MPI_SUCCESS) {
                MPI_Comm_size(shrunk, &nsize);
            }
            mark("shrink rc=%s new_size=%d dt=%.3f", errstr(src), nsize, now_s() - t);
            if (src == MPI_SUCCESS) {
                t = now_s();
                int brc = MPI_Barrier(shrunk);
                mark("shrunk_barrier_rc=%s dt=%.3f", errstr(brc), now_s() - t);
            }
#else
            mark("revoke_shrink requested but ULFM not compiled in");
#endif
        }
        if (second_barrier) {
            t = now_s();
            int rc2 = MPI_Barrier(comm);
            mark("second_barrier_rc=%s dt=%.3f", errstr(rc2), now_s() - t);
        }
    }

    mark("main_exit");
    return 0;
}
