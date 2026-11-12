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
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_rank = -1;
static double g_t0;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void mark(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void mark(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    printf("[probe r=%d pid=%d t=%.3f] %s\n", g_rank, (int)getpid(), now_s() - g_t0, buf);
    fflush(stdout);
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

static void sleep_ms(long ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

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
        for (int i = 0;; i++) {
            if (i % 5 == 0) {
                mark("all_hang heartbeat");
            }
            sleep_ms(1000);
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
