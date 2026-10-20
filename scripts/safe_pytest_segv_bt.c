// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Crash helper for run_safe_pytest.sh, loaded with LD_PRELOAD. On SIGSEGV or SIGBUS it writes the faulting
// thread's native backtrace, the fault address, the thread id and /proc/self/maps to $SEGV_BT_FILE, then
// re-raises with the default action (a core dump, exit 139).
//
// The handler is installed before Python starts, so Python's faulthandler (which pytest enables) saves it as
// the previous handler: on a crash faulthandler prints the Python stacks, restores this handler and re-raises,
// and this one adds the native stack. The output file is opened at load time, so pytest's fd capture cannot
// swallow it. Everything in the handler is async-signal-safe except backtrace(), which is preloaded below.
#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static int out_fd = -1;

static void put(const char* s, size_t n) {
    ssize_t r = write(out_fd, s, n);
    (void)r;
}

static void handler(int sig, siginfo_t* si, void* uc) {
    (void)uc;
    char buf[256];
    int n = snprintf(
        buf,
        sizeof buf,
        "SEGV_BT: signal %d at address %p in thread %ld of pid %d\n",
        sig,
        si->si_addr,
        (long)syscall(SYS_gettid),
        getpid());
    put(buf, n);
    void* frames[64];
    int k = backtrace(frames, 64);
    backtrace_symbols_fd(frames, k, out_fd);
    int fd = open("/proc/self/maps", O_RDONLY);
    if (fd >= 0) {
        char m[4096];
        ssize_t r;
        put("SEGV_BT: maps\n", 14);
        while ((r = read(fd, m, sizeof m)) > 0) {
            put(m, r);
        }
        close(fd);
    }
    put("SEGV_BT: end\n", 13);
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void install(void) {
    const char* path = getenv("SEGV_BT_FILE");
    if (path == NULL) {
        return;
    }
    out_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (out_fd < 0) {
        return;
    }
    // Fault the unwinder in now: backtrace() loads libgcc_s on first use, which is not safe in a handler.
    void* warm[1];
    backtrace(warm, 1);
    // A stack overflow leaves no stack to run the handler on.
    static char altstack[1 << 16];
    stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack, .ss_flags = 0};
    sigaltstack(&ss, NULL);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
}
