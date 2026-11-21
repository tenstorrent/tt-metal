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
// A program killed by a signal is reported with exit 128 + the signal. If the daemon that started this wrapper dies
// (a killed prted), the program is sent SIGTERM, then SIGKILL 10 s later: a rank never outlives its daemon.
//
//   tt-run-rank --guard <prterun pid> [<signal tool>]
//
// The guard tt-run starts next to the launcher, so the job still stops gracefully when tt-run itself is killed
// (SIGKILL, an abort, a crash): tt-run cannot act then, and the launcher outlives it in a session of its own. tt-run
// holds the write end of the guard's stdin and writes "done" once the job is over. If stdin closes without it, tt-run
// died: while the launcher still runs, send SIGTERM to every rank through PRRTE job control (the signal tool, by
// default this binary's --uri mode), wait the stop grace (TT_RUN_STOP_GRACE_S, 120 s by default) for the job to end,
// then force it through the launcher (SIGTERM, which kills every rank: SIGTERM, then SIGKILL 250 ms later), as tt-run
// would have.
//
//   tt-run-rank --uri <prterun PMIx server URI> --signal <N>
//
// Send signal N to every rank of a running prterun job, on every host, through PRRTE job control: tt-run's stop
// (SIGTERM) and the signals it passes on. PMIx_Job_control(PMIX_JOB_CTRL_SIGNAL) makes prterun tell every daemon to
// kill() its own ranks with it; unlike a signal sent to prterun itself (SIGTERM, then SIGKILL 250 ms later), the job is
// not aborted and no SIGKILL follows. The URI is the first line of prterun's tool rendezvous file
// (<tmpdir>/prterun.<host>.<pid>.<uid>/pmix.<host>.tool.<pid>); connecting by URI avoids PMIx's pid lookup, which
// searches the whole system tmpdir recursively. Exit code: 0 on success, 1 on a PMIx error, 2 on bad arguments.

// Without PMIx (an MPI install that has none) the graceful parts are compiled out: the wrapper still passes signals
// on, stops a rank whose daemon died and prints the [tt-stop] line, but leaves ending the job to the launcher; the
// --uri mode fails, so tt-run and the guard force a stop instead.
#ifdef TT_RUN_RANK_HAS_PMIX
#include <pmix.h>
#include <pmix_tool.h>
#endif

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <glob.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// Every signal tt-run passes on to the ranks. Caught here so it reaches the program, not this wrapper: SIGALRM would
// kill the wrapper (and its program with it) and SIGTSTP would pause only the wrapper.
constexpr int kForwarded[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2, SIGALRM, SIGTSTP, SIGCONT, SIGWINCH};
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

#ifdef TT_RUN_RANK_HAS_PMIX
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
#endif

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

// The process start time (/proc/<pid>/stat field 22), 0 when the process is gone or a zombie: with the pid, it tells
// the launcher this guard watches from an unrelated process that reused its pid.
unsigned long long start_time(pid_t pid) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/stat", static_cast<int>(pid));
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) {
        return 0;
    }
    char buf[1024] = {};
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    const char* p = std::strrchr(buf, ')');  // the command name can hold spaces and parentheses
    if (p == nullptr || p[1] == '\0' || p[2] == 'Z') {
        return 0;
    }
    p += 2;  // field 3, the state
    for (int field = 3; field < 22 && p != nullptr; ++field) {
        p = std::strchr(p, ' ');
        p = p != nullptr ? p + 1 : nullptr;
    }
    return p != nullptr ? std::strtoull(p, nullptr, 10) : 0;
}

// prterun's PMIx server URI for tools: the first line of <tmpdir>/prterun.<host>.<pid>.<uid>/pmix.<host>.tool.<pid>
// (ttrun.py prterun_tool_uri), empty while it is not there.
std::string tool_uri(pid_t pid) {
    const char* tmp = std::getenv("TMPDIR");
    const std::string pattern = std::string(tmp != nullptr && *tmp != '\0' ? tmp : "/tmp") + "/prterun.*." +
                                std::to_string(pid) + ".*/pmix.*.tool." + std::to_string(pid);
    std::string uri;
    glob_t found{};
    if (::glob(pattern.c_str(), 0, nullptr, &found) == 0) {
        for (size_t i = 0; i < found.gl_pathc && uri.empty(); ++i) {
            if (FILE* f = std::fopen(found.gl_pathv[i], "r")) {
                char line[1024] = {};
                if (std::fgets(line, sizeof line, f) != nullptr) {
                    uri = line;
                    uri.erase(uri.find_last_not_of(" \r\n") + 1);
                }
                std::fclose(f);
            }
        }
    }
    ::globfree(&found);
    return uri;
}

// Run the signal tool (tt-run-rank --uri) to send SIGTERM to every rank; "" when sent, else why not. Bounded.
std::string stop_ranks(const char* helper, pid_t launcher) {
    if (helper == nullptr || *helper == '\0') {
        return "no signal tool";
    }
    std::string uri;
    for (int i = 0; i < 50 && uri.empty(); ++i) {  // a launcher started a moment ago may not have published it yet
        uri = tool_uri(launcher);
        if (uri.empty()) {
            ::usleep(100000);
        }
    }
    if (uri.empty()) {
        return "no rendezvous file for prterun pid " + std::to_string(launcher);
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
        return std::string("fork: ") + std::strerror(errno);
    }
    if (pid == 0) {
        ::execl(helper, helper, "--uri", uri.c_str(), "--signal", "15", static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status = 0;
    for (int i = 0; i < 100; ++i) {  // 10 s
        if (::waitpid(pid, &status, WNOHANG) == pid) {
            return WIFEXITED(status) && WEXITSTATUS(status) == 0
                       ? ""
                       : "the signal tool exited " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        }
        ::usleep(100000);
    }
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &status, 0);
    return "the signal tool timed out";
}

int signal_job(int argc, char** argv) {
    const char* uri = nullptr;
    int32_t signum = SIGTERM;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--uri") == 0) {
            uri = argv[i + 1];
        } else if (std::strcmp(argv[i], "--signal") == 0) {
            signum = static_cast<int32_t>(std::atoi(argv[i + 1]));
        }
    }
    if (uri == nullptr) {
        std::fprintf(stderr, "usage: %s --uri <prterun PMIx URI> --signal <N>\n", argv[0]);
        return 2;
    }
#ifndef TT_RUN_RANK_HAS_PMIX
    std::fprintf(stderr, "tt-run-rank: built without PMIx; cannot signal the job (signal %d)\n", signum);
    return 3;
#else

    // Identify as a tool with a namespace of our own, the way PRRTE's own tools (pterm) do.
    char nspace[PMIX_MAX_NSLEN + 1];
    std::snprintf(nspace, sizeof nspace, "tt-run-rank.%d", static_cast<int>(::getpid()));
    pmix_rank_t rank = 0;
    pmix_info_t init[3];
    PMIX_INFO_LOAD(&init[0], PMIX_TOOL_NSPACE, nspace, PMIX_STRING);
    PMIX_INFO_LOAD(&init[1], PMIX_TOOL_RANK, &rank, PMIX_PROC_RANK);
    PMIX_INFO_LOAD(&init[2], PMIX_SERVER_URI, uri, PMIX_STRING);
    pmix_proc_t me;
    pmix_status_t rc = PMIx_tool_init(&me, init, 3);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(stderr, "tt-run-rank: PMIx_tool_init(%s): %s\n", uri, PMIx_Error_string(rc));
        return 1;
    }

    // No targets: every rank of the job this prterun runs.
    pmix_info_t directive;
    PMIX_INFO_LOAD(&directive, PMIX_JOB_CTRL_SIGNAL, &signum, PMIX_INT32);
    rc = PMIx_Job_control(nullptr, 0, &directive, 1, nullptr, nullptr);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(stderr, "tt-run-rank: PMIx_Job_control(signal %d): %s\n", signum, PMIx_Error_string(rc));
    }
    PMIx_tool_finalize();
    return rc == PMIX_SUCCESS ? 0 : 1;
#endif
}

int guard(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s --guard <prterun pid> [<signal tool>]\n", argv[0]);
        return 2;
    }
    const pid_t launcher = static_cast<pid_t>(std::atoi(argv[2]));
    const char* helper = argc > 3 ? argv[3] : nullptr;
    const pid_t ttrun = ::getppid();
    const unsigned long long started = start_time(launcher);
    ::signal(SIGPIPE, SIG_IGN);
    ::signal(SIGHUP, SIG_IGN);
    ::signal(SIGINT, SIG_IGN);  // a stop at tt-run is tt-run's; this guard acts only when tt-run is gone

    // Block until tt-run says the job is over ("done") or its end of the pipe closes.
    std::string said;
    char buf[64];
    for (;;) {
        const ssize_t n = ::read(STDIN_FILENO, buf, sizeof buf);
        if (n > 0) {
            said.append(buf, static_cast<size_t>(n));
            if (said.find("done") != std::string::npos) {
                return 0;
            }
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        break;
    }
    const auto launcher_running = [&] { return started != 0 && start_time(launcher) == started; };
    if (!launcher_running()) {
        return 0;  // tt-run ended with its job
    }

    const char* g = std::getenv("TT_RUN_STOP_GRACE_S");
    const double grace = (g != nullptr && *g != '\0') ? std::atof(g) : 120.0;
    std::fprintf(
        stderr,
        "[tt-run guard] tt-run (pid %d) ended while its job was still running (prterun pid %d): stopping the job\n",
        static_cast<int>(ttrun),
        static_cast<int>(launcher));
    const std::string why = stop_ranks(helper, launcher);
    if (why.empty()) {
        std::fprintf(
            stderr,
            "[tt-run guard] sent SIGTERM to all ranks; waiting up to %g s for them to exit, then forcing\n",
            grace);
        for (double waited = 0.0; waited < grace; waited += 0.2) {
            if (!launcher_running()) {
                std::fprintf(stderr, "[tt-run guard] the job ended after the stop\n");
                return 0;
            }
            ::usleep(200000);
        }
    }
    if (!launcher_running()) {
        return 0;
    }
    std::fprintf(
        stderr,
        "[tt-run guard] job forced down (%s): the launcher kills every rank (SIGTERM, then SIGKILL 250 ms later) "
        "without "
        "a graceful shutdown, so devices may be left unhealthy. Run tt-smi -r on the job's hosts before the next "
        "run.\n",
        why.empty() ? "ranks did not exit within the stop grace" : why.c_str());
    ::kill(launcher, SIGINT);  // tt-run starts prterun with SIGTERM blocked; SIGINT is its kill path too
    for (int i = 0; i < 100 && launcher_running(); ++i) {  // 10 s, tt-run's SIGNAL_FORWARD_GRACE_S
        ::usleep(100000);
    }
    if (launcher_running()) {
        ::kill(launcher, SIGKILL);
    }
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--guard") == 0) {
        return guard(argc, argv);
    }
    if (argc > 1 && std::strcmp(argv[1], "--uri") == 0) {
        return signal_job(argc, argv);
    }
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
    // Held from before the handlers are installed until the child exists, so a signal in between is delivered (and
    // forwarded) once there is a child to forward it to, never dropped.
    sigset_t forwarded{};
    sigset_t previous_mask{};
    ::sigemptyset(&forwarded);
    for (const int s : kForwarded) {
        ::sigaddset(&forwarded, s);
    }
    ::sigprocmask(SIG_BLOCK, &forwarded, &previous_mask);
    struct sigaction action{};
    action.sa_handler = forward;
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    for (const int s : kForwarded) {
        ::sigaction(s, &action, nullptr);
    }

    const pid_t wrapper = ::getpid();
    const pid_t daemon = ::getppid();  // the launcher's daemon that started this rank
    const pid_t child = ::fork();
    if (child < 0) {
        std::perror("tt-run-rank: fork");
        ::sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
        return 127;
    }
    if (child == 0) {
        for (const int s : kForwarded) {
            ::signal(s, SIG_DFL);
        }
        ::sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
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
    ::sigprocmask(SIG_SETMASK, &previous_mask, nullptr);  // a signal held since startup is forwarded now
    if (status_pipe[1] >= 0) {
        ::close(status_pipe[1]);
    }

    // Wait for the program. If the job is stopping while the program is still writing a core dump (a crash that is
    // not over yet), end it: a core dump never holds the job (teardown plan P8); the core is cut short. If the daemon
    // that started this wrapper is gone (this wrapper was reparented), nothing will ever stop the program: stop it.
    int status = 0;
    bool dump_ended = false;
    int orphaned_ticks = -1;  // 100 ms ticks since the daemon was found gone
    for (;;) {
        const pid_t done = ::waitpid(child, &status, WNOHANG);
        if (done == child || (done < 0 && errno != EINTR)) {
            break;
        }
        if (orphaned_ticks < 0 && ::getppid() != daemon) {
            orphaned_ticks = 0;
            std::fprintf(
                stderr,
                "[tt-run-rank] rank %s: the daemon that started it (pid %d) is gone; stopping the rank (SIGTERM)\n",
                std::getenv("OMPI_COMM_WORLD_RANK") != nullptr ? std::getenv("OMPI_COMM_WORLD_RANK") : "?",
                static_cast<int>(daemon));
            forward(SIGTERM);
        } else if (orphaned_ticks >= 0) {
            ++orphaned_ticks;
            if (orphaned_ticks == 100) {  // still there 10 s later
                ::kill(child, SIGKILL);
            }
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

#ifndef TT_RUN_RANK_HAS_PMIX
    (void)context_rank;
    std::fprintf(stderr, "[tt-stop] tt-run-rank: built without PMIx; the launcher ends the job\n");
    return code;
#else
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
#endif
}
