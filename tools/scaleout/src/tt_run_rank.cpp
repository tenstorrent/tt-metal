// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// tt-run-rank: the wrapper each rank runs in, so a failed rank stops the job gracefully, under tt-run or a plain
// mpirun.
//
//   tt-run-rank [--stop-grace <s>] [--on-peer-exit stop|notify] -- <program> [args...]
//
// It runs the program as its child and passes it every signal the launcher sends. It connects to its PMIx server as a
// tool for the whole run. The program fails when it writes "failed <code> <reason>" to TT_RUN_RANK_FD (read at once),
// or ends non-zero without a stop having reached it: the wrapper prints "[rank R] <reason> (on HOST, pid P); stopping
// the job (exit CODE)", which tt-run reads, and tells every other wrapper (a PMIx event). Each writes
// "exit <rank> <code> <signal> <reason>" to its program's TT_RUN_RANK_EVENTS_FD, then sends it SIGTERM (stop) or
// SIGUSR1 (notify: the job goes on). A wrapper whose program failed or was stopped holds its own exit until every rank
// has exited, so the launcher never kills a rank that is still closing; the failed rank's wrapper leaves first, so the
// launcher returns its code. A program still running --stop-grace seconds (120) after a stop reached it is killed
// (137). If the daemon that started this wrapper dies, the program is sent SIGTERM, then SIGKILL 10 s later.
//
//   tt-run-rank --guard <prterun pid> [<signal tool>] [--stop-grace <s>]
//
// The guard tt-run starts next to the launcher, so the job still stops gracefully when tt-run itself is killed
// (SIGKILL, an abort, a crash): tt-run cannot act then, and the launcher outlives it in a session of its own. tt-run
// holds the write end of the guard's stdin and writes "done" once the job is over; it also hands the guard the read end
// of the launcher's stderr (TT_RUN_GUARD_DRAIN_FD), which the guard drains once tt-run is gone, so prterun never writes
// into a pipe nobody reads (it aborts the job on SIGPIPE). If stdin closes without "done", tt-run died: while the
// launcher still runs, send SIGTERM to every rank through PRRTE job control (the signal tool, by default this binary's
// --uri mode), wait the stop grace for the job to end, then force it through the launcher, as tt-run would have: SIGINT
// (tt-run starts prterun with SIGTERM blocked), on which prterun kills every rank (SIGTERM, then SIGKILL 250 ms later),
// and SIGKILL 10 s later if prterun is still there.
//
//   tt-run-rank --uri <prterun PMIx server URI> --signal <N>
//
// Send signal N to every rank of a running prterun job, on every host, through PRRTE job control: tt-run's stop
// (SIGTERM) and the signals it passes on. Unlike a signal sent to prterun itself (SIGTERM, then SIGKILL 250 ms later),
// the job is not aborted and no SIGKILL follows. The URI is the first line of prterun's tool rendezvous file. Exit
// code: 0 on success, 1 on a PMIx error, 2 on bad arguments, 3 when built without PMIx.

// Without PMIx the wrapper still passes signals on, stops a rank whose daemon died and prints the stop line, but leaves
// ending the job to the launcher; the --uri mode fails, so tt-run and the guard force a stop instead.
#ifdef TT_RUN_RANK_HAS_PMIX
#include <pmix.h>
#include <pmix_tool.h>
#endif

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <glob.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int kDumpBoundTicks = 100;  // 10 s: how long a crash may write its core dump before it is cut short

// Every signal tt-run passes on to the ranks. Caught here so it reaches the program, not this wrapper: SIGALRM would
// kill the wrapper (and its program with it) and SIGTSTP would pause only the wrapper.
constexpr int kForwarded[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2, SIGALRM, SIGTSTP, SIGCONT, SIGWINCH};
volatile sig_atomic_t g_child = -1;       // the program while it runs; -1 once reaped (its pid may be reused)
volatile sig_atomic_t g_stop_signal = 0;  // a stop (SIGTERM, SIGINT, SIGHUP) delivered to the program
volatile sig_atomic_t g_reported = 0;     // this wrapper reported its program's failure and stopped the job

void forward(int signal_number) {
    const int saved_errno = errno;  // the main loop reads errno right after waitpid
    const bool stop = signal_number == SIGTERM || signal_number == SIGINT || signal_number == SIGHUP;
    // A SIGTERM after a stop was delivered is the same stop again (job control's, after the event's): not passed on.
    if (!(signal_number == SIGTERM && (g_stop_signal != 0 || g_reported != 0))) {
        if (stop) {
            g_stop_signal = signal_number;
        }
        if (g_child > 0) {
            ::kill(static_cast<pid_t>(g_child), signal_number);
        }
    }
    errno = saved_errno;
}

// This rank's number: MPI's, else tt-run's own (TT_RUN_RANK, set for a rank tt-run starts without mpirun).
const char* rank_name() {
    for (const char* var : {"OMPI_COMM_WORLD_RANK", "TT_RUN_RANK"}) {
        if (const char* v = std::getenv(var); v != nullptr && *v != '\0') {
            return v;
        }
    }
    return "?";
}

// A signal's name without "SIG" (sigabbrev_np needs glibc 2.32).
const char* signal_name(int signal_number) {
    switch (signal_number) {
        case SIGHUP: return "HUP";
        case SIGINT: return "INT";
        case SIGQUIT: return "QUIT";
        case SIGILL: return "ILL";
        case SIGTRAP: return "TRAP";
        case SIGABRT: return "ABRT";
        case SIGBUS: return "BUS";
        case SIGFPE: return "FPE";
        case SIGKILL: return "KILL";
        case SIGUSR1: return "USR1";
        case SIGSEGV: return "SEGV";
        case SIGUSR2: return "USR2";
        case SIGPIPE: return "PIPE";
        case SIGALRM: return "ALRM";
        case SIGTERM: return "TERM";
        case SIGCHLD: return "CHLD";
        case SIGCONT: return "CONT";
        case SIGSTOP: return "STOP";
        case SIGTSTP: return "TSTP";
        case SIGXCPU: return "XCPU";
        case SIGXFSZ: return "XFSZ";
        case SIGSYS: return "SYS";
        default: return "?";
    }
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

// Whether path is this user's own entry, not a symlink: the only kind read from the shared tmpdir.
bool owned(const char* path) {
    struct stat st{};
    return ::lstat(path, &st) == 0 && st.st_uid == ::getuid() && !S_ISLNK(st.st_mode);
}

// prterun's PMIx server URI for tools: the first line of <tmpdir>/prterun.<host>.<pid>.<uid>/pmix.<host>.tool.<pid>
// (ttrun.py LaunchedJob.prterun_tool_uri), empty while it is not there. Only this user's directory and file count (by
// name and by owner): another user cannot point the stop at a server of theirs.
std::string tool_uri(pid_t pid) {
    const char* tmp = std::getenv("TMPDIR");
    // This host's only (TMPDIR may be shared storage, where another host's prterun can have the same pid). PRRTE names
    // the directory after the host, without its domain by default.
    char host[256] = {};
    ::gethostname(host, sizeof host - 1);
    const std::string full = host, short_name = full.substr(0, full.find('.'));
    std::string uri;
    glob_t found{};
    int globbed = GLOB_NOMATCH;
    for (const std::string& h : {short_name, full}) {
        const std::string pattern = std::string(tmp != nullptr && *tmp != '\0' ? tmp : "/tmp") + "/prterun." + h + "." +
                                    std::to_string(pid) + "." + std::to_string(::getuid()) + "/pmix.*.tool." +
                                    std::to_string(pid);
        if (::glob(pattern.c_str(), globbed == 0 ? GLOB_APPEND : 0, nullptr, &found) == 0) {
            globbed = 0;
        }
        if (h == full) {
            break;
        }
    }
    if (globbed == 0) {
        for (size_t i = 0; i < found.gl_pathc && uri.empty(); ++i) {
            const std::string file = found.gl_pathv[i];
            if (!owned(file.substr(0, file.rfind('/')).c_str()) || !owned(file.c_str())) {
                continue;
            }
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
        for (auto& info : init) {
            PMIX_INFO_DESTRUCT(&info);
        }
        return 1;
    }

    // No targets: every rank of the job this prterun runs.
    pmix_info_t directive;
    PMIX_INFO_LOAD(&directive, PMIX_JOB_CTRL_SIGNAL, &signum, PMIX_INT32);
    rc = PMIx_Job_control(nullptr, 0, &directive, 1, nullptr, nullptr);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(stderr, "tt-run-rank: PMIx_Job_control(signal %d): %s\n", signum, PMIx_Error_string(rc));
    }
    PMIX_INFO_DESTRUCT(&directive);
    for (auto& info : init) {
        PMIX_INFO_DESTRUCT(&info);
    }
    PMIx_tool_finalize();
    return rc == PMIX_SUCCESS ? 0 : 1;
#endif
}

int guard(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s --guard <prterun pid> [<signal tool>] [--stop-grace <s>]\n", argv[0]);
        return 2;
    }
    const pid_t launcher = static_cast<pid_t>(std::atoi(argv[2]));
    const char* helper = argc > 3 && std::strncmp(argv[3], "--", 2) != 0 ? argv[3] : nullptr;
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
    // tt-run read the launcher's stderr; with tt-run gone nothing does, and prterun aborts the job (SIGKILL to every
    // rank) after about ten SIGPIPEs. tt-run handed this guard the pipe's read end: drain it to the terminal from now
    // on, for as long as the launcher writes, so the stop below stays graceful and its output is not lost.
    if (const char* fd_text = std::getenv("TT_RUN_GUARD_DRAIN_FD"); fd_text != nullptr && *fd_text != '\0') {
        const int fd = std::atoi(fd_text);
        std::thread([fd] {
            char chunk[4096];
            ssize_t n = 0;
            while ((n = ::read(fd, chunk, sizeof chunk)) > 0 || (n < 0 && errno == EINTR)) {
                if (n > 0) {
                    (void)!::write(STDERR_FILENO, chunk, static_cast<size_t>(n));
                }
            }
        }).detach();
    }

    const char* g = std::getenv("TT_RUN_STOP_GRACE_S");
    double grace = (g != nullptr && *g != '\0') ? std::atof(g) : 120.0;
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--stop-grace") == 0 && std::atof(argv[i + 1]) > 0) {
            grace = std::atof(argv[i + 1]);
        }
    }
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

// ---- wrap mode ----------------------------------------------------------------------------------------------------
struct Job {
    int me = -1;
    std::string nspace;
    bool notify = false;   // --on-peer-exit notify
    double grace = 120.0;  // --stop-grace
    int events_fd = -1;    // the write end of the program's TT_RUN_RANK_EVENTS_FD
    bool tool = false;     // connected to the PMIx server as a tool
    std::mutex mu;
    std::vector<char> ended;  // the ranks whose program has ended
};
Job g_job;

std::string host_name() {
    char host[256] = "?";
    ::gethostname(host, sizeof host - 1);
    return host;
}

// What a wrapper says of its program: "failed <rank> <code> <signal> <reason>" (stop or notify the others) or
// "ended <rank> <code>" (count it).
void on_peer(const std::string& what) {
    char kind[16] = {};
    int rank = -1, code = 0, sig = 0, used = 0;
    if (std::sscanf(what.c_str(), "%15s %d %d %d %n", kind, &rank, &code, &sig, &used) < 3 || rank < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_job.mu);
    if (std::strcmp(kind, "ended") == 0) {
        if (rank < static_cast<int>(g_job.ended.size())) {
            g_job.ended[rank] = 1;
        }
        return;
    }
    if (rank == g_job.me || g_child <= 0) {
        return;
    }
    const std::string line = "exit " + std::to_string(rank) + " " + std::to_string(code) + " " + std::to_string(sig) +
                             " " + what.substr(static_cast<size_t>(used)) + "\n";
    if (g_job.events_fd >= 0) {
        (void)!::write(g_job.events_fd, line.data(), line.size());
    }
    if (g_job.notify) {
        ::kill(static_cast<pid_t>(g_child), SIGUSR1);
    } else if (g_stop_signal == 0) {
        g_stop_signal = SIGTERM;
        ::kill(static_cast<pid_t>(g_child), SIGTERM);
    }
}

#ifdef TT_RUN_RANK_HAS_PMIX
constexpr pmix_status_t kPeerEvent = PMIX_EXTERNAL_ERR_BASE - 4242;  // payload "tt.peer": "<nspace> <what>"

void peer_event(
    size_t,
    pmix_status_t,
    const pmix_proc_t*,
    pmix_info_t info[],
    size_t ninfo,
    pmix_info_t*,
    size_t,
    pmix_event_notification_cbfunc_fn_t done,
    void* cbdata) {
    for (size_t i = 0; i < ninfo; ++i) {
        if (std::strcmp(info[i].key, "tt.peer") == 0 && info[i].value.type == PMIX_STRING) {
            const std::string w = info[i].value.data.string;
            if (w.compare(0, g_job.nspace.size() + 1, g_job.nspace + " ") == 0) {
                on_peer(w.substr(g_job.nspace.size() + 1));
            }
        }
    }
    if (done != nullptr) {
        done(PMIX_EVENT_ACTION_COMPLETE, nullptr, 0, nullptr, nullptr, cbdata);
    }
}
#endif

// Connect to the PMIx server that started this rank (its daemon) as a tool, through the server's rendezvous file.
void connect_tool(pid_t daemon) {
#ifdef TT_RUN_RANK_HAS_PMIX
    const char* dir = std::getenv("PMIX_SERVER_TMPDIR");
    const char* host = std::getenv("PMIX_HOSTNAME");
    const char* ns = std::getenv("PMIX_NAMESPACE");
    if (dir == nullptr || host == nullptr || ns == nullptr) {
        return;  // not launched by PMIx (tt-run's offline mode)
    }
    const std::string file = std::string(dir) + "/pmix." + host + ".tool." + std::to_string(daemon);
    std::string uri;
    if (FILE* f = owned(file.c_str()) ? std::fopen(file.c_str(), "r") : nullptr) {
        char line[1024] = {};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            uri = line;
            uri.erase(uri.find_last_not_of(" \r\n") + 1);
        }
        std::fclose(f);
    }
    g_job.nspace = ns;
    ::unsetenv("PMIX_NAMESPACE");  // a tool, not a second client for the rank
    ::unsetenv("PMIX_RANK");
    char tool_ns[PMIX_MAX_NSLEN + 1];
    std::snprintf(tool_ns, sizeof tool_ns, "tt-run-rank.%d", static_cast<int>(::getpid()));
    pmix_rank_t zero = 0;
    pmix_info_t init[3];
    PMIX_INFO_LOAD(&init[0], PMIX_TOOL_NSPACE, tool_ns, PMIX_STRING);
    PMIX_INFO_LOAD(&init[1], PMIX_TOOL_RANK, &zero, PMIX_PROC_RANK);
    PMIX_INFO_LOAD(&init[2], PMIX_SERVER_URI, uri.c_str(), PMIX_STRING);
    pmix_proc_t me;
    pmix_status_t rc = uri.empty() ? PMIX_ERR_NOT_FOUND : PMIx_tool_init(&me, init, 3);
    for (auto& i : init) {
        PMIX_INFO_DESTRUCT(&i);
    }
    pmix_status_t code = kPeerEvent;
    if (rc == PMIX_SUCCESS) {
        rc = PMIx_Register_event_handler(&code, 1, nullptr, 0, peer_event, nullptr, nullptr);
        rc = rc < 0 ? rc : PMIX_SUCCESS;  // the blocking form returns the handler's reference
    }
    if (rc != PMIX_SUCCESS) {
        std::fprintf(
            stderr,
            "[tt-run-rank] rank %s: cannot connect to the PMIx server (%s): the other ranks are not told when it "
            "ends\n",
            rank_name(),
            uri.empty() ? ("no " + file).c_str() : PMIx_Error_string(rc));
        return;
    }
    g_job.tool = true;
    if (const char* v = std::getenv("TT_RUN_RANK_VERBOSE"); v != nullptr && *v != '\0' && *v != '0') {
        std::fprintf(stderr, "[tt-run-rank] rank %s: connected to the PMIx server as a tool\n", rank_name());
    }
#else
    (void)daemon;
#endif
}

// Tell every wrapper of the job (this one included) what became of this rank's program.
void tell(const std::string& what) {
#ifdef TT_RUN_RANK_HAS_PMIX
    if (!g_job.tool) {
        return;
    }
    const std::string payload = g_job.nspace + " " + what;
    pmix_info_t info;
    PMIX_INFO_LOAD(&info, "tt.peer", payload.c_str(), PMIX_STRING);
    (void)PMIx_Notify_event(kPeerEvent, nullptr, PMIX_RANGE_GLOBAL, &info, 1, nullptr, nullptr);
    PMIX_INFO_DESTRUCT(&info);
#else
    (void)what;
#endif
}

// SIGTERM to every rank through PRRTE job control: the backstop to the event (a wrapper that missed it), or the stop
// itself when this wrapper has no tool connection (then as a client, once its program is gone).
void stop_job() {
#ifdef TT_RUN_RANK_HAS_PMIX
    const bool client = !g_job.tool;
    pmix_proc_t me;
    if (client && (g_child > 0 || PMIx_Init(&me, nullptr, 0) != PMIX_SUCCESS)) {
        return;  // the launcher ends the job
    }
    pmix_proc_t all;
    PMIX_LOAD_PROCID(&all, client ? me.nspace : g_job.nspace.c_str(), PMIX_RANK_WILDCARD);
    int32_t signum = SIGTERM;
    pmix_info_t directive;
    PMIX_INFO_LOAD(&directive, PMIX_JOB_CTRL_SIGNAL, &signum, PMIX_INT32);
    const pmix_status_t rc = PMIx_Job_control(&all, 1, &directive, 1, nullptr, nullptr);
    PMIX_INFO_DESTRUCT(&directive);
    if (rc != PMIX_SUCCESS) {
        std::fprintf(stderr, "[tt-run-rank] rank %s: PMIx_Job_control: %s\n", rank_name(), PMIx_Error_string(rc));
    }
    if (client) {
        PMIx_Finalize(nullptr, 0);
    }
#endif
}

// This rank's program failed: say so (tt-run reads the line), tell the other wrappers, and stop the job unless notify.
void report(const std::string& how, int code, int sig, pid_t pid) {
    g_reported = 1;
    std::fprintf(
        stderr,
        "[rank %d] %s (on %s, pid %d); %s (exit %d)\n",
        g_job.me,
        how.c_str(),
        host_name().c_str(),
        static_cast<int>(pid),
        g_job.notify ? "the job goes on" : "stopping the job",
        code);
    std::fflush(stderr);
    tell("failed " + std::to_string(g_job.me) + " " + std::to_string(code) + " " + std::to_string(sig) + " " + how);
    if (!g_job.notify) {
        if (g_job.tool) {
            ::usleep(200000);  // the event, with its line, reaches the ranks first
        }
        stop_job();
    }
}

// A program that failed or was stopped: hold this wrapper's exit until every rank's program has ended, so the launcher
// kills none still closing. The first failed rank leaves first, so the launcher reports its code.
void hold(bool origin) {
    const double bound = g_job.grace + 10.0;
    for (double waited = 0.0; g_job.tool && waited < bound; waited += 0.1) {
        {
            std::lock_guard<std::mutex> lock(g_job.mu);
            if (std::find(g_job.ended.begin(), g_job.ended.end(), 0) == g_job.ended.end()) {
                break;
            }
        }
        ::usleep(100000);
    }
    if (!origin) {
        ::usleep(300000);
    }
}

// Read what the program wrote on TT_RUN_RANK_FD; a "failed <code> <reason>" line reports its failure at once.
void read_words(int fd, std::string& said, pid_t child) {
    char chunk[512];
    for (ssize_t n = 0; fd >= 0 && (n = ::read(fd, chunk, sizeof chunk)) > 0;) {
        said.append(chunk, static_cast<size_t>(n));
    }
    for (size_t nl = 0; (nl = said.find('\n')) != std::string::npos; said.erase(0, nl + 1)) {
        int code = 0;
        int used = 0;
        if (g_reported == 0 && std::sscanf(said.c_str(), "failed %d %n", &code, &used) == 1 && used > 0) {
            report(said.substr(static_cast<size_t>(used), nl - static_cast<size_t>(used)), code, 0, child);
        }
    }
}

void disconnect() {
#ifdef TT_RUN_RANK_HAS_PMIX
    if (g_job.tool) {
        PMIx_tool_finalize();
        if (const char* v = std::getenv("TT_RUN_RANK_VERBOSE"); v != nullptr && *v != '\0' && *v != '0') {
            std::fprintf(stderr, "[tt-run-rank] rank %d: left the PMIx server\n", g_job.me);
        }
    }
#endif
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
    for (; first + 1 < argc && std::strncmp(argv[first], "--", 2) == 0 && argv[first][2] != '\0'; first += 2) {
        if (std::strcmp(argv[first], "--stop-grace") == 0 && std::atof(argv[first + 1]) > 0) {
            g_job.grace = std::atof(argv[first + 1]);
        } else if (std::strcmp(argv[first], "--on-peer-exit") == 0 && std::strcmp(argv[first + 1], "stop") != 0) {
            g_job.notify = std::strcmp(argv[first + 1], "notify") == 0;
            if (!g_job.notify) {
                first = argc;  // a bad mode: usage
            }
        } else if (std::strcmp(argv[first], "--on-peer-exit") != 0) {
            first = argc;
        }
    }
    if (first < argc && std::strcmp(argv[first], "--") == 0) {
        ++first;
    }
    if (first >= argc) {
        std::fprintf(
            stderr, "usage: %s [--stop-grace <s>] [--on-peer-exit stop|notify] -- <program> [args...]\n", argv[0]);
        return 2;
    }

    ::signal(SIGCHLD, SIG_DFL);  // inherited as ignored, the kernel would reap the program and hide its status
    ::signal(SIGPIPE, SIG_IGN);  // a dead daemon's pipe must not kill this wrapper before it stops its program
    int status_pipe[2] = {-1, -1};
    int events_pipe[2] = {-1, -1};
    if (::pipe2(status_pipe, O_CLOEXEC) != 0) {
        status_pipe[0] = status_pipe[1] = -1;
    }
    if (::pipe2(events_pipe, O_CLOEXEC) != 0) {
        events_pipe[0] = events_pipe[1] = -1;
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
        ::signal(SIGPIPE, SIG_DFL);  // an ignored disposition survives exec
        ::sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
        // The launcher kills the process it started, this wrapper, not its child: without this a program the
        // launcher's SIGKILL reached only through the wrapper would outlive the job.
        ::prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (::getppid() != wrapper) {
            ::_exit(128 + SIGKILL);  // the wrapper is already gone
        }
        // Without O_CLOEXEC, so the program inherits them; never a variable inherited from outside, pointing at
        // nothing.
        for (const auto& [var, end] :
             {std::pair{"TT_RUN_RANK_FD", status_pipe[1]}, {"TT_RUN_RANK_EVENTS_FD", events_pipe[0]}}) {
            const int fd = end >= 0 ? ::dup(end) : -1;
            if (fd >= 0) {
                ::setenv(var, std::to_string(fd).c_str(), 1);
            } else {
                ::unsetenv(var);
            }
        }
        ::execvp(argv[first], argv + first);
        std::fprintf(stderr, "tt-run-rank: cannot run %s: %s\n", argv[first], std::strerror(errno));
        ::_exit(127);
    }
    g_child = child;
    ::sigprocmask(SIG_SETMASK, &previous_mask, nullptr);  // a signal held since startup is forwarded now
    for (const int fd : {status_pipe[1], events_pipe[0]}) {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    if (status_pipe[0] >= 0) {
        ::fcntl(status_pipe[0], F_SETFL, O_NONBLOCK);
    }
    if (events_pipe[1] >= 0) {
        ::fcntl(events_pipe[1], F_SETFL, O_NONBLOCK);  // a program that never reads them never blocks this wrapper
    }
    g_job.events_fd = events_pipe[1];
    g_job.me = std::atoi(rank_name());
    if (const char* n = std::getenv("OMPI_COMM_WORLD_SIZE"); n != nullptr && std::atoi(n) > 0) {
        g_job.ended.assign(static_cast<size_t>(std::atoi(n)), 0);
    }
    connect_tool(daemon);

    // Wait for the program, reading what it writes on TT_RUN_RANK_FD as it writes it. A crash still writing its core
    // dump is ended after 10 s, at once when the job is stopping (teardown plan P8). A program still running the stop
    // grace after a stop reached it is killed. A program whose daemon is gone is stopped.
    int status = 0;
    bool dump_ended = false;
    bool forced = false;
    int dump_ticks = 0;
    int wait_errno = 0;       // waitpid failed: the program's status is unknown, which is a failure
    int orphaned_ticks = -1;  // 100 ms ticks since the daemon was found gone
    double stop_ticks = -1;   // 100 ms ticks since a stop reached the program
    std::string said;
    for (;;) {
        const pid_t done = ::waitpid(child, &status, WNOHANG);
        if (done == child) {
            break;
        }
        if (done < 0 && errno != EINTR) {
            wait_errno = errno;
            break;
        }
        read_words(status_pipe[0], said, child);
        if (orphaned_ticks < 0 && ::getppid() != daemon) {
            orphaned_ticks = 0;
            std::fprintf(
                stderr,
                "[tt-run-rank] rank %s: the daemon that started it (pid %d) is gone; stopping the rank (SIGTERM)\n",
                rank_name(),
                static_cast<int>(daemon));
            forward(SIGTERM);
        } else if (orphaned_ticks >= 0 && ++orphaned_ticks == 100) {  // still there 10 s later
            ::kill(child, SIGKILL);
        }
        if (!dump_ended && core_dumping(child) && (g_stop_signal != 0 || ++dump_ticks >= kDumpBoundTicks)) {
            std::fprintf(
                stderr,
                "[tt-run-rank] rank %s was still writing a core dump %s; ending it (core cut short)\n",
                rank_name(),
                g_stop_signal != 0 ? "when the job stopped" : "10 s after it crashed");
            ::kill(child, SIGKILL);
            dump_ended = true;
        }
        if (g_stop_signal != 0 && !forced && ++stop_ticks >= g_job.grace * 10) {
            std::fprintf(
                stderr,
                "[rank %s] did not exit within its stop grace of %g s: forced (SIGKILL); the devices may need tt-smi "
                "-r\n",
                rank_name(),
                g_job.grace);
            ::kill(child, SIGKILL);
            forced = true;
        }
        ::usleep(100000);
    }
    read_words(status_pipe[0], said, child);  // a word written just before the exit
    g_child = -1;                             // reaped: its pid may belong to another process from now on
    const bool signalled = wait_errno == 0 && WIFSIGNALED(status);
    const int sig = signalled ? WTERMSIG(status) : 0;
    int code = signalled ? 128 + sig : WEXITSTATUS(status);
    if (wait_errno != 0) {
        code = 1;  // the program's status is unknown: a failure
    }
    ::signal(SIGTERM, SIG_IGN);  // the job's stop comes back to this wrapper too

    // After a stop, any code is a stopped rank's; with none, a non-zero code is a failure; 0 is a clean exit.
    const bool stopped = g_stop_signal != 0 || forced;
    if (g_reported == 0 && !stopped && code != 0) {
        std::string how;
        if (wait_errno != 0) {
            how = std::string("could not be waited for (waitpid: ") + std::strerror(wait_errno) + ")";
        } else if (dump_ended) {
            how = "crashed and was still writing its core dump (core cut short)";
        } else if (signalled) {
            how = "was killed by signal " + std::to_string(sig) + " (SIG" + signal_name(sig) + ")";
        } else {
            how = "exited with code " + std::to_string(code);
        }
        report(how, code, sig, child);
    } else if (g_reported != 0 && !g_job.tool && !g_job.notify) {
        stop_job();  // its word came while it ran; without a tool connection the stop goes now, as a client
    }
    tell("ended " + std::to_string(g_job.me) + " " + std::to_string(code));
    if ((stopped || g_reported != 0) && orphaned_ticks < 0) {
        hold(g_reported != 0);
    }
    disconnect();
    return code;
}
