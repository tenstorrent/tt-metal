// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_distributed_context.hpp"
#include "job_stop_internal.hpp"
#include "dtype_size.hpp"
#include <mpi.h>
#include <mpi-ext.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <cerrno>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <limits>
#include <thread>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <tt_stl/assert.hpp>

// Use MPIX_ERR_PROC_FAILED as a proxy to detect whether OpenMPI was built with
// ULFM extensions.
#if (defined(OPEN_MPI) && OPEN_MPI && defined(MPIX_ERR_PROC_FAILED))
#define OMPI_HAS_ULFM 1
#else
#define OMPI_HAS_ULFM 0
#endif

namespace tt::tt_metal::distributed::multihost {

namespace {

// Sub-context layout comes from tt-run env vars (set per-process, no collectives):
//   TT_RUN_SUBCONTEXT_ID     -> this rank's sub-context id (absent => single-context job)
//   TT_RUN_SUBCONTEXT_SIZES  -> comma-separated sizes of all sub-contexts, in id order
struct MpiLauncherEnvLayout {
    std::optional<int> this_subcontext_id;
    std::vector<int> subcontext_sizes;
    // Prefix sum of subcontext_sizes: world_rank_prefix[s] is the first MPI_COMM_WORLD rank in sub-context s.
    // E.g. sizes {64,1} -> prefix {0,64}; decode local 3 -> world 0+3, prefill local 0 -> world 64+0.
    std::vector<int> world_rank_prefix;
};

MpiLauncherEnvLayout g_mpi_launcher_env_layout{};

// `job_world_size` is MPI_COMM_WORLD (pre-split) rank count; used to validate TT_RUN_SUBCONTEXT_SIZES and
// to build the single-context layout when TT_RUN_SUBCONTEXT_ID is unset.
MpiLauncherEnvLayout parse_launcher_env_layout_from_env(int job_world_size) {
    MpiLauncherEnvLayout m;
    TT_FATAL(job_world_size > 0, "job_world_size must be positive");

    const char* id_env = std::getenv("TT_RUN_SUBCONTEXT_ID");
    if (id_env == nullptr || id_env[0] == '\0') {
        m.subcontext_sizes = {job_world_size};
        m.world_rank_prefix = {0};
        return m;
    }

    const char* sizes_env = std::getenv("TT_RUN_SUBCONTEXT_SIZES");
    TT_FATAL(
        sizes_env != nullptr && sizes_env[0] != '\0',
        "TT_RUN_SUBCONTEXT_ID is set but TT_RUN_SUBCONTEXT_SIZES is missing");

    try {
        m.this_subcontext_id = std::stoi(std::string(id_env));
    } catch (const std::exception&) {
        TT_THROW("Invalid TT_RUN_SUBCONTEXT_ID (expected integer): {}", id_env);
    }
    TT_FATAL(*m.this_subcontext_id >= 0, "TT_RUN_SUBCONTEXT_ID must be non-negative, got {}", *m.this_subcontext_id);

    std::stringstream ss(sizes_env);
    for (std::string tok; std::getline(ss, tok, ',');) {
        const int sz = std::stoi(tok);
        TT_FATAL(sz > 0, "TT_RUN_SUBCONTEXT_SIZES: sizes must be positive, got {}", sz);
        m.subcontext_sizes.push_back(sz);
    }

    const int this_id = *m.this_subcontext_id;
    TT_FATAL(
        this_id < static_cast<int>(m.subcontext_sizes.size()),
        "TT_RUN_SUBCONTEXT_ID {} out of range for {} sub-contexts",
        this_id,
        m.subcontext_sizes.size());

    int sum_sizes = 0;
    for (int s : m.subcontext_sizes) {
        sum_sizes += s;
    }
    TT_FATAL(
        sum_sizes == job_world_size,
        "TT_RUN_SUBCONTEXT_SIZES sum {} does not match MPI_COMM_WORLD size {}",
        sum_sizes,
        job_world_size);

    // world_rank_prefix[i] = sum(subcontext_sizes[0..i-1]) so local rank L in sub-context i maps to
    // MPI world rank world_rank_prefix[i] + L (tt-run merges sub-contexts as contiguous world blocks).
    m.world_rank_prefix.assign(m.subcontext_sizes.size(), 0);
    for (std::size_t i = 1; i < m.subcontext_sizes.size(); ++i) {
        m.world_rank_prefix[i] = m.world_rank_prefix[i - 1] + m.subcontext_sizes[i - 1];
    }
    return m;
}

}  // namespace

/* ----------------------------- helpers ---------------------------------- */

constexpr MPI_Op reduce_to_mpi(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM: return MPI_SUM;
        case ReduceOp::MAX: return MPI_MAX;
        case ReduceOp::MIN: return MPI_MIN;
        case ReduceOp::PROD: return MPI_PROD;
        case ReduceOp::LAND: return MPI_LAND;
        case ReduceOp::LOR: return MPI_LOR;
        case ReduceOp::BAND: return MPI_BAND;
        case ReduceOp::BOR: return MPI_BOR;
    }
    return MPI_SUM;  // default
}

constexpr MPI_Datatype dtype_to_mpi(DType dt) noexcept {
    switch (dt) {
        case DType::INT8: return MPI_INT8_T;
        case DType::UINT8: return MPI_UINT8_T;
        case DType::INT16: return MPI_INT16_T;
        case DType::UINT16: return MPI_UINT16_T;
        case DType::INT32: return MPI_INT32_T;
        case DType::UINT32: return MPI_UINT32_T;
        case DType::INT64: return MPI_INT64_T;
        case DType::UINT64: return MPI_UINT64_T;
        case DType::FLOAT32: return MPI_FLOAT;
        case DType::FLOAT64: return MPI_DOUBLE;
        case DType::BOOL: return MPI_C_BOOL;
        case DType::BYTE: return MPI_BYTE;
        case DType::COMPLEX_FLOAT: return MPI_C_FLOAT_COMPLEX;
        case DType::COMPLEX_DOUBLE: return MPI_C_DOUBLE_COMPLEX;
    }
    return MPI_DATATYPE_NULL;
}

inline void check_size_fits_int(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TT_THROW("MPI buffer size > INT_MAX");
    }
}

inline void mpi_check(int error_code, const char* call_text) {
    if (error_code != MPI_SUCCESS) {
#ifdef MPIX_ERR_PROC_FAILED
        int error_class = error_code;
        MPI_Error_class(error_code, &error_class);
        if (error_class == MPIX_ERR_PROC_FAILED) {
            job_stop::note_peer_failed();  // bounds this rank's exit fence: the dead peer may never reach it
        }
#endif
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // throw with the textual form of the call
        throw MPIDistributedException(Rank{rank}, error_code, std::string(call_text) + " failed");
    }
}

bool was_mpi_finalized() noexcept {
    int flag = 0;
    /* Safe to call at any time—even before MPI_Init() */
    MPI_Finalized(&flag);  // sets flag = 1 if MPI_Finalize has completed
    return flag != 0;
}

#define MPI_CHECK(call) mpi_check((call), #call)

// Teardown plan P16: a per-call deadline for the collectives that wait on a peer. 0 (the default) keeps the
// unbounded wait, so a legitimately long or slow-but-arriving collective is left alone; a positive value bounds a
// collective that hangs because a specific rank never arrived.
// TODO(teardown P30): with the default 0 here and no --timeout from tt-run, a hung collective never ends and R2
// holds only when someone opts in. Decide the owner of the default: a non-zero value here sized for the longest
// legitimate collective, or tt-run setting TT_METAL_OPERATION_TIMEOUT_SECONDS per rank from the job's shape.
// Row: test_teardown_sanity.py::test_hang_without_any_timeout.
// The singleton's private Open MPI session directory (see init_env) and the job's world size, kept at file scope so
// the abort paths (MPIContext::abort, std::terminate) can clean up and decide whether there is anybody to abort for.
static std::string g_singleton_session_dir;
static int g_world_size = 0;
static bool g_launched = false;  // started by a launcher (PRRTE/PMIx), so the rank leads its own process group

static void remove_singleton_session_dir() {
    if (!g_singleton_session_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(g_singleton_session_dir, ec);
        g_singleton_session_dir.clear();
    }
}

// Teardown plan P24: a helper this rank forked (a compile worker, a profiler capture) must not outlive the rank. The
// launcher kills a rank's whole process group on an abort but only reaps the rank on a clean exit, so a helper that
// ignores SIGTERM lived on; tt-run's session sweep found it on the launch host only, and turned the clean job into a
// failure. So the rank ends its own group itself, on the clean exit path and the abort path alike: every process in
// the group this rank leads (PRRTE gives each rank its own) and every descendant gets SIGTERM, a short grace, then
// SIGKILL. Only for a launched rank (g_launched); a plain pytest never runs it. The simulators (TT_METAL_SIMULATOR)
// load in-process, so there is no simulator child to end.
// What this rank left behind: every process in the group it leads (PRRTE gives each rank its own) and every
// descendant (a helper forked through a wrapper that did not join the group, or when the rank itself sits inside a
// wrapper's group). Read from /proc; the rank itself is never in the list.
static std::vector<pid_t> own_leftover_processes() {
    const pid_t self = ::getpid();
    std::vector<std::pair<pid_t, pid_t>> parent_of;  // (pid, ppid)
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator("/proc", ec)) {
        const std::string name = entry.path().filename().string();
        if (name.empty() || name[0] < '0' || name[0] > '9') {
            continue;
        }
        const pid_t pid = static_cast<pid_t>(std::atoi(name.c_str()));
        std::ifstream stat(entry.path() / "stat");
        std::string line;
        if (!std::getline(stat, line)) {
            continue;
        }
        const auto close_paren = line.rfind(')');  // the comm field may hold spaces; ppid is the 4th field after it
        if (close_paren == std::string::npos) {
            continue;
        }
        std::istringstream rest(line.substr(close_paren + 1));
        char state = 0;
        long ppid = 0;
        if (!(rest >> state >> ppid)) {
            continue;
        }
        parent_of.emplace_back(pid, static_cast<pid_t>(ppid));
    }
    std::vector<pid_t> out;
    auto listed = [&](pid_t pid) { return std::find(out.begin(), out.end(), pid) != out.end(); };
    const pid_t group = ::getpgrp();
    for (const auto& [pid, ppid] : parent_of) {
        if (pid != self && group == self && ::getpgid(pid) == self && !listed(pid)) {
            out.push_back(pid);
        }
    }
    bool grew = true;  // descendants: close the parent chain from this rank down
    while (grew) {
        grew = false;
        for (const auto& [pid, ppid] : parent_of) {
            if (pid != self && !listed(pid) && (ppid == self || listed(ppid))) {
                out.push_back(pid);
                grew = true;
            }
        }
    }
    return out;
}

// TODO(teardown P24, CI): on two CI heads the helper outlived the rank and tt-run's backstop ended it, while both
// P24 rows pass on the sandbox; the log line below is the first diagnostic from a CI rank (run 44219417923).
static void end_own_process_group() {
    auto leftovers = own_leftover_processes();
    if (leftovers.empty()) {
        return;
    }
    std::string pids;
    for (const pid_t pid : leftovers) {
        pids += (pids.empty() ? "" : " ") + std::to_string(pid);
        ::kill(pid, SIGTERM);
    }
    std::fprintf(
        stderr,
        "[MPIContext] ending %zu process(es) this rank left behind before it exits (pids %s)\n",
        leftovers.size(),
        pids.c_str());
    for (int i = 0; i < 100; ++i) {  // 2 s grace
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (own_leftover_processes().empty()) {
            break;
        }
    }
    const auto to_kill = own_leftover_processes();
    for (const pid_t pid : to_kill) {
        ::kill(pid, SIGKILL);
    }
    // SIGKILL is asynchronous: leave only once every target is gone from /proc, or is a zombie that is not ours to reap
    // (its own parent will). Without this wait the rank exits while the helper is still being torn down and the
    // launcher's census still sees it (seen in CI). Our own children are reaped on the way.
    const pid_t self = ::getpid();
    for (int i = 0; i < 250; ++i) {  // up to 5 s, normally one or two iterations
        bool pending = false;
        for (const pid_t pid : to_kill) {
            ::waitpid(pid, nullptr, WNOHANG);  // reaps a child of ours that has exited; ECHILD for a grandchild
            std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
            std::string line;
            if (!std::getline(stat, line)) {
                continue;  // gone
            }
            const auto close_paren = line.rfind(')');
            std::istringstream rest(close_paren == std::string::npos ? "" : line.substr(close_paren + 1));
            char state = 0;
            long ppid = 0;
            rest >> state >> ppid;
            if (state == 'Z' && static_cast<pid_t>(ppid) != self) {
                continue;  // exited; its parent reaps it
            }
            pending = true;
        }
        if (!pending) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    while (::waitpid(-1, nullptr, WNOHANG) > 0) {  // reap what was ours, so the launcher counts nothing left
    }
}

// Teardown plan P25: the launcher's session directory (prterun.<host>.<pid>.<uid>) is PRRTE's. prterun removes it on
// its exit path and tt-run sweeps the launch host's stale ones (#65121); a launcher killed on a REMOTE host leaves its
// directory where only the next job's ranks can see it. So every launched rank's context sweeps the stale launcher
// directories of its own host too. PMIx tells the rank where they live: PMIX_SYSTEM_TMPDIR is the temp root the
// launcher used, PMIX_SERVER_TMPDIR is this job's own directory (live, kept). The owner pid is in the name, and a
// directory whose owner is gone is stale. Reads only, after MPI_Init (no environment write, see P29 above).
static void sweep_stale_launcher_dirs() {
    const char* root = std::getenv("PMIX_SYSTEM_TMPDIR");
    const char* own = std::getenv("PMIX_SERVER_TMPDIR");
    if (root == nullptr) {
        return;
    }
    const std::string suffix = "." + std::to_string(::getuid());
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        const std::string name = entry.path().filename().string();
        if (!entry.is_directory(ec) || name.rfind("prterun.", 0) != 0 || name.size() <= suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        if (own != nullptr && std::string(own).rfind(entry.path().string(), 0) == 0) {
            continue;  // this job's own launcher
        }
        const std::string stem = name.substr(0, name.size() - suffix.size());
        const auto dot = stem.rfind('.');
        if (dot == std::string::npos) {
            continue;
        }
        char* end = nullptr;
        const long pid = std::strtol(stem.c_str() + dot + 1, &end, 10);
        if (pid <= 0 || end == nullptr || *end != '\0') {
            continue;
        }
        if (::kill(static_cast<pid_t>(pid), 0) == -1 && errno == ESRCH) {  // the launcher is gone: stale
            std::filesystem::remove_all(entry.path(), ec);
        }
    }
}

// Teardown plan P32: a singleton that was killed could not remove its directory; the next singleton init on this host
// removes the directories whose owner (the pid written into owner.pid at creation) no longer exists.
static void sweep_stale_singleton_dirs(const std::filesystem::path& base) {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
        if (!entry.is_directory(ec) || entry.path().filename().string().rfind("tt_mpi_singleton.", 0) != 0) {
            continue;
        }
        std::ifstream owner(entry.path() / "owner.pid");
        long pid = 0;
        if (!(owner >> pid) || pid <= 0) {
            continue;
        }
        if (::kill(static_cast<pid_t>(pid), 0) == -1 && errno == ESRCH) {  // a liveness probe, no signal is sent
            std::filesystem::remove_all(entry.path(), ec);
        }
    }
}

// TODO(teardown, timeout centralisation): read this through rtoptions (get_timeout_duration_for_operations) like
// dispatch and the topology mapper do, so SetOperationTimeout at runtime reaches the context too; same for the init
// timeout below (a new rtoptions entry). The env var stays the input.
static double operation_timeout_seconds() {
    const char* v = std::getenv("TT_METAL_OPERATION_TIMEOUT_SECONDS");
    if (v == nullptr || *v == '\0') {
        return 0.0;
    }
    char* end = nullptr;
    const double s = std::strtod(v, &end);
    return (end != v && s > 0.0) ? s : 0.0;
}

// Teardown plan P10: a deadline for MPI_Init itself. MPI_Init is a cross-rank wire-up — a rank blocks in it until
// every peer also calls MPI_Init — and it is not a DistributedContext collective, so the operation timeout cannot
// reach it: a peer that never starts (a dead rank, or a start-up skew past any sane bound) would hang the job at
// start-up. 0 (the default) keeps the unbounded wait, so a legitimately slow start is left alone; a positive value
// bounds the wait for a peer to join MPI_Init.
static double distributed_init_timeout_seconds() {
    const char* v = std::getenv("TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS");
    if (v == nullptr || *v == '\0') {
        return 0.0;
    }
    char* end = nullptr;
    const double s = std::strtod(v, &end);
    return (end != v && s > 0.0) ? s : 0.0;
}

// Drive a non-blocking request to completion by polling it, so a wait can always be left. A blocking MPI call never
// returns on a signal, so every wait in this context is one of these.
//  - SIGTERM (the job is stopping, see distributed_termination.cpp): the wait throws at once, no clock involved, and
//    the caller unwinds and closes its devices. A point-to-point request is cancelled; a collective's cannot be
//    (MPI forbids it) and is left to the process exit.
//  - TT_METAL_OPERATION_TIMEOUT_SECONDS (P16, off by default): the call cannot finish because a peer is wedged or
//    gone, and every other rank is still waiting on it, so this rank stops the whole job (the other ranks get
//    SIGTERM and close) and throws; MPI_Abort when the launcher cannot be asked.
static void wait_or_abort(
    MPI_Comm comm, MPI_Request& req, const char* what, bool point_to_point = false, MPI_Status* status = nullptr) {
    const double timeout = operation_timeout_seconds();  // 0: no deadline, the wait is still a poll for the flag below
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        int flag = 0;
        MPI_CHECK(MPI_Test(&req, &flag, status != nullptr ? status : MPI_STATUS_IGNORE));
        if (flag != 0) {
            return;
        }
        if (DistributedContext::is_terminating()) {
            int rank = 0;
            MPI_Comm_rank(comm, &rank);
            if (point_to_point) {
                MPI_Cancel(&req);
                MPI_Request_free(&req);
            }
            throw std::runtime_error(
                "[MPIContext] rank " + std::to_string(rank) + ": " + what + " abandoned, the job is stopping");
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (timeout > 0.0 && elapsed >= timeout) {
            if (point_to_point) {
                MPI_Cancel(&req);
                MPI_Request_free(&req);
            }
            const std::string reason = std::string(what) +
                                       " did not complete within TT_METAL_OPERATION_TIMEOUT_SECONDS=" +
                                       std::to_string(static_cast<int>(timeout)) + " s (a peer is wedged or gone)";
            if (!DistributedContext::request_job_stop(1, reason)) {
                MPI_Abort(comm, 1);
            }
            TT_THROW("MPIContext: {} timed out after {} s (a peer did not arrive)", what, timeout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

MPIDistributedException::MPIDistributedException(Rank rank, int error_code, std::string msg) :
    rank_(rank), error_code_(error_code), message_(std::move(msg)) {
    // retrieve human-readable MPI error string
    char buf[MPI_MAX_ERROR_STRING] = {0};
    int len = 0;
    MPI_Error_string(error_code_, buf, &len);
    error_string_.assign(buf, len);
    message_ += ": " + error_string_;
}

// implement interface
Rank MPIDistributedException::rank() const noexcept { return rank_; }

int MPIDistributedException::error_code() const noexcept { return error_code_; }

const std::string& MPIDistributedException::message() const noexcept { return message_; }

const std::string& MPIDistributedException::error_string() const noexcept { return error_string_; }

/* -------------------------- MPIRequest ---------------------------------- */

Status MPIRequest::wait() {
    MPI_Status status{};
    wait_or_abort(MPI_COMM_WORLD, req_, "wait", /*point_to_point=*/true, &status);
    done_ = true;

    int count = 0;
    MPI_CHECK(MPI_Get_count(&status, MPI_CHAR, &count));
    return Status{Rank(status.MPI_SOURCE), Tag(status.MPI_TAG), count};
}

std::optional<Status> MPIRequest::test() {
    MPI_Status status{};
    int flag = 0;
    MPI_CHECK(MPI_Test(&req_, &flag, &status));
    if (!flag) {
        // A caller that polls test() in its own loop (the control plane's exchanges do) leaves it the way a wait does
        // once the job is stopping.
        if (DistributedContext::is_terminating()) {
            MPI_Cancel(&req_);
            MPI_Request_free(&req_);
            throw std::runtime_error("[MPIContext] a pending request was abandoned, the job is stopping");
        }
        return std::nullopt;
    }

    done_ = true;
    int count = 0;
    MPI_CHECK(MPI_Get_count(&status, MPI_CHAR, &count));
    return Status{Rank(status.MPI_SOURCE), Tag(status.MPI_TAG), count};
}

void MPIRequest::cancel() {
    if (done_ || req_ == MPI_REQUEST_NULL) {
        return;
    }
    MPI_CHECK(MPI_Cancel(&req_));
    MPI_CHECK(MPI_Request_free(&req_));
    done_ = true;
}

bool MPIRequest::active() const { return !done_; }

/* -------------------------- MPIContext ---------------------------------- */

inline void init_env(int& argc, char**& argv) {
    static std::once_flag mpi_once;
    static int provided_thread_level = MPI_THREAD_SINGLE;
    std::call_once(mpi_once, [&] {
        int provided = 0;

        // A process nobody launched (no PMIx rank from mpirun / tt-run) is a singleton: it has no peers, so it has
        // no use for the shared-memory transport, and Open MPI never unlinks a singleton's 16 MB segment
        // (tt-metal #63870). Ask for the self transport only; an explicit OMPI_MCA_btl in the environment wins.
        bool singleton_tmpdir_redirected = false;
        bool had_caller_tmpdir = false;
        std::string caller_tmpdir;
        // Teardown plan P29: the TCP-only transport line (btl self,tcp) stays tt-run's launch-line policy. It is not
        // set from here for launched ranks: every environment write before MPI_Init races the tracy profiler's
        // worker thread reading the environment (a segfault in getenv seen in CI), and the launch line wins anyway.
        // test_bare_launch_dead_rank_leaves_no_shm guards the outcome.
        const bool launched = std::getenv("PMIX_RANK") != nullptr || std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr;
        if (!launched) {
            setenv("OMPI_MCA_btl", "self", /*overwrite=*/0);
            // Open MPI also leaves a singleton's session directory (/tmp/ompi.<host>.<uid>/...) behind — the
            // launcher that would reap it never ran (tt-metal #63870, comprehensively fixed by #63954). Steer that
            // session dir into a private directory we own by pointing TMPDIR at it just for MPI_Init, then remove
            // the directory at exit. The caller's TMPDIR is restored immediately after init so nothing else moves.
            const char* base_env = std::getenv("TMPDIR");
            const std::string base = (base_env != nullptr && base_env[0] != '\0') ? base_env : "/tmp";
            sweep_stale_singleton_dirs(base);  // P32: what a killed singleton could not remove
            std::string tmpl = base + "/tt_mpi_singleton.XXXXXX";
            std::vector<char> buf(tmpl.c_str(), tmpl.c_str() + tmpl.size() + 1);
            if (mkdtemp(buf.data()) != nullptr) {
                g_singleton_session_dir = buf.data();
                std::ofstream(std::filesystem::path(g_singleton_session_dir) / "owner.pid") << ::getpid() << "\n";
                had_caller_tmpdir = (base_env != nullptr);
                if (had_caller_tmpdir) {
                    caller_tmpdir = base_env;
                }
                setenv("TMPDIR", g_singleton_session_dir.c_str(), /*overwrite=*/1);
                singleton_tmpdir_redirected = true;
                // Runs after the MPI_Finalize atexit below (atexit is LIFO), so the session dir is empty by then.
                // The abort paths (P31) and the next init (P32) remove it too; #63954 makes all of this moot by not
                // initialising MPI for a process no launcher started.
                std::atexit(remove_singleton_session_dir);
            }
        }
        // Teardown plan P10: bound MPI_Init so a peer that never joins the wire-up cannot hang the job at start-up.
        // MPI is not yet initialized, so we cannot MPI_Abort from here; instead a watchdog thread _Exit()s this rank
        // when the deadline passes (the launcher then ends every other rank). Disabled by default (timeout 0), so a
        // legitimately slow start is untouched; the watchdog is cancelled the moment MPI_Init returns.
        const double init_timeout = distributed_init_timeout_seconds();
        std::atomic<bool> init_complete{false};
        std::thread init_watchdog;
        if (init_timeout > 0.0) {
            init_watchdog = std::thread([init_timeout, &init_complete] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(init_timeout);
                while (std::chrono::steady_clock::now() < deadline) {
                    if (init_complete.load(std::memory_order_acquire)) {
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (!init_complete.load(std::memory_order_acquire)) {
                    std::fprintf(
                        stderr,
                        "[MPIContext] MPI_Init did not complete within TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS=%.0f "
                        "s; a peer never joined the wire-up. Exiting so the launcher ends the job.\n",
                        init_timeout);
                    std::_Exit(EXIT_FAILURE);
                }
            });
        }
        const int init_rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        init_complete.store(true, std::memory_order_release);
        if (init_watchdog.joinable()) {
            init_watchdog.join();
        }
        if (init_rc != MPI_SUCCESS) {
            TT_THROW("MPI_Init_thread failed");
        }
        provided_thread_level = provided;

        // The TMPDIR redirect was only needed to place OpenMPI's session dir at init; give the caller theirs back.
        if (singleton_tmpdir_redirected) {
            if (had_caller_tmpdir) {
                setenv("TMPDIR", caller_tmpdir.c_str(), /*overwrite=*/1);
            } else {
                unsetenv("TMPDIR");
            }
        }

        g_launched = launched;
        if (launched) {
            // P25: one rank per host is enough; the local rank is PMIx/Open MPI's, 0 when absent.
            const char* local_rank = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
            if (local_rank == nullptr || std::string(local_rank) == "0") {
                sweep_stale_launcher_dirs();
            }
        }

        // The exit of the process. on_exit, not atexit: it sees the exit code, so a rank that exits non-zero (a
        // C++ exit(1), main returning 1, a Python sys.exit(3) or uncaught exception, which end in exit too) stops
        // the job before it leaves: the other ranks get SIGTERM, close their devices, and every rank leaves
        // together (DistributedContext::leave_stopped_job). A rank whose job is stopping skips MPI_Finalize, which
        // fences on every rank and would hang on peers still in their collectives (P1). A clean exit finalizes.
        // Teardown plan P27: a rank that exits 0 while its peers are still in a collective waits in MPI_Finalize
        // until the job ends; that fence is not bounded, since a rank cannot tell a peer that is still working
        // from one that is stuck. P24: the helpers this rank forked are ended first, on every path.
        ::on_exit(
            [](int status, void*) {
                if (g_launched) {
                    end_own_process_group();
                }
                if (g_launched && g_world_size > 1) {
                    if (status != 0 && !DistributedContext::is_terminating() &&
                        !DistributedContext::request_job_stop(status, "exited with code " + std::to_string(status))) {
                        MPI_Abort(MPI_COMM_WORLD, status);  // no launcher to ask: the old way, every rank killed
                    }
                    if (DistributedContext::is_terminating()) {
                        DistributedContext::leave_stopped_job();
                    }
                }
                MPI_Finalize();
            },
            nullptr);

        // Teardown plan P1: a C++ rank that leaves through an uncaught exception (a TT_FATAL becomes one) must not
        // let its peers hang in a barrier: it stops the job and leaves with them (exit 1, or 128 + SIGTERM when the
        // exception is a wait the stop abandoned). MPI_Abort only when the launcher cannot be asked.
        // Teardown plan P8: never std::abort(): SIGABRT writes a core dump, and the whole job would wait for it.
        std::set_terminate([] {
            std::string what = "unknown exception";
            if (auto current = std::current_exception()) {
                try {
                    std::rethrow_exception(current);
                } catch (const std::exception& e) {
                    what = e.what();
                } catch (...) {
                }
            }
            std::fprintf(stderr, "[MPIContext] std::terminate: uncaught exception: %s\n", what.c_str());
            int inited = 0;
            int finalized = 0;
            MPI_Initialized(&inited);
            MPI_Finalized(&finalized);
            if (inited != 0 && finalized == 0 && g_world_size > 1) {
                if (g_launched) {
                    end_own_process_group();  // P24, the abort path
                }
                if (g_launched && (DistributedContext::is_terminating() ||
                                   DistributedContext::request_job_stop(1, "uncaught C++ exception: " + what))) {
                    DistributedContext::leave_stopped_job();
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            remove_singleton_session_dir();  // P31: a world of one has nobody to abort for; leave nothing behind
            std::_Exit(EXIT_FAILURE);
        });

        // Teardown plan P20: an error with no communicator (a peer's socket closing across hosts) is consulted on
        // MPI_COMM_SELF, which is MPI_ERRORS_ARE_FATAL by default. Make it return so the survivor sees the failure
        // and can close its device, instead of the process aborting under the fatal handler.
        MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN);
    });

    // Validate AFTER the once-block completes: MPI is initialized exactly once and the finalizer is
    // registered even when we reject the runtime. The Python bindings release the GIL around
    // barrier / send_bytes / recv_bytes and the host-socket calls, so several Python threads may be
    // inside MPI at once (DFlash relay + collective teardown). That is only legal at
    // MPI_THREAD_MULTIPLE; MPI_Init_thread may succeed while granting a weaker level.
    TT_FATAL(
        provided_thread_level >= MPI_THREAD_MULTIPLE,
        "MPI runtime provided thread level {} but MPI_THREAD_MULTIPLE ({}) is required for concurrent "
        "host-socket / barrier calls from multiple Python threads",
        provided_thread_level,
        static_cast<int>(MPI_THREAD_MULTIPLE));
}

void MPIContext::create(int argc, char** argv) {
    if (current_world_) {
        return;
    }
    init_env(argc, argv);

    ContextPtr parent = std::make_shared<MPIContext>(MPI_COMM_WORLD)->duplicate();
    const int job_world_size = static_cast<int>(*parent->size());
    g_world_size = job_world_size;
    if (g_launched && job_world_size > 1) {
        // A launched rank of a real job leaves on SIGTERM through its waits and the exit above, and stops the job
        // itself when it fails (distributed_termination.cpp). The watchdog's last resort is MPI_Abort.
        job_stop::set_force_abort([](int code) { MPI_Abort(MPI_COMM_WORLD, code); });
        DistributedContext::install_termination_handler();
        job_stop::tell_rank_wrapper("context");
    }

    g_mpi_launcher_env_layout = parse_launcher_env_layout_from_env(job_world_size);

    if (g_mpi_launcher_env_layout.this_subcontext_id.has_value()) {
        const int color = *g_mpi_launcher_env_layout.this_subcontext_id;
        const auto mpi_parent = std::dynamic_pointer_cast<MPIContext>(parent);
        TT_FATAL(mpi_parent != nullptr, "MPIContext::create: parent must be MPIContext");
        int parent_rank = 0;
        MPI_CHECK(MPI_Comm_rank(mpi_parent->comm(), &parent_rank));
        // Key = rank on parent comm so sub-context ranks follow global (parent) order within each color.
        current_world_ = mpi_parent->split(Color(color), Key(parent_rank));
        TT_FATAL(
            static_cast<int>(*current_world_->size()) ==
                g_mpi_launcher_env_layout.subcontext_sizes[static_cast<std::size_t>(color)],
            "MPI_Comm_split size {} does not match TT_RUN_SUBCONTEXT_SIZES[{}]={}",
            *current_world_->size(),
            color,
            g_mpi_launcher_env_layout.subcontext_sizes[static_cast<std::size_t>(color)]);
    } else {
        current_world_ = std::move(parent);
    }

    if (!mpi_job_world_) {
        mpi_job_world_ = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    }
}

const ContextPtr& MPIContext::get_current_world() {
    if (!current_world_) {
        // Default initialization of MPIContext if not already initialized
        MPIContext::create(0, nullptr);
    }
    return current_world_;
}

ContextPtr MPIContext::get_world_context() {
    if (!mpi_job_world_) {
        if (!current_world_) {
            MPIContext::create(0, nullptr);
        }
        mpi_job_world_ = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    }
    return mpi_job_world_;
}

std::optional<SubcontextId> MPIContext::subcontext_id() const {
    const auto& id = g_mpi_launcher_env_layout.this_subcontext_id;
    if (!id.has_value()) {
        return std::nullopt;
    }
    return SubcontextId{*id};
}

int MPIContext::subcontext_count() const { return static_cast<int>(g_mpi_launcher_env_layout.subcontext_sizes.size()); }

Size MPIContext::subcontext_size(SubcontextId subcontext_id) const {
    const auto& L = g_mpi_launcher_env_layout;
    TT_FATAL(
        *subcontext_id >= 0 && *subcontext_id < static_cast<int>(L.subcontext_sizes.size()),
        "subcontext_id {} out of range [0, {})",
        *subcontext_id,
        L.subcontext_sizes.size());
    return Size(L.subcontext_sizes[*subcontext_id]);
}

ttsl::Span<const int> MPIContext::subcontext_sizes() const {
    const auto& L = g_mpi_launcher_env_layout;
    return {L.subcontext_sizes.data(), L.subcontext_sizes.size()};
}

Rank MPIContext::local_to_world_rank(SubcontextId subcontext_id, Rank local_rank) const {
    const auto& L = g_mpi_launcher_env_layout;
    TT_FATAL(
        *subcontext_id >= 0 && *subcontext_id < static_cast<int>(L.subcontext_sizes.size()),
        "subcontext_id {} out of range [0, {})",
        *subcontext_id,
        L.subcontext_sizes.size());
    TT_FATAL(
        *local_rank >= 0 && *local_rank < L.subcontext_sizes[*subcontext_id],
        "local_rank {} out of range for sub-context {} (size {})",
        *local_rank,
        *subcontext_id,
        L.subcontext_sizes[*subcontext_id]);
    // World rank = start of this sub-context's block + offset within that block.
    return Rank{L.world_rank_prefix[*subcontext_id] + *local_rank};
}

void MPIContext::set_current_world(const ContextPtr& ctx) {
    TT_FATAL(
        ctx != nullptr && std::dynamic_pointer_cast<MPIContext>(ctx) != nullptr,
        "MPIContext::set_current_world: context is not a MPIContext or a nullptr");
    MPIContext::current_world_ = ctx;
}

bool MPIContext::is_initialized() {
    int is_mpi_initialized;
    MPI_CHECK(MPI_Initialized(&is_mpi_initialized));
    return is_mpi_initialized != 0;
}

MPIContext::MPIContext(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);  // don't abort on error
    // Cache the comm's process group for MPI_Group_incl / MPI_Group_translate_ranks (create_sub_context,
    // translate_ranks_to_other_ctx). Without a stored group those paths would need ad-hoc Comm_group + free
    // at every call. Omitting MPI_Group_free in ~MPIContext leaks one MPI_Group per wrapper instance.
    MPI_CHECK(MPI_Comm_group(comm_, &group_));
    MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
    MPI_CHECK(MPI_Comm_size(comm_, &size_));
    id_ = DistributedContext::generate_unique_id();
}

MPIContext::MPIContext(MPI_Comm comm, MPI_Group group) : comm_(comm), group_(group) {
    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);  // don't abort on error
    MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
    MPI_CHECK(MPI_Comm_size(comm_, &size_));
    id_ = DistributedContext::generate_unique_id();
}

Rank MPIContext::rank() const { return Rank(rank_); }
Size MPIContext::size() const { return Size(size_); }
bool MPIContext::supports_fault_tolerance() const { return OMPI_HAS_ULFM; }
void MPIContext::barrier() const {
    MPI_Request req{};
    MPI_CHECK(MPI_Ibarrier(comm_, &req));
    wait_or_abort(comm_, req, "barrier");
}

/* ---- point‑to‑point ---------------------------------------------------- */

void MPIContext::send(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Isend(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_, &req));
    wait_or_abort(comm_, req, "send");
}

void MPIContext::ssend(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Issend(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_, &req));
    wait_or_abort(comm_, req, "ssend", /*point_to_point=*/true);
}

void MPIContext::recv(ttsl::Span<std::byte> buf, Rank src, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Irecv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, &req));
    wait_or_abort(comm_, req, "recv");
}

RequestPtr MPIContext::isend(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Isend(
        const_cast<std::byte*>(buf.data()), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_, &req));
    return std::make_shared<MPIRequest>(req);
}

RequestPtr MPIContext::irecv(ttsl::Span<std::byte> buf, Rank src, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Irecv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, &req));
    return std::make_shared<MPIRequest>(req);
}

/* ---- collectives ------------------------------------------------------- */

void MPIContext::broadcast(ttsl::Span<std::byte> buf, Rank root) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Ibcast(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *root, comm_, &req));
    wait_or_abort(comm_, req, "broadcast");
}

void MPIContext::all_reduce(
    ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    check_size_fits_int(send_buf.size());

    TT_FATAL(
        send_buf.size() == recv_buf.size(),
        "all_reduce: send/recv sizes differ ({} vs {})",
        send_buf.size(),
        recv_buf.size());

    const int elem_size = static_cast<int>(dtype_size(dtype));  // e.g. 4 for FLOAT32
    TT_FATAL(
        send_buf.size() % elem_size == 0,
        "all_reduce: buffer size {} is not a multiple of element size {}",
        send_buf.size(),
        elem_size);

    const int count = static_cast<int>(send_buf.size() / elem_size);

    // allow in‑place (send == recv) by switching to MPI_IN_PLACE
    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : static_cast<void*>(send_buf.data());

    MPI_Request req{};
    MPI_CHECK(MPI_Iallreduce(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), comm_, &req));
    wait_or_abort(comm_, req, "all_reduce");
}

void MPIContext::reduce(
    ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype, Rank root) const {
    const int elem_sz = static_cast<int>(dtype_size(dtype));
    TT_FATAL(
        send_buf.size() % elem_sz == 0,
        "reduce: send size {} not multiple of element size {}",
        send_buf.size(),
        elem_sz);

    const int count = static_cast<int>(send_buf.size() / elem_sz);
    check_size_fits_int(count);

    // On non‑root ranks 'recv_buf' can be any pointer (or even nullptr).  On root it must fit.
    if (rank() == root) {
        TT_FATAL(
            recv_buf.size() == send_buf.size(),
            "reduce: on root rank, recv size {} != send size {}",
            recv_buf.size(),
            send_buf.size());
    }

    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : send_buf.data();

    MPI_Request req{};
    MPI_CHECK(
        MPI_Ireduce(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), *root, comm_, &req));
    wait_or_abort(comm_, req, "reduce");
}

void MPIContext::gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, Rank root) const {
    const int send_count = static_cast<int>(send_buf.size());
    check_size_fits_int(send_count);

    // Root must have room for 'size()' times the per‑rank payload.
    if (rank() == root) {
        const std::size_t expected = static_cast<std::size_t>(send_count) * (*size());
        TT_FATAL(
            recv_buf.size() == expected, "gather: root recv buffer {} bytes, expected {}", recv_buf.size(), expected);
    }

    MPI_Request req{};
    MPI_CHECK(
        MPI_Igather(send_buf.data(), send_count, MPI_CHAR, recv_buf.data(), send_count, MPI_CHAR, *root, comm_, &req));
    wait_or_abort(comm_, req, "gather");
}

void MPIContext::scatter(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, Rank root) const {
    const int recv_count = static_cast<int>(recv_buf.size());
    check_size_fits_int(recv_count);

    if (rank() == root) {
        const std::size_t expected = static_cast<std::size_t>(recv_count) * (*size());
        TT_FATAL(
            send_buf.size() == expected, "scatter: root send buffer {} bytes, expected {}", send_buf.size(), expected);
    }

    MPI_Request req{};
    MPI_CHECK(
        MPI_Iscatter(send_buf.data(), recv_count, MPI_CHAR, recv_buf.data(), recv_count, MPI_CHAR, *root, comm_, &req));
    wait_or_abort(comm_, req, "scatter");
}

void MPIContext::all_gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const {
    const int send_count = static_cast<int>(send_buf.size());
    check_size_fits_int(send_count);

    const std::size_t expected_recv = static_cast<std::size_t>(send_count) * (*size());

    TT_FATAL(
        recv_buf.size() == expected_recv,
        "all_gather: recv buffer {} bytes, expected {} (world × send)",
        recv_buf.size(),
        expected_recv);

    // allow MPI_IN_PLACE if caller wants to receive in the same buffer
    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : send_buf.data();

    MPI_Request req{};
    MPI_CHECK(MPI_Iallgather(send_ptr, send_count, MPI_CHAR, recv_buf.data(), send_count, MPI_CHAR, comm_, &req));
    wait_or_abort(comm_, req, "all_gather");
}

void MPIContext::all_to_all(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const {
    const int world = *size();

    TT_FATAL(
        send_buf.size() % world == 0 && recv_buf.size() % world == 0,
        "all_to_all: send ({}) or recv ({}) size not divisible by world size {}",
        send_buf.size(),
        recv_buf.size(),
        world);

    TT_FATAL(
        send_buf.size() == recv_buf.size(),
        "all_to_all: send size {} != recv size {}",
        send_buf.size(),
        recv_buf.size());

    const int block = static_cast<int>(send_buf.size() / world);
    check_size_fits_int(block);

    MPI_Request req{};
    MPI_CHECK(MPI_Ialltoall(send_buf.data(), block, MPI_CHAR, recv_buf.data(), block, MPI_CHAR, comm_, &req));
    wait_or_abort(comm_, req, "all_to_all");
}

void MPIContext::reduce_scatter(
    ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    const int world = *size();
    const int elem_sz = static_cast<int>(dtype_size(dtype));

    TT_FATAL(
        send_buf.size() % elem_sz == 0,
        "reduce_scatter: send size {} is not multiple of element size {}",
        send_buf.size(),
        elem_sz);

    const std::size_t total_elems = send_buf.size() / elem_sz;
    TT_FATAL(
        total_elems % world == 0,
        "reduce_scatter: element count {} not divisible by world size {}",
        total_elems,
        world);

    const std::size_t recv_elems = total_elems / world;
    const std::size_t expected_recv_bytes = recv_elems * elem_sz;

    TT_FATAL(
        recv_buf.size() == expected_recv_bytes,
        "reduce_scatter: recv size {} != expected {}",
        recv_buf.size(),
        expected_recv_bytes);

    const int recv_count = static_cast<int>(recv_elems);  // per-rank element count
    check_size_fits_int(recv_count);

    MPI_Request req{};
    MPI_CHECK(MPI_Ireduce_scatter_block(
        send_buf.data(),      // sendbuf
        recv_buf.data(),      // recvbuf
        recv_count,           // elements per rank
        dtype_to_mpi(dtype),  // element datatype
        reduce_to_mpi(op),    // operation (SUM, MAX, …)
        comm_,                // communicator
        &req));
    wait_or_abort(comm_, req, "reduce_scatter");
}

void MPIContext::scan(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    TT_FATAL(
        send_buf.size() == recv_buf.size(), "scan: send size {} != recv size {}", send_buf.size(), recv_buf.size());

    const int elem_sz = static_cast<int>(dtype_size(dtype));
    TT_FATAL(
        send_buf.size() % elem_sz == 0,
        "scan: buffer size {} not multiple of element size {}",
        send_buf.size(),
        elem_sz);

    const int count = static_cast<int>(send_buf.size() / elem_sz);
    check_size_fits_int(count);

    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : send_buf.data();

    MPI_Request req{};
    MPI_CHECK(MPI_Iscan(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), comm_, &req));
    wait_or_abort(comm_, req, "scan");
}
/* ---- communicator management ------------------------------------------ */

ContextPtr MPIContext::duplicate() const {
    MPI_Comm dup = MPI_COMM_NULL;
    MPI_Request req{};
    MPI_CHECK(MPI_Comm_idup(comm_, &dup, &req));
    wait_or_abort(comm_, req, "duplicate");
    return std::make_shared<MPIContext>(dup);
}

ContextPtr MPIContext::split(Color color, Key key) const {
    MPI_Comm split_comm;
    if (*color == SPLIT_COLOR_UNDEFINED) {
        color = Color(MPI_UNDEFINED);
    }
    // MPI has no non-blocking split, so this is the one wait in the context a SIGTERM cannot end; it runs only while
    // contexts are being set up, when every rank is between the same two calls.
    MPI_CHECK(MPI_Comm_split(comm_, *color, *key, &split_comm));
    return std::make_shared<MPIContext>(split_comm);
}

ContextPtr MPIContext::create_sub_context(ttsl::Span<int> ranks) const {
    MPI_Group sub_grp = MPI_GROUP_NULL;
    MPI_Comm sub_comm = MPI_COMM_NULL;

    MPI_CHECK(MPI_Group_incl(group_, static_cast<int>(ranks.size()), ranks.data(), &sub_grp));
    if (MPI_Comm_create_group(comm_, sub_grp, 0 /*tag*/, &sub_comm) != MPI_SUCCESS) {
        // sub_comm is not valid, we are not in the group
        MPI_Group_free(&sub_grp);
        throw MPIDistributedException(rank(), MPI_ERR_GROUP, "MPI_Comm_create_group failed: not in the group");
    }
    if (sub_comm == MPI_COMM_NULL) {
        // we are not in the group
        MPI_Group_free(&sub_grp);
        throw MPIDistributedException(rank(), MPI_ERR_GROUP, "MPI_Comm_create_group returned empty comm");
    }
    MPI_Group_free(&sub_grp);
    return std::make_shared<MPIContext>(sub_comm);
}

void MPIContext::translate_ranks_to_other_ctx(
    ttsl::Span<int> ranks, const ContextPtr& other_ctx, ttsl::Span<int> translated_ranks) const {
    TT_FATAL(
        ranks.size() == translated_ranks.size(),
        "translate_ranks_to_other_ctx: ranks size {} != translated_ranks size {}",
        ranks.size(),
        translated_ranks.size());
    auto mpi_context = std::dynamic_pointer_cast<MPIContext>(other_ctx);
    TT_FATAL(mpi_context != nullptr, "translate_ranks_to_other_ctx: other_ctx is not a MPIContext");

    MPI_CHECK(MPI_Group_translate_ranks(
        group_, static_cast<int>(ranks.size()), ranks.data(), mpi_context->group(), translated_ranks.data()));
}

void MPIContext::abort(int error_code) const {
    if (g_launched) {
        end_own_process_group();  // P24: the abort path ends this rank's helpers too
    }
    if (size_ <= 1) {
        // Teardown plan P31: a singleton has no peers to end; an MPI_Abort here would only skip the atexit that removes
        // its private session directory. Exit normally with the code instead, so MPI_Finalize and the cleanup run.
        std::exit(error_code);
    }
    // Stop the job rather than kill it: the other ranks get SIGTERM and close their devices, then every rank leaves
    // together with this code. MPI_Abort (every rank killed where it stands) only when the launcher cannot be asked.
    if (g_launched && DistributedContext::request_job_stop(error_code, "abort(" + std::to_string(error_code) + ")")) {
        DistributedContext::leave_stopped_job();
    }
    MPI_Abort(comm_, error_code);
}

void MPIContext::revoke_and_shrink() {
#if (!OMPI_HAS_ULFM)
    TT_THROW("revoke_and_shrink() requires MPI ULFM support which is not available in this build");
#else
    int rc = MPIX_Comm_revoke(comm_);
    if (rc != MPI_SUCCESS && rc != MPI_ERR_REVOKED) {  // another rank may have revoked first
        abort(rc);
    }

    MPI_Comm new_comm = MPI_COMM_NULL;
    MPI_Group new_group = MPI_GROUP_NULL;
    MPI_CHECK(MPIX_Comm_shrink(comm_, &new_comm));
    MPI_Comm_group(new_comm, &new_group);

    MPI_Comm_set_errhandler(new_comm, MPI_ERRORS_RETURN);

    // overall probably I don't need MPI_CHECK, we are recovering here. If we cannot recover, we should abort
    // and not throw an exception.
    int new_rank = 0;
    int new_size = 0;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    // Free the old communicator *after* shrink completes
    MPI_Comm old_comm = this->comm_;

    if (old_comm != MPI_COMM_NULL && old_comm != new_comm) {
        MPI_Comm_free(&old_comm);
        MPI_Group_free(&group_);
    }
    this->comm_ = new_comm;
    this->group_ = new_group;
    this->rank_ = new_rank;
    this->size_ = new_size;
#endif
}

bool MPIContext::is_revoked() {
#if (!OMPI_HAS_ULFM)
    TT_THROW("is_revoked() requires MPI ULFM support which is not available in this build");
#else
    int flag = 0;
    // MPI_Comm_test_inter is safe to call even if the communicator is revoked
    // don't need to check error code
    MPI_Comm_test_inter(comm_, &flag);
    return flag != 0;
#endif
}

std::size_t MPIContext::snoop_incoming_msg_size(Rank source, Tag tag) const {
    int size_bytes = 0;
    MPI_Status status;
    // A polled probe, for the same reason as wait_or_abort: a blocking MPI_Probe never returns on a signal.
    for (int found = 0; found == 0;) {
        MPI_CHECK(MPI_Iprobe(*source, *tag, comm_, &found, &status));
        if (found == 0) {
            if (DistributedContext::is_terminating()) {
                throw std::runtime_error(
                    "[MPIContext] rank " + std::to_string(rank_) + ": probe abandoned, the job is stopping");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    MPI_CHECK(MPI_Get_count(&status, MPI_CHAR, &size_bytes));
    return static_cast<std::size_t>(size_bytes);
}

MPIContext::~MPIContext() {
    if (was_mpi_finalized()) {
        return;  // MPI_Finalize() already called
    }
    // comm_ vs group_: different object types. MPI_COMM_WORLD is predefined at MPI_Init; MPI_Finalize tears
    // down the whole runtime — never MPI_Comm_free(MPI_COMM_WORLD). MPI_Comm_group(comm_, &group_) allocates
    // a separate MPI_Group (same process set as comm_) that we own and must MPI_Group_free — not the comm.
    // group_ is kept for APIs that take MPI_Group (e.g. Group_incl, translate_ranks).
    if (group_ != MPI_GROUP_NULL) {
        MPI_Group_free(&group_);
        group_ = MPI_GROUP_NULL;
    }
    // Omitting Group_free above (or dropping group_ without redesign) leaks one MPI_Group per context — handle
    // growth and possible resource exhaustion in long-lived or many-context workloads.
    // Only Comm_dup / Comm_split / … handles; never MPI_COMM_WORLD or MPI_COMM_NULL.
    if (comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&comm_);
        comm_ = MPI_COMM_NULL;
    }
}
}  // namespace tt::tt_metal::distributed::multihost
