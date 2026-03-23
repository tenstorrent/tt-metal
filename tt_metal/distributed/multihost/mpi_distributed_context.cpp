// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_distributed_context.hpp"
#include <mpi.h>
#include <mpi-ext.h>

#include <algorithm>
#include <charconv>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <string_view>
#include <unistd.h>
#include <vector>
#include <fmt/format.h>
#include <tt_stl/assert.hpp>

// Use MPIX_ERR_PROC_FAILED as a proxy to detect whether OpenMPI was built with
// ULFM extensions.
#if (defined(OPEN_MPI) && OPEN_MPI && defined(MPIX_ERR_PROC_FAILED))
#define OMPI_HAS_ULFM 1
#else
#define OMPI_HAS_ULFM 0
#endif

namespace tt::tt_metal::distributed::multihost {

// Exit code used by all ULFM fast-fail paths.  Matches EX_SOFTWARE (70) from
// <sysexits.h> and is recognised by ttrun.py to emit a targeted diagnostic
// rather than a generic "non-zero exit" message.
static constexpr int kUlfmExitCode = 70;

/* ----------------------------- helpers ---------------------------------- */

[[nodiscard]] constexpr MPI_Op reduce_to_mpi(ReduceOp op) noexcept {
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

[[nodiscard]] constexpr MPI_Datatype dtype_to_mpi(DType dt) noexcept {
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

[[nodiscard]] constexpr int mpi_dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::INT8:
        case DType::UINT8:
        case DType::BOOL:
        case DType::BYTE: return 1;
        case DType::INT16:
        case DType::UINT16: return 2;
        case DType::INT32:
        case DType::UINT32:
        case DType::FLOAT32: return 4;
        case DType::INT64:
        case DType::UINT64:
        case DType::FLOAT64:
        case DType::COMPLEX_FLOAT: return 8;
        case DType::COMPLEX_DOUBLE: return 16;
    }
    return 0;
}

inline void check_size_fits_int(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TT_THROW("MPI buffer size > INT_MAX");
    }
}

// ---------------------------------------------------------------------------
// ULFM helpers: fast-fail with diagnostics when a remote rank dies.
// ---------------------------------------------------------------------------

#if OMPI_HAS_ULFM

[[nodiscard]] inline bool is_ulfm_failure(int rc) noexcept {
    return rc == MPIX_ERR_PROC_FAILED || rc == MPIX_ERR_PROC_FAILED_PENDING || rc == MPIX_ERR_REVOKED;
}

// Identify which world-ranks have failed. Best-effort: returns "unknown" on
// any secondary failure (comm may already be torn down).
// RAII owner for an MPI_Group handle.  Frees the group on destruction if it
// has been populated (i.e. is not MPI_GROUP_NULL).
struct MpiGroupGuard {
    MPI_Group g{MPI_GROUP_NULL};
    MpiGroupGuard() = default;
    explicit MpiGroupGuard(MPI_Group grp) : g(grp) {}
    ~MpiGroupGuard() {
        if (g != MPI_GROUP_NULL) {
            MPI_Group_free(&g);
        }
    }
    MpiGroupGuard(const MpiGroupGuard&) = delete;
    MpiGroupGuard& operator=(const MpiGroupGuard&) = delete;
    MpiGroupGuard(MpiGroupGuard&& o) noexcept : g(o.g) { o.g = MPI_GROUP_NULL; }
    MpiGroupGuard& operator=(MpiGroupGuard&& o) noexcept {
        if (this != &o) {
            if (g != MPI_GROUP_NULL) {
                MPI_Group_free(&g);
            }
            g = o.g;
            o.g = MPI_GROUP_NULL;
        }
        return *this;
    }
};

static std::string identify_failed_ranks(MPI_Comm comm) {
    // Try to ack pending failures. On a revoked communicator this may return
    // MPIX_ERR_REVOKED; in that case we still try get_acked below because
    // prior acks from earlier operations may have recorded failure information
    // that we can retrieve.
    MPIX_Comm_failure_ack(comm);  // best-effort; ignore return code
    MpiGroupGuard failed_guard;
    if (MPIX_Comm_failure_get_acked(comm, &failed_guard.g) != MPI_SUCCESS) {
        return "unknown";
    }
    int failed_size = 0;
    MPI_Group_size(failed_guard.g, &failed_size);
    if (failed_size == 0) {
        return "unknown (no acked failures)";
    }
    MpiGroupGuard world_guard;
    MPI_Comm_group(comm, &world_guard.g);
    std::vector<int> local_indices(failed_size);
    std::iota(local_indices.begin(), local_indices.end(), 0);
    std::vector<int> world_ranks(failed_size, MPI_UNDEFINED);
    if (world_guard.g != MPI_GROUP_NULL) {
        MPI_Group_translate_ranks(failed_guard.g, failed_size, local_indices.data(), world_guard.g, world_ranks.data());
    }

    fmt::memory_buffer buf;
    for (int i = 0; i < failed_size; ++i) {
        if (i > 0) {
            fmt::format_to(std::back_inserter(buf), ", ");
        }
        if (world_ranks[i] == MPI_UNDEFINED) {
            fmt::format_to(std::back_inserter(buf), "?");
        } else {
            fmt::format_to(std::back_inserter(buf), "{}", world_ranks[i]);
        }
    }
    return fmt::to_string(buf);
}

// Handle a detected rank failure according to the active FailurePolicy.
//
// FAST_FAIL mode: revoke communicator, print diagnostic, call _exit(kUlfmExitCode).
//   _exit() is used instead of exit()/throw to avoid MPI_Finalize() deadlock:
//   once a communicator is revoked, MPI_Finalize() blocks indefinitely because
//   it tries to synchronise with ranks that no longer exist.  _exit() bypasses
//   all atexit handlers (including the one that calls MPI_Finalize).
//
// FAULT_TOLERANT mode: revoke communicator, print diagnostic, throw
//   MPIRankFailureException.  Caller contract: the catcher MUST call
//   revoke_and_shrink() before making any further MPI calls on this
//   communicator — the old comm is revoked and unusable.
// Parse a comma-separated list of rank numbers (e.g. "1, 3") into a vector.
// Returns empty if the string is "unknown", "unknown (no acked failures)", or unparseable.
static std::vector<Rank> parse_failed_ranks_string(std::string_view s) {
    std::vector<Rank> result;
    if (s.empty() || s.find("unknown") != std::string_view::npos) {
        return result;
    }

    auto take_next_comma_separated_field = [](std::string_view& remainder) -> std::string_view {
        const auto comma = remainder.find(',');
        const auto field = remainder.substr(0, comma);
        remainder = (comma == std::string_view::npos) ? std::string_view{} : remainder.substr(comma + 1);
        return field;
    };

    auto trim_whitespace = [](std::string_view tok) -> std::string_view {
        const auto start = tok.find_first_not_of(" \t");
        if (start == std::string_view::npos) {
            return {};
        }
        const auto end = tok.find_last_not_of(" \t");
        return tok.substr(start, end - start + 1);
    };

    auto parse_whole_token_as_rank_int = [](std::string_view tok) -> std::optional<int> {
        int value = 0;
        const char* const first = tok.data();
        const char* const last = first + tok.size();
        const auto [ptr, ec] = std::from_chars(first, last, value);
        if (ec != std::errc{} || ptr != last) {
            return std::nullopt;
        }
        return value;
    };

    while (!s.empty()) {
        const std::string_view raw_field = take_next_comma_separated_field(s);
        const std::string_view token = trim_whitespace(raw_field);
        if (token.empty() || token == "?") {
            continue;
        }
        const std::optional<int> rank_value = parse_whole_token_as_rank_int(token);
        if (!rank_value) {
            continue;
        }
        result.push_back(Rank{*rank_value});
    }
    return result;
}

static void handle_rank_failure(
    MPI_Comm comm,
    int cached_rank,
    int error_code,
    std::string_view operation,
    FailurePolicy policy,
    std::vector<Rank>* failed_ranks_cache,
    std::mutex* failed_ranks_cache_mutex,
    std::atomic_flag* revoked_flag) {
    // Identify who died before revoking (failure_get_acked requires pre-revoke comm).
    // Always attempt identify_failed_ranks() regardless of error code — even for
    // MPIX_ERR_REVOKED.  On a revoked comm, MPIX_Comm_failure_ack() may still succeed
    // (the ack is local state); if it does, subsequent calls to failed_ranks() will
    // find the acked group.  If it fails, identify_failed_ranks() gracefully returns
    // "unknown".  Skipping the ack for REVOKED caused failed_ranks() to return empty
    // on ranks that saw REVOKED instead of PROC_FAILED, because the post-revoke ack
    // in failed_ranks() would fail.
    std::string failed = identify_failed_ranks(comm);

    // Cache any successfully identified failed ranks so that failed_ranks()
    // can return them even if the communicator is already revoked by the time
    // the caller queries.  This is the key fix: for ranks that see REVOKED,
    // MPIX_Comm_failure_ack() inside failed_ranks() will fail, but we may
    // have captured the information here before the revoke propagated.
    if (failed_ranks_cache) {
        auto parsed = parse_failed_ranks_string(failed);
        if (!parsed.empty()) {
            // Scoped so the mutex is never held across MPI revoke / throw / _exit below.
            {
                std::lock_guard lock(*failed_ranks_cache_mutex);
                *failed_ranks_cache = std::move(parsed);
            }
        }
    }

    // Mark the communicator revoked before throwing or shrinking so callers can
    // observe that the current communicator is no longer healthy.
    if (revoked_flag) {
        revoked_flag->test_and_set(std::memory_order_release);
    }

    // Revoke so all survivors unblock. Ignore return -- another rank may have revoked first.
    MPIX_Comm_revoke(comm);

    fmt::print(
        stderr,
        "\n"
        "================================================================\n"
        "{}: MPI rank failure detected\n"
        "  Detecting rank : {}\n"
        "  Failed rank(s) : {}\n"
        "  During         : {}\n"
        "  MPI error code : {}\n"
        "================================================================\n",
        (policy == FailurePolicy::FAST_FAIL) ? "FATAL" : "WARNING",
        cached_rank,
        failed,
        operation,
        error_code);
    std::fflush(stderr);

    if (policy == FailurePolicy::FAULT_TOLERANT) {
        // Caller must catch MPIRankFailureException and call revoke_and_shrink()
        // before making any further MPI calls on this communicator.
        throw MPIRankFailureException(Rank{cached_rank}, error_code, failed);
    }

    // FAST_FAIL: exit code kUlfmExitCode (EX_SOFTWARE) flags ULFM-initiated shutdown so
    // ttrun.py can emit a targeted diagnostic.
    _exit(kUlfmExitCode);
}

#endif  // OMPI_HAS_ULFM

// ---------------------------------------------------------------------------
// mpi_check variants
// ---------------------------------------------------------------------------

// Generic check: used outside MPIContext member functions (static methods,
// constructors before comm_ is valid, MPIRequest methods).
inline void mpi_check(int error_code, std::string_view call_text) {
    if (error_code != MPI_SUCCESS) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        throw MPIDistributedException(Rank{rank}, error_code, fmt::format("{} failed", call_text));
    }
}

// Context-aware check: used in MPIContext member functions. Detects ULFM rank
// failures and dispatches to handle_rank_failure with the active policy.
inline void mpi_check_ctx(
    int error_code,
    std::string_view call_text,
    [[maybe_unused]] MPI_Comm comm,
    int cached_rank,
    [[maybe_unused]] FailurePolicy policy,
    [[maybe_unused]] std::vector<Rank>* failed_ranks_cache = nullptr,
    [[maybe_unused]] std::mutex* failed_ranks_cache_mutex = nullptr,
    [[maybe_unused]] std::atomic_flag* revoked_flag = nullptr) {
    if (error_code == MPI_SUCCESS) {
        return;
    }
#if OMPI_HAS_ULFM
    if (is_ulfm_failure(error_code)) {
        handle_rank_failure(
            comm,
            cached_rank,
            error_code,
            call_text,
            policy,
            failed_ranks_cache,
            failed_ranks_cache_mutex,
            revoked_flag);
        // In FAST_FAIL mode this is [[noreturn]].
        // In FAULT_TOLERANT mode handle_rank_failure throws, so we also never reach here.
        return;  // unreachable, but silences compiler warnings
    }
#endif
    throw MPIDistributedException(Rank{cached_rank}, error_code, fmt::format("{} failed", call_text));
}

[[nodiscard]] bool was_mpi_finalized() noexcept {
    int flag = 0;
    /* Safe to call at any time -- even before MPI_Init() */
    MPI_Finalized(&flag);  // sets flag = 1 if MPI_Finalize has completed
    return flag != 0;
}

MPIContext::CommunicatorState::~CommunicatorState() {
    if (was_mpi_finalized()) {
        return;
    }
    if (comm != MPI_COMM_WORLD && comm != MPI_COMM_NULL) {
        if (group != MPI_GROUP_NULL) {
            MPI_Group_free(&group);
        }
        MPI_Comm_free(&comm);
    }
}

std::shared_ptr<MPIContext::CommunicatorState> MPIContext::build_state(MPI_Comm comm, MPI_Group group) {
    auto state = std::make_shared<CommunicatorState>();
    state->comm = comm;
    // Publish the communicator to the RAII state before the first checked MPI
    // call so failures unwind through CommunicatorState cleanup.
    mpi_check(MPI_Comm_set_errhandler(state->comm, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
    if (group == MPI_GROUP_NULL) {
        mpi_check(MPI_Comm_group(state->comm, &state->group), "MPI_Comm_group");
    } else {
        state->group = group;
    }
    mpi_check(MPI_Comm_rank(state->comm, &state->rank), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(state->comm, &state->size), "MPI_Comm_size");
    return state;
}

std::shared_ptr<MPIContext::CommunicatorState> MPIContext::snapshot_state() const {
    std::lock_guard lock(comm_mutex_);
    TT_FATAL(state_ != nullptr, "MPIContext has no active communicator state");
    return state_;
}

// MPI_CHECK     -- generic, for static methods and non-MPIContext contexts
// MPI_CHECK_CTX -- ULFM-aware, for MPIContext member functions
#define MPI_CHECK(call) mpi_check((call), #call)
#define MPI_CHECK_STATE(state, call)                     \
    mpi_check_ctx(                                       \
        (call),                                          \
        #call,                                           \
        (state)->comm,                                   \
        (state)->rank,                                   \
        failure_policy_.load(std::memory_order_acquire), \
        &cached_failed_ranks_,                           \
        &failed_ranks_cache_mutex_,                      \
        &revoked_)

MPIDistributedException::MPIDistributedException(Rank rank, int error_code, std::string msg) :
    rank_(rank), error_code_(error_code), message_(std::move(msg)) {
    // retrieve human-readable MPI error string
    char buf[MPI_MAX_ERROR_STRING] = {0};
    int len = 0;
    MPI_Error_string(error_code_, buf, &len);
    error_string_.assign(buf, len);
    message_ = fmt::format("{}: {}", message_, error_string_);
}

// implement interface
Rank MPIDistributedException::rank() const noexcept { return rank_; }

int MPIDistributedException::error_code() const noexcept { return error_code_; }

const std::string& MPIDistributedException::message() const noexcept { return message_; }

const std::string& MPIDistributedException::error_string() const noexcept { return error_string_; }

/* ---------------------- MPIRankFailureException -------------------------- */

MPIRankFailureException::MPIRankFailureException(Rank detecting_rank, int error_code, std::string failed_ranks_str) :
    rank_(detecting_rank), error_code_(error_code), failed_ranks_(std::move(failed_ranks_str)) {
    char buf[MPI_MAX_ERROR_STRING] = {0};
    int len = 0;
    MPI_Error_string(error_code_, buf, &len);
    error_string_.assign(buf, len);
    message_ = fmt::format("MPI rank failure detected (failed ranks: {}): {}", failed_ranks_, error_string_);
}

Rank MPIRankFailureException::rank() const noexcept { return rank_; }
int MPIRankFailureException::error_code() const noexcept { return error_code_; }
const std::string& MPIRankFailureException::message() const noexcept { return message_; }
const std::string& MPIRankFailureException::error_string() const noexcept { return error_string_; }
const std::string& MPIRankFailureException::failed_ranks() const noexcept { return failed_ranks_; }

/* -------------------------- MPIRequest ---------------------------------- */

Status MPIRequest::wait() {
    MPI_Status status{};
    MPI_CHECK(MPI_Wait(&req_, &status));
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
        return std::nullopt;
    }

    done_ = true;
    int count = 0;
    MPI_CHECK(MPI_Get_count(&status, MPI_CHAR, &count));
    return Status{Rank(status.MPI_SOURCE), Tag(status.MPI_TAG), count};
}

void MPIRequest::cancel() {
    if (done_) {
        return;
    }
    MPI_CHECK(MPI_Cancel(&req_));
    MPI_CHECK(MPI_Request_free(&req_));
    done_ = true;
}

bool MPIRequest::active() const { return !done_; }

/* -------------------------- MPIContext ---------------------------------- */

// ---------------------------------------------------------------------------
// Fast-exit helpers for non-MPI fatal errors
//
// Two mechanisms work together to prevent a hanging rank from stalling the
// whole job when a non-MPI error occurs (e.g. ESTALE during JIT compilation,
// OOM, any unhandled C++ exception):
//
//  1. std::set_terminate — fires for any uncaught C++ exception or explicit
//     std::terminate() call.  Revokes MPI_COMM_WORLD so blocked ranks receive
//     ERR_REVOKED, then calls _exit(kUlfmExitCode) to bypass all atexit handlers.
//
//  2. MPI_Finalize watchdog — the atexit handler that calls MPI_Finalize is
//     collective: if one rank crashed or was killed without calling it, every
//     other rank hangs there forever.  We arm a SIGALRM before calling
//     MPI_Finalize; if it doesn't return within MPI_FINALIZE_TIMEOUT_SECS the
//     alarm fires, we log, and _exit(kUlfmExitCode).
//
// Together these cover the cases ULFM can't: a rank that is alive-but-stuck
// (e.g. blocked in Python teardown while holding an MPI_Finalize call).
// ---------------------------------------------------------------------------

// Seconds to wait for MPI_Finalize before assuming a remote rank is dead.
// Chosen to be well under a typical CI step timeout but long enough for
// legitimate slow-finalize scenarios (large communicator teardown).
static constexpr unsigned MPI_FINALIZE_TIMEOUT_SECS = 30;

// SIGALRM handler: MPI_Finalize watchdog fired — another rank is dead/stuck.
// async-signal-safe: only write() and _exit() are called.
static void mpi_finalize_alarm_handler(int /*sig*/) noexcept {
    static const char msg[] =
        "[ULFM] MPI_Finalize watchdog: another rank appears dead or stuck. "
        "Exiting kUlfmExitCode (70) to unblock the job.\n";
    [[maybe_unused]] auto w = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    // Note: MPIX_Comm_revoke is NOT async-signal-safe; skip it here.
    // _exit skips all atexit handlers (including the MPI_Finalize atexit),
    // avoiding a second hang.
    _exit(kUlfmExitCode);
}

// std::terminate handler: catches uncaught C++ exceptions and explicit
// std::terminate() calls (e.g. exceptions thrown in thread-pool workers).
// Revokes MPI_COMM_WORLD so surviving ranks detect the failure via ULFM,
// then calls _exit(kUlfmExitCode) to bypass MPI_Finalize.
//
// Async-signal-safety note: This is NOT a POSIX signal handler. Per C++
// [except.terminate], std::terminate runs in the thread that invoked it, after
// exception handling is abandoned — not in the async-signal execution context.
// MPIX_Comm_revoke is therefore appropriate here (contrast mpi_finalize_alarm_handler,
// which must only use async-signal-safe operations). Do not call std::terminate
// from a custom signal handler and expect MPI calls here to be safe; if you need
// that pattern, revoke from normal thread context instead.
//
// To switch to fault-tolerant mode: remove this handler and instead set
// FailurePolicy::FAULT_TOLERANT on your MPIContext, then catch
// MPIRankFailureException in your collective loops.
static void mpi_terminate_handler() noexcept {
#if OMPI_HAS_ULFM
    // Revoke so that any rank blocked in a collective receives ERR_REVOKED
    // and our MPI_CHECK_CTX macro triggers the FailurePolicy dispatch.
    // This is best-effort: MPI_COMM_WORLD may already be revoked or
    // MPI may not be initialized yet.
    MPIX_Comm_revoke(MPI_COMM_WORLD);
#endif
    static const char msg[] =
        "================================================================\n"
        "FATAL: std::terminate called in MPI context\n"
        "  Cause  : uncaught exception or explicit std::terminate()\n"
        "  Action : revoked MPI_COMM_WORLD (if ULFM available), exiting kUlfmExitCode (70)\n"
        "================================================================\n";
    [[maybe_unused]] auto w = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    _exit(kUlfmExitCode);
}

inline void init_env(int& argc, char**& argv) {
    static std::once_flag mpi_once;

    std::call_once(mpi_once, [&] {
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) {
            TT_THROW("MPI_Init_thread failed");
        }

        // Install the terminate handler so that any uncaught exception
        // (including those thrown in thread-pool workers via std::async)
        // revokes the world communicator and calls _exit(kUlfmExitCode) rather than
        // letting the process hang in teardown.
        std::set_terminate(mpi_terminate_handler);

        // Install the finalize watchdog handler once at init. The atexit path
        // only arms alarm(2) before MPI_Finalize; tests (e.g. FinalizeWatchdogPath)
        // expect sigaction(SIGALRM) to show a custom handler while the process runs.
        // SA_RESETHAND: one-shot — kernel resets to SIG_DFL after first delivery,
        // so a second SIGALRM (e.g. from a spurious alarm) takes the default action
        // rather than re-invoking the handler.  sigaction() is preferred over
        // signal() for portable, well-defined handler persistence semantics.
        struct sigaction sa = {};
        sa.sa_handler = mpi_finalize_alarm_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESETHAND;
        sigaction(SIGALRM, &sa, nullptr);

        // Ensure MPI_Finalize is called when the program exits.
        // Guard with a watchdog: if MPI_Finalize does not return within
        // MPI_FINALIZE_TIMEOUT_SECS, another rank is presumed dead/stuck
        // and we exit instead of hanging the job.
        //
        // Switching to fault-tolerant mode: replace the atexit below with
        // an explicit call site that handles MPIRankFailureException from
        // MPI_Finalize (not yet possible — MPI_Finalize does not throw).
        // For now, the watchdog is the pragmatic solution.
        std::atexit([] {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (finalized) {
                return;  // already called (e.g. explicit finalize earlier)
            }
            // RAII guard: arm the watchdog on construction, disarm on destruction.
            // Ensures alarm(0) is called even if future code between here and
            // MPI_Finalize is restructured (MPI_Finalize itself does not throw,
            // but defensive RAII is idiomatic C++).
            struct AlarmGuard {
                explicit AlarmGuard(unsigned secs) { alarm(secs); }
                ~AlarmGuard() { alarm(0); }
                AlarmGuard(const AlarmGuard&) = delete;
                AlarmGuard& operator=(const AlarmGuard&) = delete;
                AlarmGuard(AlarmGuard&&) = delete;
                AlarmGuard& operator=(AlarmGuard&&) = delete;
            } guard(MPI_FINALIZE_TIMEOUT_SECS);
            MPI_Finalize();
            // alarm(0) is called by ~AlarmGuard above.
            // Do NOT call signal(SIGALRM, SIG_DFL) here: calling signal() inside
            // an atexit handler is unnecessary (the process is exiting) and
            // potentially hazardous if other atexit handlers interact with SIGALRM.
        });
    });
}

void MPIContext::create(int argc, char** argv) {
    init_env(argc, argv);
    // it is a good idea to duplicate the world communicator
    // don't want to rely on the global comm_world which cannot be replaced
    current_world_ = std::make_shared<MPIContext>(MPI_COMM_WORLD)->duplicate();
}

const ContextPtr& MPIContext::get_current_world() {
    if (!current_world_) {
        // Default initialization of MPIContext if not already initialized
        MPIContext::create(0, nullptr);
    }
    return current_world_;
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

MPIContext::MPIContext(MPI_Comm comm) : state_(build_state(comm)) { id_ = DistributedContext::generate_unique_id(); }

MPIContext::MPIContext(MPI_Comm comm, MPI_Group group) : state_(build_state(comm, group)) {
    id_ = DistributedContext::generate_unique_id();
}

MPI_Comm MPIContext::comm() const { return snapshot_state()->comm; }

MPI_Group MPIContext::group() const { return snapshot_state()->group; }

Rank MPIContext::rank() const { return Rank(snapshot_state()->rank); }
Size MPIContext::size() const { return Size(snapshot_state()->size); }
bool MPIContext::supports_fault_tolerance() const { return OMPI_HAS_ULFM; }

void MPIContext::barrier() const {
    const auto state = snapshot_state();
    MPI_CHECK_STATE(state, MPI_Barrier(state->comm));
}

/* ---- point‑to‑point ---------------------------------------------------- */

void MPIContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    const auto state = snapshot_state();
    MPI_CHECK_STATE(state, MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, state->comm));
}

void MPIContext::ssend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    const auto state = snapshot_state();
    MPI_CHECK_STATE(state, MPI_Ssend(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, state->comm));
}

void MPIContext::recv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const {
    check_size_fits_int(buf.size());
    const auto state = snapshot_state();
    MPI_CHECK_STATE(
        state,
        MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, state->comm, MPI_STATUS_IGNORE));
}

RequestPtr MPIContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    const auto state = snapshot_state();
    MPI_Request req{};
    MPI_CHECK_STATE(
        state,
        MPI_Isend(
            const_cast<std::byte*>(buf.data()),
            static_cast<int>(buf.size()),
            MPI_CHAR,
            *dest,
            *tag,
            state->comm,
            &req));
    return std::make_shared<MPIRequest>(req);
}

RequestPtr MPIContext::irecv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const {
    check_size_fits_int(buf.size());
    const auto state = snapshot_state();
    MPI_Request req{};
    MPI_CHECK_STATE(
        state, MPI_Irecv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, state->comm, &req));
    return std::make_shared<MPIRequest>(req);
}

/* ---- collectives ------------------------------------------------------- */

void MPIContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const {
    check_size_fits_int(buf.size());
    const auto state = snapshot_state();
    MPI_CHECK_STATE(state, MPI_Bcast(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *root, state->comm));
}

void MPIContext::all_reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    check_size_fits_int(send_buf.size());
    const auto state = snapshot_state();

    TT_FATAL(
        send_buf.size() == recv_buf.size(),
        "all_reduce: send/recv sizes differ ({} vs {})",
        send_buf.size(),
        recv_buf.size());

    const int elem_size = mpi_dtype_size(dtype);  // e.g. 4 for FLOAT32
    TT_FATAL(
        send_buf.size() % elem_size == 0,
        "all_reduce: buffer size {} is not a multiple of element size {}",
        send_buf.size(),
        elem_size);

    const int count = static_cast<int>(send_buf.size() / elem_size);

    // allow in‑place (send == recv) by switching to MPI_IN_PLACE
    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : static_cast<void*>(send_buf.data());

    MPI_CHECK_STATE(
        state, MPI_Allreduce(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), state->comm));
}

void MPIContext::reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype, Rank root) const {
    const auto state = snapshot_state();
    const int elem_sz = mpi_dtype_size(dtype);
    TT_FATAL(
        send_buf.size() % elem_sz == 0,
        "reduce: send size {} not multiple of element size {}",
        send_buf.size(),
        elem_sz);

    const int count = static_cast<int>(send_buf.size() / elem_sz);
    check_size_fits_int(count);

    // On non‑root ranks 'recv_buf' can be any pointer (or even nullptr).  On root it must fit.
    if (state->rank == *root) {
        TT_FATAL(
            recv_buf.size() == send_buf.size(),
            "reduce: on root rank, recv size {} != send size {}",
            recv_buf.size(),
            send_buf.size());
    }

    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : send_buf.data();

    MPI_CHECK_STATE(
        state,
        MPI_Reduce(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), *root, state->comm));
}

void MPIContext::gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    const auto state = snapshot_state();
    const int send_count = static_cast<int>(send_buf.size());
    check_size_fits_int(send_count);

    // Root must have room for 'size()' times the per‑rank payload.
    if (state->rank == *root) {
        const std::size_t expected = static_cast<std::size_t>(send_count) * state->size;
        TT_FATAL(
            recv_buf.size() == expected, "gather: root recv buffer {} bytes, expected {}", recv_buf.size(), expected);
    }

    MPI_CHECK_STATE(
        state,
        MPI_Gather(send_buf.data(), send_count, MPI_CHAR, recv_buf.data(), send_count, MPI_CHAR, *root, state->comm));
}

void MPIContext::scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    const auto state = snapshot_state();
    const int recv_count = static_cast<int>(recv_buf.size());
    check_size_fits_int(recv_count);

    if (state->rank == *root) {
        const std::size_t expected = static_cast<std::size_t>(recv_count) * state->size;
        TT_FATAL(
            send_buf.size() == expected, "scatter: root send buffer {} bytes, expected {}", send_buf.size(), expected);
    }

    MPI_CHECK_STATE(
        state,
        MPI_Scatter(send_buf.data(), recv_count, MPI_CHAR, recv_buf.data(), recv_count, MPI_CHAR, *root, state->comm));
}

void MPIContext::all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    const auto state = snapshot_state();
    const int send_count = static_cast<int>(send_buf.size());
    check_size_fits_int(send_count);

    const std::size_t expected_recv = static_cast<std::size_t>(send_count) * state->size;

    TT_FATAL(
        recv_buf.size() == expected_recv,
        "all_gather: recv buffer {} bytes, expected {} (world × send)",
        recv_buf.size(),
        expected_recv);

    // allow MPI_IN_PLACE if caller wants to receive in the same buffer
    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : send_buf.data();

    MPI_CHECK_STATE(
        state, MPI_Allgather(send_ptr, send_count, MPI_CHAR, recv_buf.data(), send_count, MPI_CHAR, state->comm));
}

void MPIContext::all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    const auto state = snapshot_state();
    const int world = state->size;

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

    MPI_CHECK_STATE(
        state, MPI_Alltoall(send_buf.data(), block, MPI_CHAR, recv_buf.data(), block, MPI_CHAR, state->comm));
}

void MPIContext::reduce_scatter(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    const auto state = snapshot_state();
    const int world = state->size;
    const int elem_sz = mpi_dtype_size(dtype);

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

    // --- fixed call -------------------------------------------------------
    MPI_CHECK_STATE(
        state,
        MPI_Reduce_scatter_block(
            send_buf.data(),      // sendbuf
            recv_buf.data(),      // recvbuf
            recv_count,           // elements per rank
            dtype_to_mpi(dtype),  // element datatype
            reduce_to_mpi(op),    // operation (SUM, MAX, …)
            state->comm));        // communicator
}

void MPIContext::scan(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    const auto state = snapshot_state();
    TT_FATAL(
        send_buf.size() == recv_buf.size(), "scan: send size {} != recv size {}", send_buf.size(), recv_buf.size());

    const int elem_sz = mpi_dtype_size(dtype);
    TT_FATAL(
        send_buf.size() % elem_sz == 0,
        "scan: buffer size {} not multiple of element size {}",
        send_buf.size(),
        elem_sz);

    const int count = static_cast<int>(send_buf.size() / elem_sz);
    check_size_fits_int(count);

    void* send_ptr = (send_buf.data() == recv_buf.data()) ? MPI_IN_PLACE : send_buf.data();

    MPI_CHECK_STATE(
        state, MPI_Scan(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), state->comm));
}
/* ---- communicator management ------------------------------------------ */

ContextPtr MPIContext::duplicate() const {
    const auto state = snapshot_state();
    MPI_Comm dup = MPI_COMM_NULL;
    MPI_CHECK_STATE(state, MPI_Comm_dup(state->comm, &dup));
    return std::make_shared<MPIContext>(dup);
}

ContextPtr MPIContext::split(Color color, Key key) const {
    const auto state = snapshot_state();
    MPI_Comm split_comm;
    if (*color == SPLIT_COLOR_UNDEFINED) {
        color = Color(MPI_UNDEFINED);
    }
    MPI_CHECK_STATE(state, MPI_Comm_split(state->comm, *color, *key, &split_comm));
    return std::make_shared<MPIContext>(split_comm);
}

ContextPtr MPIContext::create_sub_context(tt::stl::Span<int> ranks) const {
    const auto state = snapshot_state();
    MPI_Group sub_grp = MPI_GROUP_NULL;
    MPI_Comm sub_comm = MPI_COMM_NULL;

    MPI_CHECK_STATE(state, MPI_Group_incl(state->group, static_cast<int>(ranks.size()), ranks.data(), &sub_grp));
    if (MPI_Comm_create_group(state->comm, sub_grp, 0 /*tag*/, &sub_comm) != MPI_SUCCESS) {
        // sub_comm is not valid, we are not in the group
        MPI_Group_free(&sub_grp);
        throw MPIDistributedException(
            Rank{state->rank}, MPI_ERR_GROUP, "MPI_Comm_create_group failed: not in the group");
    }
    if (sub_comm == MPI_COMM_NULL) {
        // we are not in the group
        MPI_Group_free(&sub_grp);
        throw MPIDistributedException(Rank{state->rank}, MPI_ERR_GROUP, "MPI_Comm_create_group returned empty comm");
    }
    MPI_Group_free(&sub_grp);
    return std::make_shared<MPIContext>(sub_comm);
}

void MPIContext::translate_ranks_to_other_ctx(
    tt::stl::Span<int> ranks, const ContextPtr& other_ctx, tt::stl::Span<int> translated_ranks) const {
    const auto state = snapshot_state();
    TT_FATAL(
        ranks.size() == translated_ranks.size(),
        "translate_ranks_to_other_ctx: ranks size {} != translated_ranks size {}",
        ranks.size(),
        translated_ranks.size());
    auto mpi_context = std::dynamic_pointer_cast<MPIContext>(other_ctx);
    TT_FATAL(mpi_context != nullptr, "translate_ranks_to_other_ctx: other_ctx is not a MPIContext");

    MPI_CHECK_STATE(
        state,
        MPI_Group_translate_ranks(
            state->group, static_cast<int>(ranks.size()), ranks.data(), mpi_context->group(), translated_ranks.data()));
}

void MPIContext::abort(int error_code) const {
    const auto state = snapshot_state();
    MPI_Abort(state->comm, error_code);
}

void MPIContext::revoke_and_shrink() {
#if (!OMPI_HAS_ULFM)
    TT_THROW("revoke_and_shrink() requires MPI ULFM support which is not available in this build");
#else
    // Detect concurrent invocations: revoke_and_shrink() must be called from a
    // single designated thread with no concurrent MPI calls in flight.
    // shrink_in_progress_ catches re-entrant or concurrent shrink calls; broader
    // "no MPI in flight" invariant is documented in the header and enforced by the
    // caller.
    if (shrink_in_progress_.test_and_set(std::memory_order_acquire)) {
        TT_THROW("revoke_and_shrink() invoked concurrently — violates single-caller invariant");
    }
    struct ShrinkGuard {
        std::atomic_flag& flag;
        explicit ShrinkGuard(std::atomic_flag& f) noexcept : flag(f) {}
        ~ShrinkGuard() { flag.clear(std::memory_order_release); }
        ShrinkGuard(const ShrinkGuard&) = delete;
        ShrinkGuard& operator=(const ShrinkGuard&) = delete;
        ShrinkGuard(ShrinkGuard&&) = delete;
        ShrinkGuard& operator=(ShrinkGuard&&) = delete;
    } shrink_guard{shrink_in_progress_};

    const auto old_state = snapshot_state();

    int rc = MPIX_Comm_revoke(old_state->comm);
    if (rc != MPI_SUCCESS && rc != MPI_ERR_REVOKED) {  // another rank may have revoked first
        abort(rc);
    }
    revoked_.test_and_set(std::memory_order_release);

    MPI_Comm new_comm = MPI_COMM_NULL;
    MPI_CHECK_STATE(old_state, MPIX_Comm_shrink(old_state->comm, &new_comm));
    // Route replacement-communicator setup through build_state(new_comm)
    // rather than MPI_CHECK_STATE(old_state, ...) so any setup failure is
    // attributed to the new communicator and unwinds through the same RAII
    // cleanup path as other communicator factories.
    auto new_state = build_state(new_comm);

    // Swap the current communicator snapshot atomically. Old readers keep the
    // pre-shrink communicator alive through their shared_ptr snapshots.
    {
        std::lock_guard lock(comm_mutex_);
        state_ = std::move(new_state);
    }

    revoked_.clear(std::memory_order_release);  // cleared: new communicator is healthy
    {
        std::lock_guard cache_lock(failed_ranks_cache_mutex_);
        cached_failed_ranks_.clear();  // new comm has no known failures
    }
#endif
}

bool MPIContext::is_revoked() {
    // revoked_ is set by both failure detection (handle_rank_failure) and
    // explicit revoke_and_shrink(), then cleared after a successful shrink.
    // This is correct for both ULFM and non-ULFM builds: without ULFM, revocation
    // never happens so the flag is always clear.
    // NOTE: This is a snapshot read; the communicator may become revoked between
    // this call and the next MPI operation.  Callers must not rely on this value
    // being stable across multiple statements.
    return revoked_.test(std::memory_order_acquire);
}

void MPIContext::set_failure_policy(FailurePolicy policy) {
#if (!OMPI_HAS_ULFM)
    if (policy == FailurePolicy::FAULT_TOLERANT) {
        TT_THROW("FAULT_TOLERANT failure policy requires ULFM support which is not available in this build");
    }
#endif
    failure_policy_.store(policy, std::memory_order_release);
}

std::optional<bool> MPIContext::agree([[maybe_unused]] bool local_value) const {
#if OMPI_HAS_ULFM
    const auto state = snapshot_state();
    int flag = local_value ? 1 : 0;
    int rc = MPIX_Comm_agree(state->comm, &flag);
    if (rc != MPI_SUCCESS) {
        mpi_check_ctx(
            rc,
            "MPIX_Comm_agree",
            state->comm,
            state->rank,
            failure_policy_.load(std::memory_order_acquire),
            &cached_failed_ranks_,
            &failed_ranks_cache_mutex_,
            &revoked_);
    }
    return flag != 0;
#else
    // Without ULFM support, agreement is not available.
    // Return nullopt so callers can distinguish "no ULFM" from a real result,
    // consistent with the base-class default and the header doc.
    return std::nullopt;
#endif
}

std::vector<Rank> MPIContext::failed_ranks() const {
#if OMPI_HAS_ULFM
    const auto state = snapshot_state();
    // Query ULFM for currently-acked failed ranks on this communicator.
    //
    // Attempt MPIX_Comm_failure_ack() first (best-effort, ignore return code).
    // On a non-revoked comm this records any pending failures; on an already-
    // revoked comm it returns MPIX_ERR_REVOKED, which we silently ignore.
    // Then call MPIX_Comm_failure_get_acked() to retrieve the acked group.
    //
    // POTENTIAL FAILURE CASE — REVOKED-path ranks:
    //   A rank that sees MPIX_ERR_REVOKED (77) receives a communicator that was
    //   revoked by a peer *before* this rank processed the failure.  By the time
    //   this method runs, MPIX_Comm_failure_ack() returns MPIX_ERR_REVOKED, so
    //   MPIX_Comm_failure_get_acked() sees no acked failures and returns an empty
    //   group → failed_size == 0 → we fall through to cached_failed_ranks_.
    //
    //   cached_failed_ranks_ was populated by handle_rank_failure() before
    //   MPIX_Comm_revoke() propagated — so for ranks where identify_failed_ranks()
    //   succeeded at detection time, the cache will carry the right answer.  If
    //   identify_failed_ranks() also failed (e.g., the comm was already fully
    //   revoked when handle_rank_failure() ran), the cache is empty and this
    //   method returns {}.
    //
    //   When reliable failed-rank identification is required, compare communicator
    //   size before vs. after revoke_and_shrink() — the delta is always accurate.
    MPIX_Comm_failure_ack(state->comm);  // best-effort; ignore return code
    MPI_Group failed_group = MPI_GROUP_NULL;
    if (MPIX_Comm_failure_get_acked(state->comm, &failed_group) != MPI_SUCCESS) {
        std::lock_guard cache_lock(failed_ranks_cache_mutex_);
        return cached_failed_ranks_;
    }
    int failed_size = 0;
    MPI_Group_size(failed_group, &failed_size);
    if (failed_size == 0) {
        MPI_Group_free(&failed_group);
        std::lock_guard cache_lock(failed_ranks_cache_mutex_);
        return cached_failed_ranks_;
    }
    MPI_Group world_group = MPI_GROUP_NULL;
    MPI_Comm_group(state->comm, &world_group);
    std::vector<int> local_indices(failed_size);
    std::iota(local_indices.begin(), local_indices.end(), 0);
    std::vector<int> world_ranks(failed_size, MPI_UNDEFINED);
    if (world_group != MPI_GROUP_NULL) {
        MPI_Group_translate_ranks(
            failed_group, failed_size, local_indices.data(), world_group, world_ranks.data());
        MPI_Group_free(&world_group);
    }
    MPI_Group_free(&failed_group);

    std::vector<Rank> result;
    result.reserve(failed_size);
    for (int r : world_ranks) {
        if (r != MPI_UNDEFINED) {
            result.push_back(Rank{r});
        }
    }
    return result;
#else
    return {};
#endif
}

std::size_t MPIContext::snoop_incoming_msg_size(Rank source, Tag tag) const {
    const auto state = snapshot_state();
    int size_bytes = 0;
    MPI_Status status;
    MPI_CHECK_STATE(state, MPI_Probe(*source, *tag, state->comm, &status));
    MPI_CHECK_STATE(state, MPI_Get_count(&status, MPI_CHAR, &size_bytes));
    return static_cast<std::size_t>(size_bytes);
}

MPIContext::~MPIContext() {
    std::lock_guard lock(comm_mutex_);
    state_.reset();
}
}  // namespace tt::tt_metal::distributed::multihost
