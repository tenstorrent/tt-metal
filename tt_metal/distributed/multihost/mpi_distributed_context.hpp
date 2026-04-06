// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <csignal>
#include <cstdint>
#include <mpi.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "api/tt-metalium/distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {

class MPIContext;

class MPIDistributedException : public DistributedException {
public:
    MPIDistributedException(Rank rank, int error_code, std::string msg);

    // implement interface
    Rank rank() const noexcept override;

    int error_code() const noexcept override;

    const std::string& message() const noexcept override;

    const std::string& error_string() const noexcept override;

private:
    Rank rank_{0};
    int error_code_{0};
    std::string message_;
    std::string error_string_;
};

// ---------------------------------------------------------------------------
// Failure policy: controls how MPIContext reacts when a remote rank dies.
//
// FAST_FAIL (default):
//   Detect failure → revoke communicator → log diagnostic → _exit(70).
//   Clean, fast, CI-friendly.  No application code changes needed.
//
// FAULT_TOLERANT:
//   Detect failure → revoke communicator → log diagnostic → throw
//   MPIRankFailureException.  The caller catches, calls revoke_and_shrink()
//   to obtain a healthy communicator, and continues work with a reduced
//   world.  Requires ULFM support at build time.
//
// To switch, call ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT)
// before entering your communication loop.
// ---------------------------------------------------------------------------
enum class FailurePolicy {
    FAST_FAIL,       // Detect failure → revoke → log → _exit(70).
    FAULT_TOLERANT,  // Detect failure → revoke → log → throw MPIRankFailureException.
};

// Exception thrown in FAULT_TOLERANT mode when a remote rank dies.
// Carries enough context for the caller to decide how to recover.
class MPIRankFailureException : public DistributedException {
public:
    MPIRankFailureException(Rank detecting_rank, int error_code, std::string failed_ranks_str);

    Rank rank() const noexcept override;
    int error_code() const noexcept override;
    const std::string& message() const noexcept override;
    const std::string& error_string() const noexcept override;

    // Comma-separated list of world-ranks that failed (e.g. "2, 5").
    const std::string& failed_ranks() const noexcept;

private:
    Rank rank_{0};
    int error_code_{0};
    std::string message_;
    std::string error_string_;
    std::string failed_ranks_;
};

// Communicator snapshot at isend/irecv post time — used when completing the
// request so mpi_check_ctx revokes the same MPI_Comm the operation used (the
// live MPIContext may already point at a shrunk communicator).
struct MPIRequestCommSnapshot {
    MPI_Comm comm{MPI_COMM_NULL};
    int rank{0};
    std::vector<std::string> rank_hostnames;
};

namespace detail {

// std::terminate() uses MPIX_Comm_revoke as a best-effort wakeup for blocked
// peers. Success and "already revoked" are the expected outcomes; any other
// MPI error still leads to the same process-local _exit(70) fallback.
[[nodiscard]] bool terminate_revoke_result_is_nonfatal(int rc) noexcept;

// MPI_Finalize should only be treated as clean when it returns MPI_SUCCESS.
// Any non-success return is surfaced in diagnostics and converted into the
// same fast-fail exit path used by the watchdog.
[[nodiscard]] bool finalize_return_is_nonfatal(int rc) noexcept;

// Tracks whether teardown should skip MPI_Finalize entirely because the process
// is already known to be in an unsafe state (e.g. a revoked communicator that
// was never recovered).
void mark_finalize_unsafe() noexcept;
void clear_finalize_unsafe() noexcept;
[[nodiscard]] bool finalize_is_unsafe() noexcept;

// Tracks whether a process-wide MPI teardown is in progress so new MPI entry
// points can fail closed instead of entering collectives during finalization.
[[nodiscard]] bool finalize_is_in_progress() noexcept;

// Arms the SIGALRM-based finalize watchdog only for a lexical scope, temporarily
// unmasks SIGALRM in the current thread, starts a detached backup watchdog
// thread, and restores prior process/thread state on normal return.
class ScopedFinalizeAlarmHandler {
public:
    explicit ScopedFinalizeAlarmHandler(unsigned secs) noexcept;
    ~ScopedFinalizeAlarmHandler() noexcept;

    ScopedFinalizeAlarmHandler(const ScopedFinalizeAlarmHandler&) = delete;
    ScopedFinalizeAlarmHandler& operator=(const ScopedFinalizeAlarmHandler&) = delete;
    ScopedFinalizeAlarmHandler(ScopedFinalizeAlarmHandler&&) = delete;
    ScopedFinalizeAlarmHandler& operator=(ScopedFinalizeAlarmHandler&&) = delete;

    [[nodiscard]] bool armed() const noexcept { return armed_; }

private:
    struct sigaction old_sigalrm_action_{};
    sigset_t old_sigmask_{};
    unsigned old_alarm_secs_{0};
    std::uint64_t watchdog_generation_{0};
    bool finalize_scope_owner_{false};
    bool sigalrm_unmasked_{false};
    bool armed_{false};
};

// Test helper: verify that the scoped watchdog has installed its SIGALRM
// handler at the current instant.
[[nodiscard]] bool is_finalize_alarm_handler_installed_for_testing() noexcept;
[[nodiscard]] bool is_sigalrm_unblocked_for_testing() noexcept;

}  // namespace detail

// ---------------------------------------------------------------------
//                       Main distributed context
// ---------------------------------------------------------------------
class MPIContext : public DistributedContext, public std::enable_shared_from_this<MPIContext> {
public:
    // factory (initialises MPI environment once per process)
    static void create(int argc, char** argv);
    static const ContextPtr& get_current_world();
    static bool is_initialized();

    // destructor – communicator MPI_COMM_WORLD is freed automatically by MPI_Finalize
    // All other communicators are freed here
    ~MPIContext() override;

    /* ---------------- basic info / sync ---------------- */
    [[nodiscard]] Rank rank() const override;
    [[nodiscard]] Size size() const override;
    [[nodiscard]] bool supports_fault_tolerance() const override;
    void barrier() const override;

    /* ---------------- point‑to‑point ------------------- */
    void send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    void ssend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    void recv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const override;

    [[nodiscard]] RequestPtr isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    [[nodiscard]] RequestPtr irecv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const override;

    /* ---------------- collectives ---------------------- */
    void broadcast(tt::stl::Span<std::byte> buf, Rank root) const override;
    void all_reduce(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;
    void reduce(
        tt::stl::Span<std::byte> send_buf,
        tt::stl::Span<std::byte> recv_buf,
        ReduceOp op,
        DType dtype,
        Rank root) const override;
    void gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const override;
    void scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const override;
    void all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;
    void all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;
    void reduce_scatter(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;
    void scan(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;

    void translate_ranks_to_other_ctx(
        tt::stl::Span<int> ranks, const ContextPtr& other_ctx, tt::stl::Span<int> translated_ranks) const override;

    /* ------------- communicator management ------------- */
    [[nodiscard]] ContextPtr duplicate() const override;
    [[nodiscard]] ContextPtr split(Color color, Key key) const override;
    [[nodiscard]] ContextPtr create_sub_context(tt::stl::Span<int> ranks) const override;
    void abort(int error_code) const override;
    void revoke_and_shrink() override;
    [[nodiscard]] bool is_revoked() override;
    void set_failure_policy(FailurePolicy policy);

    // Returns the agreed-upon value across all surviving ranks.
    // Essential before calling revoke_and_shrink() to ensure all survivors
    // are coordinated.  If any rank passes false, the result is false.
    // Returns std::nullopt if not compiled with ULFM support.
    std::optional<bool> agree(bool local_value) const override;

    // Returns the set of ranks detected as failed since the last
    // revoke_and_shrink() call.  Only populated under FAULT_TOLERANT policy.
    // Cleared when revoke_and_shrink() succeeds.
    //
    // KNOWN LIMITATION — REVOKED-path ranks may return empty:
    //   When a rank detects failure via MPIX_ERR_REVOKED (error 77) rather than
    //   MPIX_ERR_PROC_FAILED (error 75), it means another rank already called
    //   MPIX_Comm_revoke() before this rank saw the failure.  On an already-revoked
    //   communicator, MPIX_Comm_ack_failed() returns MPIX_ERR_REVOKED itself, so
    //   MPIX_Comm_get_failed() cannot record the failed-rank set.
    //
    //   To mitigate this, handle_rank_failure() caches identified ranks in
    //   cached_failed_ranks_ *before* calling MPIX_Comm_revoke(), and this
    //   method falls back to that cache if the live ULFM query returns empty.
    //   However, if identify_failed_ranks() also failed (e.g., the communicator
    //   was already revoked when handle_rank_failure() ran), the cache will be
    //   empty and this method returns {}.
    //
    //   Reliable alternative: compare communicator size before vs. after
    //   revoke_and_shrink() — the delta gives the count of lost ranks, even
    //   when ULFM cannot identify them by rank number.
    std::vector<Rank> failed_ranks() const override;

    /* ------------- message snooping ------------- */
    std::size_t snoop_incoming_msg_size(Rank source, Tag tag) const override;

    /* ----------------- mpi constructors ---------------- */
    explicit MPIContext(MPI_Comm comm);
    explicit MPIContext(MPI_Comm comm, MPI_Group group);
    MPI_Comm comm() const;
    MPI_Group group() const;

    static void set_current_world(const ContextPtr& ctx);

private:
    friend class MPIRequest;

    struct CommunicatorState {
        MPI_Comm comm{MPI_COMM_NULL};
        MPI_Group group{MPI_GROUP_NULL};
        int rank{0};
        int size{0};
        std::vector<std::string> rank_hostnames;

        ~CommunicatorState();
    };

    [[nodiscard]] static std::shared_ptr<CommunicatorState> build_state(
        MPI_Comm comm, MPI_Group group = MPI_GROUP_NULL);

    // Grabs a shared_ptr to the current CommunicatorState under comm_mutex_, then
    // release the mutex before any blocking or long-running MPI call.
    //
    // Open MPI ULFM makes this essential: revoke_and_shrink() may replace state_
    // with a new communicator (and the destructor of the old CommunicatorState
    // frees the previous MPI_Comm / MPI_Group) while another thread or an
    // in-flight operation still uses handles from the pre-shrink world.
    //
    // **Holding only raw comm_/group_ members would invite use-after-free once
    // the swap completes**.
    //
    // A snapshot keeps the old state object alive until every call
    // that took a snapshot returns, so MPI always sees valid handles even if a
    // rank failure triggers concurrent revoke, shrink, and replacement.
    //
    // Separately, ULFM error handling (mpi_check_ctx / handle_rank_failure) must
    // run against the same communicator the failing operation used; the
    // snapshot ties each entry point to one consistent comm/rank view for the
    // duration of that MPI call and its error path.
    [[nodiscard]] std::shared_ptr<CommunicatorState> snapshot_state() const;

    std::shared_ptr<CommunicatorState> state_;
    mutable std::atomic_flag revoked_ = {};  // set when MPIX_Comm_revoke() is called; cleared after shrink
    // Atomic so that set_failure_policy() on one thread and MPI_CHECK_CTX reads
    // on another thread do not race.  Use memory_order_release on write,
    // memory_order_acquire on read.
    std::atomic<FailurePolicy> failure_policy_{FailurePolicy::FAST_FAIL};
    // Set while revoke_and_shrink() is in progress; caught by TT_ASSERT to
    // detect concurrent invocations, which violate the single-caller invariant.
    std::atomic_flag shrink_in_progress_ = {};

    // Detection-time cache of failed ranks, populated by handle_rank_failure()
    // *before* MPIX_Comm_revoke() is called and before throwing
    // MPIRankFailureException.
    //
    // This cache exists to work around the REVOKED-path failure case (see the
    // failed_ranks() comment above).  Ranks that receive MPIX_ERR_REVOKED have
    // a revoked communicator by the time they enter handle_rank_failure(); the
    // best-effort MPIX_Comm_ack_failed() in identify_failed_ranks() may still
    // succeed at that point (the ack is local state), allowing the failed group
    // to be captured here.  If identify_failed_ranks() also fails, the cache
    // remains empty and failed_ranks() will return {}.
    //
    // Cleared by revoke_and_shrink() when the communicator is replaced.
    //
    // Protects cached_failed_ranks_ across three concurrent access paths:
    //   - Write: handle_rank_failure() via MPI_CHECK_CTX on any thread executing
    //            a failed MPI operation.
    //   - Read:  failed_ranks() const member (any thread).
    //   - Clear: revoke_and_shrink() when replacing the communicator.
    //
    // Kept separate from comm_mutex_ to avoid holding a mutex during blocking MPI
    // calls: comm_mutex_ gates comm_/group_/rank_/size_ updates; this mutex gates
    // only the failed-rank cache, which is a fast in-memory operation.  Acquiring
    // both simultaneously is never required, so there is no lock-ordering concern.
    mutable std::mutex failed_ranks_cache_mutex_;
    mutable std::vector<Rank> cached_failed_ranks_;

    // Protects state_ replacement and snapshot acquisition.
    //
    // Individual MPI entry points take a shared_ptr snapshot under this mutex,
    // then release the mutex before entering MPI. This avoids use-after-free on
    // MPI_Comm / MPI_Group when revoke_and_shrink() swaps in a new communicator.
    //
    // Concurrent shrink+MPI still requires caller-level coordination for
    // semantics: in-flight operations may continue against the pre-shrink
    // communicator and legitimately observe MPI_ERR_REVOKED / rank-failure
    // exceptions. What this mutex guarantees is safe handle lifetime, not
    // transparent multi-threaded recovery.
    mutable std::mutex comm_mutex_;

    // caching our own world communicator which is duplicator of MPI_COMM_WORLD
    inline static ContextPtr current_world_;
};

// ---------------------------------------------------------------------
//                           Non‑blocking request
// ---------------------------------------------------------------------
class MPIRequest : public Request {
public:
    MPIRequest(MPI_Request req, std::weak_ptr<const MPIContext> owner, MPIRequestCommSnapshot comm_snapshot) noexcept :
        req_(req), owner_(std::move(owner)), comm_snapshot_(std::move(comm_snapshot)) {}

    Status wait() override;
    std::optional<Status> test() override;
    void cancel() override;
    bool active() const override;

private:
    mutable MPI_Request req_{};
    bool done_{};
    std::weak_ptr<const MPIContext> owner_;
    MPIRequestCommSnapshot comm_snapshot_;
};

}  // namespace tt::tt_metal::distributed::multihost
