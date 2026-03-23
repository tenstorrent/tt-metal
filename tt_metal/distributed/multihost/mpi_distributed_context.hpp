// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <mpi.h>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include "api/tt-metalium/distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {

class MPIContext;
class MPIRequest;

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

// ---------------------------------------------------------------------
//                           Non‑blocking request
// ---------------------------------------------------------------------
class MPIRequest : public Request {
public:
    explicit MPIRequest(MPI_Request req) : req_(req) {}

    Status wait() override;
    std::optional<Status> test() override;
    void cancel() override;
    bool active() const override;

private:
    mutable MPI_Request req_{};
    bool done_{};
};

// ---------------------------------------------------------------------
//                       Main distributed context
// ---------------------------------------------------------------------
class MPIContext : public DistributedContext {
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
    //   communicator, MPIX_Comm_failure_ack() returns MPIX_ERR_REVOKED itself, so
    //   MPIX_Comm_failure_get_acked() cannot record the failed-rank set.
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
    struct CommunicatorState {
        MPI_Comm comm{MPI_COMM_NULL};
        MPI_Group group{MPI_GROUP_NULL};
        int rank{0};
        int size{0};

        ~CommunicatorState();
    };

    [[nodiscard]] static std::shared_ptr<CommunicatorState> build_state(
        MPI_Comm comm, MPI_Group group = MPI_GROUP_NULL);
    [[nodiscard]] std::shared_ptr<CommunicatorState> snapshot_state() const;

    std::shared_ptr<CommunicatorState> state_;
    std::atomic_flag revoked_{};  // set when MPIX_Comm_revoke() is called; cleared after shrink
    // Atomic so that set_failure_policy() on one thread and MPI_CHECK_CTX reads
    // on another thread do not race.  Use memory_order_release on write,
    // memory_order_acquire on read.
    std::atomic<FailurePolicy> failure_policy_{FailurePolicy::FAST_FAIL};
    // Set while revoke_and_shrink() is in progress; caught by TT_ASSERT to
    // detect concurrent invocations, which violate the single-caller invariant.
    std::atomic_flag shrink_in_progress_{};

    // Detection-time cache of failed ranks, populated by handle_rank_failure()
    // *before* MPIX_Comm_revoke() is called and before throwing
    // MPIRankFailureException.
    //
    // This cache exists to work around the REVOKED-path failure case (see the
    // failed_ranks() comment above).  Ranks that receive MPIX_ERR_REVOKED have
    // a revoked communicator by the time they enter handle_rank_failure(); the
    // best-effort MPIX_Comm_failure_ack() in identify_failed_ranks() may still
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

}  // namespace tt::tt_metal::distributed::multihost
