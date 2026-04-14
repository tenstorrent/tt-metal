// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mpi.h>
#include <memory>
#include <optional>
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
    /// MPI communicator spanning the full job (`MPI_COMM_WORLD`). Rank/size are global even when
    /// `get_current_world()` was split via `TT_RUN_SUBCONTEXT_ID`. Safe to use for cross-subcontext p2p.
    static ContextPtr get_world_context();
    static bool is_initialized();

    [[nodiscard]] std::optional<SubcontextId> subcontext_id() const override;
    [[nodiscard]] int subcontext_count() const override;
    [[nodiscard]] Size subcontext_size(SubcontextId subcontext_id) const override;
    [[nodiscard]] tt::stl::Span<const int> subcontext_sizes() const override;
    [[nodiscard]] Rank local_to_world_rank(SubcontextId subcontext_id, Rank local_rank) const override;

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

    /* ------------- message snooping ------------- */
    std::size_t snoop_incoming_msg_size(Rank source, Tag tag) const override;

    /* ----------------- mpi constructors ---------------- */
    explicit MPIContext(MPI_Comm comm);
    explicit MPIContext(MPI_Comm comm, MPI_Group group);
    const MPI_Comm& comm() const { return comm_; }
    const MPI_Group& group() const { return group_; }

    static void set_current_world(const ContextPtr& ctx);

private:
    static void refresh_launcher_layout_from_env();

    MPI_Comm comm_{MPI_COMM_NULL};
    MPI_Group group_{MPI_GROUP_NULL};
    int rank_{0};
    int size_{0};

    // Subcommunicator for this process after optional TT_RUN_SUBCONTEXT_ID split (or duplicate of world).
    inline static ContextPtr current_world_;
    // Lazily-created view of MPI_COMM_WORLD (not freed in destructor).
    inline static ContextPtr mpi_job_world_;
};

}  // namespace tt::tt_metal::distributed::multihost
