// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mpi.h>
#include "api/tt-metalium/distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {

// ---------------------------------------------------------------------
//                           Non‑blocking request
// ---------------------------------------------------------------------
class MPIRequest : public Request {
public:
    explicit MPIRequest(MPI_Request req) : req_(req), done_(false) {}

    Status               wait()  override;
    std::optional<Status> test()  override;
    void                 cancel() override;
    bool                 active() const override;

private:
    mutable MPI_Request req_{};
    bool                done_{};
};

// ---------------------------------------------------------------------
//                       Main distributed context
// ---------------------------------------------------------------------
class MPIContext : public DistributedContext {
public:
    // factory (initialises MPI environment once per process)
    static std::shared_ptr<DistributedContext> create(int argc, char** argv);

    // destructor – communicator freed automatically by MPI_Finalize
    ~MPIContext() override = default;

    /* ---------------- basic info / sync ---------------- */
    [[nodiscard]] Rank rank() const override;
    [[nodiscard]] Size size() const override;
    void barrier() const override;

    /* ---------------- point‑to‑point ------------------- */
    void send (tt::stl::Span<std::byte> buf, Rank dest,   Tag tag) const override;
    void recv (tt::stl::Span<std::byte> buf, Rank source, Tag tag) const override;

    [[nodiscard]] RequestPtr isend(tt::stl::Span<std::byte> buf, Rank dest,   Tag tag) const override;
    [[nodiscard]] RequestPtr irecv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const override;

    /* ---------------- collectives ---------------------- */
    void broadcast    (tt::stl::Span<std::byte> buf,                              Rank root) const override;
    void all_reduce   (tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf,
                       ReduceOp op) const override;
    void reduce       (tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf,
                       ReduceOp op, Rank root) const override;
    void gather       (tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf,
                       Rank root) const override;
    void scatter      (tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf,
                       Rank root) const override;
    void all_gather   (tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;
    void all_to_all   (tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;
    void reduce_scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf,
                        ReduceOp op) const override;
    void scan         (tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf,
                       ReduceOp op) const override;

    /* ------------- communicator management ------------- */
    [[nodiscard]] std::shared_ptr<DistributedContext> duplicate() const override;
    [[nodiscard]] std::shared_ptr<DistributedContext> split(Color color, Key key) const override;
    [[nodiscard]] std::shared_ptr<DistributedContext> create_sub_context(tt::stl::Span<Rank> ranks) const override;

    explicit MPIContext(MPI_Comm comm);

private:
    MPI_Comm comm_{MPI_COMM_NULL};
    int      rank_{0};
    int      size_{0};
};

} // namespace tt::tt_metal::distributed::multihost
