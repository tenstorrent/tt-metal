// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/mpi.hpp>
#include "api/tt-metalium/distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {

// --- non-blocking request wrapper ----------------------------------------
class MPIRequest : public IRequest {
public:
    explicit MPIRequest(boost::mpi::request&& r) : req_(std::move(r)), done_(false) {}

    Status wait() override;

    std::optional<Status> test() override;

    void cancel() override;

    bool active() const override;

private:
    mutable boost::mpi::request req_;
    bool done_{};
};

// --- MPIContext implementation ------------------------------------------
class MPIContext : public IDistributedContext {
public:
    // single init of MPI

    static void init(int& argc, char**& argv);

    // factory
    static std::shared_ptr<IDistributedContext> create(int argc, char** argv);

    // destructor (communicator and environment cleaned up automatically)
    ~MPIContext() override = default;

    // rank & size
    [[nodiscard]] Rank rank() const override;
    [[nodiscard]] Size size() const override;

    // barrier
    void barrier() const override;

    // blocking send/recv
    void send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;

    void recv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const override;

    // non-blocking send/recv
    [[nodiscard]] RequestPtr isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;

    [[nodiscard]] RequestPtr irecv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const override;

    // --- collectives via Boost.MPI free functions --------------------------

    void broadcast(tt::stl::Span<std::byte> buf, Rank root) const override;

    void all_reduce(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const override;

    void reduce(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, Rank root) const override;

    void gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const override;

    void scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const override;

    void all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;

    void all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;

    void reduce_scatter(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const override;

    void scan(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const override;

    // --- communicator management via raw MPI calls -------------------------

    [[nodiscard]] std::shared_ptr<IDistributedContext> duplicate() const override;

    [[nodiscard]] std::shared_ptr<IDistributedContext> split(Color color, Key key) const override;

    [[nodiscard]] std::shared_ptr<IDistributedContext> create_sub_context(tt::stl::Span<Rank> ranks) const override;

    explicit MPIContext(boost::mpi::communicator c);

private:
    boost::mpi::communicator comm_;
};

}  // namespace tt::tt_metal::distributed::multihost
