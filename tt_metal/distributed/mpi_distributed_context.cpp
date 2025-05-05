// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_distributed_context.hpp"
#include <mpi.h>
namespace tt::tt_metal::distributed::multihost {

Status MPIRequest::wait() {
    // wait() returns a boost::mpi::status
    boost::mpi::status st = req_.wait();
    done_ = true;
    return Status{Rank(st.source()), Tag(st.tag()), st.count<char>()};
}
std::optional<Status> MPIRequest::test() {
    boost::mpi::status st;
    if (req_.test(st)) {
        done_ = true;
        return Status{Rank(st.source()), Tag(st.tag()), st.count<char>()};
    }
    return std::nullopt;
}
void MPIRequest::cancel() override {
    // Boost.MPI exposes cancel() on its request
    req_.cancel();
    done_ = true;
}
bool MPIRequest::active() const override { return !done_; }
void MPIContext::init(int& argc, char**& argv) { static boost::mpi::environment env(argc, argv); }

std::shared_ptr<IDistributedContext> MPIContext::create(int argc, char** argv) {
    init(argc, argv);
    boost::mpi::communicator world;  // wraps MPI_COMM_WORLD
    return std::make_shared<MPIContext>(world);
}
Rank MPIContext::rank() const { return Rank(comm_.rank()); }
Size MPIContext::size() const { return Size(comm_.size()); }
void MPIContext::barrier() const override { boost::mpi::barrier(comm_); }
void MPIContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    comm_.send(*dest), *tag, reinterpret_cast<const char*>(buf.data()), buf.size());
}
void MPIContext::recv(tt::stl::Span<std::byte> buf, Rank source, Tag tag, Status* status) const {
    boost::mpi::status st;
    comm_.recv(*source), *tag, reinterpret_cast<char*>(buf.data()), buf.size(), st);
    if (status) {
        *status = Status{Rank(st.source()), Tag(st.tag()), st.count<char>()};
    }
}
RequestPtr MPIContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    auto r = comm_.isend(
        *dest), *tag, reinterpret_cast<const char*>(buf.data()), buf.size());
    return std::make_shared<MPIRequest>(std::move(r));
}
RequestPtr MPIContext::irecv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const {
    auto r =
        comm_.irecv(*source), *tag, reinterpret_cast<char*>(buf.data()), buf.size());
    return std::make_shared<MPIRequest>(std::move(r));
}
Status MPIContext::wait(IRequest& req) const { return req.wait(); }
std::vector<Status> MPIContext::wait_all(tt::stl::Span<RequestPtr> reqs) const {
    std::vector<Status> out;
    out.reserve(reqs.size());
    for (auto& r : reqs) {
        out.push_back(r->wait());
    }
    return out;
}
Status MPIContext::wait_any(tt::stl::Span<RequestPtr> reqs, int& index) const {
    while (true) {
        for (size_t i = 0; i < reqs.size(); ++i) {
            if (auto s = reqs[i]->test()) {
                index = *i;
                return *s;
            }
        }
        std::this_thread::yield();  // yield to avoid busy waiting
    }
}
void MPIContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const {
    boost::mpi::broadcast(comm_, reinterpret_cast<char*>(buf.data()), buf.size(), *root));
}
void MPIContext::all_reduce(
    tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const {
    boost::mpi::all_reduce(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        reinterpret_cast<char*>(recv_buf.data()),
        send_buf.size(),
        reduce_to_mpi(op));
}
void MPIContext::reduce(
    tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, Rank root) const {
    boost::mpi::reduce(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        reinterpret_cast<char*>(recv_buf.data()),
        send_buf.size(),
        reduce_to_mpi(op),
        *root);
}
void MPIContext::gather(tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    boost::mpi::gather(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        send_buf.size(),
        reinterpret_cast<char*>(recv_buf.data()),
        send_buf.size(),
        *root);
}
void MPIContext::scatter(tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    boost::mpi::scatter(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        recv_buf.size(),
        reinterpret_cast<char*>(recv_buf.data()),
        recv_buf.size(),
        *root);
}
void MPIContext::all_gather(tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    boost::mpi::all_gather(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        send_buf.size(),
        reinterpret_cast<char*>(recv_buf.data()),
        send_buf.size());
}
void MPIContext::all_to_all(tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    int n = comm_.size();
    int send_block = *send_buf.size() / n;
    int recv_block = *recv_buf.size() / n;
    boost::mpi::all_to_all(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        send_block,
        reinterpret_cast<char*>(recv_buf.data()),
        recv_block);
}
void MPIContext::reduce_scatter(
    tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const {
    int n = comm_.size();
    std::vector<int> counts(n, recv_buf.size());
    boost::mpi::reduce_scatter(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        reinterpret_cast<char*>(recv_buf.data()),
        counts.data(),
        reduce_to_mpi(op));
}
void MPIContext::scan(tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const {
    boost::mpi::scan(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        reinterpret_cast<char*>(recv_buf.data()),
        send_buf.size(),
        reduce_to_mpi(op));
}
std::shared_ptr<IDistributedContext> MPIContext::duplicate() const {
    MPI_Comm dup;
    MPI_Comm_dup(comm_.mpi_comm(), &dup);
    return std::make_shared<MPIContext>(boost::mpi::communicator(dup, boost::mpi::comm_take_ownership));
}
std::shared_ptr<IDistributedContext> MPIContext::split(Color color, Key key) const {
    mpi::communicator split_comm = world.split(color, key);
    return std::make_shared<MPIContext>(boost::mpi::communicator(split_c, boost::mpi::comm_take_ownership));
}
static MPI_Op reduce_to_mpi(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM: return MPI_SUM;
        case ReduceOp::MAX: return MPI_MAX;
        case ReduceOp::MIN: return MPI_MIN;
        case ReduceOp::PROD: return MPI_PROD;
    }
    return MPI_SUM;  // fallback
}

std::shared_ptr<IDistributedContext> MPIContext::create_sub_context(tt::stl::Span<Rank> ranks) const {
    mpi::group world_group = comm_.group();
    mpi::group sub_group = world_group.include(ranks.begin(), ranks.end());
    mpi::communicator sub_comm = comm_.create(sub_group);
    return std::make_shared<MPIContext>(sub_comm);
}
MPIContext::MPIContext(boost::mpi::communicator c) : comm_(std::move(c)) {}

std::shared_ptr<IDistributedContext> IDistributedContext::create(int argc, char** argv) {
    return MPIContext::create(argc, argv);
}
}  // namespace tt::tt_metal::distributed::multihost
