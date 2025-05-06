// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_distributed_context.hpp"
#include <mpi.h>

namespace tt::tt_metal::distributed::multihost {

static MPI_Op reduce_to_mpi(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM: return MPI_SUM;
        case ReduceOp::MAX: return MPI_MAX;
        case ReduceOp::MIN: return MPI_MIN;
        case ReduceOp::PROD: return MPI_PROD;
        default: throw std::logic_error("Wrong reduce Op type");
    }
    return MPI_SUM;
}

template <class T>
inline std::optional<T> to_std_optional(boost::optional<T>&& opt) {
    if (opt) {
        return std::move(opt.value());
    } else {
        return std::nullopt;
    }
}
Status MPIRequest::wait() {
    // wait() returns a boost::mpi::status
    boost::mpi::status st = req_.wait();
    done_ = true;
    auto boost_opt = st.count<char>();
    int size = boost_opt.has_value() ? boost_opt.value() : 0;
    return Status{Rank(st.source()), Tag(st.tag()), size};
}
std::optional<Status> MPIRequest::test() {
    auto st = req_.test();
    if (!st.has_value()) {
        return std::nullopt;
    };
    auto boost_opt = st->count<char>();
    int size = boost_opt.has_value() ? boost_opt.value() : 0;
    return Status{Rank(st->source()), Tag(st->tag()), size};
}
void MPIRequest::cancel() {
    // Boost.MPI exposes cancel() on its request
    req_.cancel();
    done_ = true;
}
bool MPIRequest::active() const { return !done_; }
void MPIContext::init(int& argc, char**& argv) { static boost::mpi::environment env(argc, argv); }

std::shared_ptr<IDistributedContext> MPIContext::create(int argc, char** argv) {
    init(argc, argv);
    boost::mpi::communicator world;  // wraps MPI_COMM_WORLD
    return std::make_shared<MPIContext>(world);
}
Rank MPIContext::rank() const { return Rank(comm_.rank()); }
Size MPIContext::size() const { return Size(comm_.size()); }
void MPIContext::barrier() const { comm_.barrier(); }

void MPIContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    comm_.send(*dest, *tag, reinterpret_cast<const char*>(buf.data()), buf.size());
}
void MPIContext::recv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const {
    comm_.recv(*source, *tag, reinterpret_cast<char*>(buf.data()), buf.size());
}
RequestPtr MPIContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    auto r = comm_.isend(*dest, *tag, reinterpret_cast<const char*>(buf.data()), buf.size());
    return std::make_shared<MPIRequest>(std::move(r));
}
RequestPtr MPIContext::irecv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const {
    auto r = comm_.irecv(*source, *tag, reinterpret_cast<char*>(buf.data()), buf.size());
    return std::make_shared<MPIRequest>(std::move(r));
}

void MPIContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const {
    boost::mpi::broadcast(comm_, reinterpret_cast<char*>(buf.data()), buf.size(), *root);
}
void MPIContext::all_reduce(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const {
    throw std::logic_error(" MPIContext::all_reduce not yet implemented");
    // overall there is no sense to introduce reduce ops on bytes.
    // If we need any in fututre need to add overloads for other types of the data
    /*
    boost::mpi::all_reduce(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        send_buf.size(),
        reinterpret_cast<char*>(recv_buf.data()),
        reduce_to_mpi(op));
    */
}
void MPIContext::reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, Rank root) const {
    throw std::logic_error(" MPIContext::reduce not yet implemented");
    // overall there is no sense to introduce reduce ops on bytes.
    // If we need any in fututre need to add overloads for other types of the data
    /*
boost::mpi::reduce(
    comm_,
    reinterpret_cast<const char*>(send_buf.data()),
    send_buf.size(),
    reinterpret_cast<char*>(recv_buf.data()),
    reduce_to_mpi(op),
    *root);
    */
}
void MPIContext::gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    boost::mpi::gather(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        send_buf.size(),
        reinterpret_cast<char*>(recv_buf.data()),
        *root);
}
void MPIContext::scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    boost::mpi::scatter(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        reinterpret_cast<char*>(recv_buf.data()),
        send_buf.size(),
        *root);
}
void MPIContext::all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    boost::mpi::all_gather(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        send_buf.size(),
        reinterpret_cast<char*>(recv_buf.data()));
}
void MPIContext::all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    int n = comm_.size();
    int send_block = send_buf.size() / n;
    int recv_block = recv_buf.size() / n;
    boost::mpi::all_to_all(
        comm_, reinterpret_cast<char*>(send_buf.data()), send_block, reinterpret_cast<char*>(recv_buf.data()));
}

void MPIContext::reduce_scatter(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const {
    throw std::logic_error(" MPIContext::reduce_scatter not yet implemented");
    /*
    // I didn't find it
int n = comm_.size();
std::vector<int> counts(n, recv_buf.size());
boost::mpi::reduce_scatter(
    comm_,
    reinterpret_cast<const char*>(send_buf.data()), send_buf.size(),
    reinterpret_cast<char*>(      recv_buf.data()),
    counts.data(),
    reduce_to_mpi(op));*/
}

void MPIContext::scan(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const {
    throw std::logic_error(" MPIContext::scan not yet implemented");
    // overall there is no sense to introduce any reduce ops on bytes.
    // If we need any in fututre need to add overloads for other types of the data
    /*

    boost::mpi::scan(
        comm_,
        reinterpret_cast<const char*>(send_buf.data()),
        send_buf.size(),
        reinterpret_cast<char*>(recv_buf.data()),
        reduce_to_mpi(op));
        */
}

std::shared_ptr<IDistributedContext> MPIContext::duplicate() const {
    return std::make_shared<MPIContext>(boost::mpi::communicator(MPI_Comm(comm_), boost::mpi::comm_duplicate));
}

std::shared_ptr<IDistributedContext> MPIContext::split(Color color, Key key) const {
    boost::mpi::communicator split_comm = comm_.split(*color, *key);
    return std::make_shared<MPIContext>(split_comm);
}

std::shared_ptr<IDistributedContext> MPIContext::create_sub_context(tt::stl::Span<Rank> ranks) const {
    boost::mpi::group world_group = comm_.group();
    std::vector<int> int_ranks(ranks.size());
    std::transform(ranks.begin(), ranks.end(), int_ranks.begin(), [](Rank r) { return *r; });
    boost::mpi::group sub_group = world_group.include(int_ranks.begin(), int_ranks.end());

    return std::make_shared<MPIContext>(boost::mpi::communicator(comm_, sub_group));
}
MPIContext::MPIContext(boost::mpi::communicator c) : comm_(std::move(c)) {}

std::shared_ptr<IDistributedContext> IDistributedContext::create(int argc, char** argv) {
    return MPIContext::create(argc, argv);
}
}  // namespace tt::tt_metal::distributed::multihost
