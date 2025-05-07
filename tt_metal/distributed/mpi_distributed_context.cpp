// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "mpi_distributed_context.hpp"
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <limits>
#include "assert.hpp"
namespace tt::tt_metal::distributed::multihost {

namespace {

/* ----------------------------- helpers ---------------------------------- */

inline MPI_Op reduce_to_mpi(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:  return MPI_SUM;
        case ReduceOp::MAX:  return MPI_MAX;
        case ReduceOp::MIN:  return MPI_MIN;
        case ReduceOp::PROD: return MPI_PROD;
    }
    return MPI_SUM; // default
}

inline void check_size_fits_int(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TT_THROW("MPI buffer size > INT_MAX");
    }
}

inline void throw_if_error(
    int rc, const char* call_name, std::source_location where = std::source_location::current()) {
    if (rc == MPI_SUCCESS) {
        return;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ostringstream msg;
    msg << call_name << " failed on rank " << rank << " (" << where.file_name() << ':' << where.line() << ')';

    throw MPIDistributedException(Rank{rank}, rc, msg.str());
}

template <auto MpiFunc, class... Args>
void mpi_call(Args&&... args, std::source_location where = std::source_location::current()) {
    static_assert(std::is_pointer_v<decltype(MpiFunc)>, "Template parameter must be a (constexpr) function pointer");
    static_assert(std::is_same_v<int, decltype(MpiFunc(std::forward<Args>(args)...))>, "MPI functions must return int");

    int rc = MpiFunc(std::forward<Args>(args)...);
    throw_if_error(rc, /*call name*/ __func__, where);
}

} // namespace

MPIDistributedException::MPIDistributedException(Rank rank, int error_code, std::string msg) :
    rank_(rank), error_code_(error_code), message_(std::move(msg)) {
    // retrieve human-readable MPI error string
    char buf[MPI_MAX_ERROR_STRING] = {0};
    int len = 0;
    MPI_Error_string(error_code_, buf, &len);
    error_string_.assign(buf, len);
}

// implement interface
Rank MPIDistributedException::rank() const noexcept override { return rank_; }

int MPIDistributedException::error_code() const noexcept override { return error_code_; }

const std::string& MPIDistributedException::message() const noexcept override { return message_; }

const std::string& MPIDistributedException::error_string() const noexcept override { return error_string_; }

/* -------------------------- MPIRequest ---------------------------------- */

Status MPIRequest::wait() {
    MPI_Status status{};
    mpi_call<MPI_Wait>(&req_, &status);
    done_ = true;

    int count = 0;
    mpi_call<MPI_Get_count>(&status, MPI_CHAR, &count);
    return Status{Rank(status.MPI_SOURCE), Tag(status.MPI_TAG), count};
}

std::optional<Status> MPIRequest::test() {
    MPI_Status status{};
    int flag = 0;
    mpi_call<MPI_Test>(&req_, &flag, &status);
    if (!flag) return std::nullopt;

    done_ = true;
    int count = 0;
    mpi_call<MPI_Get_count>(&status, MPI_CHAR, &count);
    return Status{Rank(status.MPI_SOURCE), Tag(status.MPI_TAG), count};
}

void MPIRequest::cancel() {
    if (done_) return;
    mpi_call<MPI_Cancel>(&req_);
    mpi_call<MPI_Request_free>(&req_);
    done_ = true;
}

bool MPIRequest::active() const { return !done_; }

/* -------------------------- MPIContext ---------------------------------- */

static void init_env(int& argc, char**& argv)
{
    static bool initialized = false;
    if (!initialized) {
        int provided = 0;
        int rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if (rc != MPI_SUCCESS) {
            TT_THROW("MPI_Init_thread failed");
        }
        initialized = true;
        std::atexit([] { MPI_Finalize(); });
    }
}

std::shared_ptr<DistributedContext> MPIContext::create(int argc, char** argv)
{
    init_env(argc, argv);
    return std::make_shared<MPIContext>(MPI_COMM_WORLD);
}

MPIContext::MPIContext(MPI_Comm comm) : comm_(comm)
{
    mpi_call<MPI_Comm_rank>(comm_, &rank_);
    mpi_call<MPI_Comm_size>(comm_, &size_);
}

Rank  MPIContext::rank()  const { return Rank(rank_); }
Size  MPIContext::size()  const { return Size(size_); }
void MPIContext::barrier() const { mpi_call<MPI_Barrier>(comm_); }

/* ---- point‑to‑point ---------------------------------------------------- */

void MPIContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const
{
    check_size_fits_int(buf.size());
    mpi_call<MPI_Send>(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_);
}

void MPIContext::recv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const
{
    check_size_fits_int(buf.size());
    mpi_call<MPI_Recv>(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, MPI_STATUS_IGNORE);
}

RequestPtr MPIContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const
{
    check_size_fits_int(buf.size());
    MPI_Request req{};
    mpi_call<MPI_Isend>(
        const_cast<std::byte*>(buf.data()), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_, &req);
    return std::make_shared<MPIRequest>(req);
}

RequestPtr MPIContext::irecv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const
{
    check_size_fits_int(buf.size());
    MPI_Request req{};
    mpi_call<MPI_Irecv>(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, &req);
    return std::make_shared<MPIRequest>(req);
}

/* ---- collectives ------------------------------------------------------- */

void MPIContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const
{
    check_size_fits_int(buf.size());
    mpi_call<MPI_Bcast>(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *root, comm_);
}

void MPIContext::all_reduce(tt::stl::Span<std::byte> send_buf,
                            tt::stl::Span<std::byte> recv_buf,
                            ReduceOp op) const
{
    if (send_buf.size() != recv_buf.size()) {
        TT_THROW("send/recv buffer size mismatch, {} != {}", send_buf.size(), recv_buf.size());
    }
    check_size_fits_int(send_buf.size());

    mpi_call<MPI_Allreduce>(
        send_buf.data(), recv_buf.data(), static_cast<int>(send_buf.size()), MPI_CHAR, reduce_to_mpi(op), comm_);
}

void MPIContext::reduce(tt::stl::Span<std::byte> send_buf,
                         tt::stl::Span<std::byte> recv_buf,
                         ReduceOp op, Rank root) const
{
    if (recv_buf.size() < send_buf.size()) {
        TT_THROW("recv buffer too small: {} < {}", recv_buf.size(), send_buf.size());
    }
    check_size_fits_int(send_buf.size());

    mpi_call<MPI_Reduce>(
        send_buf.data(), recv_buf.data(), static_cast<int>(send_buf.size()), MPI_CHAR, reduce_to_mpi(op), *root, comm_);
}

void MPIContext::gather(tt::stl::Span<std::byte> send_buf,
                        tt::stl::Span<std::byte> recv_buf, Rank root) const
{
    check_size_fits_int(send_buf.size());
    mpi_call<MPI_Gather>(
        send_buf.data(),
        static_cast<int>(send_buf.size()),
        MPI_CHAR,
        recv_buf.data(),
        static_cast<int>(send_buf.size()),
        MPI_CHAR,
        *root,
        comm_);
}

void MPIContext::scatter(tt::stl::Span<std::byte> send_buf,
                         tt::stl::Span<std::byte> recv_buf, Rank root) const
{
    check_size_fits_int(recv_buf.size());
    mpi_call<MPI_Scatter>(
        send_buf.data(),
        static_cast<int>(recv_buf.size()),
        MPI_CHAR,
        recv_buf.data(),
        static_cast<int>(recv_buf.size()),
        MPI_CHAR,
        *root,
        comm_);
}

void MPIContext::all_gather(tt::stl::Span<std::byte> send_buf,
                            tt::stl::Span<std::byte> recv_buf) const
{
    check_size_fits_int(send_buf.size());
    mpi_call<MPI_Allgather>(
        send_buf.data(),
        static_cast<int>(send_buf.size()),
        MPI_CHAR,
        recv_buf.data(),
        static_cast<int>(send_buf.size()),
        MPI_CHAR,
        comm_);
}

void MPIContext::all_to_all(tt::stl::Span<std::byte> send_buf,
                            tt::stl::Span<std::byte> recv_buf) const
{
    int n = size_;
    if (send_buf.size() % n || recv_buf.size() % n) {
        TT_THROW("send buffer size {} or recv buffer size {} not divisible by world size {}", send_buf.size(), recv_buf.size(), n);
    }

    int block = static_cast<int>(send_buf.size() / n);
    mpi_call<MPI_Alltoall>(send_buf.data(), block, MPI_CHAR, recv_buf.data(), block, MPI_CHAR, comm_);
}

void MPIContext::reduce_scatter(tt::stl::Span<std::byte> send_buf,
                                tt::stl::Span<std::byte> recv_buf,
                                ReduceOp op) const
{
    int n = size_;
    if (send_buf.size() % n) {
        TT_THROW("send buffer size {} not divisible by world size {}", send_buf.size(), n);
    }

    int recv_count = static_cast<int>(send_buf.size() / n);
    check_size_fits_int(recv_count);

    mpi_call<MPI_Reduce_scatter>(send_buf.data(), recv_buf.data(), &recv_count, MPI_CHAR, reduce_to_mpi(op), comm_);
}

void MPIContext::scan(tt::stl::Span<std::byte> send_buf,
                      tt::stl::Span<std::byte> recv_buf,
                      ReduceOp op) const
{
    if (send_buf.size() != recv_buf.size()) {
        TT_THROW("send {}/recv {} buffer size mismatch ", send_buf.size(), recv_buf.size());
    }
    check_size_fits_int(send_buf.size());

    mpi_call<MPI_Scan>(
        send_buf.data(), recv_buf.data(), static_cast<int>(send_buf.size()), MPI_CHAR, reduce_to_mpi(op), comm_);
}

/* ---- communicator management ------------------------------------------ */

std::shared_ptr<DistributedContext> MPIContext::duplicate() const
{
    MPI_Comm dup;
    mpi_call<MPI_Comm_dup>(comm_, &dup);
    return std::make_shared<MPIContext>(dup);
}

std::shared_ptr<DistributedContext> MPIContext::split(Color color, Key key) const
{
    MPI_Comm split_comm;
    mpi_call<MPI_Comm_split>(comm_, *color, *key, &split_comm);
    return std::make_shared<MPIContext>(split_comm);
}

std::shared_ptr<DistributedContext>
MPIContext::create_sub_context(tt::stl::Span<Rank> ranks) const
{
    MPI_Group world_grp, sub_grp;
    mpi_call<MPI_Comm_group>(comm_, &world_grp);

    std::vector<int> int_ranks(ranks.size());
    std::transform(ranks.begin(), ranks.end(), int_ranks.begin(),
                   [](Rank r) { return *r; });

    mpi_call<MPI_Group_incl>(world_grp, static_cast<int>(int_ranks.size()), int_ranks.data(), &sub_grp);

    MPI_Comm sub_comm;
    mpi_call<MPI_Comm_create_group>(comm_, sub_grp, 0 /*tag*/, &sub_comm);
    mpi_call<MPI_Group_free>(&sub_grp);
    mpi_call<MPI_Group_free>(&world_grp);

    return std::make_shared<MPIContext>(sub_comm);
}

void MPIContext::abort(int error_code) const { mpi_call<MPI_Abort>(comm_, error_code); }
/* -------------------- factory for generic interface --------------------- */
std::shared_ptr<DistributedContext> DistributedContext::create(int argc, char** argv)
{
    return MPIContext::create(argc, argv);
}

} // namespace tt::tt_metal::distributed::multihost
