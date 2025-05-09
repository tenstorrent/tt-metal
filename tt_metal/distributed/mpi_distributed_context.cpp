// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "mpi_distributed_context.hpp"
#include <mpi.h>
#include <algorithm>
#include <limits>
#include "assert.hpp"

namespace tt::tt_metal::distributed::multihost {

/* ----------------------------- helpers ---------------------------------- */

constexpr MPI_Op reduce_to_mpi(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:  return MPI_SUM;
        case ReduceOp::MAX:  return MPI_MAX;
        case ReduceOp::MIN:  return MPI_MIN;
        case ReduceOp::PROD: return MPI_PROD;
        case ReduceOp::LAND: return MPI_LAND;
        case ReduceOp::LOR: return MPI_LOR;
        case ReduceOp::BAND: return MPI_BAND;
        case ReduceOp::BOR: return MPI_BOR;
    }
    return MPI_SUM; // default
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

constexpr inline int mpi_dtype_size(DType dt) noexcept {
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
        case DType::FLOAT64: return 8;
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

inline void mpi_check(int error_code, const char* call_text) {
    if (error_code != MPI_SUCCESS) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // throw with the textual form of the call
        throw MPIDistributedException(Rank{rank}, error_code, std::string(call_text) + " failed");
    }
}

#define MPI_CHECK(call) mpi_check((call), #call)

MPIDistributedException::MPIDistributedException(Rank rank, int error_code, std::string msg) :
    rank_(rank), error_code_(error_code), message_(std::move(msg)) {
    // retrieve human-readable MPI error string
    char buf[MPI_MAX_ERROR_STRING] = {0};
    int len = 0;
    MPI_Error_string(error_code_, buf, &len);
    error_string_.assign(buf, len);
}

// implement interface
Rank MPIDistributedException::rank() const noexcept { return rank_; }

int MPIDistributedException::error_code() const noexcept { return error_code_; }

const std::string& MPIDistributedException::message() const noexcept { return message_; }

const std::string& MPIDistributedException::error_string() const noexcept { return error_string_; }

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
    if (!flag) return std::nullopt;

    done_ = true;
    int count = 0;
    MPI_CHECK(MPI_Get_count(&status, MPI_CHAR, &count));
    return Status{Rank(status.MPI_SOURCE), Tag(status.MPI_TAG), count};
}

void MPIRequest::cancel() {
    if (done_) return;
    MPI_CHECK(MPI_Cancel(&req_));
    MPI_CHECK(MPI_Request_free(&req_));
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

MPIContext::MPIContext(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);  // don't abort on error
    MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
    MPI_CHECK(MPI_Comm_size(comm_, &size_));
}

Rank  MPIContext::rank()  const { return Rank(rank_); }
Size  MPIContext::size()  const { return Size(size_); }
void MPIContext::barrier() const { MPI_CHECK(MPI_Barrier(comm_)); }

/* ---- point‑to‑point ---------------------------------------------------- */

void MPIContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const
{
    check_size_fits_int(buf.size());
    MPI_CHECK(MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_));
}

void MPIContext::recv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const
{
    check_size_fits_int(buf.size());
    MPI_CHECK(MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, MPI_STATUS_IGNORE));
}

RequestPtr MPIContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const
{
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Isend(
        const_cast<std::byte*>(buf.data()), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_, &req));
    return std::make_shared<MPIRequest>(req);
}

RequestPtr MPIContext::irecv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const
{
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Irecv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, &req));
    return std::make_shared<MPIRequest>(req);
}

/* ---- collectives ------------------------------------------------------- */

void MPIContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const
{
    check_size_fits_int(buf.size());
    MPI_CHECK(MPI_Bcast(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *root, comm_));
}

void MPIContext::all_reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    check_size_fits_int(send_buf.size());

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

    MPI_CHECK(MPI_Allreduce(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), comm_));
}

void MPIContext::reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype, Rank root) const {
    const int elem_sz = mpi_dtype_size(dtype);
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

    MPI_CHECK(MPI_Reduce(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), *root, comm_));
}

void MPIContext::gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    const int send_count = static_cast<int>(send_buf.size());
    check_size_fits_int(send_count);

    // Root must have room for 'size()' times the per‑rank payload.
    if (rank() == root) {
        const std::size_t expected = static_cast<std::size_t>(send_count) * (*size());
        TT_FATAL(
            recv_buf.size() == expected, "gather: root recv buffer {} bytes, expected {}", recv_buf.size(), expected);
    }

    MPI_CHECK(MPI_Gather(send_buf.data(), send_count, MPI_CHAR, recv_buf.data(), send_count, MPI_CHAR, *root, comm_));
}

void MPIContext::scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    const int recv_count = static_cast<int>(recv_buf.size());
    check_size_fits_int(recv_count);

    if (rank() == root) {
        const std::size_t expected = static_cast<std::size_t>(recv_count) * (*size());
        TT_FATAL(
            send_buf.size() == expected, "scatter: root send buffer {} bytes, expected {}", send_buf.size(), expected);
    }

    MPI_CHECK(MPI_Scatter(send_buf.data(), recv_count, MPI_CHAR, recv_buf.data(), recv_count, MPI_CHAR, *root, comm_));
}

void MPIContext::all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
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

    MPI_CHECK(MPI_Allgather(send_ptr, send_count, MPI_CHAR, recv_buf.data(), send_count, MPI_CHAR, comm_));
}

void MPIContext::all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
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

    MPI_CHECK(MPI_Alltoall(send_buf.data(), block, MPI_CHAR, recv_buf.data(), block, MPI_CHAR, comm_));
}

void MPIContext::reduce_scatter(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    const int world = *size();
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

    const int recv_count = static_cast<int>(recv_elems);
    check_size_fits_int(recv_count);

    MPI_CHECK(MPI_Reduce_scatter(
        send_buf.data(), recv_buf.data(), &recv_count, dtype_to_mpi(dtype), reduce_to_mpi(op), comm_));
}

void MPIContext::scan(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
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

    MPI_CHECK(MPI_Scan(send_ptr, recv_buf.data(), count, dtype_to_mpi(dtype), reduce_to_mpi(op), comm_));
}
/* ---- communicator management ------------------------------------------ */

std::shared_ptr<DistributedContext> MPIContext::duplicate() const
{
    MPI_Comm dup;
    MPI_CHECK(MPI_Comm_dup(comm_, &dup));
    return std::make_shared<MPIContext>(dup);
}

std::shared_ptr<DistributedContext> MPIContext::split(Color color, Key key) const
{
    MPI_Comm split_comm;
    MPI_CHECK(MPI_Comm_split(comm_, *color, *key, &split_comm));
    return std::make_shared<MPIContext>(split_comm);
}

std::shared_ptr<DistributedContext>
MPIContext::create_sub_context(tt::stl::Span<Rank> ranks) const
{
    MPI_Group world_grp, sub_grp;
    MPI_CHECK(MPI_Comm_group(comm_, &world_grp));

    std::vector<int> int_ranks(ranks.size());
    std::transform(ranks.begin(), ranks.end(), int_ranks.begin(),
                   [](Rank r) { return *r; });

    MPI_CHECK(MPI_Group_incl(world_grp, static_cast<int>(int_ranks.size()), int_ranks.data(), &sub_grp));

    MPI_Comm sub_comm;
    MPI_CHECK(MPI_Comm_create_group(comm_, sub_grp, 0 /*tag*/, &sub_comm));

    // TODO: unsafe free calls. now mpi can throw
    MPI_CHECK(MPI_Group_free(&sub_grp));
    MPI_CHECK(MPI_Group_free(&world_grp));

    return std::make_shared<MPIContext>(sub_comm);
}

void MPIContext::abort(int error_code) const { MPI_Abort(comm_, error_code); }

/* -------------------- factory for generic interface --------------------- */
std::shared_ptr<DistributedContext> DistributedContext::create(int argc, char** argv) {
    return MPIContext::create(argc, argv);
}

void MPIContext::revoke_and_shrink() {
    // need to understand that MPI_WORLD_COMM is not a valid communicator anymore after this call
    // and that the context is not valid anymore
    /**
        int rc = MPIX_Comm_revoke(comm_);
        if (rc != MPI_SUCCESS && rc != MPI_ERR_REVOKED) {  // another rank may have revoked first
            abort(rc);
        }


        MPI_Comm new_comm = MPI_COMM_NULL;
        MPI_CHECK(MPIX_Comm_shrink(comm_, &new_comm));

        MPI_Comm_set_errhandler(new_comm, MPI_ERRORS_RETURN);

        // overall probably I don't neet MPI_CHECK, we are recovering here. If we cannot recover, we should abort
        // and not throw an exception.
        int new_rank = 0, new_size = 0;
        MPI_Comm_rank(new_comm, &new_rank);
        MPI_Comm_size(new_comm, &new_size);

        // Free the old communicator *after* shrink completes
        MPI_Comm old_comm = this->comm_;
        if (old_comm != MPI_COMM_NULL && old_comm != new_comm) {
            MPI_Comm_free(&old_comm);
        }

        this->comm_ = new_comm;
        this->rank_ = new_rank;
        this->size_ = new_size;
        */
}

MPIContext::~MPIContext() {
    if (comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&comm_);
    }
}
}  // namespace tt::tt_metal::distributed::multihost
