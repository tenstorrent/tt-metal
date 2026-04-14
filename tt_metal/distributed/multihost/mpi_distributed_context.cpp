// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_distributed_context.hpp"
#include <mpi.h>
#include <mpi-ext.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <tt_stl/assert.hpp>

// Use MPIX_ERR_PROC_FAILED as a proxy to detect whether OpenMPI was built with
// ULFM extensions.
#if (defined(OPEN_MPI) && OPEN_MPI && defined(MPIX_ERR_PROC_FAILED))
#define OMPI_HAS_ULFM 1
#else
#define OMPI_HAS_ULFM 0
#endif

namespace tt::tt_metal::distributed::multihost {

namespace {

struct MpiLauncherEnvLayout {
    bool split_active = false;
    std::optional<int> this_subcontext_id;
    int subcontext_count = 1;
    std::vector<int> subcontext_sizes;
    std::vector<int> world_rank_prefix;
};

MpiLauncherEnvLayout g_mpi_launcher_env_layout{};

std::vector<int> parse_csv_ints(std::string_view csv) {
    std::vector<int> out;
    while (!csv.empty()) {
        auto comma = csv.find(',');
        std::string_view token = csv.substr(0, comma);
        while (!token.empty() && token.front() == ' ') {
            token.remove_prefix(1);
        }
        while (!token.empty() && token.back() == ' ') {
            token.remove_suffix(1);
        }
        TT_FATAL(!token.empty(), "TT_RUN_SUBCONTEXT_SIZES: empty token");
        int v = 0;
        const char* begin = token.data();
        const char* end = begin + token.size();
        auto [ptr, ec] = std::from_chars(begin, end, v);
        TT_FATAL(ec == std::errc{} && ptr == end, "TT_RUN_SUBCONTEXT_SIZES: invalid integer in '{}'", token);
        TT_FATAL(v > 0, "TT_RUN_SUBCONTEXT_SIZES: size must be positive, got {}", v);
        out.push_back(v);
        if (comma == std::string_view::npos) {
            break;
        }
        csv.remove_prefix(comma + 1);
    }
    return out;
}

MpiLauncherEnvLayout parse_launcher_env_layout(int current_context_world_size) {
    MpiLauncherEnvLayout m;
    TT_FATAL(current_context_world_size > 0, "current_context_world_size must be positive");

    const char* id_env = std::getenv("TT_RUN_SUBCONTEXT_ID");
    if (id_env == nullptr || id_env[0] == '\0') {
        m.split_active = false;
        m.this_subcontext_id = std::nullopt;
        m.subcontext_count = 1;
        m.subcontext_sizes = {current_context_world_size};
        m.world_rank_prefix = {0};
        return m;
    }

    m.split_active = true;
    int this_id = 0;
    try {
        this_id = std::stoi(std::string(id_env));
    } catch (const std::exception&) {
        TT_THROW("Invalid TT_RUN_SUBCONTEXT_ID (expected integer): {}", id_env);
    }
    TT_FATAL(this_id >= 0, "TT_RUN_SUBCONTEXT_ID must be non-negative");
    m.this_subcontext_id = this_id;

    const char* count_env = std::getenv("TT_RUN_SUBCONTEXT_COUNT");
    const char* sizes_env = std::getenv("TT_RUN_SUBCONTEXT_SIZES");
    TT_FATAL(
        count_env != nullptr && count_env[0] != '\0',
        "TT_RUN_SUBCONTEXT_ID is set but TT_RUN_SUBCONTEXT_COUNT is missing");
    TT_FATAL(
        sizes_env != nullptr && sizes_env[0] != '\0',
        "TT_RUN_SUBCONTEXT_ID is set but TT_RUN_SUBCONTEXT_SIZES is missing");

    try {
        m.subcontext_count = std::stoi(std::string(count_env));
    } catch (const std::exception&) {
        TT_THROW("Invalid TT_RUN_SUBCONTEXT_COUNT: {}", count_env);
    }
    TT_FATAL(m.subcontext_count > 0, "TT_RUN_SUBCONTEXT_COUNT must be positive");
    m.subcontext_sizes = parse_csv_ints(sizes_env);
    TT_FATAL(
        static_cast<int>(m.subcontext_sizes.size()) == m.subcontext_count,
        "TT_RUN_SUBCONTEXT_SIZES length {} does not match TT_RUN_SUBCONTEXT_COUNT {}",
        m.subcontext_sizes.size(),
        m.subcontext_count);
    TT_FATAL(
        this_id < m.subcontext_count, "TT_RUN_SUBCONTEXT_ID {} out of range for count {}", this_id, m.subcontext_count);

    m.world_rank_prefix.resize(m.subcontext_count);
    int acc = 0;
    for (int i = 0; i < m.subcontext_count; i++) {
        m.world_rank_prefix[i] = acc;
        acc += m.subcontext_sizes[i];
    }

    TT_FATAL(
        m.subcontext_sizes[this_id] == current_context_world_size,
        "TT_RUN_SUBCONTEXT_SIZE / communicator size {} does not match TT_RUN_SUBCONTEXT_SIZES for this id ({})",
        current_context_world_size,
        m.subcontext_sizes[this_id]);

    return m;
}

}  // namespace

/* ----------------------------- helpers ---------------------------------- */

constexpr MPI_Op reduce_to_mpi(ReduceOp op) {
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

constexpr int mpi_dtype_size(DType dt) noexcept {
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

inline void mpi_check(int error_code, const char* call_text) {
    if (error_code != MPI_SUCCESS) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // throw with the textual form of the call
        throw MPIDistributedException(Rank{rank}, error_code, std::string(call_text) + " failed");
    }
}

bool was_mpi_finalized() noexcept {
    int flag = 0;
    /* Safe to call at any time—even before MPI_Init() */
    MPI_Finalized(&flag);  // sets flag = 1 if MPI_Finalize has completed
    return flag != 0;
}

#define MPI_CHECK(call) mpi_check((call), #call)

MPIDistributedException::MPIDistributedException(Rank rank, int error_code, std::string msg) :
    rank_(rank), error_code_(error_code), message_(std::move(msg)) {
    // retrieve human-readable MPI error string
    char buf[MPI_MAX_ERROR_STRING] = {0};
    int len = 0;
    MPI_Error_string(error_code_, buf, &len);
    error_string_.assign(buf, len);
    message_ += ": " + error_string_;
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

inline void init_env(int& argc, char**& argv) {
    static std::once_flag mpi_once;

    std::call_once(mpi_once, [&] {
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) {
            TT_THROW("MPI_Init_thread failed");
        }

        // Ensure MPI_Finalize is called when the program exits
        std::atexit([] { MPI_Finalize(); });
    });
}

void MPIContext::create(int argc, char** argv) {
    if (current_world_) {
        return;
    }
    init_env(argc, argv);

    ContextPtr parent = std::make_shared<MPIContext>(MPI_COMM_WORLD)->duplicate();

    const char* sub_id_str = std::getenv("TT_RUN_SUBCONTEXT_ID");
    if (sub_id_str != nullptr && sub_id_str[0] != '\0') {
        int color = 0;
        try {
            color = std::stoi(std::string(sub_id_str));
        } catch (const std::exception&) {
            TT_THROW("Invalid TT_RUN_SUBCONTEXT_ID (expected integer): {}", sub_id_str);
        }
        TT_FATAL(color >= 0, "TT_RUN_SUBCONTEXT_ID must be non-negative, got {}", color);

        const auto mpi_parent = std::dynamic_pointer_cast<MPIContext>(parent);
        TT_FATAL(mpi_parent != nullptr, "MPIContext::create: parent must be MPIContext");
        int parent_rank = 0;
        MPI_CHECK(MPI_Comm_rank(mpi_parent->comm(), &parent_rank));
        // Key = rank on parent comm so sub-context ranks follow global (parent) order within each color.
        current_world_ = mpi_parent->split(Color(color), Key(parent_rank));
    } else {
        current_world_ = std::move(parent);
    }
    refresh_launcher_layout_from_env();
}

void MPIContext::refresh_launcher_layout_from_env() {
    TT_FATAL(current_world_ != nullptr, "MPIContext: current world not set");
    if (!mpi_job_world_) {
        mpi_job_world_ = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    }
    const int sub_sz = static_cast<int>(*current_world_->size());
    g_mpi_launcher_env_layout = parse_launcher_env_layout(sub_sz);
    if (g_mpi_launcher_env_layout.split_active) {
        int total_ranks = 0;
        for (int s : g_mpi_launcher_env_layout.subcontext_sizes) {
            total_ranks += s;
        }
        TT_FATAL(
            total_ranks == *mpi_job_world_->size(),
            "Sum of TT_RUN_SUBCONTEXT_SIZES ({}) does not match MPI job world size ({})",
            total_ranks,
            *mpi_job_world_->size());
    }
}

const ContextPtr& MPIContext::get_current_world() {
    if (!current_world_) {
        // Default initialization of MPIContext if not already initialized
        MPIContext::create(0, nullptr);
    }
    return current_world_;
}

ContextPtr MPIContext::get_world_context() {
    if (!mpi_job_world_) {
        if (!current_world_) {
            MPIContext::create(0, nullptr);
        }
        mpi_job_world_ = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    }
    return mpi_job_world_;
}

std::optional<SubcontextId> MPIContext::subcontext_id() const {
    const auto& id = g_mpi_launcher_env_layout.this_subcontext_id;
    if (!id.has_value()) {
        return std::nullopt;
    }
    return SubcontextId{*id};
}

int MPIContext::subcontext_count() const { return g_mpi_launcher_env_layout.subcontext_count; }

Size MPIContext::subcontext_size(SubcontextId subcontext_id) const {
    const auto& L = g_mpi_launcher_env_layout;
    TT_FATAL(
        *subcontext_id >= 0 && *subcontext_id < L.subcontext_count,
        "subcontext_id {} out of range [0, {})",
        *subcontext_id,
        L.subcontext_count);
    return Size(L.subcontext_sizes[*subcontext_id]);
}

tt::stl::Span<const int> MPIContext::subcontext_sizes() const {
    const auto& L = g_mpi_launcher_env_layout;
    return {L.subcontext_sizes.data(), L.subcontext_sizes.size()};
}

Rank MPIContext::local_to_world_rank(SubcontextId subcontext_id, Rank local_rank) const {
    const auto& L = g_mpi_launcher_env_layout;
    TT_FATAL(
        *subcontext_id >= 0 && *subcontext_id < L.subcontext_count,
        "subcontext_id {} out of range [0, {})",
        *subcontext_id,
        L.subcontext_count);
    TT_FATAL(
        *local_rank >= 0 && *local_rank < L.subcontext_sizes[*subcontext_id],
        "local_rank {} out of range for sub-context {} (size {})",
        *local_rank,
        *subcontext_id,
        L.subcontext_sizes[*subcontext_id]);
    return Rank{L.world_rank_prefix[*subcontext_id] + *local_rank};
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

MPIContext::MPIContext(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);  // don't abort on error
    MPI_CHECK(MPI_Comm_group(comm_, &group_));
    MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
    MPI_CHECK(MPI_Comm_size(comm_, &size_));
    id_ = DistributedContext::generate_unique_id();
}

MPIContext::MPIContext(MPI_Comm comm, MPI_Group group) : comm_(comm), group_(group) {
    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);  // don't abort on error
    MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
    MPI_CHECK(MPI_Comm_size(comm_, &size_));
    id_ = DistributedContext::generate_unique_id();
}

Rank MPIContext::rank() const { return Rank(rank_); }
Size MPIContext::size() const { return Size(size_); }
bool MPIContext::supports_fault_tolerance() const { return OMPI_HAS_ULFM; }
void MPIContext::barrier() const { MPI_CHECK(MPI_Barrier(comm_)); }

/* ---- point‑to‑point ---------------------------------------------------- */

void MPIContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_CHECK(MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_));
}

void MPIContext::ssend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_CHECK(MPI_Ssend(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_));
}

void MPIContext::recv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_CHECK(MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, MPI_STATUS_IGNORE));
}

RequestPtr MPIContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Isend(
        const_cast<std::byte*>(buf.data()), static_cast<int>(buf.size()), MPI_CHAR, *dest, *tag, comm_, &req));
    return std::make_shared<MPIRequest>(req);
}

RequestPtr MPIContext::irecv(tt::stl::Span<std::byte> buf, Rank src, Tag tag) const {
    check_size_fits_int(buf.size());
    MPI_Request req{};
    MPI_CHECK(MPI_Irecv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, *src, *tag, comm_, &req));
    return std::make_shared<MPIRequest>(req);
}

/* ---- collectives ------------------------------------------------------- */

void MPIContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const {
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

    const int recv_count = static_cast<int>(recv_elems);  // per-rank element count
    check_size_fits_int(recv_count);

    // --- fixed call -------------------------------------------------------
    MPI_CHECK(MPI_Reduce_scatter_block(
        send_buf.data(),      // sendbuf
        recv_buf.data(),      // recvbuf
        recv_count,           // elements per rank
        dtype_to_mpi(dtype),  // element datatype
        reduce_to_mpi(op),    // operation (SUM, MAX, …)
        comm_));              // communicator
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

ContextPtr MPIContext::duplicate() const {
    MPI_Comm dup = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_dup(comm_, &dup));
    return std::make_shared<MPIContext>(dup);
}

ContextPtr MPIContext::split(Color color, Key key) const {
    MPI_Comm split_comm;
    if (*color == SPLIT_COLOR_UNDEFINED) {
        color = Color(MPI_UNDEFINED);
    }
    MPI_CHECK(MPI_Comm_split(comm_, *color, *key, &split_comm));
    return std::make_shared<MPIContext>(split_comm);
}

ContextPtr MPIContext::create_sub_context(tt::stl::Span<int> ranks) const {
    MPI_Group sub_grp = MPI_GROUP_NULL;
    MPI_Comm sub_comm = MPI_COMM_NULL;

    MPI_CHECK(MPI_Group_incl(group_, static_cast<int>(ranks.size()), ranks.data(), &sub_grp));
    if (MPI_Comm_create_group(comm_, sub_grp, 0 /*tag*/, &sub_comm) != MPI_SUCCESS) {
        // sub_comm is not valid, we are not in the group
        MPI_Group_free(&sub_grp);
        throw MPIDistributedException(rank(), MPI_ERR_GROUP, "MPI_Comm_create_group failed: not in the group");
    }
    if (sub_comm == MPI_COMM_NULL) {
        // we are not in the group
        MPI_Group_free(&sub_grp);
        throw MPIDistributedException(rank(), MPI_ERR_GROUP, "MPI_Comm_create_group returned empty comm");
    }
    MPI_Group_free(&sub_grp);
    return std::make_shared<MPIContext>(sub_comm);
}

void MPIContext::translate_ranks_to_other_ctx(
    tt::stl::Span<int> ranks, const ContextPtr& other_ctx, tt::stl::Span<int> translated_ranks) const {
    TT_FATAL(
        ranks.size() == translated_ranks.size(),
        "translate_ranks_to_other_ctx: ranks size {} != translated_ranks size {}",
        ranks.size(),
        translated_ranks.size());
    auto mpi_context = std::dynamic_pointer_cast<MPIContext>(other_ctx);
    TT_FATAL(mpi_context != nullptr, "translate_ranks_to_other_ctx: other_ctx is not a MPIContext");

    MPI_CHECK(MPI_Group_translate_ranks(
        group_, static_cast<int>(ranks.size()), ranks.data(), mpi_context->group(), translated_ranks.data()));
}

void MPIContext::abort(int error_code) const { MPI_Abort(comm_, error_code); }

void MPIContext::revoke_and_shrink() {
#if (!OMPI_HAS_ULFM)
    TT_THROW("revoke_and_shrink() requires MPI ULFM support which is not available in this build");
#else
    int rc = MPIX_Comm_revoke(comm_);
    if (rc != MPI_SUCCESS && rc != MPI_ERR_REVOKED) {  // another rank may have revoked first
        abort(rc);
    }

    MPI_Comm new_comm = MPI_COMM_NULL;
    MPI_Group new_group = MPI_GROUP_NULL;
    MPI_CHECK(MPIX_Comm_shrink(comm_, &new_comm));
    MPI_Comm_group(new_comm, &new_group);

    MPI_Comm_set_errhandler(new_comm, MPI_ERRORS_RETURN);

    // overall probably I don't need MPI_CHECK, we are recovering here. If we cannot recover, we should abort
    // and not throw an exception.
    int new_rank = 0;
    int new_size = 0;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    // Free the old communicator *after* shrink completes
    MPI_Comm old_comm = this->comm_;

    if (old_comm != MPI_COMM_NULL && old_comm != new_comm) {
        MPI_Comm_free(&old_comm);
        MPI_Group_free(&group_);
    }
    this->comm_ = new_comm;
    this->group_ = new_group;
    this->rank_ = new_rank;
    this->size_ = new_size;
#endif
}

bool MPIContext::is_revoked() {
#if (!OMPI_HAS_ULFM)
    TT_THROW("is_revoked() requires MPI ULFM support which is not available in this build");
#else
    int flag = 0;
    // MPI_Comm_test_inter is safe to call even if the communicator is revoked
    // don't need to check error code
    MPI_Comm_test_inter(comm_, &flag);
    return flag != 0;
#endif
}

std::size_t MPIContext::snoop_incoming_msg_size(Rank source, Tag tag) const {
    int size_bytes = 0;
    MPI_Status status;
    MPI_CHECK(MPI_Probe(*source, *tag, comm_, &status));
    MPI_CHECK(MPI_Get_count(&status, MPI_CHAR, &size_bytes));
    return static_cast<std::size_t>(size_bytes);
}

MPIContext::~MPIContext() {
    if (was_mpi_finalized()) {
        return;  // MPI_Finalize() already called
    }
    if (comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_NULL) {
        MPI_Group_free(&group_);
        MPI_Comm_free(&comm_);
    }
}
}  // namespace tt::tt_metal::distributed::multihost
