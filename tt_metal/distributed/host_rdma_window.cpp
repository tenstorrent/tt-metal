// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_rdma_window.hpp"

#include <vector>

#include <mpi.h>

#include <fmt/format.h>

#include "tt_metal/distributed/host_uva_layout.hpp"

namespace tt::tt_metal::experimental {

namespace {

// Staged credit words, one slot per in-flight small put. An Rput reads its origin after
// the call returns, so a caller's temporary would be gone by then.
constexpr uint32_t kWordSlots = 256;

std::string mpi_error_text(const char* what, int rc) {
    char buf[MPI_MAX_ERROR_STRING] = {};
    int len = 0;
    MPI_Error_string(rc, buf, &len);
    return std::string(what) + " failed: " + std::string(buf, static_cast<size_t>(len));
}

}  // namespace

struct RdmaWindow::Impl {
    MPI_Win win = MPI_WIN_NULL;
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 0;
    bool locked = false;
    uint8_t* base = nullptr;
    uint64_t bytes = 0;

    // Slot 0 is reserved so Op::id == 0 can mean "none".
    std::vector<MPI_Request> reqs{MPI_REQUEST_NULL};
    std::vector<uint32_t> free_slots;

    uint64_t words[kWordSlots] = {};
    uint32_t next_word = 0;

    uint32_t acquire() {
        if (!free_slots.empty()) {
            const uint32_t s = free_slots.back();
            free_slots.pop_back();
            return s;
        }
        reqs.push_back(MPI_REQUEST_NULL);
        return static_cast<uint32_t>(reqs.size() - 1);
    }
    void release(uint32_t slot) {
        reqs[slot] = MPI_REQUEST_NULL;
        free_slots.push_back(slot);
    }
};

RdmaWindow::RdmaWindow() : impl_(std::make_unique<Impl>()) {}

bool RdmaWindow::agree_value(const uint64_t local, std::string& err) {
    uint64_t lo = 0;
    uint64_t hi = 0;
    if (const int rc = MPI_Allreduce(&local, &lo, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD); rc != MPI_SUCCESS) {
        err = mpi_error_text("MPI_Allreduce", rc);
        return false;
    }
    if (const int rc = MPI_Allreduce(&local, &hi, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD); rc != MPI_SUCCESS) {
        err = mpi_error_text("MPI_Allreduce", rc);
        return false;
    }
    return lo == hi;
}

bool RdmaWindow::agree(const bool local_ok, std::string& err) {
    const uint8_t mine = local_ok ? 1 : 0;
    uint8_t all = 0;
    if (const int rc = MPI_Allreduce(&mine, &all, 1, MPI_UINT8_T, MPI_MIN, MPI_COMM_WORLD); rc != MPI_SUCCESS) {
        err = mpi_error_text("MPI_Allreduce", rc);
        return false;
    }
    if (all == 0 && local_ok) {
        err = "a peer host failed socket bringup; this host was ready";
    }
    return all != 0;
}

std::unique_ptr<RdmaWindow> RdmaWindow::create(
    uint8_t* region_base, uint64_t region_bytes, uint32_t expect_rank, uint32_t expect_size, std::string& err) {
    err.clear();
    if (region_base == nullptr || region_bytes == 0) {
        err = "RdmaWindow::create: the region is empty";
        return nullptr;
    }

    std::unique_ptr<RdmaWindow> w(new RdmaWindow());
    Impl& im = *w->impl_;
    im.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(im.comm, &im.rank);
    MPI_Comm_size(im.comm, &im.size);
    im.base = region_base;
    im.bytes = region_bytes;

    // Host ids are passed straight to MPI as ranks, so a mismatch silently puts every frame
    // on the wrong peer. Checked before the window exists, so there is nothing to free.
    if (static_cast<uint32_t>(im.rank) != expect_rank || static_cast<uint32_t>(im.size) != expect_size) {
        err = fmt::format(
            "RdmaWindow::create: this rank is {} of {} in MPI_COMM_WORLD but the socket topology calls it "
            "host {} of {}",
            im.rank,
            im.size,
            expect_rank,
            expect_size);
        return nullptr;
    }

    // disp_unit 1: displacements are the byte offsets the layout already computes.
    if (const int rc =
            MPI_Win_create(region_base, static_cast<MPI_Aint>(region_bytes), 1, MPI_INFO_NULL, im.comm, &im.win);
        rc != MPI_SUCCESS) {
        err = mpi_error_text("MPI_Win_create", rc);
        return nullptr;
    }

    // Both checks below are rank-local, but freeing the window is collective. Neither may
    // return early: a rank freeing alone leaves the others waiting in a different collective.
    std::string local_err;

    // The trailing flag needs the target to observe window memory with ordinary loads,
    // which is defined only under the unified model.
    int* model = nullptr;
    int flag = 0;
    // void* by way of the address of the pointer: MPI's attribute out-param is void*, and
    // int** converts to it only through an explicit cast.
    MPI_Win_get_attr(im.win, MPI_WIN_MODEL, static_cast<void*>(&model), &flag);
    if (flag == 0 || model == nullptr || *model != MPI_WIN_UNIFIED) {
        local_err =
            "RdmaWindow::create: this MPI provides a SEPARATE window memory model. The arrival "
            "flag is read with an ordinary load on the target, which that model leaves undefined.";
    }

    // Passive target for the whole run, so no access needs an epoch of its own.
    if (local_err.empty()) {
        if (const int rc = MPI_Win_lock_all(MPI_MODE_NOCHECK, im.win); rc != MPI_SUCCESS) {
            local_err = mpi_error_text("MPI_Win_lock_all", rc);
        } else {
            im.locked = true;
        }
    }

    // The one collective every rank reaches whatever it found. On failure they all destroy
    // `w`, whose destructor unlocks and frees -- one path, taken by everyone or no one.
    std::string peer_err;
    if (!agree(local_err.empty(), peer_err)) {
        err = local_err.empty() ? peer_err : local_err;
        return nullptr;
    }
    return w;
}

RdmaWindow::~RdmaWindow() {
    Impl& im = *impl_;
    if (im.win == MPI_WIN_NULL) {
        return;
    }
    if (im.locked) {
        MPI_Win_unlock_all(im.win);
    }
    MPI_Win_free(&im.win);
}

std::string RdmaWindow::put(const void* src, uint64_t bytes, uint32_t peer_rank, uint64_t target_offset, Op& op) {
    Impl& im = *impl_;
    const uint32_t slot = im.acquire();
    const int rc = MPI_Rput(
        src,
        static_cast<int>(bytes),
        MPI_BYTE,
        static_cast<int>(peer_rank),
        static_cast<MPI_Aint>(target_offset),
        static_cast<int>(bytes),
        MPI_BYTE,
        im.win,
        &im.reqs[slot]);
    if (rc != MPI_SUCCESS) {
        im.release(slot);
        op = Op{};
        return mpi_error_text("MPI_Rput", rc);
    }
    op.id = slot;
    return {};
}

// Fire and forget: a credit is an absolute count, so a lost or duplicated one is a no-op
// and nothing waits on its completion.
std::string RdmaWindow::put_word(uint64_t value, uint32_t peer_rank, uint64_t target_offset) {
    Impl& im = *impl_;
    uint64_t* const staged = &im.words[im.next_word];
    im.next_word = (im.next_word + 1) % kWordSlots;
    *staged = value;

    const int rc = MPI_Put(
        staged,
        sizeof(uint64_t),
        MPI_BYTE,
        static_cast<int>(peer_rank),
        static_cast<MPI_Aint>(target_offset),
        sizeof(uint64_t),
        MPI_BYTE,
        im.win);
    if (rc != MPI_SUCCESS) {
        return mpi_error_text("MPI_Put(credit)", rc);
    }
    // Local completion only: the staging slot has to be free before it wraps round.
    if (const int frc = MPI_Win_flush_local(static_cast<int>(peer_rank), im.win); frc != MPI_SUCCESS) {
        return mpi_error_text("MPI_Win_flush_local(credit)", frc);
    }
    return {};
}

bool RdmaWindow::test(Op& op) {
    Impl& im = *impl_;
    if (!op.valid() || op.id >= im.reqs.size()) {
        return true;
    }
    int done = 0;
    MPI_Test(&im.reqs[op.id], &done, MPI_STATUS_IGNORE);
    if (done != 0) {
        im.release(op.id);
        op = Op{};
        return true;
    }
    return false;
}

std::string RdmaWindow::flush(uint32_t peer_rank) {
    const int rc = MPI_Win_flush(static_cast<int>(peer_rank), impl_->win);
    return rc == MPI_SUCCESS ? std::string{} : mpi_error_text("MPI_Win_flush", rc);
}

std::string RdmaWindow::barrier() {
    const int rc = MPI_Barrier(impl_->comm);
    return rc == MPI_SUCCESS ? std::string{} : mpi_error_text("MPI_Barrier", rc);
}

std::string RdmaWindow::describe() const {
    return fmt::format(
        "MPI RMA, rank {} of {}, unified window over {} MiB", impl_->rank, impl_->size, impl_->bytes >> 20);
}

}  // namespace tt::tt_metal::experimental
